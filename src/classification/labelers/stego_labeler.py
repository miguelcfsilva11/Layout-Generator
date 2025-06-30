from src.classification.labelers.labeler import Labeler
from torchvision.transforms.functional import to_tensor, normalize
from torchvision import transforms
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import clip
from pydensecrf.utils import unary_from_softmax
from src.models.STEGO.src.crf import dense_crf
from src.models.STEGO.src.utils import get_transform

from src.models.STEGO.src.train_segmentation import LitUnsupervisedSegmenter
from os.path import join
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torchvision import transforms

class STEGOLabeler(Labeler):

    def __init__(self, image: Image.Image, text_labels: list[str], model_ckpt_path: str):

        super().__init__(image, text_labels)

        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.text_labels = text_labels
        self.model       = LitUnsupervisedSegmenter.load_from_checkpoint(model_ckpt_path).to(self.device).eval()

        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.segment_map = None
        self.label_map = None

    def segment_image(self):
        original_size = self.image.size
        transform     = get_transform(448, False, "center")
        img_tensor    = transform(self.image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            code1 = self.model(img_tensor)
            code2 = self.model(img_tensor.flip(dims=[3]))
            code  = (code1 + code2.flip(dims=[3])) / 2
            code  = F.interpolate(code, img_tensor.shape[-2:], mode='bilinear', align_corners=False)

            linear_probs = torch.log_softmax(self.model.linear_probe(code), dim=1).cpu()
            single_img   = img_tensor[0].cpu()
            pred         = dense_crf(single_img, linear_probs[0]).argmax(0)
            pred_tensor  = torch.tensor(pred).unsqueeze(0).unsqueeze(0).float()
            upsampled    = F.interpolate(pred_tensor, size=(original_size[1], original_size[0]), mode='nearest') 
            upsampled    = upsampled.squeeze().long().numpy()  

        self.segment_map = upsampled
        return upsampled

    def assign_labels_with_clip(self):
        if self.segment_map is None:
            self.segment_image()

        np_img      = np.array(self.image.convert("RGB"))
        h, w        = self.segment_map.shape
        label_map   = np.empty((h, w), dtype=object)
        text_tokens = clip.tokenize(self.text_labels).to(self.device)
        
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        unique_segments = np.unique(self.segment_map)
        for seg_id in unique_segments:
            mask = self.segment_map == seg_id
            coords = np.argwhere(mask)
            if coords.size == 0:
                continue

            ymin, xmin  = coords.min(axis=0)
            ymax, xmax  = coords.max(axis=0)
            cropped     = self.image.crop((xmin, ymin, xmax + 1, ymax + 1))
            image_input = self.clip_preprocess(cropped).unsqueeze(0).to(self.device)

            with torch.no_grad():

                image_features = self.clip_model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                similarity     = (100.0 * image_features @ text_features.T).squeeze(0)
                best_idx       = similarity.argmax().item()
                best_label     = self.text_labels[best_idx]

            label_map[mask]    = best_label

        self.label_map = label_map
        for y in range(448):
            for x in range(448):
                if len(label_map[y, x]) == 0:
                    print(f"⚠️ Empty cell at ({y}, {x})")

        return label_map

    def segment_and_label_grid(self):
        self.assign_labels_with_clip()
        self.update_grid_labels(self.label_map)
        return self.get_grid_labels()

    def visualize_segmentation(self, alpha=0.5):


        if self.label_map is None:
            self.assign_labels_with_clip()

        label_to_color = {
            label: np.random.randint(0, 255, (3,), dtype=np.uint8)
            for label in self.text_labels
        }

        color_mask = np.zeros((*self.label_map.shape, 3), dtype=np.uint8)
        for label, color in label_to_color.items():
            color_mask[self.label_map == label] = color

        blended = (
            0.5 * np.array(self.image.convert("RGB")) + 0.5 * color_mask
        ).astype(np.uint8)

        plt.figure(figsize=(8, 8))
        plt.imshow(blended)
        plt.axis('off')
        plt.title("STEGO + CLIP Labeling")
        plt.show()
