from src.classification.labelers.labeler import Labeler
import numpy as np
import torch
from PIL import Image
import clip
from torchvision import transforms
from src.models.lseg.lseg_net import LSegNet  # adjust import path if needed
import cv2
import matplotlib.pyplot as plt

class LSegLabeler(Labeler):
    """
    Labeler using LSeg with CLIP for full-scene segmentation with no label type distinction.
    """
    def __init__(self, image: Image.Image, text_labels: list[str] = None, model_ckpt_path: str = "data/checkpoints/lseg_minimal_e200.ckpt"):
        
        super().__init__(image, text_labels)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.text_labels = text_labels if text_labels else []

        self.model = LSegNet(
            arch_option=0,
            block_depth=0,
            activation="lrelu",
            backbone="clip_vitl16_384",
            num_features=256,
            aux_layers=[],
        ).to(self.device)

        checkpoint = torch.load(model_ckpt_path, map_location=torch.device('cpu'))

        self.model.load_state_dict(checkpoint)
        self.model = self.model.float()
        self.model.clip_pretrained = self.model.clip_pretrained.float()  # <-- This is key

        self.model.eval()

        tokens = clip.tokenize(self.text_labels).to(self.device)
        with torch.no_grad():
            self.text_features = self.model.clip_pretrained.encode_text(tokens)


        self.seg_prob_maps = None
        self.label_map = None

        self.transform = transforms.Compose([
            transforms.Resize((480, 480)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def _compute_segmentation_maps(self):
        img_tensor = self.transform(self.image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_feats = self.model(img_tensor)
            image_feats /= image_feats.norm(dim=1, keepdim=True)
            logits = self.model.logit_scale * torch.einsum("bchw,nc->bnhw", image_feats, self.text_features)

        probs = torch.sigmoid(logits.squeeze(0)).cpu().numpy()
        self.seg_prob_maps = {label: probs[i] for i, label in enumerate(self.text_labels)}
        
        original_size = self.image.size
        resized_maps = {}
        for label, prob_map in self.seg_prob_maps.items():
            resized = cv2.resize(prob_map, original_size, interpolation=cv2.INTER_LINEAR)
            resized_maps[label] = resized
        self.seg_prob_maps = resized_maps

        return self.seg_prob_maps

    def build_label_map(self):
        if self.seg_prob_maps is None:
            self._compute_segmentation_maps()

        h, w = next(iter(self.seg_prob_maps.values())).shape
        label_map = np.full((h, w), None, dtype=object)

        stacked = np.stack([self.seg_prob_maps[label] for label in self.text_labels])
        best_idx = np.argmax(stacked, axis=0)
        best_labels = np.array(self.text_labels)[best_idx]

        for i in range(h):
            for j in range(w):
                label_map[i, j] = best_labels[i, j]

        self.label_map = label_map
        return label_map

    def segment_and_label_grid(self):
        full_label_map = self.build_label_map()
        self.update_grid_labels(full_label_map)
        return self.get_grid_labels()

    def visualize_overlays(self, alpha=0.5):

        if self.seg_prob_maps is None:
            self._compute_segmentation_maps()

        num      = len(self.text_labels) + 1
        fig, axs = plt.subplots(1, num, figsize=(5 * num, 5))
        axs[0].imshow(self.image)
        axs[0].set_title('Original')
        axs[0].axis('off')

        for idx, label in enumerate(self.text_labels, start=1):
            mask = self.seg_prob_maps[label]
            axs[idx].imshow(self.image)
            axs[idx].imshow(mask, cmap='viridis', alpha=alpha)
            axs[idx].set_title(label)
            axs[idx].axis('off')

        plt.tight_layout()
        plt.show()
