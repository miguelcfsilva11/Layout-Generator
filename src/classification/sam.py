import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2


np.random.seed(3)


class SAM2Model:
    def __init__(self, model_name="facebook/sam2-hiera-tiny", device="cpu"):

        self.device          = torch.device(device)
        self.predictor       = SAM2ImagePredictor.from_pretrained(model_name)
        self.predictor.model = self.predictor.model.to(self.device)
        self.image           = None
        self.image_np        = None
        self.mask_generator  = SAM2AutomaticMaskGenerator(self.predictor.model)


    def load_image(self, image_path):

        self.image    = Image.open(image_path)
        self.image_np = np.array(self.image.convert("RGB"))
        self.predictor.set_image(self.image_np)
        return self.image, self.image_np

    def predict(self, point_coords=None, point_labels=None, box_coords=None, multimask_output=True):

        with torch.inference_mode(), torch.autocast(self.device.type, dtype=torch.bfloat16):
            masks, scores, logits = self.predictor.predict(

                point_coords      = point_coords,
                point_labels      = point_labels,
                box               = box_coords,
                multimask_output  = multimask_output,
            )

        sorted_ind = np.argsort(scores)[::-1]
        return masks[sorted_ind], scores[sorted_ind], logits[sorted_ind]

    def show_masks(self, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):

        if self.image is None:
            raise ValueError("No image loaded. Please call load_image() first.")
            
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10, 10))
            plt.imshow(self.image)

            self.show_mask(mask, plt.gca(), borders=borders)

            if point_coords is not None:
                if input_labels is None:
                    raise ValueError("Input labels must be provided along with point coordinates.")
                self.show_points(point_coords, input_labels, plt.gca())
            if box_coords is not None:
                self.show_box(box_coords, plt.gca())
            if len(scores) > 1:
                plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)

            plt.axis('off')
            plt.show()

    @staticmethod
    def show_mask(mask, ax, random_color=False, borders=True):


        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])

        h, w       = mask.shape[-2:]
        mask       = mask.astype(np.uint8)
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

        if borders:

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours    = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            mask_image  = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)

        ax.imshow(mask_image)

    @staticmethod
    def show_points(coords, labels, ax, marker_size=375):


        pos_points = coords[labels==1]
        neg_points = coords[labels==0]

        ax.scatter(pos_points[:, 0], pos_points[:, 1],
                   color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1],
                   color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

    @staticmethod
    def show_box(box, ax):

        x0, y0  = box[0], box[1]
        w, h    = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

    @staticmethod
    def show_anns(anns, borders=True):
        if len(anns) == 0:
            return

        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:, :, 3] = 0
        for ann in sorted_anns:

            m          = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.5]])
            img[m]     = color_mask 

            if borders:
                import cv2
                contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
                contours    = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
                cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 
        ax.imshow(img)

