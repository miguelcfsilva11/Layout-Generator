from src.classification.labelers.labeler import Labeler
import numpy as np
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import torch

class CLIPSegLabeler(Labeler):
    """
    A Labeler subclass that uses CLIPSeg for multi-label semantic segmentation
    and grid-based labeling.
    """
    def __init__(self, image: Image.Image, text_labels: list[str] = None, object_labels: list[str] = None, terrain_labels: list[str] = None):
        """
        Args:
            image: PIL Image to segment and label.
        """
        super().__init__(image, text_labels)
        self.seg_prob_maps  = None
        self.label_map      = None
        self.text_labels    = text_labels if text_labels else []
        self.processor      = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model          = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.device         = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.object_labels  = object_labels if object_labels else []
        self.terrain_labels = terrain_labels if terrain_labels else []

    def set_labels(self, text_labels: list[str]):
        """
        Set the textual labels
        """
        self.text_labels = text_labels

    def _compute_segmentation_maps(self):
        """
        Generate raw probability maps for each text label over the image.
        Populates self.seg_prob_maps as a dict[label] -> 2D numpy array.
        """
        prob_maps = {}
        for label in self.text_labels:
            inputs = self.processor(text=label, images=self.image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs  = torch.sigmoid(logits)

            if probs.ndim == 2:
                probs = probs.unsqueeze(0).unsqueeze(0)
            elif probs.ndim == 3:
                probs = probs.unsqueeze(1)

            probs_resized = torch.nn.functional.interpolate(
                probs,
                size=self.image.size[::-1],
                mode='bilinear',
                align_corners=False
            )
            prob_maps[label] = probs_resized.squeeze().cpu().numpy()

        self.seg_prob_maps = prob_maps
        return prob_maps

    def build_label_map(self):
        if self.seg_prob_maps is None:
            self._compute_segmentation_maps()

        h, w         = self.image.size[::-1]
        label_map    = np.full((h, w), None, dtype=object)

        object_maps  = [self.seg_prob_maps[label] for label in self.object_labels]
        terrain_maps = [self.seg_prob_maps[label] for label in self.terrain_labels]

        if object_maps:
            object_stack    = np.stack(object_maps, axis=0)
            best_obj_idx    = np.argmax(object_stack, axis=0)
            best_obj_probs  = np.max(object_stack, axis=0)
            best_obj_labels = np.array(self.object_labels)[best_obj_idx]
        else:
            best_obj_probs  = np.zeros((h, w))
            best_obj_labels = np.full((h, w), None)

        if terrain_maps:
            terrain_stack    = np.stack(terrain_maps, axis=0)
            best_terr_idx    = np.argmax(terrain_stack, axis=0)
            best_terr_probs  = np.max(terrain_stack, axis=0)
            best_terr_labels = np.array(self.terrain_labels)[best_terr_idx]
        else:
            best_terr_probs  = np.zeros((h, w))
            best_terr_labels = np.full((h, w), None)

        for i in range(h):
            for j in range(w):
                if best_obj_probs[i, j] > 0.2:
                    label_map[i, j] = best_obj_labels[i, j]
                else:
                    label_map[i, j] = best_terr_labels[i, j]

        self.label_map = label_map
        return label_map


    def segment_and_label_grid(self):
        """
        Full pipeline: compute label_map, update grid labels, and return grid_labels.
        """
        full_label_map = self.build_label_map()
        self.update_grid_labels(full_label_map)
        return self.get_grid_labels()

    def visualize_overlays(self, alpha: float = 0.5):
        """
        Visualize each label overlay on the image.
        """
        import matplotlib.pyplot as plt

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