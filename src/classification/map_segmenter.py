from src.classification.sam import SAM2Model
from src.classification.labelers.slic_labeler import SlicLabeler
from src.classification.labelers.cell_labeler import CellLabeler
import numpy as np
import clip
import torch
from PIL import Image

n_segments = [300]

class MapSegmenter:
    def __init__(
        self,
        sam_model_name : str  = "facebook/sam2-hiera-tiny",
        device         : str  = "cpu",
        clip_model_name: str  = "ViT-B/32",
        use_cell       : bool = False,
        use_adaptive   : bool = False,
        slic_growing   : bool = False,
    ):

        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=device)

        self.sam             = SAM2Model(sam_model_name, device)
        self.device          = device
        self.use_cell        = use_cell
        self.use_adaptive    = use_adaptive
        self.slic_growing    = slic_growing
        self.slic            = None
        self.grid_labels     = None
        self.combined_labels = None

    def show_sam_masks(self, sam_masks, pil_img, np_img):

        base    = pil_img.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))

        for mask_dict in sam_masks:
            mask       = mask_dict["segmentation"]
            mask_img   = Image.fromarray((mask * 255).astype(np.uint8)).convert("L")
            green_mask = Image.new("RGBA", base.size, (0, 255, 0, 100))
            overlay.paste(green_mask, (0, 0), mask_img)

        result = Image.alpha_composite(base, overlay)
        result.show()

    def process_image(
        self,
        image_path    : str,
        object_labels : list[str],
        terrain_labels: list[str],
        grid_size     : int = 16,
        n_segments    : list[int] = n_segments,
        guided        : bool = True,
    ):
        self.object_labels  = object_labels
        self.terrain_labels = terrain_labels
        pil_img, np_img     = self.sam.load_image(image_path)
        height, width       = np_img.shape[:2]

        if not guided:

            sam_masks         = self.sam.mask_generator.generate(np_img)
            combined_obj_mask = self._create_combined_sam_mask(sam_masks, height, width)
            background_mask   = ~combined_obj_mask

            self._process_background(pil_img, combined_obj_mask, background_mask, n_segments, terrain_labels)
            self._label_sam_regions(sam_masks, np_img)

        else:

            obj_masks         = self._clip_guided_sam_masks(np_img, object_labels)
            combined_obj_mask = np.zeros((height, width), dtype=bool)
            for info in obj_masks.values():
                combined_obj_mask |= info["mask"]
            background_mask   = ~combined_obj_mask

            self._process_background(pil_img, combined_obj_mask, background_mask, n_segments, terrain_labels)

            for label, info in obj_masks.items():
                mask                         = info["mask"].astype(bool)
                self.slic.pixel_labels[mask] = label
                self.slic.label_to_masks.setdefault(label, np.zeros_like(mask, bool))[mask] = True

        self._create_final_grid(grid_size)
        return self.grid_labels

    def _create_combined_sam_mask(self, sam_masks, height, width):
        combined_mask = np.zeros((height, width), dtype=bool)
        for mask in sam_masks:
            combined_mask |= mask["segmentation"]
        return combined_mask

    def _process_background(
        self, pil_img, combined_sam_mask, background_mask, n_segments, terrain_labels
    ):
        if self.use_cell:
            self.slic = CellLabeler(pil_img, n_segments, terrain_labels, combined_sam_mask)
        else:
            self.slic = SlicLabeler(pil_img, n_segments, terrain_labels, combined_sam_mask)

        if self.use_adaptive:
            self.slic.adaptive_labelling(textual=False)
        else:
            if self.use_cell:
                self.slic.labelling(textual=False)
            else:
                self.slic.labelling(
                    n_segments[0], flag=False, output=False, growing=self.slic_growing
                )

        self.slic.pixel_labels[~background_mask] = ""

    def _label_sam_regions(self, sam_masks, np_img):
        text_embeddings = self.slic.compute_text_features(self.object_labels)

        for mask in sam_masks:
            seg                    = mask["segmentation"]
            region_img             = self.slic.image_from_mask(seg, np_img)
            region_embedding       = self.slic.retrieve_image_embedding(region_img)
            label, _               = self.slic.retrieve_closest_label(
                region_embedding, self.object_labels, text_embeddings
            )

            self.slic.pixel_labels[seg] = label
            self.slic.label_to_masks.setdefault(label, seg.copy())

    def _clip_guided_sam_masks(
        self, np_img, labels, crop_size=(128, 128), stride=32, score_threshold=0.3
    ):
        h, w           = np_img.shape[:2]
        crop_w, crop_h = crop_size
        results        = {}

        for txt in labels:
            candidate_points = []
            best_score       = -1
            best_box         = None

            for y in range(crop_h // 2, h, stride):
                for x in range(crop_w // 2, w, stride):
                    left    = max(x - crop_w // 2, 0)
                    upper   = max(y - crop_h // 2, 0)
                    right   = min(x + crop_w // 2, w)
                    lower   = min(y + crop_h // 2, h)
                    patch   = Image.fromarray(np_img).crop((left, upper, right, lower))
                    clip_in = self.clip_preprocess(patch).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        img_f  = self.clip_model.encode_image(clip_in)
                        txt_f  = self.clip_model.encode_text(
                            clip.tokenize([txt]).to(self.device)
                        )
                        img_f /= img_f.norm(dim=-1, keepdim=True)
                        txt_f /= txt_f.norm(dim=-1, keepdim=True)
                        score  = (img_f @ txt_f.T).item()

                    if score >= score_threshold:
                        candidate_points.append(((x, y), score))
                    if score > best_score:
                        best_score = score
                        best_box = (left, upper, right, lower)

            if not candidate_points:
                results[txt] = {"mask": np.zeros((h, w), dtype=bool), "score": 0.0, "point": None, "bbox": None}
                continue

            coords = np.array([pt for pt, _ in candidate_points])
            labels_arr = np.ones(len(coords))
            masks, scores, _ = self.sam.predict(point_coords=coords, point_labels=labels_arr)
            final_mask = np.any(masks > 0.5, axis=0)
            results[txt] = {
                "mask": final_mask,
                "score": best_score,
                "point": coords[scores.argmax()].tolist(),
                "bbox": best_box,
            }

        return results

    def _create_final_grid(self, grid_size):
        height, width = self.slic.pixel_labels.shape
        self.grid_labels = np.full((height // grid_size, width // grid_size), "", dtype=object)

        self.slic.update_grid_labels(self.slic.pixel_labels)
        self.grid_labels = self.slic.grid_labels
