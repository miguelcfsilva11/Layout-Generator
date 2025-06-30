from src.classification.labelers.labeler import Labeler
from fast_slic import Slic
import numpy as np
from src.classification.sam import SAM2Model
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

class SlicLabeler(Labeler):

    def __init__(self, image, n_segments, text_labels, sam_mask = None):

        super().__init__(image, text_labels)
        if type(n_segments) != list or 0 in n_segments:
            raise ValueError("Invalid number of segments.")

        self.assignments    = [Slic(num_components=x, compactness=8).iterate(self.image_np) for x in n_segments]
        self.pixel_labels   = np.full_like(self.assignments[0], "", dtype=object)
        self.assignments    = {n_segments[i]: self.assignments[i] for i in range(len(n_segments))}
        self.label_to_masks = {}
        self.sam_mask       = sam_mask


    def show_slic_segmentation_overlay(self, num_components=10):
        assignment = self.assignments[num_components]
        image_np   = self.image_np.copy()
        overlay    = mark_boundaries(image_np, assignment, color=(0, 1, 0), mode='thick')  # Green in RGB normalized [0, 1]

        plt.figure(figsize=(8, 8))
        plt.imshow(overlay)
        plt.title(f"SLIC Segmentation Overlay - {num_components} segments")
        plt.axis('off')
        plt.show()

    def adaptive_labelling(self, output = True):

        text_embeddings = self.compute_text_features(self.text_labels)
        confidence_map  = np.zeros((self.height, self.width), dtype=np.float32)
        label_map       = np.full((self.height, self.width), "", dtype=object)
        level_map       = np.full((self.height, self.width), -1, dtype=np.int32)
        exclusion_mask  = self.sam_mask if self.sam_mask is not None else np.zeros_like(self.pixel_labels, dtype=bool)

        for key in self.assignments:
            assignment      = self.assignments[key]
            unique_segments = np.unique(assignment)
            for seg in unique_segments:
                
                block_mask      = (assignment == seg)
                block_mask     &= ~exclusion_mask

                if not np.any(block_mask):
                    continue

                y_indices, x_indices = np.where(block_mask)
                x_min, x_max = x_indices.min(), x_indices.max()
                y_min, y_max = y_indices.min(), y_indices.max()

                if x_max == x_min or y_max == y_min:
                    continue

                block_img                 = self.image_from_mask(block_mask, self.image_np)
                block_embedding           = self.retrieve_image_embedding(block_img)
                closest_label, confidence = self.retrieve_closest_label(block_embedding, self.text_labels, text_embeddings)        
                current_confidences       = confidence_map[y_indices, x_indices]
                update_mask               = confidence > current_confidences

                if np.any(update_mask):
                    confidence_map[y_indices[update_mask], x_indices[update_mask]] = confidence
                    label_map[y_indices[update_mask], x_indices[update_mask]]      = closest_label
                    level_map[y_indices[update_mask], x_indices[update_mask]]      = key
        
        for lbl in self.text_labels:
            mask = np.where(label_map == lbl, 255, 0).astype(np.uint8)
            if np.any(mask):
                self.label_to_masks[lbl] = self.image_from_mask(mask, self.image_np)

        self.update_grid_labels(label_map)
        for label in self.text_labels:
            mask = label_map == label
            self.pixel_labels[mask] = label
    
        if output:
            self.output_labelling("adaptive_")
    

    def labelling(self, num_components=10, flag=True, output = True, growing=True):

        unique_labels       = np.unique(self.assignments[num_components])
        self.assignment     = self.assignments[num_components]        
        if growing:

            segment_embeddings  = {}

            for label in unique_labels:
                
                labels_image              = self.image_from_labels([label])
                segment_embeddings[label] = self.retrieve_image_embedding(labels_image)

            self.calculate_neighbors(self.assignments[num_components], unique_labels)  

            self.parent           = {label: label for label in unique_labels}
            self.group_segments   = {label: [label] for label in unique_labels}
            self.group_embeddings = {label: segment_embeddings[label] for label in unique_labels}
            threshold             = 0.9
            
            if flag:
                self.improved_merging(unique_labels, threshold)
            else: self.normal_merging(unique_labels, threshold)

            merged_groups    = {}
            for label in unique_labels:
                root = self.find(label)
                merged_groups.setdefault(root, []).append(label)
                
            self.merged_groups = merged_groups
        else:
            
            self.merged_groups = {label: [label] for label in np.unique(self.assignments[num_components])}

        exclusion_mask     = self.sam_mask if self.sam_mask is not None else np.zeros_like(self.assignment, dtype=bool)
        final_regions      = self.compute_final_regions(self.merged_groups)
        self.pixel_labels  = np.full_like(self.assignment, "", dtype=object)

        for region_img in final_regions:
    
            region_img_np      = np.array(region_img)                
            original_mask      = region_img_np[..., 3] == 255
            clean_mask         = original_mask & ~exclusion_mask
            
            if not np.any(clean_mask):
                continue

            y_inds, x_inds     = np.where(clean_mask)
            x_min, x_max       = x_inds.min(), x_inds.max()
            y_min, y_max       = y_inds.min(), y_inds.max()

            if x_max == x_min or y_max == y_min:
                continue

            region_crop      = region_img.crop((x_min, y_min, x_max+1, y_max+1))
            region_embedding = self.retrieve_image_embedding(region_crop)
            closest_label, _ = self.retrieve_closest_label(
                region_embedding, 
                self.text_labels, 
                self.compute_text_features(self.text_labels)
            )


            self.pixel_labels[clean_mask] = closest_label

            if closest_label in self.label_to_masks:
                self.label_to_masks[closest_label] |= clean_mask
            else:
                self.label_to_masks[closest_label] = clean_mask
                
        self.update_grid_labels(self.pixel_labels)
        if output:
            self.output_labelling(f"growing_{num_components}_")
    