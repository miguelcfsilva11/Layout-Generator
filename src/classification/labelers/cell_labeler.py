import numpy as np
from src.classification.labelers.labeler import Labeler

class CellLabeler(Labeler):
    def __init__(self, image, n_segments, text_labels=None, sam_mask=None):
        super().__init__(image, text_labels)
        self.sam_mask       = sam_mask if sam_mask is not None else np.zeros(self.image_np.shape[:2], bool)
        h, w                = self.image_np.shape[:2]
        self.pixel_labels   = np.full((h, w), "", dtype=object)
        self.label_to_masks = {}
        self.grid_labels    = None

    def labelling(self, textual=False):
        """
        Label only background (terrain) grid cells and update pixel-level labels and masks.
        """

        gh, gw = self.grid_size
        h, w   = self.image_np.shape[:2]
        cell_h = h // gh
        cell_w = w // gw

        if self.sam_mask is None:
            self.sam_mask   = np.zeros((h, w), bool)
        background_mask     = ~self.sam_mask

        if textual:
            text_embeddings = self.compute_text_features(self.text_labels)
        self.grid_labels    = np.full((gh, gw), None, dtype=object)

        for i in range(gh):
            for j in range(gw):
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                cell_bg_mask = background_mask[y1:y2, x1:x2]
                if not cell_bg_mask.any():
                    continue
                cell_img = self.crop_cell(self.image, i, j)
                if textual:
                    query_features = self.retrieve_image_embedding(cell_img)
                    best_label, _  = self.retrieve_closest_label(
                        query_features,
                        self.text_labels,
                        text_embeddings
                    )
                else:
                    best_label, _ = self.find_best_tile(cell_img)

                self.grid_labels[i, j]                        = best_label
                self.pixel_labels[y1:y2, x1:x2][cell_bg_mask] = best_label
                mask                                          = (self.pixel_labels == best_label)
                self.label_to_masks.setdefault(best_label, np.zeros_like(mask, bool))[mask] = True

        self.update_grid_labels(self.pixel_labels)

    def adaptive_labelling(self, k=1, temp=0.02, max_block_size=16, textual=False):
        """
        Adaptively label only background (terrain) grid cells in multi-scale blocks.
        """
        gh, gw = self.grid_size
        h, w   = self.image_np.shape[:2]
        cell_h = h // gh
        cell_w = w // gw

        if self.sam_mask is None:
            self.sam_mask = np.zeros((h, w), bool)
        background_mask = ~self.sam_mask
        grid_scores     = [[(None, np.inf) for _ in range(gw)] for _ in range(gh)]
 
        if textual:
            text_embeddings = self.compute_text_features(self.text_labels)

        current_block = k
        while current_block <= max_block_size:
            for i in range(gh - current_block + 1):
                for j in range(gw - current_block + 1):

                    y1, y2   = i * cell_h, (i + current_block) * cell_h
                    x1, x2   = j * cell_w, (j + current_block) * cell_w
                    block_bg = background_mask[y1:y2, x1:x2]

                    if not block_bg.any():
                        continue

                    block_img = self.crop_cell(self.image, i, j, current_block)
                    if textual:
                        query_features   = self.retrieve_image_embedding(block_img)
                        best_label, dist = self.retrieve_closest_label(
                            query_features,
                            self.text_labels,
                            text_embeddings
                        )
                    else:
                        best_label, dist = self.find_best_tile(block_img)

                    for di in range(current_block):
                        for dj in range(current_block):
                            x, y = i + di, j + dj

                            if not background_mask[y*cell_h:(y+1)*cell_h, x*cell_w:(x+1)*cell_w].any():
                                continue
                            prev_label, prev_score = grid_scores[x][y]
                            if dist < prev_score - temp * current_block:
                                grid_scores[x][y] = (best_label, dist)
            current_block *= 2

        self.grid_labels = np.full((gh, gw), None, dtype=object)
        for i in range(gh):
            for j in range(gw):
                lbl, _ = grid_scores[i][j]
                if lbl is None:
                    continue
                self.grid_labels[i, j] = lbl

        for i in range(gh):
            for j in range(gw):
                lbl = self.grid_labels[i, j]
                if lbl is None:
                    continue
                y1, y2          = i * cell_h, (i + 1) * cell_h
                x1, x2          = j * cell_w, (j + 1) * cell_w
                cell_bg_mask    = background_mask[y1:y2, x1:x2]

                self.pixel_labels[y1:y2, x1:x2][cell_bg_mask] = lbl
                mask                                          = (self.pixel_labels == lbl)
                self.label_to_masks.setdefault(lbl, np.zeros_like(mask, bool))[mask] = True

        self.update_grid_labels(self.pixel_labels)
