from torch.utils.data import Dataset
import numpy as np
import torch

class GridDataset(Dataset):
    def __init__(self, data, label_to_idx, latent_h, latent_w, mode="masked"):

        self.label_to_idx             = label_to_idx
        self.num_labels               = len(label_to_idx)
        self.data                     = data
        self.mode                     = mode
        self.mask_label               = self.num_labels
        self.vocab_size               = self.num_labels + 1
        self.latent_h                 = latent_h
        self.latent_w                 = latent_w
        sample_grid                   = data[0]["grid_labels"]
        self.grid_h, self.grid_w      = sample_grid.shape
        self.ds_h                     = self.grid_h             // self.latent_h
        self.ds_w                     = self.grid_w             // self.latent_w
        self.current_stage            = 0

    def set_curriculum_masking(self, epoch):
        """
        Update the current curriculum stage based on epoch:
          epoch < 2  → stage 0 (small patch)
          epoch < 4  → stage 1 (medium)
          epoch < 6  → stage 2 (large)
          otherwise  → stage 3 (quadrant)
        """
        if epoch < 2:
            self.current_stage = 0
        elif epoch < 4:
            self.current_stage = 1
        elif epoch < 6:
            self.current_stage = 2
        else:
            self.current_stage = 3

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        item        = self.data[idx]
        raw_grid    = item["grid_labels"]
        description = item.get("description", "") or ""
        grid        = np.vectorize(lambda x: self.label_to_idx.get(x, self.num_labels))(raw_grid)
        grid        = grid.astype(np.int64)
        h, w        = grid.shape

        if self.mode == "full":
            masked_grid = grid.copy()
            mask        = np.zeros((h, w), dtype=np.int64)
        else:
            stage       = self.current_stage
            mask        = np.zeros((h, w), dtype=np.int64)

            if stage == 0:
                ph   = self.ds_h
                pw   = self.ds_w
                top  = np.random.randint(0, h - ph + 1)
                left = np.random.randint(0, w - pw + 1)
                mask[top:top+ph, left:left+pw] = 1

            elif stage == 1:
                ph   = self.ds_h * 2
                pw   = self.ds_w * 2
                top  = np.random.randint(0, h - ph + 1)
                left = np.random.randint(0, w - pw + 1)

                mask[top:top+ph, left:left+pw] = 1
            elif stage == 2:

                ph   = (self.latent_h // 2) * self.ds_h
                pw   = (self.latent_w // 2) * self.ds_w
                top  = np.random.randint(0, h - ph + 1)
                left = np.random.randint(0, w - pw + 1)
                mask[top:top+ph, left:left+pw] = 1
            else:
                if np.random.rand() < 0.5:
                    if np.random.rand() < 0.5:
                        mask[:h//2, :] = 1
                    else:
                        mask[h//2:, :] = 1
                else:
                    if np.random.rand() < 0.5:
                        mask[:, :w//2] = 1
                    else:
                        mask[:, w//2:] = 1

            masked_grid = grid.copy()
            masked_grid[mask == 1] = self.num_labels

        masked_tensor = torch.from_numpy(masked_grid)
        mask_tensor   = torch.from_numpy(mask).unsqueeze(0).float()
        true_tensor   = torch.from_numpy(grid)

        return masked_tensor, mask_tensor, true_tensor, description
