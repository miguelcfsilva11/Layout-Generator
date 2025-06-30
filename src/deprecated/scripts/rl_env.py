import gym
import numpy as np
import torch
from torch.distributions import Categorical
from src.models.procedural_generator import ProceduralGenerator
from transformers import BertTokenizer
import random
from scipy.ndimage import label as connected_components

class MapRNN(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=256, num_layers=1):
        super().__init__()

        self.lstm        = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, past_latent_embeddings):
        """
        past_latent_embeddings: Tensor of shape [B, T, input_dim]
        """
        lstm_out, _      = self.lstm(past_latent_embeddings)
        final            = self.output_proj(lstm_out[:, -1])
        return final


class GridGenEnv(gym.Env):
    def __init__(self, model: ProceduralGenerator, tokenizer, label_to_idx, idx_to_label,
                grid_H=64, grid_W=64, grids_per_episode=9):
        super().__init__()
        self.model             = model.eval()
        self.tokenizer         = tokenizer
        self.label_to_idx      = label_to_idx
        self.idx_to_label      = idx_to_label
        self.grid_H            = grid_H
        self.grid_W            = grid_W
        self.num_labels        = len(label_to_idx)
        self.grids_per_episode = grids_per_episode
        self.side              = int(grids_per_episode ** 0.5)
        self.full_grid         = np.full((grid_H * self.side, grid_W * self.side), self.num_labels, dtype=np.int64)
        self.grid              = np.copy(self.full_grid)
        self.mask              = np.zeros_like(self.full_grid)

        self.reset()



    def reset(self):
        self.history      = []
        self.current_step = 0

        self.full_grid.fill(self.num_labels)
        self.grid = np.copy(self.full_grid)
        self.mask = np.zeros_like(self.full_grid)

        y_start, y_end, x_start, x_end = self._current_tile_coords()
        self.mask[y_start:y_end, x_start:x_end] = 1

        self._sample_description()

        return self._get_obs()

    def _current_tile_coords(self):

        row     = self.current_step // self.side
        col     = self.current_step % self.side
        y_start = row * self.grid_H
        y_end   = y_start + self.grid_H
        x_start = col * self.grid_W
        x_end   = x_start + self.grid_W

        return y_start, y_end, x_start, x_end


    def _sample_description(self):

        themes           = ["castle", "forest", "lake", "desert", "village", "mountain", "swamp", "plains", "volcano"]
        adj              = ["dark", "ancient", "quiet", "vast", "mystical", "abandoned", "icy", "lush", "foggy"]
        theme            = random.choice(themes)
        adj_1            = random.choice(adj)
        adj_2            = random.choice([a for a in adj if a != adj_1])
        self.description = f"a {adj_1} {theme} with {adj_2} surroundings"

    def _get_obs(self):
        text_batch                     = self.tokenizer([self.description], padding=True, truncation=True, return_tensors="pt")
        y_start, y_end, x_start, x_end = self._current_tile_coords()

        grid_tile   = self.grid[y_start:y_end, x_start:x_end]
        mask_tile   = self.mask[y_start:y_end, x_start:x_end]
        grid_tensor = torch.tensor(grid_tile).unsqueeze(0)
        mask_tensor = torch.tensor(mask_tile).unsqueeze(0)

        return grid_tensor, mask_tensor, text_batch

    def step(self, grid_completion):
        self.history.append(torch.tensor(grid_completion))

        y_start, y_end, x_start, x_end               = self._current_tile_coords()
        self.full_grid[y_start:y_end, x_start:x_end] = torch.tensor(grid_completion)
        self.current_step                           += 1
        done                                         = self.current_step >= self.grids_per_episode

        if not done:
            self.grid = np.copy(self.full_grid)
            self._update_mask_for_next_tile()
            reward = 0.0
        else:
            full_map = torch.tensor(self.full_grid)
            reward   = self._compute_reward(full_map)

        return self._get_obs(), reward, done, {}

    def _update_mask_for_next_tile(self):

        self.mask                      = np.zeros_like(self.full_grid)
        y_start, y_end, x_start, x_end = self._current_tile_coords()

        if self.current_step == 0:
            self.mask[y_start:y_end, x_start:x_end] = 1
        else:
            if x_start > 0:   self.mask[y_start:y_end, x_start + self.grid_W // 2:x_end] = 1
            elif y_start > 0: self.mask[y_start + self.grid_H // 2:y_end, x_start:x_end] = 1
            else:             self.mask[y_start:y_end, x_start:x_end] = 1


    def _stitch_history(self):
        return torch.tensor(self.full_grid)

    
    def _compute_reward(self, full_map):

        total_labels          = self.num_labels
        used_labels           = full_map.unique()
        valid_labels          = used_labels[used_labels < total_labels]
        diversity_score       = len(valid_labels) / total_labels

        grid_np               = full_map.cpu().numpy()
        structure             = np.ones((3, 3))
        binary_grid           = (grid_np != total_labels).astype(int)
        labeled, num_features = connected_components(binary_grid, structure)
        connectivity_score    = 1.0 / num_features if num_features > 0 else 0.0

        unique, counts        = np.unique(grid_np[grid_np < total_labels], return_counts=True)
        probs                 = counts / counts.sum() if counts.sum() > 0 else [1.0]
        entropy_score         = -np.sum(probs * np.log(probs + 1e-8)) / np.log(len(probs) + 1e-8) 

        reward                = (
            0.4 * diversity_score +
            0.3 * connectivity_score +
            0.3 * entropy_score
        )

        return float(reward)
