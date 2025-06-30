import pickle
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from src.models.procedural_generator import ProceduralGenerator
from transformers import BertTokenizer
from tqdm import tqdm
from matplotlib import pyplot as plt
from src.dataset.grid_dataset import GridDataset
from src.models.components import GridVAE
import torch.nn as nn


def expand_embedding_weights(old_embedding, new_vocab_size):
    old_weight = old_embedding.weight.data
    new_embedding = nn.Embedding(new_vocab_size, old_weight.shape[1])
    new_embedding.weight.data[:old_weight.shape[0]] = old_weight
    new_embedding.weight.data[old_weight.shape[0]:].uniform_(-0.02, 0.02)
    return new_embedding

device = "cpu"

with open("data/output/specific_processed_data.pkl", "rb") as f:
    data = pickle.load(f)

all_labels                                      = {cell for item in data for row in item["grid_labels"] for cell in row if cell is not None}
label_to_idx                                    = {label: i for i, label in enumerate(sorted(all_labels))}

grid_H, grid_W                                  = 64, 64
embed_dim, num_heads, ff_dim                    = 64, 4, 128
time_embed_dim, diffusion_hidden_dim, timesteps = 128, 256, 1000
batch_size, num_epochs                          = 4, 10

dataset         = GridDataset(data, label_to_idx)
dataloader      = DataLoader(dataset, batch_size=batch_size, shuffle=True)
num_labels_with_mask = len(label_to_idx) + 1

vae_ckpt = "data/checkpoints/vae_best_model.pth"
vae      = GridVAE(len(label_to_idx), embed_dim, grid_H, grid_W, latent_channels=32)
state_dict = torch.load(vae_ckpt, map_location="cpu")  # ✅ Always load to CPU first
vae.load_state_dict(state_dict)

vae.embedding.embedding = expand_embedding_weights(vae.embedding.embedding, len(label_to_idx) + 1)

vae.to(device)
vae.eval()

for p in vae.parameters():
    p.requires_grad = False

model           = ProceduralGenerator(embed_dim, num_heads, ff_dim, time_embed_dim,
                            diffusion_hidden_dim, (grid_H, grid_W), len(label_to_idx), grid_H, grid_W, 32, timesteps)
model.vae       = vae

tokenizer       = BertTokenizer.from_pretrained("bert-base-uncased")
optimizer = optim.Adam(model.diffusion.parameters(), lr=1e-4)

#checkpoint_path = "data/checkpoints/model_125.pth"

#model.load_state_dict(torch.load(checkpoint_path, map_location=device))
#print(f"✅ Loaded checkpoint: {checkpoint_path}")
model.to(device)

def visualize_comparison(prediction, original, label_to_idx):

    idx_to_label = {v: k for k, v in label_to_idx.items()}
    pred         = prediction.argmax(dim=1)[0].cpu().numpy()
    orig         = original[0].cpu().numpy()
    fig, axs     = plt.subplots(1, 2)
    key_pressed  = {"continue": False}

    axs[0].imshow(orig, cmap="tab20")
    axs[0].set_title("Original Grid")
    axs[1].imshow(pred, cmap="tab20")
    axs[1].set_title("Predicted Grid")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()

    def on_key(event):
        if event.key == 'tab':
            key_pressed["continue"] = True
            plt.close()

    fig.canvas.mpl_connect('key_press_event', on_key)
    print("👀 Press Tab while the image is open to continue training...")
    plt.show()
    while not key_pressed["continue"]:
        pass
    
def train():
    best_loss  = float('inf')
    best_epoch = -1
    for epoch in range(num_epochs):
        total_loss = 0

        dataset.set_curriculum_masking(epoch)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}", leave=False)

        for masked_grids, masks, targets, descriptions in progress_bar:

            masked_grids = masked_grids.to(device)
            masks        = masks.to(device)
            targets      = targets.to(device)
            text_batch   = tokenizer(list(descriptions), padding=True, truncation=True, return_tensors="pt").to(masked_grids.device)
            t            = torch.randint(0, timesteps, (masked_grids.size(0),), device=masked_grids.device)
            loss         = model.train_step(optimizer, masked_grids, t, text_batch, targets, masks)
            total_loss  += loss
            progress_bar.set_postfix(loss=f"{loss:.4f}")    

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss  = avg_loss
            best_epoch = epoch + 1
            print(f"🔁 New best model at epoch {best_epoch} with loss {best_loss:.4f}")
            torch.save(model.state_dict(), f"data/checkpoints/procedural_generator_model_epoch_{epoch + 1}.pth")

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), "data/checkpoints/last_procedural_generator_model.pth")
    print("Model saved")

def sample():
    """
    To run this method, set batch_size to 1 in the dataloader.
    """

    model.eval()

    for masked_grids, masks, targets, descriptions in dataloader:
        masked_grids = masked_grids.to(device)
        masks        = masks.to(device)
        targets      = targets.to(device)
        text_batch   = tokenizer(list(descriptions), padding=True, truncation=True, return_tensors="pt").to(device)
        t            = torch.randint(0, timesteps, (1,), device=device)

        with torch.no_grad():
            predictions, _ = model(masked_grids, t, text_batch, masks, use_blue_noise=False)

        visualize_comparison(predictions, targets, label_to_idx)


if __name__ == "__main__":
    train()
