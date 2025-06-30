# train_diffusion.py

import math
import pickle
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import clip
from tqdm import tqdm

from src.dataset.grid_dataset import GridDataset
from src.models.components import GridVAE
from src.models.components import DiffusionModel  # Assumes updated DiffusionModel with skip_mlp → latent_channels
from src.generators.proc_generator import ProcGenerator

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# 1. Load and prepare the dataset
# -------------------------------
with open("data/output/maps.pkl", "rb") as f:
    data = pickle.load(f)

print(f"Loaded {len(data)} items from procedural_data.pkl")
# Build a label-to-index mapping from all grid labels
labels = ProcGenerator.labels
label_to_idx = {label: idx for idx, label in enumerate(labels)}
num_labels = len(label_to_idx)

mask_index = num_labels
full_vocab_size = num_labels + 1

grid_H, grid_W = 64, 64
embed_dim = 32
latent_channels = 32
num_heads = 8
ff_dim = 128
time_embed_dim = 128
diffusion_hidden_dim = 256
timesteps = 1000

batch_size = 4
num_epochs = 1
learning_rate = 1e-4

dataset = GridDataset(data, label_to_idx) 
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# -----------------------------------------
# 2. Load the pretrained VAE and freeze it
# -----------------------------------------
vae_ckpt = "vae_best_model_epoch.pth"
vae = GridVAE(
    num_labels=num_labels + 1,  # include <MASK>
    embed_dim=embed_dim,
    grid_H=grid_H,
    grid_W=grid_W,
    latent_channels=latent_channels
)
state_dict = torch.load(vae_ckpt, map_location="cpu")
vae.load_state_dict(state_dict)
vae.to(device).eval()
for p in vae.parameters():
    p.requires_grad = False


# -------------------------------------------------
# 3. Dynamically determine the VAE's latent spatial size
# -------------------------------------------------
with torch.no_grad():
    dummy = torch.zeros(1, grid_H, grid_W, dtype=torch.long, device=device)
    z_dummy, _, _ = vae.encode(dummy)      # shape: (1, latent_channels, H_lat, W_lat)
latent_h, latent_w = z_dummy.shape[-2:]   # e.g. (16, 16) if VAE downsamples 64→16


# ------------------------------------
# 4. Load a frozen CLIP text encoder
# ------------------------------------
# Load CLIP model and get the text encoder
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()
for param in clip_model.parameters():
    param.requires_grad = False

# Get CLIP text embedding dimension
clip_text_dim = clip_model.transformer.width  # 512 for ViT-B/32


# -------------------------------------------------
# 5. Instantiate the DiffusionModel (denoiser) class
# -------------------------------------------------
diffusion_model = DiffusionModel(
    num_labels=full_vocab_size,
    embed_dim=embed_dim,
    latent_channels=latent_channels,
    grid_h=latent_h,
    grid_w=latent_w,
    time_embed_dim=time_embed_dim,
    num_heads=num_heads,
    hidden_dim=diffusion_hidden_dim,
    timesteps=timesteps,
    device=device
)
diffusion_model.to(device)

# Include only diffusion model parameters in optimization
optimizer = optim.Adam(diffusion_model.parameters(), lr=learning_rate)


# --------------------------------------------------
# 6. Helper: cosine beta/alpha schedule (from class)
# --------------------------------------------------
def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, device=device)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)

betas = cosine_beta_schedule(timesteps)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

def encode_text_with_clip(descriptions):
    """
    Encode text descriptions using CLIP.
    
    Args:
        descriptions: List of text strings
    
    Returns:
        text_embeddings: (B, seq_len, clip_text_dim) tensor
    """
    # Tokenize text for CLIP
    text_tokens = clip.tokenize(descriptions, truncate=True).to(device)
    
    with torch.no_grad():
        # Get CLIP text features
        text_features = clip_model.encode_text(text_tokens)  # (B, clip_text_dim)
        text_features = text_features.float()
    
    # Add sequence dimension to match expected format (B, seq_len, clip_text_dim)
    # Since CLIP gives us a single vector per text, we'll unsqueeze to add seq_len=1
    text_embeddings = text_features.unsqueeze(1)  # (B, 1, clip_text_dim)
    
    return text_embeddings

def train():
    # --------------------------------------------------
    # 7. Training loop for the diffusion denoising model
    # --------------------------------------------------
    best_loss = float("inf")
    best_epoch = -1

    for epoch in range(num_epochs):
        diffusion_model.train()
        total_loss = 0.0

        # (Optional) Curriculum masking if implemented inside GridDataset
        dataset.set_curriculum_masking(epoch)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for masked_grids, masks, targets, descriptions in progress_bar:
            # masked_grids:  (B, H, W)  LongTensor with <MASK> where masked
            # masks:         (B, 1, H, W) FloatTensor: 1 = masked/unknown, 0 = known
            # targets:       (B, H, W)  LongTensor of true labels
            # descriptions:  list[str] length B

            masked_grids = masked_grids.to(device).long()
            masks = masks.to(device).float()
            targets = targets.to(device).long()
            B = targets.size(0)

            # 1) Build `grid_labels` for cross-attention: 
            #    We supply masked labels during training so unknown cells use <MASK> index.
            grid_labels = targets.clone()
            grid_labels[masks.squeeze(1) == 1] = mask_index  # (B, H, W)

            # 2) Encode full target grid into VAE latent (frozen)
            with torch.no_grad():
                z_true, mu, logvar = vae.encode(targets)  
                # z_true: (B, latent_channels, H_lat, W_lat)

            # 3) Downsample mask to latent spatial size
            mask_latent = F.interpolate(masks, size=z_true.shape[-2:], mode="nearest")  # (B,1,H_lat,W_lat)

            # 4) Sample timestep t uniformly
            t = torch.randint(0, timesteps, (B,), device=device)

            # 5) Generate noise in latent space
            noise = torch.randn_like(z_true)

            # 6) Compute noisy latent: z_noisy = sqrt(alpha_cumprod[t])*z_true + sqrt(1 - alpha_cumprod[t])*noise
            alpha_cum = alphas_cumprod[t].view(B, 1, 1, 1)
            z_noisy = torch.sqrt(alpha_cum) * z_true + torch.sqrt(1 - alpha_cum) * noise

            # 7) Combine with mask: keep z_true where unmasked, z_noisy where masked
            z_input = z_noisy * mask_latent + z_true * (1 - mask_latent)

            # 8) Get text embeddings from CLIP (frozen)
            text_emb = encode_text_with_clip(descriptions)  # (B, 1, clip_text_dim)

            optimizer.zero_grad()

            # 9) Predict noise via diffusion_model
            pred_noise = diffusion_model(
                latent=z_input,
                mask=mask_latent,
                t=t,
                text_emb=text_emb,
                grid_labels=grid_labels
            )  # → (B, latent_channels, H_lat, W_lat)

            # 10) Compute masked MSE loss
            loss = F.mse_loss(pred_noise * mask_latent, noise * mask_latent, reduction="mean")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"📘 Epoch {epoch+1}, Avg Diffusion Loss: {avg_loss:.6f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            best_path = "data/checkpoints/diffusion_model_best.pth"
            torch.save(diffusion_model.state_dict(), best_path)
            print(f"✅ New best diffusion model saved to {best_path} at epoch {best_epoch}")

    # Always save last epoch
    torch.save(diffusion_model.state_dict(), "data/checkpoints/diffusion_model_last.pth")
    print(f"Training complete. Best epoch: {best_epoch}, Best loss: {best_loss:.6f}")

@torch.no_grad()
def sample_inpaint_no_vae_encode():
    """
    Sample from the diffusion model WITHOUT encoding the known region via the VAE.
    Instead, we pass the discrete `grid_labels` (with <MASK> indices for unknown cells)
    into the diffusion network at every step for cross-attention. We do not clamp the latent.
    After t=0, we decode, argmax over labels, and finally overwrite known cells in the discrete grid.

    Returns:
        completed_grid:   (H, W) LongTensor of discrete labels, where known cells are exactly as in masked_grid
    """

    global betas, alphas, alphas_cumprod, timesteps
    global vae

    ckpt_path = "data/checkpoints/diffusion_model_best.pth"
    state = torch.load(ckpt_path, map_location=device)
    diffusion_model.load_state_dict(state)
    diffusion_model.to(device).eval()
    print(f"✅ Loaded diffusion model from {ckpt_path}")
    
    timesteps = diffusion_model.timesteps  # Use the model's timesteps

    sample_idx = 0
    masked_grid, mask, target_grid, description = dataset[sample_idx]
    timesteps = diffusion_model.timesteps  # Use the model's timesteps
    
    # Ensure shapes match (add batch dimension)
    if masked_grid.ndim == 2:
        masked_grid = masked_grid.unsqueeze(0)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(0)

    diffusion_model.eval()
    text_projector.eval()
    vae.eval()
    clip_model.eval()

    # 1) Encode text with CLIP
    text_emb = encode_text_with_clip([description])  # (1, 1, target_dim)

    # 2) Build `grid_labels` for cross-attention:
    grid_labels = masked_grid.clone().to(device).long()  # (1, H, W)
    grid_labels[mask.squeeze(1).to(device) == 1] = mask_index

    # 3) Prepare the "pure noise" initial latent z_T
    H_lat, W_lat = diffusion_model.grid_h, diffusion_model.grid_w
    z_t = torch.randn(1, diffusion_model.latent_channels, H_lat, W_lat, device=device)

    # 4) Downsample the mask to latent resolution
    mask_latent = F.interpolate(mask.to(device), size=(H_lat, W_lat), mode="nearest")  # (1,1,H_lat,W_lat)

    # 5) Reverse diffusion loop: t = T−1 ... 0
    for t in range(timesteps - 1, -1, -1):
        t_batch = torch.tensor([t], device=device, dtype=torch.long)

        # Predict noise
        pred_noise = diffusion_model(
            latent=z_t,            # current noisy latent
            mask=mask_latent,      # latent‐space mask (1=unknown)
            t=t_batch,             # (1,)
            text_emb=text_emb,     # (1, 1, target_dim)
            grid_labels=grid_labels  # (1, H, W) with <MASK> for unknown
        )

        # Compute the previous latent z_{t−1} using the usual DDPM formula
        a_t = alphas[t]                            # scalar
        a_cum = alphas_cumprod[t]                  # scalar
        a_prev = alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0, device=device)
        beta_t = betas[t]                          # scalar

        pred_x0 = (z_t - beta_t.sqrt() * pred_noise) / a_cum.sqrt()
        if t > 0:
            noise = torch.randn_like(z_t)
            z_prev = (
                a_prev.sqrt() * beta_t * pred_x0
                + a_t.sqrt() * (1 - a_prev) * z_t
                + (beta_t * (1 - a_prev)).sqrt() * noise
            ) / (1 - a_cum)
        else:
            z_prev = pred_x0

        z_t = z_prev

    # 6) Decode latent to logits
    logits = vae.decode(z_t)  # (1, full_vocab_size, H, W)

    # 7) Convert logits → discrete labels
    generated_grid = logits.argmax(dim=1).cpu().squeeze(0)  # (H, W)

    # 8) Finally, overwrite known cells (mask=0) to ensure they remain exactly as `masked_grid`
    mask_2d = mask.squeeze(0).squeeze(0).cpu() 
    mg_2d   = masked_grid.squeeze(0).cpu()
    completed_grid = generated_grid.clone()           
    completed_grid[mask_2d == 0] = mg_2d[mask_2d == 0]
    
    # 9) Visualize or save the completed grid
    proc = ProcGenerator(size=64)
    proc.grid = completed_grid
    vis = proc.create_visualization()  # Returns a matplotlib figure or similar

    # If create_visualization returns a Matplotlib-compatible array, display it:
    if vis is not None:
        plt.figure(figsize=(4, 4))
        plt.imshow(vis)
        plt.axis("off")
        plt.title(f"Completed Grid for: \"{description}\"")
        plt.show()
    else:
        print("Completed grid (label indices):")
        print(completed_grid)

if __name__ == "__main__":
    sample_inpaint_no_vae_encode()