import torch
import torch.nn as nn
import numpy as np
import transformers
from src.models.components import GridVAE, DiffusionModel, NoiseGenerator
from torch.cuda.amp import autocast, GradScaler

def cosine_beta_schedule(timesteps, s=0.008):

    steps          = timesteps + 1
    x              = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas          = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

    return torch.clamp(betas, 0.0001, 0.9999)

class ProceduralGenerator(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim,
                 time_embed_dim, diffusion_hidden_dim,
                 output_shape, num_labels, grid_H, grid_W, latent_channels=32,
                 timesteps=1000, inference=False, device="cpu"):
        super(ProceduralGenerator, self).__init__()

        self.device          = device
        self.inference       = inference
        self.embed_dim       = embed_dim
        self.grid_H          = grid_H
        self.grid_W          = grid_W
        self.latent_dim      = grid_H * grid_W * latent_channels

        self.scaler          = GradScaler(device)
        self.vae             = GridVAE(num_labels, embed_dim, grid_H, grid_W, latent_channels)
        self.diffusion       = DiffusionModel(embed_dim, self.latent_dim, time_embed_dim, diffusion_hidden_dim, num_heads, latent_channels)
        self.bert            = transformers.BertModel.from_pretrained('bert-base-uncased', torch_dtype=torch.float32)

        for param in self.bert.parameters():
            param.requires_grad = False

        betas                   = cosine_beta_schedule(timesteps)
        alphas                  = 1.0 - betas
        alphas_cumprod          = torch.cumprod(alphas, dim=0)
        self.timesteps          = timesteps
  
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)

        self.noise_gen  = NoiseGenerator()

    def forward(self, grid, t, text_input, mask=None, use_blue_noise=True):

        batch   = grid.size(0)
        z, _, _ = self.vae.encode(grid)
        flat    = z.reshape(batch, -1)

        with torch.no_grad():
            text_embedding = self.bert(**text_input).last_hidden_state

        if use_blue_noise:
            noise = self.noise_gen.generate_blue_noise(flat.shape, device=flat.device)
        else:
            noise = torch.randn_like(flat)

        t_idx         = t.long()
        alpha         = self.alphas_cumprod[t_idx].view(-1, 1)
        noisy_latent  = torch.sqrt(alpha) * flat + torch.sqrt(1 - alpha) * noise
        diffusion_out = self.diffusion(noisy_latent, t, text_embedding, map_embedding)
        pred_noise    = diffusion_out

        if self.inference:
            denoised = (noisy_latent - torch.sqrt(1 - alpha) * pred_noise) / torch.sqrt(alpha)
        else:
            denoised = self.diffusion(noisy_latent, t, text_embedding, map_embedding)

        predictions = self.vae.decode(denoised)
        if self.inference:
            with torch.no_grad():
                predictions = predictions.argmax(dim=1).int()

        if mask is not None:
            boolean_mask  = mask.bool().unsqueeze(1)
            boolean_mask  = boolean_mask.expand(-1, predictions.size(1), -1, -1)
            grid_expanded = grid.unsqueeze(1).expand(-1, predictions.size(1), -1, -1)
            predictions   = torch.where(boolean_mask, predictions, grid_expanded)

        return predictions, pred_noise
        
def train_step(self, grid, text_input, mask, optimizer, use_blue_noise=True, map_embedding=None):
    """
    Training step for the diffusion model with frozen VAE.
    
    Args:
        grid: Ground truth grid [B, H, W]
        text_input: Text conditioning (tokenized for BERT)
        mask: Binary mask indicating regions to inpaint [B, 1, H, W]
        optimizer: Optimizer for diffusion model parameters
        use_blue_noise: Whether to use blue noise or white noise
        map_embedding: Optional map context embedding [B, 32]
    
    Returns:
        Dictionary with loss components
    """
    self.train()
    optimizer.zero_grad()
    
    batch_size = grid.size(0)
    
    for param in self.vae.parameters():
        param.requires_grad = False
    
    with autocast(device_type=self.device.type if hasattr(self.device, 'type') else 'cuda'):
        with torch.no_grad():
            z_true, mu, logvar = self.vae.encode(grid)
        
        with torch.no_grad(): 
            text_embedding = self.bert(**text_input).last_hidden_state
        
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device)
        
        if use_blue_noise:
            noise = self.noise_gen.generate_blue_noise(z_true.shape, device=z_true.device)
        else:
            noise = torch.randn_like(z_true)
        
        alpha_cumprod  = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        z_noisy        = torch.sqrt(alpha_cumprod) * z_true + torch.sqrt(1 - alpha_cumprod) * noise
        mask_latent    = F.interpolate(mask.float(), size=z_true.shape[-2:], mode='nearest')
        z_input        = z_noisy * mask_latent + z_true * (1 - mask_latent)
        grid_embedding = None

        if hasattr(self, 'use_grid_conditioning') and self.use_grid_conditioning:
            with torch.no_grad():
                grid_emb = self.vae.embedding(grid)
                grid_embedding = grid_emb.view(batch_size, -1, self.embed_dim)
        
        pred_noise = self.diffusion(
            z_input, 
            mask_latent, 
            t, 
            text_embedding, 
            map_embedding, 
            grid_embedding
        )
        
        noise_target = noise * mask_latent
        pred_masked  = pred_noise * mask_latent
        loss         = F.mse_loss(pred_masked, noise_target, reduction='mean')
    
    self.scaler.scale(loss).backward()
    self.scaler.step(optimizer)
    self.scaler.update()
    
    return {
        'total_loss': loss.item(),
        'mse_loss': loss.item()
    }

@torch.no_grad()
def sample(self, text_input, mask, reference_grid=None, num_inference_steps=50, 
           guidance_scale=1.0, use_blue_noise=True, map_embedding=None, 
           batch_size=1, temperature=1.0):
    """
    Sample from the diffusion model to generate grid completions.
    
    Args:
        text_input: Text conditioning (tokenized for BERT)
        mask: Binary mask indicating regions to inpaint [B, 1, H, W]
        reference_grid: Optional reference grid for unmasked regions [B, H, W]
        num_inference_steps: Number of denoising steps
        guidance_scale: Classifier-free guidance strength
        use_blue_noise: Whether to start from blue noise
        map_embedding: Optional map context embedding [B, 32]
        batch_size: Batch size for generation
        temperature: Sampling temperature
    
    Returns:
        Generated grids [B, H, W]
    """
    self.eval()
    
    device = next(self.parameters()).device
    
    with torch.no_grad():
        text_embedding = self.bert(**text_input).last_hidden_state
    
    latent_shape = (batch_size, self.vae.latent_channels, self.grid_H // 4, self.grid_W // 4)
    
    if use_blue_noise:
        z = self.noise_gen.generate_blue_noise(latent_shape, device=device) * temperature
    else:
        z = torch.randn(latent_shape, device=device) * temperature
    
    if reference_grid is not None:
        with torch.no_grad():
            z_ref, _, _ = self.vae.encode(reference_grid)
        
        mask_latent = F.interpolate(mask.float(), size=z.shape[-2:], mode='nearest')
        z = z * mask_latent + z_ref * (1 - mask_latent)

    grid_embedding = None
    if reference_grid is not None:
        with torch.no_grad():
            grid_emb = self.vae.embedding(reference_grid)
            grid_embedding = grid_emb.view(batch_size, -1, self.embed_dim)
    
    timesteps = torch.linspace(self.timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=device)
    
    for i, t in enumerate(timesteps):
        t_batch = t.repeat(batch_size)
        
        mask_input = F.interpolate(mask.float(), size=z.shape[-2:], mode='nearest')
        pred_noise = self.diffusion(
            z, 
            mask_input, 
            t_batch, 
            text_embedding, 
            map_embedding, 
            grid_embedding
        )
        
        alpha              = self.alphas[t]
        alpha_cumprod      = self.alphas_cumprod[t]
        alpha_cumprod_prev = self.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0, device=device)
        pred_x0            = (z - torch.sqrt(1 - alpha_cumprod) * pred_noise) / torch.sqrt(alpha_cumprod)
        
        if t > 0:
            beta    = self.betas[t]
            noise   = torch.randn_like(z) * temperature
            z_prev  = (
                torch.sqrt(alpha_cumprod_prev) * beta * pred_x0 +
                torch.sqrt(alpha) * (1 - alpha_cumprod_prev) * z +
                torch.sqrt(beta * (1 - alpha_cumprod_prev)) * noise
            ) / (1 - alpha_cumprod)
        else:
            z_prev = pred_x0
        
        if reference_grid is not None:
            z_prev = z_prev * mask_latent + z_ref * (1 - mask_latent)
        
        z = z_prev
    
    generated_grids = self.vae.decode(z)
    generated_grids = generated_grids.argmax(dim=1)

    if reference_grid is not None:
        mask_2d = mask.squeeze(1).bool()
        generated_grids = torch.where(mask_2d, generated_grids, reference_grid)
    
    return generated_grids

def train_epoch(self, dataloader, optimizer, epoch, log_interval=100):
    """
    Train for one epoch.
    
    Args:
        dataloader: Training data loader
        optimizer: Optimizer
        epoch: Current epoch number
        log_interval: Logging frequency
    
    Returns:
        Average loss for the epoch
    """
    self.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):

        grid          = batch['grid'].to(self.device)
        text_input    = {k: v.to(self.device) for k, v in batch['text_input'].items()}
        mask          = batch['mask'].to(self.device)
        map_embedding = batch.get('map_embedding', None)

        if map_embedding is not None:
            map_embedding = map_embedding.to(self.device)
        
        losses       = self.train_step(grid, text_input, mask, optimizer, map_embedding=map_embedding)
        total_loss  += losses['total_loss']
        num_batches += 1
        
        if batch_idx % log_interval == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, '
                  f'Loss: {losses["total_loss"]:.6f}')
    
    avg_loss = total_loss / num_batches
    print(f'Epoch {epoch} completed. Average Loss: {avg_loss:.6f}')
    
    return avg_loss

@torch.no_grad()
def evaluate(self, dataloader, num_samples=None):
    """
    Evaluate the model on a validation set.
    
    Args:
        dataloader: Validation data loader
        num_samples: Maximum number of samples to evaluate (None for all)
    
    Returns:
        Dictionary with evaluation metrics
    """
    self.eval()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        if num_samples is not None and batch_idx >= num_samples:
            break
            
        grid = batch['grid'].to(self.device)
        text_input = {k: v.to(self.device) for k, v in batch['text_input'].items()}
        mask = batch['mask'].to(self.device)
        map_embedding = batch.get('map_embedding', None)
        if map_embedding is not None:
            map_embedding = map_embedding.to(self.device)
        
        with autocast(device_type=self.device.type if hasattr(self.device, 'type') else 'cuda'):
            batch_size = grid.size(0)
            
            z_true, _,          _ = self.vae.encode(grid)
                   text_embedding = self.bert(**text_input).last_hidden_state
            
            t     = torch.randint(0, self.timesteps, (batch_size,), device=self.device)
            noise = torch.randn_like(z_true)
            
            alpha_cumprod = self.alphas_cumprod[t].view(-1, 1, 1, 1)
            z_noisy       = torch.sqrt(alpha_cumprod) * z_true + torch.sqrt(1 - alpha_cumprod) * noise
            mask_latent   = F.interpolate(mask.float(), size=z_true.shape[-2:], mode='nearest')
            z_input       = z_noisy * mask_latent + z_true * (1 - mask_latent)
            pred_noise    = self.diffusion(z_input, mask_latent, t, text_embedding, map_embedding)
            noise_target  = noise * mask_latent
            pred_masked   = pred_noise * mask_latent
            loss          = F.mse_loss(pred_masked, noise_target, reduction='mean')
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    
    return {
        'avg_loss'   : avg_loss,
        'num_samples': num_batches * dataloader.batch_size
    }