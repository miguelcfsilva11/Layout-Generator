import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.models.attention import MultiHeadAttention, ScaledDotProductAttention
import math

def expand_embedding_weights(old_embedding, new_vocab_size):

    old_weight                                      = old_embedding.weight.data
    new_embedding                                   = nn.Embedding(new_vocab_size, old_weight.shape[1])
    new_embedding.weight.data[:old_weight.shape[0]] = old_weight
    new_embedding.weight.data[old_weight.shape[0]:].uniform_(-0.02, 0.02)
    return new_embedding
    
class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim):

        super(TimeEmbedding, self).__init__()
        self.embed_dim = embed_dim

    def forward(self, t):

        half_dim   = self.embed_dim // 2
        emb_factor = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        exponents  = torch.exp(torch.arange(half_dim, dtype=torch.float, device=t.device) * -emb_factor)
        emb        = t.unsqueeze(1) * exponents.unsqueeze(0)
        emb_sin    = torch.sin(emb)
        emb_cos    = torch.cos(emb)

        return torch.cat([emb_sin, emb_cos], dim=1)

class GridEmbedding(nn.Module):
    def __init__(self, num_labels, embed_dim):

        super(GridEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_labels, embed_dim)
        self.norm      = nn.LayerNorm(embed_dim)

    def forward(self, grid):
        embedded = self.embedding(grid)
        return self.norm(embedded)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act   = nn.SiLU()
        self.skip  = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        h          = self.act(self.norm1(self.conv1(x)))
        h          = self.act(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class AxialAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.row_attn = TransformerBlock(embed_dim, num_heads, embed_dim * 4, dropout)
        self.col_attn = TransformerBlock(embed_dim, num_heads, embed_dim * 4, dropout)

    def forward(self, x):

        B, H, W, D = x.shape
        x          = x.permute(0, 2, 1, 3).reshape(B * W, H, D)
        x          = self.row_attn(x)
        x          = x.view(B, W, H, D).permute(0, 2, 1, 3)
        x          = x.reshape(B * H, W, D)
        x          = self.col_attn(x)
        x          = x.view(B, H, W, D)
        return x

class GridVAE(nn.Module):
    def __init__(self, num_labels, embed_dim=64, grid_H=64, grid_W=64, latent_channels=32, num_heads=8, ff_dim=512, dropout=0.1):
        super().__init__()

        self.embed_dim    = embed_dim
        self.grid_H       = grid_H
        self.grid_W       = grid_W
        self.embedding    = GridEmbedding(num_labels, embed_dim)
        self.pos_enc      = PositionalEncoding2D(embed_dim, grid_H, grid_W)

        self.encoder_conv = nn.Sequential(
            ResidualBlock(embed_dim, 64),
            ResidualBlock(64, 128),
            nn.Conv2d(128, latent_channels * 2, 3, padding=1)
        )

        self.encoder_attn         = AxialAttention(embed_dim, num_heads, dropout)
        self.encoder_attn_proj    = nn.Conv2d(embed_dim, latent_channels * 2, kernel_size=1)
        self.latent_to_embed      = nn.Conv2d(latent_channels, embed_dim, 1)
        self.decoder_transformer1 = TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
        self.decoder_transformer2 = TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
        self.decoder_cnn          = nn.Sequential(

            ResidualBlock(embed_dim, embed_dim),
            ResidualBlock(embed_dim, embed_dim),
            nn.Conv2d(embed_dim, num_labels, 1)
        )
        self.decoder_pos_enc = PositionalEncoding2D(embed_dim, grid_H, grid_W) 


    def encode(self, grid):
        x          = self.embedding(grid)
        x          = self.pos_enc(x)
        x          = x.permute(0, 3, 1, 2)
        conv_feat  = self.encoder_conv(x)
        x_attn     = x.permute(0, 2, 3, 1)
        x_attn     = self.encoder_attn(x_attn).permute(0, 3, 1, 2)
        x_attn     = self.encoder_attn_proj(x_attn)
        fused      = conv_feat + x_attn
        mu, logvar = torch.chunk(fused, 2, dim=1)

        return self.reparameterize(mu, logvar), mu, logvar

    def decode(self, z):
        
        x = self.latent_to_embed(z)
        x = x.permute(0, 2, 3, 1)
        x = self.decoder_pos_enc(x)
        x = x.reshape(z.size(0), -1, self.embed_dim)
        x = self.decoder_transformer1(x)
        x = self.decoder_transformer2(x)
        x = x.view(z.size(0), self.grid_H, self.grid_W, self.embed_dim).permute(0, 3, 1, 2)

        return self.decoder_cnn(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, grid):
        z, mu, logvar = self.encode(grid)
        out = self.decode(z)
        return out, mu, logvar

    def get_latent(self, grid):
        with torch.no_grad():
            x = self.embedding(grid)
            x = self.pos_enc(x)
            x = x.permute(0, 3, 1, 2)
            h = self.encoder_conv(x)
            mu, _ = torch.chunk(h, 2, dim=1)
        return mu

    def reconstruct(self, grid):
        with torch.no_grad():
            z, _, _ = self.encode(grid)
            return self.decode(z)

class PositionalEncoding2D(nn.Module):
    def __init__(self, embed_dim, grid_h, grid_w):
        super().__init__()

        self.embed_dim  = embed_dim
        self.grid_h     = grid_h
        self.grid_w     = grid_w
        
        pe              = torch.zeros(grid_h, grid_w, embed_dim)
        div_term        = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))
        pos_h           = torch.arange(0, grid_h).unsqueeze(1).unsqueeze(2).float()
        pos_w           = torch.arange(0, grid_w).unsqueeze(0).unsqueeze(2).float()
        
        pe[:, :, 0::2]  = torch.sin(pos_h * div_term)
        pe[:, :, 1::2]  = torch.cos(pos_h * div_term)
        pe[:, :, 0::2] += torch.sin(pos_w * div_term)
        pe[:, :, 1::2] += torch.cos(pos_w * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe

class DiffusionModel(nn.Module):
    def __init__(
        self,
        num_labels     : int,
        embed_dim      : int,
        latent_channels: int,
        grid_h         : int,
        grid_w         : int,
        time_embed_dim : int,
        num_heads      : int,
        hidden_dim     : int,
        timesteps      : int = 1000,
        device         : str = "cpu"
    ):
        super().__init__()

        self.device          = device
        self.embed_dim       = embed_dim
        self.latent_channels = latent_channels
        self.grid_h          = grid_h
        self.grid_w          = grid_w
        self.timesteps       = timesteps
        self.pos_enc         = PositionalEncoding2D(embed_dim=embed_dim, grid_h=grid_h, grid_w=grid_w)
        self.grid_embedding  = GridEmbedding(num_labels=num_labels, embed_dim=embed_dim)
        self.time_embed      = TimeEmbedding(time_embed_dim)
        self.mask_proj       = nn.Conv2d(1, time_embed_dim, kernel_size=3, padding=1)
        stem_in_channels     = latent_channels + time_embed_dim + time_embed_dim

        self.stem_conv       = nn.Sequential(
            nn.Conv2d(stem_in_channels, 64, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.SiLU(),
            nn.Conv2d(128, embed_dim, kernel_size=3, padding=1)
        )

        self.self_attn = MultiHeadAttention(
            e_dim   = embed_dim,
            q_dim   = embed_dim,
            v_dim   = embed_dim,
            n_heads = num_heads
        )


        self.text_attn = CrossAttentionLayer(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

        self.grid_attn = CrossAttentionLayer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            context_dim=embed_dim
        )

        self.skip_mlp = nn.Sequential(
            nn.Linear(embed_dim + time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_channels)
        )

        self.out_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, latent_channels)
        )

        betas          = self._cosine_beta_schedule(timesteps)
        alphas         = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)                  
        self.register_buffer("alphas", alphas)               
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    @staticmethod
    def _cosine_beta_schedule(timesteps, s=0.008):
        """
        Build a cosine‐based schedule for betas, as in Nichol & Dhariwal (2021).
        """
        steps          = timesteps + 1
        x              = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas          = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)

    def forward(
        self,
        latent     : torch.Tensor,
        mask       : torch.Tensor,
        t          : torch.Tensor,
        text_emb   : torch.Tensor,
        grid_labels: torch.LongTensor
    ) -> torch.Tensor:
        B, C_lat, H, W = latent.shape
        assert C_lat == self.latent_channels, \
            f"latent_channels ({C_lat}) != self.latent_channels ({self.latent_channels})"
    
        N = H * W

        grid_emb    = self.grid_embedding(grid_labels)
        grid_emb    = self.pos_enc(grid_emb)
        grid_tokens = grid_emb.view(B, H * W, self.embed_dim)
        t_emb       = self.time_embed(t)
        mask_float  = mask.float()
        mask_emb    = self.mask_proj(mask_float)
        t_map       = t_emb[:, :, None, None].expand(-1, -1, H, W)

        stem_in     = torch.cat([latent, mask_emb, t_map], dim=1)
        h_conv      = self.stem_conv(stem_in)
        h_tokens    = h_conv.view(B, self.embed_dim, N).permute(0, 2, 1)
        pos         = self.pos_enc.pe.view(N, self.embed_dim)
        h_tokens    = h_tokens + pos.unsqueeze(0).to(h_tokens.device)
        h_tokens    = self.self_attn(h_tokens, h_tokens, h_tokens)
        h_tokens    = self.text_attn(h_tokens, text_emb)
        h_tokens    = self.grid_attn(h_tokens, grid_tokens)
        t_broadcast = t_emb.unsqueeze(1).expand(-1, N, -1)

        skip_in     = torch.cat([h_tokens, t_broadcast], dim=-1)
        skip_flat   = skip_in.view(-1, skip_in.size(-1))
        skip_out    = self.skip_mlp(skip_flat)
        skip_out    = skip_out.view(B, N, self.latent_channels).permute(0, 2, 1)
        skip_out    = skip_out.view(B, self.latent_channels, H, W)
        out_flat    = self.out_mlp(h_tokens.view(-1, self.embed_dim))
        out         = out_flat.view(B, self.latent_channels, N).permute(0, 2, 1)
        out         = out.reshape(B, self.latent_channels, H, W)


        return out + skip_out



class NoiseGenerator:
    def __init__(self, noise_dim=64):

        self.noise_dim = noise_dim

    def generate_white_noise(self, shape, device='cpu'):

        white = torch.randn(*shape, device=device)
        dims = list(range(1, white.ndim))
        mean = white.mean(dim=dims, keepdim=True)
        std  = white.std(dim=dims, keepdim=True) + 1e-8
        return (white - mean) / std

    def generate_blue_noise(self, shape, device='cpu'):

        white_noise = torch.randn(*shape, device=device)

        dims        = list(range(2, len(shape)))
        noise_fft   = torch.fft.fftn(white_noise, dim=dims)

        grid_shape  = shape[2:]
        freq_grids  = torch.meshgrid(
            *[torch.fft.fftfreq(n, d=1.0, device=device) for n in grid_shape],
            indexing='ij'
        )
        radius         = torch.sqrt(sum(g**2 for g in freq_grids)).unsqueeze(0).unsqueeze(0)
        radius         = torch.where(radius == 0, torch.tensor(1.0, device=device), radius)
        filtered_fft   = noise_fft * radius
        filtered_noise = torch.fft.ifftn(filtered_fft, dim=dims).real
        dims_all       = list(range(2, filtered_noise.ndim))
        mean           = filtered_noise.mean(dim=dims_all, keepdim=True)
        std            = filtered_noise.std(dim=dims_all, keepdim=True) + 1e-8

        return (filtered_noise - mean) / std


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()

        self.att        = MultiHeadAttention(embed_dim, embed_dim, embed_dim, num_heads)
        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout1   = nn.Dropout(dropout)
        self.ffn        = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout2   = nn.Dropout(dropout)

    def forward(self, x):

        attn_output = self.att(x, x, x)
        x           = self.layernorm1(x + self.dropout1(attn_output))
        ffn_output  = self.ffn(x)
        x           = self.layernorm2(x + self.dropout2(ffn_output))
        return x

class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, context_dim=768):
        super(CrossAttentionLayer, self).__init__()

        self.context_proj = nn.Linear(context_dim, embed_dim)
        self.att          = MultiHeadAttention(embed_dim, embed_dim, embed_dim, num_heads)
        self.layernorm    = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, query, context):

        context     = self.context_proj(context)
        attn_output = self.att(query, context, context)
        return self.layernorm(query + attn_output)



class DoubleConv(nn.Module):
    """(Conv -> GN -> SiLU) * 2 with optional FiLM conditioning."""
    def __init__(self, in_ch, out_ch, time_dim=None):
        super().__init__()

        self.time_dim = time_dim
        self.conv1    = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.gn1      = nn.GroupNorm(8, out_ch)
        self.act1     = nn.SiLU()
        self.conv2    = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.gn2      = nn.GroupNorm(8, out_ch)
        self.act2     = nn.SiLU()

        if time_dim is not None:
            self.time_proj = nn.Linear(time_dim, out_ch * 2)

    def forward(self, x, t_emb=None):
        h = self.conv1(x)
        h = self.gn1(h)

        if self.time_dim is not None and t_emb is not None:

            scale, shift = self.time_proj(t_emb).chunk(2, dim=-1)

            scale = scale.unsqueeze(-1).unsqueeze(-1)
            shift = shift.unsqueeze(-1).unsqueeze(-1)
            h     = h * (1 + scale) + shift

        h = self.act1(h)
        h = self.conv2(h)
        h = self.gn2(h)
        h = self.act2(h)
        return h

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim=None):
        super().__init__()
        self.maxpool     = nn.MaxPool2d(2)
        self.double_conv = DoubleConv(in_ch, out_ch, time_dim)
    def forward(self, x, t_emb=None):
        x = self.maxpool(x)
        return self.double_conv(x, t_emb)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim=None):
        super().__init__()
        self.up          = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_ch, out_ch, time_dim)
    def forward(self, x, skip, t_emb=None):
        x = self.up(x)

        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        x     = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x     = torch.cat([skip, x], dim=1)
        return self.double_conv(x, t_emb)

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class DiffusionModelUNet(nn.Module):
    def __init__(
        self,
        num_labels     : int,
        latent_channels: int,
        base_channels  : int = 64,
        grid_h         : int = 16,
        grid_w         : int = 16,
        time_embed_dim : int = 128,
        text_embed_dim : int = 512,
        timesteps      : int = 1000,
        device         : str = "cpu"
    ):
        super().__init__()
        self.device          = device
        self.latent_channels = latent_channels
        self.timesteps       = timesteps
        self.pos_enc         = PositionalEncoding2D(embed_dim=base_channels, grid_h=grid_h, grid_w=grid_w)
        self.grid_embedding  = GridEmbedding(num_labels=num_labels, embed_dim=base_channels)
        self.time_embed      = TimeEmbedding(time_embed_dim)

        self.mask_proj       = nn.Conv2d(1, time_embed_dim, kernel_size=3, padding=1)
        self.inc             = DoubleConv(latent_channels + time_embed_dim, base_channels, time_dim=time_embed_dim)

        self.down1           = Down(base_channels, base_channels * 2, time_dim=time_embed_dim)
        self.down2           = Down(base_channels * 2, base_channels * 4, time_dim=time_embed_dim)
        self.down3           = Down(base_channels * 4, base_channels * 8, time_dim=time_embed_dim)

        self.self_attn       = MultiHeadAttention(e_dim=base_channels * 8, q_dim=base_channels * 8,
                                             v_dim=base_channels * 8, n_heads=8)
        self.text_attn       = CrossAttentionLayer(embed_dim=base_channels * 8,
                                             num_heads=8, context_dim=text_embed_dim)
        self.grid_attn       = CrossAttentionLayer(embed_dim=base_channels * 8,
                                             num_heads=8, context_dim=base_channels)


        self.up3       = Up(base_channels * 8, base_channels * 4, time_dim=time_embed_dim)
        self.up2       = Up(base_channels * 4, base_channels * 2, time_dim=time_embed_dim)
        self.up1       = Up(base_channels * 2, base_channels, time_dim=time_embed_dim)
        self.outc      = OutConv(base_channels, latent_channels)
        betas          = self._cosine_beta_schedule(timesteps)
        alphas         = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    @staticmethod
    def _cosine_beta_schedule(timesteps, s=0.008):
        steps          = timesteps + 1
        x              = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas          = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)

    def forward(self, latent, mask, t, text_emb, grid_labels):

        B , C_lat, H,  W  = latent.shape
        t_emb             = self.time_embed(t)
        mask_emb          = self.mask_proj(mask.float())
        in_map            = torch.cat([latent, mask_emb], dim=1)
        x1                = self.inc(in_map, t_emb)
        x2                = self.down1(x1, t_emb)
        x3                = self.down2(x2, t_emb)
        x4                = self.down3(x3, t_emb)
        B_, C_, H_, W_    = x4.shape
        tokens            = x4.view(B_, C_, H_ * W_).permute(0, 2, 1)
        tokens            = self.self_attn(tokens, tokens, tokens)
        tokens            = self.text_attn(tokens, text_emb)
        grid_emb          = self.grid_embedding(grid_labels)
        grid_emb          = self.pos_enc(grid_emb)
        grid_tokens       = grid_emb.view(grid_emb.shape[0], -1, grid_emb.shape[-1])
        tokens            = self.grid_attn(tokens, grid_tokens)
        x4                = tokens.permute(0, 2, 1).view(B_, C_, H_, W_)
        x                 = self.up3(x4, x3, t_emb)
        x                 = self.up2(x, x2, t_emb)
        x                 = self.up1(x, x1, t_emb)
        out               = self.outc(x)
        return out



class MaskedAutoencoderGridViT(nn.Module):
    """
    MAE adapted for 64x64 labeled grids (each cell is a categorical label)
    """
    def __init__(
        self,
        num_labels,
        grid_size         = 64,
        embed_dim         = 768,
        depth             = 12,
        num_heads         = 12,
        mlp_ratio         = 4.,
        decoder_embed_dim = 512,
        decoder_depth     = 8,
        decoder_num_heads = 16,
        mask_ratio        = 0.75,
        dropout           = 0.0,
        layer_norm_eps    = 1e-6,
        init_std          = 0.02
    ):
        super().__init__()
        self.grid_size   = grid_size
        self.num_patches = grid_size * grid_size
        self.mask_ratio  = mask_ratio
        self.init_std    = init_std


        self.grid_embed = GridEmbedding(num_labels, embed_dim)
        self.pos_embed  = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, int(embed_dim * mlp_ratio), dropout)
            for _ in range(depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

        self.decoder_embed      = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed  = nn.Parameter(torch.zeros(1, self.num_patches, decoder_embed_dim))
        self.decoder_mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_blocks     = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, decoder_num_heads, int(decoder_embed_dim * mlp_ratio), dropout)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim, eps=layer_norm_eps)
        self.decoder_pred = nn.Linear(decoder_embed_dim, num_labels, bias=True)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=self.init_std)
        nn.init.trunc_normal_(self.mask_token, std=self.init_std)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=self.init_std)
        nn.init.trunc_normal_(self.decoder_mask_token, std=self.init_std)
        nn.init.xavier_uniform_(self.decoder_embed.weight)
        nn.init.constant_(self.decoder_embed.bias, 0)
        nn.init.xavier_uniform_(self.decoder_pred.weight)
        nn.init.constant_(self.decoder_pred.bias, 0)

    def random_masking(self, x, mask_ratio, device):
        """
        x: [B, N, D]
        returns masked tokens, mask, ids_restore
        """
        B, N,          D = x.shape

        len_keep    = int(N * (1 - mask_ratio))
        noise       = torch.rand(B, N, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep    = ids_shuffle[:, :len_keep]
        x_masked    = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask        = torch.ones([B, N], device=device)

        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward(self, grids, mask_ratio=None):
        """
        grids: LongTensor of shape (B, grid_size, grid_size)
        mask_ratio: override default mask ratio; if None, uses training mask or no mask in eval
        returns:
          logits: (B, N, num_labels)
          mask: (B, N)
        """
        assert grids.dim() == 3 and grids.size(1) == grids.size(2) == self.grid_size,
            f"Expected input shape (B, {self.grid_size}, {self.grid_size}), got {tuple(grids.shape)}"

        B      = grids.size(0)
        device = grids.device
        if mask_ratio is None: 
            mask_ratio = self.mask_ratio if self.training else 0.0

        x = self.grid_embed(grids)
        x = x.view(B, self.num_patches, -1)
        x = x + self.pos_embed.to(device)

        x_masked, mask, ids_restore = self.random_masking(x, mask_ratio, device)

        for blk in self.blocks: 
        x_masked  = blk(x_masked)
        x_encoded = self.encoder_norm(x_masked)

        x_dec       = self.decoder_embed(x_encoded)
        mask_tokens = self.decoder_mask_token.to(device).repeat(B, self.num_patches - x_dec.shape[1], 1)
        x_combined  = torch.cat([x_dec, mask_tokens], dim=1)
        x_combined  = torch.gather(
            x_combined,
            dim   = 1,
            index = ids_restore.unsqueeze(-1).repeat(1, 1, x_combined.shape[2])
        )
        x_combined = x_combined + self.decoder_pos_embed.to(device)

        for blk in self.decoder_blocks: 
        x_combined = blk(x_combined)
        x_combined = self.decoder_norm(x_combined)

        logits = self.decoder_pred(x_combined)
        return logits, mask

    def compute_loss(self, grids, logits, mask):
        """
        Compute cross-entropy loss only on masked tokens
        grids: (B, H, W)
        logits: (B, N, num_labels)
        mask: (B, N)
        """
        B         = grids.size(0)
        target    = grids.view(B * self.num_patches)
        pred      = logits.view(B * self.num_patches, -1)
        mask_flat = mask.view(B * self.num_patches)
        loss      = F.cross_entropy(pred, target, reduction='none')
        loss      = (loss * mask_flat).sum() / (mask_flat.sum() + 1e-8)
        return loss