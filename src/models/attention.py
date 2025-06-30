import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F


T = torch.FloatTensor

class ScaledDotProductAttention(nn.Module):
    
    def __init__(self, q_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.q_dim = q_dim
        self.dropout = dropout
        
    def forward(self, q: T, k: T, v: T, mask: Optional[T] = None) -> T:
        return F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0
        )

class MultiHeadAttention(nn.Module):
    def __init__(self, e_dim: int, q_dim: int, v_dim: int, n_heads: int) -> None:

        super().__init__()

        self.q_dim      = q_dim
        self.v_dim      = v_dim
        self.e_dim      = e_dim
        self.n_heads    = n_heads
        self.q_head_dim = q_dim    // n_heads
        self.v_head_dim = v_dim    // n_heads

        assert self.q_head_dim * n_heads == q_dim
        assert self.v_head_dim * n_heads == v_dim

        self.q_proj = nn.Linear(e_dim, q_dim)
        self.k_proj = nn.Linear(e_dim, q_dim)
        self.v_proj = nn.Linear(e_dim, v_dim)
        self.attn   = ScaledDotProductAttention(q_dim)
        self.linear = nn.Linear(v_dim, e_dim)

    def forward(self, q: T, k: T, v: T, mask: Optional[T] = None) -> T:

        q_proj: T = self.q_proj(q)
        k_proj: T = self.k_proj(k)
        v_proj: T = self.v_proj(v)

        batch_size, q_length, _ = q_proj.shape
        _, k_length, _ = k_proj.shape

        q_proj = q_proj.view(batch_size, q_length, self.n_heads, self.q_head_dim).transpose(1, 2) 
        k_proj = k_proj.view(batch_size, k_length, self.n_heads, self.q_head_dim).transpose(1, 2)  
        v_proj = v_proj.view(batch_size, k_length, self.n_heads, self.v_head_dim).transpose(1, 2)
        x: T   = self.attn(q_proj, k_proj, v_proj, mask)

        x      = x.transpose(1, 2).contiguous().view(batch_size, q_length, self.v_dim)
        x      = self.linear(x)

        return x