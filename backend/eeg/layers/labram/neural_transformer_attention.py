import torch
import torch.nn as nn
import torch.nn.functional as F

from eeg.layers.labram.relative_position_bias import RelativePositionBias


__all__ = ["NeuralTransformerAttention"]

class NeuralTransformerAttention(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 use_qkv_bias: bool = False,
                 qk_norm = None,
                 qk_scale = None,
                 attn_drop = 0.0,
                 proj_drop = 0.0,
                 window_size: tuple[int, int] | None = None,
                 attn_head_dim: int | None = None):
        super().__init__()
        
        self.num_heads = num_heads
        
        head_dim = attn_head_dim if attn_head_dim is not None else dim // num_heads
        
        all_head_dim = head_dim * self.num_heads
        
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        
        self.q_bias = nn.Parameter(torch.zeros(all_head_dim)) if use_qkv_bias else None
        self.v_bias = nn.Parameter(torch.zeros(all_head_dim)) if use_qkv_bias else None
        
        self.q_norm = qk_norm(head_dim) if qk_norm is not None else None
        self.k_norm = qk_norm(head_dim) if qk_norm is not None else None

        self.window_size = window_size
        
        self.relative_position_bias = RelativePositionBias(window_size, num_heads) if window_size is not None else None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attention=False, return_qkv=False):
        B, N, C = x.shape
        
        qkv_bias = None
        if self.q_bias is not None and self.v_bias is not None:
            qkv_bias = torch.cat((
                self.q_bias,
                torch.zeros_like(self.v_bias, requires_grad=False),
                self.v_bias
            ))
    
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        # qkv is of shape (B, N, 3 * all_head_dim)
        
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1)
        # qkv is of shape (B, N, 3, num_heads, head_dim)
        
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # qkv is of shape (3, B, num_heads, N, head_dim)
        
        q, k, v = qkv[0], qkv[1], qkv[2] # (B, nH, N, C)
        
        if self.q_norm is not None:
            q = self.q_norm(q).type_as(v)
        if self.k_norm is not None:
            k = self.k_norm(k).type_as(v)

        q = q * self.scale
        
        # (B, nH, N, C) @ (B, nH, C, N) -> (B, nH, N, N)
        attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias is not None:
            attn += self.relative_position_bias()
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if return_attention:
            return attn

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)

        if return_qkv:
            return x, qkv

        return x