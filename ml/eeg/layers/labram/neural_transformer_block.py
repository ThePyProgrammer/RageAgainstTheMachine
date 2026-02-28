import torch
import torch.nn as nn

from eeg.layers.drop_path import DropPath

from eeg.layers.labram.neural_transformer_attention import NeuralTransformerAttention


__all__ = ["NeuralTransformerBlock"]

class NeuralTransformerBlock(nn.Module):
    """
    Finally, the sequence of embeddings will be directly fed into the 
    Transformer encoder. To make the training of Transformer more stable 
    and efficient, we incorporate some modifications. First, we add 
    layer normalization to the queries and keys before the dot-product
    attention mechanism, which avoids over-large values in attention logits:
    
    Attention(Q, K, V) = softmax(
        (LN(Q) * LN(K)^T) / sqrt(d_head)
    ) * V
    
    where d_head is the dimension of one head in the multi-head attention 
    and LN denotes the LayerNorm. Next, we omit the bias term in QKV 
    computations, which accelerates the training without performance 
    degradation. For downstream tasks, we use average pooling on the output
    embeddings followed by task-specific prediction heads.
    """
    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 use_qkv_bias: bool = False,
                 qk_norm = None,
                 qk_scale = None,
                 drop: float = 0.0,
                 attn_drop: float = 0.0,
                 drop_path = 0.0,
                 init_values: float = 1.0,
                 act_layer = nn.GELU,
                 norm_layer = nn.LayerNorm,
                 window_size: tuple[int, int] | None = None,
                 attn_head_dim: int | None = None
                 
                 ):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        
        # drop path for stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        self.attn = NeuralTransformerAttention(
            dim=dim,
            num_heads=num_heads,
            use_qkv_bias=use_qkv_bias,
            qk_norm=qk_norm,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            window_size=window_size,
            attn_head_dim=attn_head_dim
        )
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            # nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True) if init_values > 0 else None
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True) if init_values > 0 else None


    def forward(self, x, return_attention=False, return_qkv=False):
        x = self.norm1(x)
        
        if return_attention:
            return self.attn(x, return_attention=True)

        if return_qkv:
            y, qkv = self.attn(x, return_qkv=return_qkv)
            x = x + self.drop_path(self.gamma_1 * y)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            return x, qkv

        x_attn = self.attn(x)
        x_attn = self.gamma_1 * x_attn if self.gamma_1 is not None else x_attn
        x = x + self.drop_path(x_attn)
        
        x = self.norm2(x)
        
        x_mlp = self.mlp(x)
        x_mlp = self.gamma_2 * x_mlp if self.gamma_2 is not None else x_mlp
        x = x + self.drop_path(x_mlp)
        
        return x