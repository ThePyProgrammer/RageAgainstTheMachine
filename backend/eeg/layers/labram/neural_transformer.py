import math
import torch
import torch.nn as nn
from einops import rearrange
from functools import partial
from eeg.layers.labram.utils import trunc_normal_, init_weights_
from eeg.layers.labram.temporal_encoder import TemporalEncoder
from eeg.layers.labram.patch_embed import PatchEmbed
from eeg.layers.labram.masking_process import MaskingProcess
from eeg.layers.labram.append_class_token import AppendClassToken
from eeg.layers.labram.spatial_temporal_embedding import SpatialTemporalEmbedding
from eeg.layers.labram.neural_transformer_block import NeuralTransformerBlock


__all__ = ["NeuralTransformer"]

class NeuralTransformer(nn.Module):
    def __init__(self,
                 eeg_size: int = 1600,
                 patch_size: int = 200,
                 embed_dim: int = 200,
                 vocab_size: int = 8192,
                 in_channels: int = 1,
                 out_channels: int = 8,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 use_qkv_bias: bool = True,
                 qk_norm = None,
                 qk_scale = None,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.1,
                 norm_layer = nn.LayerNorm,
                 init_values = 1.0,
                 attn_head_dim: int | None = None,
                 use_abs_pos_emb: bool = True,
                 init_std: float = 0.02,
                 ):
        super().__init__()
        
        assert patch_size % out_channels == 0, f"To ensure T' = T, T MUST be divisible by C (out_channels). Got T={patch_size}, C={out_channels}"
        
        self.eeg_size = eeg_size
        self.patch_size = patch_size
        self.num_features = self.embed_dim = embed_dim
        self.init_std = init_std
        
        if in_channels == 1:
            # expected in LaBraM and neural tokenizer
            self.patch_embed = TemporalEncoder(in_channels=in_channels,
                                               out_channels=out_channels)
        else:
            # expected in neural decoder
            self.patch_embed = PatchEmbed(eeg_size=eeg_size,
                                          patch_size=patch_size,
                                          in_channels=in_channels,
                                          embed_dim=embed_dim)

        self.masking_process = MaskingProcess(embed_dim=embed_dim,
                                              init_std=init_std)
        
        self.append_cls_token = AppendClassToken(embed_dim=embed_dim,
                                                 init_std=init_std)
        
        self.embedding = SpatialTemporalEmbedding(embed_dim=embed_dim,
                                                  use_abs_pos_emb=use_abs_pos_emb,
                                                  init_std=init_std,)
        
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        
        self.blocks = nn.ModuleList([
            NeuralTransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                use_qkv_bias=use_qkv_bias,
                qk_norm=qk_norm,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values,
                window_size=None,
                attn_head_dim=attn_head_dim,
            )
            for i in range(depth)
        ])
        
        self.norm = norm_layer(embed_dim)
        
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        
        trunc_normal_(self.lm_head.weight, std=self.init_std)
        self.apply(partial(init_weights_, init_std=self.init_std))
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1) # type: ignore
            rescale(layer.mlp[2].weight.data, layer_id + 1) # type: ignore

    def get_num_layers(self):
        return len(self.blocks)
    
    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor | None = None,
                input_channels: list | None = None,
                return_all_tokens: bool = True,
                return_patch_tokens: bool = False,
                return_all_patch_tokens: bool = False):
        B, N, P, T = x.size()
        # B = batch size
        # N = number of channels (electrodes)
        # P = number of patches
        # T = patch_size
        assert T == self.patch_size, f"Input T (patch size) must be equal to the specified patch_size. Got T={T}, patch_size={self.patch_size}"
        
        x = self.patch_embed(x)
        B, seq_len, embed_dim = x.size()
        # seq_len = N * P
        # embed_dim = ceil(T / C) * C
        # C = out_channels as specified
        
        # note that embed_dim should be equal to self.embed_dim
        assert embed_dim == self.embed_dim, f"embed_dim should be equal to self.embed_dim, but got {embed_dim} and {self.embed_dim}"
        
        # mask x (if provided)
        if mask is not None:
            x = self.masking_process(x, mask)
        # x is still of shape (B, seq_len, embed_dim)
        
        # append cls token to x
        x = self.append_cls_token(x)
        # x is now of shape (B, seq_len + 1, embed_dim)
        
        # x = rearrange(x, "B (N P) E -> B N P E", N=N, P=P)
        x = self.embedding(x,
                           N=N, P=P,
                           input_channels=input_channels)
        # x = rearrange(x, "B N P E -> B (N P) E")
        # x is now still of shape (B, N*P+1, embed_dim)
        
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        # x is now still of shape (B, N*P+1, embed_dim)
        
        if return_all_patch_tokens:
            # (B, N*P+1, embed_dim)
            return x
        
        x = x[:, 1:]
        
        if return_patch_tokens:
            # (B, N*P, embed_dim)
            return x
        # (B, N*P, vocab_size)
        return self.lm_head(x if return_all_tokens else x[mask])
        
        
        


