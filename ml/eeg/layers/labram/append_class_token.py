import torch
import torch.nn as nn
from eeg.layers.labram.utils import trunc_normal_


__all__ = ["AppendClassToken"]

class AppendClassToken(nn.Module):
    def __init__(self,
                 embed_dim: int = 200,
                 init_std: float = 0.02):
        super().__init__()
        self.embed_dim = embed_dim
        self.init_std = init_std
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=self.init_std)

    @torch.jit.ignore # type: ignore
    def no_weight_decay(self):
        return {'cls_token'}
    
    def forward(self, x):
        B, seq_len, embed_dim = x.size()
        
        # note that embed_dim should be equal to self.embed_dim
        assert embed_dim == self.embed_dim, f"embed_dim should be equal to self.embed_dim, but got {embed_dim} and {self.embed_dim}"
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # cls_tokens of size (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)
        # x is of size (B, 1+seq_len, embed_dim)
        
        return x