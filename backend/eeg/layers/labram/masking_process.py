import torch
import torch.nn as nn
from eeg.layers.labram.utils import trunc_normal_


__all__ = ["MaskingProcess", "random_mask_generation"]

def random_mask_generation(x, mask_ratio: float = 0.5):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [B, NP, T], sequence
        """
        B, NP, T = x.shape  # batch, length, dim
        len_keep = int(NP * (1 - mask_ratio))
        
        # generate "noise" of shape (B, NP)
        # noise is generated between [0, 1]
        noise = torch.rand(B, NP, device=x.device)
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, T))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, NP], device=x.device)
        mask[:, :len_keep] = 0
        # mask.scatter_(1, ids_keep, 0)  # unmask the kept ids

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask.to(torch.bool)


class MaskingProcess(nn.Module):
    def __init__(self,
                 embed_dim: int = 200,
                 init_std: float = 0.02):
        super().__init__()

        self.init_std = init_std
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.mask_token, std=self.init_std)
    
    def forward(self, x, mask = None):
        B, seq_len, embed_dim = x.size()
        
        # note that embed_dim should be equal to self.embed_dim
        assert embed_dim == self.embed_dim, f"embed_dim should be equal to self.embed_dim, but got {embed_dim} and {self.embed_dim}"
        
        mask_token = self.mask_token.expand(B, seq_len, -1)
        # mask_token of size (B, seq_len, embed_dim)
        
        if mask is None:
            mask = torch.zeros((
                B,
                seq_len
            ), dtype=torch.bool).to(x.device)
        
        # mask of size (B, seq_len)
        mask = mask.unsqueeze(-1).type_as(mask_token)
        # mask of size (B, seq_len, 1)
        
        # replace masked positions (represented by mask == 1) with mask_token 
        x = x * (1 - mask) + mask_token * mask
        # x is of size (B, seq_len, embed_dim)
        
        return x