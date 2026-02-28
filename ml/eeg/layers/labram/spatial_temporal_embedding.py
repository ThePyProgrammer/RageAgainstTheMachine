import torch
import torch.nn as nn
from einops import repeat
from eeg.layers.labram.utils import trunc_normal_


__all__ = ["SpatialTemporalEmbedding"]

class SpatialTemporalEmbedding(nn.Module):
    """
    In order to enable the model to be aware of the temporal and spatial
    information of patch embeddings, we initialize the following d-dimension
    lists that are learnable during training:
    
    1) The Spatial Embedding list
    SE = {se_1, se_2, ..., se_|C|} (pos_embed)
    
    where |C| is the number of channels (electrodes).
    
    For each channel c_i, we can find its corresponding spatial embedding
    se_i in the spatial embedding list SE.
    
    2) The Temporal Embedding list
    TE = {te_1, te_2, ..., te_tmax} (time_embed)
    
    Here, tmax is the maximum number of time patches (max_patches).
    
    Note: floor(t/w) <= tmax, where t is the length of the EEG signal and 
    w is the length of each time patch (patch_size).
    
    The identified spatial embedding se and temporal embedding te are then added
    to the output embeddings from the temporal encoder, with them acting as absolute
    position encoding.
    """
    def __init__(self,
                 embed_dim: int = 200,
                 use_abs_pos_emb: bool = True,
                 drop_rate: float = 0.0,
                 init_std: float = 0.02,
                 max_channels: int = 128,
                 max_patches: int = 16):
        super().__init__()
        
        self.embed_dim = embed_dim

        self.pos_embed = nn.Parameter(torch.zeros(max_channels+1, embed_dim)) if use_abs_pos_emb else None
        self.time_embed = nn.Parameter(torch.zeros(max_patches, embed_dim), requires_grad=True)
        
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=init_std)
        trunc_normal_(self.time_embed, std=init_std)
        
        self.pos_drop = nn.Dropout(p=drop_rate)
    
    @torch.jit.ignore # type: ignore
    def no_weight_decay(self):
        return {'pos_embed', 'time_embed'}
    
    def forward(self,
                x: torch.Tensor,
                N: int,
                P: int,
                input_channels: list[int] | None = None):
        """
        Args:
            x: Input tensor of shape (B, N*P+1, embed_dim)
            P: Number of patches per channel
            N: Number of channels
            input_channels: Optional channel indices to subset from pos_embed
        Returns:
            Tensor of shape (B, N, P, embed_dim) with added positional + temporal embeddings
        """
        B, seq_len_plus_one, embed_dim = x.size()
        # note that embed_dim should be equal to self.embed_dim
        assert embed_dim == self.embed_dim, f"embed_dim should be equal to self.embed_dim, but got {embed_dim} and {self.embed_dim}"
        # note that seq_len_plus_one = N*P+1
        assert seq_len_plus_one == N*P+1, f"sequence length {seq_len_plus_one-1} not equal to {N}*{P}"
        
        # Spatial Embedding
        if self.pos_embed is not None:
            pos_embed = self.pos_embed[1:, :]
            if input_channels is not None:
                input_channels = [
                    ch-1
                    for ch in input_channels
                    if ch > 0
                ]
                pos_embed = pos_embed[input_channels]
            # pos_embed is of shape (N, embed_dim)
            pos_embed = repeat(pos_embed, 'N E -> B N P E', B=B, P=P)
            # pos_embed = pos_embed.unsqueeze(2).expand(B, -1, P, -1)
            # pos_embed is now of shape (B, N, P, embed_dim)
            pos_embed = pos_embed.flatten(1, 2)
            # pos_embed is now of shape (B, N*P, embed_dim)
            
            appendage = repeat(self.pos_embed[0, :], "E -> B 1 E", B=B)
            # (B, 1, embed_dim)
            
            pos_embed = torch.cat((
                appendage,
                pos_embed
            ), dim=1)
            # pos_embed is now of shape (B, N*P+1, embed_dim)
            x = x + pos_embed
        
        # Temporal Embedding
        if self.time_embed is not None:
            time_embed = self.time_embed[:P]
            # time_embed is of shape (P, embed_dim)
            time_embed = repeat(time_embed, 'P E -> B N P E', B=B, N=N).flatten(1,2)
            # time_embed is now of shape (B, N*P, embed_dim)
            x[:, 1:, :] = x[:, 1:, :] + time_embed
            
        x = self.pos_drop(x)
        
        return x