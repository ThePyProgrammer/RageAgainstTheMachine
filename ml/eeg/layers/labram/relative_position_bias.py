import torch
import torch.nn as nn


__all__ = ["RelativePositionBias"]

class RelativePositionBias(nn.Module):
    def __init__(self, window_size: tuple[int, int], num_heads: int):
        super().__init__()
        
        self.window_size = window_size
        H, W = self.window_size
        
        self.num_relative_distance = (2 * H - 1) * (2 * W - 1) + 3
        
        self.relative_position_bias_table = nn.Parameter(torch.zeros(
            self.num_relative_distance, num_heads
        ))  # (2*W-1) * (2*H-1) + 3, num_heads
        
        # cls to token & token to cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(H) # H
        coords_w = torch.arange(W) # W
        
        coords_h, coords_w = torch.meshgrid([coords_h, coords_w]) # Each is H, W
        
        coords_h = coords_h.flatten()  # H*W
        coords_w = coords_w.flatten()  # H*W
        
        relative_coords_h = coords_h[:, None] - coords_h[None, :]  # H*W, H*W
        relative_coords_w = coords_w[:, None] - coords_w[None, :]  # H*W, H*W
        
        relative_coords_h += H - 1  # shift to start from 0
        relative_coords_w += W - 1
        relative_coords_h *= 2 * W - 1
        
        relative_position_index = torch.zeros(
            size=(H*W + 1, ) * 2, dtype=relative_coords_h.dtype
        )
        relative_position_index[1:, 1:] = relative_coords_h + relative_coords_w  # H*W, H*W
        relative_position_index[0, :] = self.num_relative_distance - 3
        relative_position_index[:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1
        # self.relative_position_index = relative_position_index
        self.register_buffer("relative_position_index", relative_position_index)
        
    def forward(self):
        if self.relative_position_bias_table is None or self.relative_position_index is None:
            return None
        
        H, W = self.window_size
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1) # type: ignore
        ].view(H * W + 1, H * W + 1, -1) # H*W, H*W, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous() # nH, H*W, H*W
        
        return relative_position_bias.unsqueeze(0)