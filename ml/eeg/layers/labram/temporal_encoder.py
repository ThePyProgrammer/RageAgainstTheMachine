import torch.nn as nn
from einops import rearrange


__all__ = ["TemporalConvolutionBlock", "TemporalEncoder"]

class TemporalConvolutionBlock(nn.Module):
    """
    Temporal Convolution Block for Temporal Encoder.
    
    The temporal convolution block is composed of a 1D convolution layer,
    a group normalization layer, and a GELU activation function.
    
    Note: this implementation enforces the length of the first dimension,
    which is considered as the batch size, to remain unchanged after.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=(1, kernel_size),
                              stride=(1, stride),
                              padding=(0, padding))
        self.norm = nn.GroupNorm(4, out_channels)
        self.gelu = nn.GELU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.gelu(x)
        return x


class TemporalEncoder(nn.Module):
    """
    Temporal Encoder for EEG Patch Embedding.
    
    As EEG is of high resolution in the temporal domain, it is vital to extract 
    temporal features before patch-wise interaction by self-attention. We employ a
    temporal encoder which consists of several temporal convolution blocks to encode 
    each EEG patch into a patch embedding.
    
    Assuming an input of shape (B, N, P, T),
    where:
    - B = batch size
    - N = number of channels (electrodes)
    - P = number of patches
    - T = number of time points per patch
    
    This temporal encoder outputs a tensor of shape (B, N*P, T'),
    where T' is the reduced number of time points after temporal encoding.
    
    T' = (floor((T - 1)/C) + 1) * C = ceil(T / C) * C.
    
    Where C = out_channels as specified.
    
    To ensure T' = T, T MUST be divisible by C.
    """
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 8):
        super().__init__()
        
        self.in_channels = in_channels
        
        assert in_channels == 1, "Input channels must be 1 for Temporal Encoder."
        
        # To ensure that the output T' = T where T is divisible by out_channels C,
        # T' = out_channels * [floor((T + 2*padding - kernel_size ) / stride) + 1]
        # one solution identified:
        # - kernel_size = 2 * out_channels - 1
        # - stride = out_channels
        # - padding = (kernel_size - 1) // 2 = out_channels - 1
        # Example: kernel_size = 15, stride = 8, padding = 7 for out_channels = 8
        
        self.entry_block = TemporalConvolutionBlock(in_channels, out_channels,
                                                    kernel_size=2 * out_channels - 1,
                                                    stride=out_channels,
                                                    padding=out_channels - 1)
        
        self.encoder_blocks = nn.Sequential(
            TemporalConvolutionBlock(out_channels, out_channels),
            TemporalConvolutionBlock(out_channels, out_channels)
        )

    def forward(self, x):
        # x is of shape (B, N, P, T)
        # B = batch size
        # N = number of channels (electrodes)
        # P = number of patches
        # T = number of time points per patch
        # C = out_channels
        x = rearrange(x, 'B N P T -> B (N P) T')
        x = x.unsqueeze(1)
        # x is now B, 1, N*P, T
        x = self.entry_block(x)
        # x is now B, out_channels, N*P, T', where T' = floor((T - 1)/C) + 1
        x = self.encoder_blocks(x)
        # x is still B, out_channels, N*P, T'
        # C = out_channels
        x = rearrange(x, 'B C NP T -> B NP (T C)')
        # output of shape (B, N * P, C*(floor((T - 1)/C) + 1) * C) = (B, N * P, ceil(T / C) * C)
        return x