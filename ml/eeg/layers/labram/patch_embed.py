import torch.nn as nn

class PatchEmbed(nn.Module):
    """
    EEG to Patch Embedding
    """
    def __init__(self,
                 eeg_size: int = 2000,
                 patch_size: int = 200,
                 in_channels: int = 1,
                 embed_dim: int = 200):
        super().__init__()
        self.eeg_size = eeg_size
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=(1, patch_size), stride=(1, patch_size))

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # B = batch size
        # C = number of channels (electrodes)
        # H = 1
        # W = number of time points (eeg_size)
        x = self.proj(x)
        # B, embed_dim, 1, W'
        # W' = eeg_size // patch_size = num_patches
        x = x.flatten(2)
        # B, embed_dim, W'
        x = x.transpose(1, 2)
        # B, num_patches, embed_dim
        return x