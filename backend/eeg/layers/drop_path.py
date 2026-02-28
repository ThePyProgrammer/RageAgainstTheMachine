import torch
import torch.nn as nn


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training:
            return x
        if self.drop_prob == 0.0:
            return x
    
        keep_prob = 1 - self.drop_prob
        # (B, 1, 1, 1, ...) mask broadcast along non-batch dims
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_mask = random_tensor.floor()
        return x.div(keep_prob) * binary_mask
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)