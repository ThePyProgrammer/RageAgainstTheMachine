import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "trunc_normal_",
    "l2norm",
    "sample_vectors",
    "init_weights_"
]

def trunc_normal_(tensor, mean=0., std=1.):
    nn.init.trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)
    return tensor

def l2norm(t: torch.Tensor) -> torch.Tensor:
    """
    Unit L2 Normalization along the last dimension of the tensor t.
    (*, ..., D)
    """
    return F.normalize(t, p=2, dim=-1)

def sample_vectors(samples: torch.Tensor, num: int) -> torch.Tensor:
    """
    Randomly sample `num` vectors from the dataset `samples`.
    - If samples >= num: sample without replacement.
    - If samples < num: sample with replacement.

    (N, D) -> (num, D)
    """
    num_samples = samples.shape[0]
    device = samples.device
    
    if num_samples >= num:
        # sample unique indices (i.e. without replacement)
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        # sample with replacement
        indices = torch.randint(0, num_samples, (num,), device=device)
    
    return samples[indices]

def init_weights_(m, init_std: float = 0.02):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=init_std)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        trunc_normal_(m.weight, std=init_std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)