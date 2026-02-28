import torch
from einops import rearrange, repeat
from eeg.layers.labram.utils import l2norm, sample_vectors


__all__ = [
    "kmeans"
]

def kmeans(samples: torch.Tensor,
           num_clusters: int,
           num_iters: int = 10,
           use_cosine_sim: bool = False):
    """
    Run K-means clustering on the input samples.
    
    Assuming samples is of shape (N, D), this returns:
    - means (num_clusters, D)
    - bins (num_clusters,)
    """
    dim = samples.shape[-1]
    dtype = samples.dtype

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            # cosine similarity: (N, num_clusters)
            dists = samples @ means.t()
        else:
            # negative squared L2 distance: (N, num_clusters)
            diffs = rearrange(samples, 'n d -> n () d') - rearrange(means, 'c d -> () c d')
            dists = -(diffs ** 2).sum(dim=-1)

        # assign clusters: (N,)
        buckets = dists.max(dim=-1).indices

        # cluster counts: (num_clusters,)
        bins = torch.bincount(buckets, minlength=num_clusters)

        # avoid divide by 0
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        # accumulate sums per cluster: (num_clusters, D)
        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, 'n -> n d', d=dim), samples)
        new_means /= bins_min_clamped[..., None]

        if use_cosine_sim:
            new_means = l2norm(new_means)

        # keep old mean if cluster is empty
        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins # type: ignore