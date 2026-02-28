import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from eeg.layers.labram.utils import l2norm
from eeg.layers.labram.kmeans import kmeans


# ------------------------
# NeuralCodebook: codebook container
# ------------------------

class NeuralCodebook(nn.Module):
    """
    Maintains the codebook for vector quantization, updated via EMA.
    """
    def __init__(self,
                 codebook_size: int,
                 codebook_dim: int,
                 kmeans_init: bool = True,
                 codebook_init_path: str | None = None):
        super().__init__()

        # K: number of code vectors (i.e. size of codebook)
        self.codebook_size = codebook_size
        # D: dimension of each vector in the codebook
        self.codebook_dim = codebook_dim

        initialized = False
        weight = torch.zeros(codebook_size, codebook_dim)

        # Initialization path 1: load from a checkpoint
        if codebook_init_path is not None:
            print(f"load init codebook weight from {codebook_init_path}")
            weight = torch.load(codebook_init_path, map_location='cpu').clone()
            initialized = True

        # Initialization path 2: randomize L2-normalized vectors
        elif not kmeans_init:
            weight = torch.randn(codebook_size, codebook_dim)
            weight = l2norm(weight)
            initialized = True
            
        # Initialization path 3 (not present here): defer to k-means for initialization

        # Track whether initialization has happened (bool stored in buffer)
        self.register_buffer('initialized', torch.Tensor([initialized]))

        # Codebook weights: (K, D), not trainable by gradient
        self.weight = nn.Parameter(weight, requires_grad=False)
        
        # Cluster size statistics: (K,)
        self.cluster_size = nn.Parameter(torch.zeros(codebook_size), requires_grad=False)
        
        self.update = True # toggle for whether codebook updates should occur
    
    @torch.jit.ignore # type: ignore
    def init_embed_(self, data):
        """
        Initialize codebook weights using k-means on provided data.
        data: (N, D) flattened input vectors
        """
        if self.initialized:
            return
        
        print("Performing k-means initialization for codebook")

        # expected shape of data: (N, D)
        embed, cluster_size = kmeans(data,
                                     num_clusters=self.codebook_size,
                                     num_iters=10,
                                     use_cosine_sim=True)
        
        # (K, D)
        self.weight.data.copy_(embed)
        # (K,)
        self.cluster_size.data.copy_(cluster_size)
        self.initialized.fill_(True)
    
    def forward(self, embed_id: torch.Tensor):
        """
        Lookup codebook vectors by indices.
        embed_id: (N,), where each element is an index
        
        Returns (N, D), i.e. the vector corresponding to each index
        """
        return F.embedding(embed_id, self.weight)
    
    def nearest_neighbor_lookup(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute squared L2 distance between input vectors and codebook,
        and perform nearest neighbor lookup.
        
        Assuming x is of shape (N, D) and codebook is of shape (K, D),
        where:
        - N is the number of elements in x
        - K is the size of the codebook
        - D is the size of each vector in the codebook
        """
        # [0/2] if k-means, initialize codebook
        if not self.initialized:
            self.init_embed_(x)
        
        # [1/2] Compute distances between inputs and codebook
        # Standard Squared Euclidean distance,
        # |x_i|^2 + |e_j|^2 - 2<x_i, e_j>
        # (N, 1)
        x_squared = x.pow(2).sum(dim=1, keepdim=True)
        # (1, K)
        codebook_squared = self.weight.pow(2).sum(dim=1).unsqueeze(0)
        # (N, K)
        x_times_codebook = torch.einsum('nd,kd->nk', x, self.weight)
        # (N, K)
        dists = x_squared + codebook_squared - 2 * x_times_codebook
        
        # [2/2] Assign each vector to the nearest codebook entry
        # (N,)
        encoding_indices = torch.argmin(dists, dim=1)
        
        return encoding_indices
        

    def update_weight(self,
                      zero_mask: torch.Tensor,
                      new_weight: torch.Tensor,
                      decay: float):
        if not self.update:
            return
        
        # Replace empty clusters with old weights
        # to avoid dead codes
        new_weight = torch.where(
            zero_mask[..., None], # (K,)
            self.weight, # (K, D)
            new_weight # (K, D)
        )
        
        self.weight.data.mul_(decay).add_(new_weight, alpha=(1-decay))
        self.weight.data.copy_(l2norm(self.weight.data))


# ------------------------
# Quantizer module
# ------------------------

class NormEMAVectorQuantizer(nn.Module):
    def __init__(self,
                 codebook_size: int,
                 codebook_dim: int,
                 beta,
                 decay: float = 0.99,
                 statistic_code_usage: bool = True,
                 kmeans_init: bool = False,
                 codebook_init_path: str | None = None):
        super().__init__()
        
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self.beta = beta
        self.decay = decay
        
        self.codebook = NeuralCodebook(
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            kmeans_init=kmeans_init,
            codebook_init_path=codebook_init_path
        )
        
        # For monitoring cluster usage statistics
        self.statistic_code_usage = statistic_code_usage
        if statistic_code_usage:
            self.register_buffer('cluster_size', torch.zeros(codebook_size))
        
        # Placeholder for distributed reduction (multi-GPU training)
        self.all_reduce_fn = nn.Identity()
    
    def reset_cluster_size(self, device):
        if self.statistic_code_usage:
            self.register_buffer('cluster_size', torch.zeros(self.codebook_size))
            self.cluster_size = self.cluster_size.to(device)
    
    def compute_cluster_size(self, encodings: torch.Tensor):
        # Per-cluster counts
        # (K,)
        cluster_size = encodings.sum(0)
        self.all_reduce_fn(cluster_size)
        
        # EMA update
        self.cluster_size.data.mul_(self.decay).add_(cluster_size, alpha=(1-self.decay))

        return cluster_size
    
    def forward(self, z):
        """
        Outputs:
        - z_q: quantized output (B, C, H, W)
        - loss: scalar tensor
        - encoding_indices: (B*H*W,)
        """
        
        # z is of shape (B, C, H, W)
        # B = batch size
        # C = channels / feature dim (== codebook_dim)
        # H, W = spatial dimensions (e.g. patches)
        
        # Rearrange to channel-last for convenience
        # reshape z -> (B, H, W, C)
        z = rearrange(z, 'b c h w -> b h w c')
        
        # apply L2 normalization to vectors
        z = l2norm(z)
        
        # flatten z to shape (N, D)
        # where N = B * H * W
        z_flattened = z.reshape(-1, self.codebook_dim)
        
        # Nearest Neighbor Lookup
        # (N, D) -> (N,)
        encoding_indices = self.codebook.nearest_neighbor_lookup(z_flattened)

        # (N, K)
        encodings = F.one_hot(encoding_indices, self.codebook_size).type(z.dtype)
        
        # Quantize: lookup codebook vectors and reshape
        # (N, D) -> (B, H, W, C)
        z_q = self.codebook(encoding_indices).view(z.shape)

        # -------------------------
        # Codebook updates
        # -------------------------
        if self.training and self.codebook.update:
            # Compute per-cluster statistics
            
            cluster_size = self.compute_cluster_size(encodings)
            
            # Per-cluster embedding sum
            # Compute sum of all vectors assigned to each cluster
            # (D, N) @ (N, K) -> (D, K)
            embed_sum = z_flattened.t() @ encodings
            self.all_reduce_fn(embed_sum)
            
            # Handle empty clusters
            zero_mask = cluster_size == 0
            cluster_size = cluster_size.masked_fill(zero_mask, 1.0)
            
            # Normalize sums into means
            # (K, D) / (K, 1) -> (K, D)
            embed_norm = embed_sum.t() / cluster_size.unsqueeze(1)
            embed_norm = l2norm(embed_norm)
            
            self.codebook.update_weight(
                zero_mask=zero_mask,
                new_weight=embed_norm,
                decay=self.decay
            )
        
        if not self.training:
            with torch.no_grad():
                self.compute_cluster_size(encodings)
        
        # -----------------------------------
        # Commitment loss + straight-through
        # -----------------------------------

        # Compute commitment loss between 
        # encoder outputs and quantized vectors
        # prevents encoder outputs from wandering too far from codes
        # We want gradients to update the encoder parameters only
        # (not the codebook via gradients), so z_q.detach() stops 
        # gradients from flowing into the codebook.
        loss = self.beta * F.mse_loss(z_q.detach(), z)
        
        # Straight-through estimator
        # in the forward pass, this is simply z_q (the quantized vector)
        # The gradient w.r.t. upstream parameters is the gradient of the identity 
        # (i.e. gradients flow to z, assuming quantization was identity)
        # source: https://arxiv.org/abs/1308.3432
        z_q = z + (z_q - z).detach()
        
        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w')
        
        return z_q, loss, encoding_indices