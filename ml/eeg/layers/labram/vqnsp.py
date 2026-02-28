"""
Vector-Quantized Neural Spectrum Prediction (VQ-NSP).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from eeg.layers.labram.neural_transformer import NeuralTransformer
from eeg.layers.labram.norm_ema_quantizer import NormEMAVectorQuantizer
from eeg.layers.labram.utils import init_weights_


def make_task_layer(dim1: int, dim2: int):
    task_layer = nn.Sequential(
        nn.Linear(dim1, dim1),
        nn.Tanh(),
        nn.Linear(dim1, dim2)
    )
    task_layer.apply(init_weights_)
    return task_layer

class VQNSP(nn.Module):
    def __init__(self,
                 eeg_size: int = 1600,
                 patch_size: int = 200,
                 encoder_embed_dim: int = 200,
                 encoder_depth: int = 12,
                 decoder_embed_dim: int = 200,
                 decoder_depth: int = 3,
                 codebook_size: int = 8192,
                 codebook_dim: int = 32,
                 decay: float = 0.99,
                 quantize_kmeans_init: bool = True,
                 decoder_out_dim: int = 200):
        super().__init__()
        
        self.patch_size = patch_size
        
        self.encoder = NeuralTransformer(
            eeg_size=eeg_size,
            patch_size=patch_size,
            embed_dim=encoder_embed_dim,
            in_channels=1,
            out_channels=8,
            depth=encoder_depth
        )
        
        self.quantizer = NormEMAVectorQuantizer(
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            beta=1.0,
            decay=decay,
            kmeans_init=quantize_kmeans_init
        )
        
        self.decoder = NeuralTransformer(
            eeg_size=eeg_size // patch_size,
            patch_size=1,
            embed_dim=decoder_embed_dim,
            in_channels=codebook_dim,
            out_channels=8,
            depth=decoder_depth
        )
        
        # task layer
        self.encode_task_layer = make_task_layer(encoder_embed_dim, codebook_dim)
        self.decode_task_layer = make_task_layer(decoder_embed_dim, decoder_out_dim)
        self.decode_task_layer_angle = make_task_layer(decoder_embed_dim, decoder_out_dim)

    def encode(self, x, input_channels: list | None = None):
        N = x.shape[1]
        # N = number of channels (electrodes)
        
        encoder_features = self.encoder(
            x,
            input_channels=input_channels,
            return_patch_tokens=True
        )
        # (B, N*P, encoder_embed_dim)
        
        with torch.cuda.amp.autocast(enabled=False):
            to_quantizer_features = self.encode_task_layer(encoder_features.type_as(self.encode_task_layer[-1].weight))
            # (B, N*P, codebook_dim)

        h, w = N, to_quantizer_features.shape[1] // N
        to_quantizer_features = rearrange(to_quantizer_features, 'b (h w) c -> b c h w', h=h, w=w)
        quantize, loss, embed_ind = self.quantizer(to_quantizer_features)
        
        return quantize, embed_ind, loss
    
    def decode(self, quantize, input_channels: list | None = None):
        decoder_features = self.decoder(
            quantize,
            input_channels=input_channels,
            return_patch_tokens=True
        )
        
        rec = self.decode_task_layer(decoder_features)
        rec_angle = self.decode_task_layer_angle(decoder_features)
        return rec, rec_angle
    
    def forward(self, x, input_channels: list | None = None):
        B, N, P, T = x.shape
        # B = batch size
        # N = number of channels (electrodes)
        # P = number of patches
        # T = patch_size
        assert T == self.patch_size, f"Input T (patch size) must be equal to the specified patch_size. Got T={T}, patch_size={self.patch_size}"
        
        quantize, embed_ind, emb_loss = self.encode(x, input_channels=input_channels)
        
        xrec, xrec_angle = self.decode(quantize, input_channels=input_channels)
        
        return xrec, xrec_angle, emb_loss
    
    def get_codebook_indices(self, x, input_channels: list | None = None):
        # used for LaBraM pre-training
        quantize, embed_ind, loss = self.encode(x, input_channels=input_channels)
        return embed_ind.view(x.shape[0], -1)