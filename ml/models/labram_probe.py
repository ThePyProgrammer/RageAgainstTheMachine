"""LaBraM embedding probe for EEG motor imagery classification."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from eeg.layers.labram_encoder import LaBraMEncoder


class LaBraMProbe(nn.Module):
    def __init__(
        self,
        checkpoint_path: str | Path,
        channel_names: list[str],
        num_classes: int = 3,
        freeze_encoder: bool = True,
        unfreeze_last_n_blocks: int = 0,
        pooling: str = "mean",
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.encoder = LaBraMEncoder.from_pretrained(str(checkpoint_path))
        self.channel_names = channel_names
        self.pooling = pooling
        self.freeze_encoder = freeze_encoder
        self.unfreeze_last_n_blocks = max(0, int(unfreeze_last_n_blocks))

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if self.unfreeze_last_n_blocks > 0:
            total_blocks = len(self.encoder.model.blocks)
            n = min(self.unfreeze_last_n_blocks, total_blocks)
            for block in self.encoder.model.blocks[-n:]:
                for param in block.parameters():
                    param.requires_grad = True

            # Keep encoder output scale adaptable when any encoder blocks are trainable.
            for param in self.encoder.model.norm.parameters():
                param.requires_grad = True

        self.encoder_has_trainable_params = any(
            p.requires_grad for p in self.encoder.parameters()
        )

        embed_dim = self.encoder.model.embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, N, P, E)
        if self.pooling == "mean":
            return tokens.mean(dim=(1, 2))
        if self.pooling == "max":
            return tokens.amax(dim=(1, 2))
        raise ValueError(f"Unsupported pooling method: {self.pooling}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, T)
        if self.encoder_has_trainable_params:
            tokens = self.encoder(
                x, channel_names=self.channel_names, return_patch_tokens=True
            )
        else:
            with torch.no_grad():
                tokens = self.encoder(
                    x, channel_names=self.channel_names, return_patch_tokens=True
                )

        pooled = self._pool_tokens(tokens)
        return self.classifier(pooled)
