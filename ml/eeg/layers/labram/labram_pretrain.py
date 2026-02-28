import torch
import torch.nn as nn
from einops import rearrange

from functools import partial
import pytorch_lightning as pl

from eeg.layers.labram.neural_transformer import NeuralTransformer
from eeg.layers.labram.masking_process import random_mask_generation


__all__ = ["LaBraMPretrain"]

class LaBraMPretrain(pl.LightningModule):
    def __init__(self,
                 eeg_size: int = 1600,
                 patch_size: int = 200,
                 embed_dim: int = 200,
                 vocab_size: int = 8192,
                 in_channels: int = 1,
                 out_channels: int = 8,
                 depth: int = 12,
                 num_heads: int = 10,
                 mlp_ratio: float = 4.0,
                 use_qkv_bias: bool = True,
                 qk_norm = partial(nn.LayerNorm, eps=1e-6),
                 qk_scale = None,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 norm_layer = partial(nn.LayerNorm, eps=1e-6),
                 init_values: float = 0.1,
                 attn_head_dim: int | None = None,
                 use_abs_pos_emb: bool = True,
                 init_std: float = 0.02
                 ):
        super().__init__()
        
        self.patch_size = patch_size
        
        self.student = NeuralTransformer(
            eeg_size=eeg_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            vocab_size=vocab_size,
            in_channels=in_channels,
            out_channels=out_channels,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            use_qkv_bias=use_qkv_bias,
            qk_norm=qk_norm,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer, # type: ignore
            init_values=init_values,
            attn_head_dim=attn_head_dim,
            use_abs_pos_emb=use_abs_pos_emb,
            init_std=init_std
        )
        
        # TODO: implement vqnsp properly
        self.vqnsp = None
        
        # self.projection_head = nn.Sequential(
        #     nn.Linear(embed_dim, embed_dim),
        #     nn.ReLU()
        # )
        
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, mask = None, input_channels = None):
        x_rec = self.student(
            x, mask=mask, input_channels=input_channels
        )
        
        # symmetric
        x_rec_sym = self.student(
            x, mask=~mask, input_channels=input_channels # type: ignore
        )
        
        return x_rec, x_rec_sym

    def training_step(self, batch, batch_idx):
        samples, input_channels = batch
        
        # TODO: figure out why it scales down by 100?
        samples = samples.float() / 100
        
        # Patching based on patch_size
        samples = rearrange(samples, "B N (P T) -> B N P T", T=self.patch_size)
        
        mask = random_mask_generation(samples.flatten(1, 2), mask_ratio=0.5)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                input_ids = self.vqnsp.get_codebook_indices(samples, input_channels)
            
            labels = input_ids[mask]
            labels_sym = input_ids[~mask]
        
        self.train()
        
        outputs = self(
            samples,
            mask=mask,
            input_channels=input_channels
        )
        
        x_rec, x_rec_sym = outputs
        
        loss_rec = self.loss_fn(x_rec, labels)
        loss_rec_sym = self.loss_fn(x_rec_sym, labels_sym)
        
        loss = loss_rec + loss_rec_sym
        
        self.log("train_loss", loss, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        samples, input_channels = batch
        
        # TODO: figure out why it scales down by 100?
        samples = samples.float() / 100
        
        # Patching based on patch_size
        samples = rearrange(samples, "B N (P T) -> B N P T", T=self.patch_size)
        
        mask = random_mask_generation(samples.flatten(1, 2), mask_ratio=0.5)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                input_ids = self.vqnsp.get_codebook_indices(samples, input_channels)
            
            labels = input_ids[mask]
            labels_sym = input_ids[~mask]

        self.eval()
        
        outputs = self(
            samples,
            mask=mask,
            input_channels=input_channels
        )
        
        x_rec, x_rec_sym = outputs
        
        loss_rec = self.loss_fn(x_rec, labels)
        loss_rec_sym = self.loss_fn(x_rec_sym, labels_sym)
        
        val_loss = loss_rec + loss_rec_sym
        
        self.log('val_loss', val_loss, prog_bar=True)
        
        return val_loss
    
    # TODO: actually implement this
    # def configure_optimizers(self): # type: ignore
    #     optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #         optimizer, mode='min', factor=0.5, patience=5
    #     )
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": {
    #             "scheduler": scheduler,
    #             "monitor": "val_loss",
    #             "frequency": 1
    #         }
    #     }