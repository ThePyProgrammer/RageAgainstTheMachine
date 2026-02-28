import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import pytorch_lightning as pl

from eeg.layers.labram.vqnsp import VQNSP

class LaBraMVQNSP(pl.LightningModule):
    def __init__(self,
                 eeg_size: int = 1600,
                 patch_size: int = 200,
                 encoder_embed_dim: int = 200,
                 encoder_depth: int = 12,
                 decoder_embed_dim: int = 200,
                 decoder_depth: int = 3,
                 n_embed: int = 8192,
                 embed_dim: int = 32,
                 decay: float = 0.99,
                 quantize_kmeans_init: bool = True,
                 decoder_out_dim: int = 200,
                 smooth_l1_loss: bool = False):
        super().__init__()
        
        self.patch_size = patch_size
        
        self.vqnsp = VQNSP(
            eeg_size=eeg_size,
            patch_size=patch_size,
            encoder_embed_dim=encoder_embed_dim,
            encoder_depth=encoder_depth,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            n_embed=n_embed,
            embed_dim=embed_dim,
            decay=decay,
            quantize_kmeans_init=quantize_kmeans_init,
            decoder_out_dim=decoder_out_dim
        )
        
        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss
    
    def run_dft(self, x):
        x_fft = torch.fft.fft(x, dim=-1)
        
        amplitude = torch.abs(x_fft)
        mean_amplitude = torch.mean(amplitude, dim=(1,2,3), keepdim=True)
        std_amplitude = torch.std(amplitude, dim=(1,2,3), keepdim=True)
        amplitude = (amplitude - mean_amplitude) / std_amplitude
        
        angle = torch.angle(x_fft)
        mean_angle = torch.mean(angle, dim=(1,2,3), keepdim=True)
        std_angle = torch.std(angle, dim=(1,2,3), keepdim=True)
        angle = (angle - mean_angle) / std_angle
        
        return amplitude, angle
    
    def forward(self, x, input_channels = None):
        B, N, P, T = x.shape
        # B = batch size
        # N = number of channels (electrodes)
        # P = number of patches
        # T = patch_size
        assert T == self.patch_size, f"Input T (patch size) must be equal to the specified patch_size. Got T={T}, patch_size={self.patch_size}"
        
        amplitude, angle = self.run_dft(x)
        
        xrec, xrec_angle, emb_loss = self.vqnsp(x, input_channels=input_channels)
        
        target = rearrange(amplitude, 'b n a c -> b (n a) c')
        rec_loss = self.loss_fn(xrec, target)
        
        target_angle = rearrange(angle, 'b n a c -> b (n a) c')
        rec_angle_loss = self.loss_fn(xrec_angle, target_angle)
        
        return emb_loss, rec_loss, rec_angle_loss
    
    def training_step(self, batch, batch_idx):
        samples, input_channels = batch
        
        # TODO: figure out why it scales down by 100?
        samples = samples.float() / 100
        
        # Patching based on patch_size
        samples = rearrange(samples, "B N (P T) -> B N P T", T=self.patch_size)
        
        self.train()
        
        with torch.cuda.amp.autocast(enabled=True):
            emb_loss, rec_loss, rec_angle_loss = self(
                samples,
                input_channels=input_channels
            )
        
        total_loss = emb_loss + rec_loss + rec_angle_loss
        
        self.log("quant_loss", emb_loss, prog_bar=True)
        self.log("rec_loss", rec_loss, prog_bar=True)
        self.log("rec_angle_loss", rec_angle_loss, prog_bar=True)
        self.log("train_loss", total_loss, prog_bar=True)
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        samples, input_channels = batch
        
        # TODO: figure out why it scales down by 100?
        samples = samples.float() / 100
        
        # Patching based on patch_size
        samples = rearrange(samples, "B N (P T) -> B N P T", T=self.patch_size)
        
        self.eval()
        
        with torch.cuda.amp.autocast(enabled=True):
            emb_loss, rec_loss, rec_angle_loss = self(
                samples,
                input_channels=input_channels
            )
        
        total_loss = emb_loss + rec_loss + rec_angle_loss
        
        self.log("val_quant_loss", emb_loss, prog_bar=True)
        self.log("val_rec_loss", rec_loss, prog_bar=True)
        self.log("val_rec_angle_loss", rec_angle_loss, prog_bar=True)
        self.log("val_loss", total_loss, prog_bar=True)
        return total_loss
    
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


        