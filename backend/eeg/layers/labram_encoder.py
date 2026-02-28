import os
from functools import partial
from pathlib import Path
import torch
from torch import nn
from einops import rearrange

from eeg.layers.labram.neural_transformer import NeuralTransformer
from eeg.layers.labram.constants import CHANNEL_NAMES


class LaBraMEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = NeuralTransformer(
            num_heads=10,
            qk_norm=partial(nn.LayerNorm, eps=1e-6)
        )

    @classmethod
    def from_pretrained(cls, checkpoint_path: str | Path):
        # Verify checkpoint exists
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Pre-trained LaBraM checkpoint not found at: {checkpoint_path}")
        
        # Load checkpoint to extract feature dimensions
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Create the encoder with pre-trained weights
        encoder = cls()

        # Extract state dict
        if 'model_state_dict' in checkpoint_data:
            full_state_dict = checkpoint_data['model_state_dict']
        elif 'model' in checkpoint_data:
            full_state_dict = checkpoint_data['model']
        else:
            full_state_dict = checkpoint_data

        state_keys = list(full_state_dict.keys())
        has_student_prefix = any(k.startswith('student.') for k in state_keys)
        has_patch_keys = any(k.startswith('patch_embed.') or k.startswith('blocks.') for k in state_keys)
        has_core_tokens = 'cls_token' in full_state_dict and 'mask_token' in full_state_dict

        # If the checkpoint already matches this encoder, load directly.
        if not has_student_prefix and not (has_core_tokens and has_patch_keys):
            missing, unexpected = encoder.load_state_dict(full_state_dict, strict=False)
            if missing or unexpected:
                print(f"[LaBraMEncoder] Loaded with missing keys: {missing}, unexpected keys: {unexpected}")
            return encoder

        # Load pre-trained patch embedding weights (student-style or flat keys)
        if has_student_prefix:
            student_keys = {k.replace('student.', ''): v for k, v in full_state_dict.items() if k.startswith('student.')}
        else:
            student_keys = full_state_dict

        model_dict = {}

        for k, v in student_keys.items():
            path = k.split(".")
            current_node = model_dict
            for el_idx in range(len(path)-1):
                current_node[path[el_idx]] = current_node.get(path[el_idx], {})
                current_node = current_node[path[el_idx]]

            current_node[path[-1]] = v

        encoder.model.append_cls_token.cls_token.data = model_dict["cls_token"]
        encoder.model.masking_process.mask_token.data = model_dict["mask_token"]
        
        encoder.model.embedding.pos_embed.data = model_dict["pos_embed"].squeeze(0)
        encoder.model.embedding.time_embed.data = model_dict["time_embed"].squeeze(0)
        
        encoder.model.patch_embed.entry_block.conv.weight.data = model_dict["patch_embed"]["conv1"]["weight"]
        encoder.model.patch_embed.entry_block.conv.bias.data = model_dict["patch_embed"]["conv1"]["bias"]
        encoder.model.patch_embed.entry_block.norm.weight.data = model_dict["patch_embed"]["norm1"]["weight"]
        encoder.model.patch_embed.entry_block.norm.bias.data = model_dict["patch_embed"]["norm1"]["bias"]

        encoder.model.patch_embed.encoder_blocks[0].conv.weight.data = model_dict["patch_embed"]["conv2"]["weight"]
        encoder.model.patch_embed.encoder_blocks[0].conv.bias.data = model_dict["patch_embed"]["conv2"]["bias"]
        encoder.model.patch_embed.encoder_blocks[0].norm.weight.data = model_dict["patch_embed"]["norm2"]["weight"]
        encoder.model.patch_embed.encoder_blocks[0].norm.bias.data = model_dict["patch_embed"]["norm2"]["bias"]

        encoder.model.patch_embed.encoder_blocks[1].conv.weight.data = model_dict["patch_embed"]["conv3"]["weight"]
        encoder.model.patch_embed.encoder_blocks[1].conv.bias.data = model_dict["patch_embed"]["conv3"]["bias"]
        encoder.model.patch_embed.encoder_blocks[1].norm.weight.data = model_dict["patch_embed"]["norm3"]["weight"]
        encoder.model.patch_embed.encoder_blocks[1].norm.bias.data = model_dict["patch_embed"]["norm3"]["bias"]

        for block_idx in range(12):
            block = encoder.model.blocks[block_idx]
            block_weights = model_dict["blocks"][str(block_idx)]
            block.gamma_1.data = block_weights["gamma_1"]
            block.gamma_2.data = block_weights["gamma_2"]
            
            block.norm1.weight.data = block_weights["norm1"]["weight"]
            block.norm1.bias.data = block_weights["norm1"]["bias"]
            block.norm2.weight.data = block_weights["norm2"]["weight"]
            block.norm2.bias.data = block_weights["norm2"]["bias"]
            
            block.attn.qkv.weight.data = block_weights["attn"]["qkv"]["weight"]
            block.attn.q_norm.weight.data = block_weights["attn"]["q_norm"]["weight"]
            block.attn.q_norm.bias.data = block_weights["attn"]["q_norm"]["bias"]
            block.attn.k_norm.weight.data = block_weights["attn"]["k_norm"]["weight"]
            block.attn.k_norm.bias.data = block_weights["attn"]["k_norm"]["bias"]
            block.attn.proj.weight.data = block_weights["attn"]["proj"]["weight"]
            block.attn.proj.bias.data = block_weights["attn"]["proj"]["bias"]
            
            block.mlp[0].weight.data = block_weights["mlp"]["fc1"]["weight"]
            block.mlp[0].bias.data = block_weights["mlp"]["fc1"]["bias"]
            block.mlp[2].weight.data = block_weights["mlp"]["fc2"]["weight"]
            block.mlp[2].bias.data = block_weights["mlp"]["fc2"]["bias"]

        encoder.model.norm.weight.data = model_dict["norm"]["weight"]
        encoder.model.norm.bias.data = model_dict["norm"]["bias"]

        encoder.model.lm_head.weight.data = model_dict["lm_head"]["weight"]
        encoder.model.lm_head.bias.data = model_dict["lm_head"]["bias"]

        return encoder
    
    def forward(self, x: torch.Tensor, channel_names: list[str]):
        """
        Expected Format(s) for $x$:
        - (number of channels, sequence length)
        - (batch size, number of channels, sequence length)
        """
        assert 2 <= x.ndim <= 3, f"Tensor has {x.ndim} dimensions, expected on 2 or 3."
        if x.ndim == 2:
            # x is (N, SL)
            x = x.unsqueeze(0)
        
        # x is (B, N, SL)
        B, N, SL = x.size()
        assert N == len(channel_names), f"Tensor has {N} channels, provided {len(channel_names)} channel names."
        
        T = self.model.patch_size
        P = min(SL // T, 16)
        x = x[:, :, -P*T:]
        x = rearrange(x, "B N (P T) -> B N P T", T=T)
        
        chs = []
        for channel in channel_names:
            # Case-insensitive lookup
            channel_upper = channel.upper()
            assert channel_upper in CHANNEL_NAMES, f"{channel} not found in valid channel list."
            chs.append(CHANNEL_NAMES.index(channel_upper)+1)
        # x is (B, N, P, T)
        y = self.model(x, input_channels=chs)
        # y is (B, N*P, vocab_size)
        y = rearrange(y, "B (N P) V -> B N P V", N=N)
        # y is (B, N, P, vocab_size)
        return y