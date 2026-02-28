import sys
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import numpy as np
import cv2
from typing import Optional

from shared.config.logging import get_logger

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../external/")
)
TOOLBOX_PATH = os.path.join(PROJECT_ROOT, "rPPG-Toolbox")
sys.path.append(TOOLBOX_PATH)

from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX

logger = get_logger(__name__)


class DLInference:
    def __init__(self, model_path: str, device: str = "mps"):
        self.device = torch.device(device)
        self.model = None
        self.buffer = []
        self.buffer_size = (
            130  # PhysNet usually needs 128 frames (power of 2) + overlap
        )
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        try:
            # Initialize PhysNet (standard config usually 128 frames)
            self.model = PhysNet_padding_Encoder_Decoder_MAX(frames=128).to(self.device)
            # Load weights
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint)
                self.model.eval()
                logger.info("Loaded PhysNet model from %s", self.model_path)
            else:
                logger.warning("Model weights not found at %s", self.model_path)
        except Exception as e:
            logger.error("Error loading model: %s", e, exc_info=True)
            self.model = None

    def preprocess_frame(self, frame_img: np.ndarray) -> torch.Tensor:
        """
        Preprocess frame for PhysNet (Spatial Step):
        1. Resize to 128x128
        2. Convert to Tensor (C, H, W)
        3. Keep as Float (0-255) for downstream temporal DiffNormalization
        """
        # 1. Resize to 128x128 as expected by the model architecture
        resized = cv2.resize(frame_img, (128, 128))

        # 2. Convert to Float Tensor
        # Keep the range 0-255 here because the temporal difference formula
        # (C(t+1) - C(t)) / (C(t+1) + C(t)) is scale-invariant,
        # and retaining larger magnitudes avoids precision loss before the division.
        tensor_img = torch.from_numpy(resized).float()

        # 3. Permute Dimensions: [H, W, C] -> [C, H, W]
        tensor_img = tensor_img.permute(2, 0, 1)

        return tensor_img

    def add_frame(self, frame_img: np.ndarray) -> Optional[float]:
        """
        Add frame to buffer and run inference if buffer is full.
        Returns the latset BVP value if available.
        """
        if self.model is None:
            return None

        tensor_img = self.preprocess_frame(frame_img)
        self.buffer.append(tensor_img)

        # Sliding window inference
        if len(self.buffer) >= 128:
            # Create batch
            # PhysNet input: [B, 3, T, H, W]
            # Current buffer: list of [3, 128, 128] tensors

            # Stack to [T, 3, H, W] -> [128, 3, 128, 128]
            t_tensor = torch.stack(self.buffer[-128:])

            # Compute Diff Normalized Data
            # D(t) = (I(t) - I(t-1)) / (I(t) + I(t-1))
            # Output T will be 127

            # Shifted tensors
            curr = t_tensor[1:]  # t from 1 to 127
            prev = t_tensor[:-1]  # t from 0 to 126

            diff = (curr - prev) / (curr + prev + 1e-7)

            # Standardize
            std = torch.std(diff)
            if std > 0:
                diff = diff / std

            diff[torch.isnan(diff)] = 0

            # Shape is now [127, 3, 128, 128]
            # PhysNet needs [B, 3, T, H, W]
            # Permute to [1, 3, 127, 128, 128]
            input_tensor = diff.permute(1, 0, 2, 3).unsqueeze(0)

            # Pad to 128 frames if model expects strictly 128
            # The loaded model is 'PhysNet_padding_Encoder_Decoder_MAX(frames=128)'
            # It expects T=128. If we give 127, it might fail or padding layer handles it.
            # Pad the last frame again to reach 128
            last_frame = input_tensor[:, :, -1:, :, :]
            input_tensor = torch.cat((input_tensor, last_frame), dim=2)

            input_tensor = input_tensor.to(self.device)

            with torch.no_grad():
                rppg, _, _, _ = self.model(input_tensor)
                # rppg is [B, T] -> [1, 128]

                # Get the last value
                bvp_seq = rppg.cpu().detach().numpy()[0]
                return float(bvp_seq[-1])

        return None
