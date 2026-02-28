import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import signal
from sklearn.manifold import TSNE

from shared.config.app_config import BOARD_SAMPLING_RATE
from shared.config.logging import get_logger
from eeg.layers.labram_encoder import LaBraMEncoder

logger = get_logger(__name__)


class EmbeddingProcessor:
    """
    Accumulates 1600 samples at 250Hz (Cyton Board Streaming Frequency) and processes as tensor:
    [batch_size, num_channels, sequence_length]
    """

    def __init__(self, window_size=3200, channel_names=None, channel_mapping=None):
        """
        Args:
            window_size: Number of samples to accumulate (default: 3200 = 12.8 seconds at 250Hz)
            channel_names: List of channel names matching 10-20 system
            channel_mapping: Dict mapping electrode IDs to channel IDs (e.g., {'FP1': '1', 'C3': '3'})
        """
        self.window_size = window_size
        self.sampling_rate = BOARD_SAMPLING_RATE
        self.channel_names = channel_names or []  # No default channels
        self.channel_mapping = channel_mapping or {}  # {electrode_id: channel_id}
        self.num_channels = len(self.channel_names)

        # If we have a mapping, compute channel indices to extract
        # channel_mapping: {'FP1': '1', 'C3': '3'} means use channels 0 and 2 (0-indexed)
        self.channel_indices = (
            self._compute_channel_indices() if channel_mapping else list(range(8))
        )

        # Rolling buffer for accumulating samples
        # Buffer stores data as (num_channels, sequence_length)
        self.buffer = deque(maxlen=window_size)
        self.lock = threading.Lock()

        # Model
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Embedding storage
        self.latest_embedding = None
        self.embedding_history = []

        # Runtime tracking
        self.start_time = None
        self.elapsed_time = 0.0

        # Processing control
        self.enabled = False
        self.processing_thread = None
        self._stop_event = threading.Event()

    def _compute_channel_indices(self):
        """Compute which channel indices to extract based on mapping."""
        if not self.channel_mapping:
            return []

        # channel_mapping: {'FP1': '1', 'C3': '3'}
        # We want to extract channels [0, 2] (0-indexed)
        indices = [int(ch_id) - 1 for ch_id in self.channel_mapping.values()]
        indices.sort()  # Keep in order
        logger.debug(
            "[EmbeddingProcessor] Using channel indices: %s for electrodes: %s",
            indices,
            list(self.channel_mapping.keys()),
        )
        return indices

    def _resample_eeg(
        self, eeg_data: np.ndarray, original_fs: int, target_fs: int
    ) -> np.ndarray:
        """Resample EEG data from original to target sampling rate."""
        if original_fs == target_fs:
            return eeg_data
        n_channels = eeg_data.shape[0]
        n_samples_target = int(eeg_data.shape[1] * target_fs / original_fs)
        resampled = np.zeros((n_channels, n_samples_target), dtype=eeg_data.dtype)
        for ch in range(n_channels):
            resampled[ch] = signal.resample(eeg_data[ch], n_samples_target)
        return resampled

    def _normalize_eeg(self, eeg_data: np.ndarray) -> np.ndarray:
        """Normalize EEG data per channel to zero-mean, unit-variance."""
        mean = eeg_data.mean(axis=1, keepdims=True)
        std = eeg_data.std(axis=1, keepdims=True)
        std = np.where(std == 0, 1.0, std)
        return (eeg_data - mean) / std

    def load_model(self, checkpoint_path: str | Path):
        logger.info("[EmbeddingProcessor] Loading model from %s", checkpoint_path)

        # Convert to Path object
        checkpoint = Path(checkpoint_path)

        # If the path is relative, resolve it from the backend directory
        if not checkpoint.is_absolute():
            # Get the backend directory (three levels up: services -> bci -> backend)
            backend_dir = Path(__file__).resolve().parent.parent.parent
            checkpoint = backend_dir / checkpoint

        logger.info("[EmbeddingProcessor] Resolved checkpoint path: %s", checkpoint)

        if "labram" in str(checkpoint_path).lower():
            self.model = LaBraMEncoder.from_pretrained(str(checkpoint))
        else:
            raise ValueError(
                f"Model not implemented for checkpoint: {checkpoint_path}. Use LaBraM model."
            )

        self.model.to(self.device)
        self.model.eval()

        # Initialize projection head from checkpoint if available
        checkpoint_data = torch.load(str(checkpoint), map_location=self.device)

        if "model_state_dict" in checkpoint_data:
            self.model.load_state_dict(checkpoint_data["model_state_dict"])

        logger.info("[EmbeddingProcessor] Model loaded on %s", self.device)

    def add_samples(self, samples: np.ndarray):
        """
        Add new samples to the buffer.

        Args:
            samples: Array of shape (8, num_samples) - all 8 channels
        """
        # Skip if no channels configured
        if len(self.channel_indices) == 0:
            return

        # Filter to only the channels we care about
        if len(self.channel_indices) < 8:
            samples = samples[self.channel_indices, :]

        with self.lock:
            # Transpose to (num_samples, num_channels) and add each sample
            samples_transposed = samples.T  # (num_samples, num_channels)
            for sample in samples_transposed:
                self.buffer.append(sample)

    def is_ready(self) -> bool:
        """Check if buffer has enough samples for processing."""
        return len(self.buffer) >= self.window_size

    def process_window(self, store_history=True) -> dict:
        """
        Process accumulated window through selected model.

        Returns:
            dict: {
                'embeddings': (B, N, P, vocab_size) tensor,
                'window_data': (num_channels, window_size) array,
                'timestamp': processing timestamp
            }
        """
        if not self.is_ready():
            return None

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        with self.lock:
            # Convert buffer to numpy array
            window_data = np.array(list(self.buffer))  # (window_size, num_channels)
            window_data = window_data.T  # (num_channels, window_size)

            # Resample + normalize to match model expectations
            # Model expects 200 Hz sampling
            original_fs = self.sampling_rate
            target_fs = 200
            window_data = self._resample_eeg(window_data, original_fs, target_fs)
            window_data = self._normalize_eeg(window_data)

        # Convert to tensor and add batch dimension
        # Shape: [batch_size=1, num_channels, sequence_length=1600]
        x = torch.from_numpy(window_data).float()  # (num_channels, window_size)
        x = x.unsqueeze(0)  # (1, num_channels, window_size)
        x = x.to(self.device)

        # Process through model
        with torch.no_grad():
            embeddings = self.model(x, channel_names=self.channel_names)
            # embeddings: (B, N, P, vocab_size) = (1, 8, 8, 8192)

        # Convert to numpy for processing
        embeddings_np = embeddings.cpu().numpy()
        B, N, P, V = embeddings_np.shape

        # Apply t-SNE to reduce vocab_size dimension: 8192 -> 2
        # Reshape to (B*N*P, vocab_size) for t-SNE
        embeddings_reshaped = embeddings_np.reshape(-1, V)  # (64, 8192)

        # t-SNE reduction
        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=min(30, embeddings_reshaped.shape[0] - 1),
        )
        embeddings_2d = tsne.fit_transform(embeddings_reshaped)  # (64, 2)

        # Reshape back to (B, N, P, 2)
        embeddings_reduced = embeddings_2d.reshape(B, N, P, 2)  # (1, 8, 8, 2)

        # Average across channels and patches for single 2D point per window
        # This gives us one (x, y) coordinate for visualization
        embeddings_avg = embeddings_reduced.mean(
            axis=(1)
        )  # (1, 2) -> average over N and P
        embeddings_avg = torch.tensor(embeddings_avg.flatten())

        result = {
            "raw": {
                "embeddings": embeddings_np,  # (B, N, P, V=8192)
                "embeddings_flat": embeddings_np.flatten(),  # 524,288 dims
                "shape": {
                    "batch_size": B,
                    "num_channels": N,
                    "num_patches": P,
                    "vocab_size": V,
                },
            },
            "reduced": {
                "embeddings_2d": embeddings_avg.tolist(),  # [x, y] - single 2D point for plotting
                "embeddings_full": embeddings_reduced,  # (B, N, P, 2) - full structure
                "shape": {
                    "batch_size": B,
                    "num_channels": N,
                    "num_patches": P,
                    "reduced_dims": 2,
                },
            },
            "window_data": window_data,
        }

        # Update elapsed time
        if self.start_time is not None:
            self.elapsed_time = time.time() - self.start_time

        # Store latest embedding
        self.latest_embedding = result

        # Optionally store in history
        if store_history:
            self.embedding_history.append(result)
            # Keep only last 100 embeddings to prevent memory issues
            if len(self.embedding_history) > 100:
                self.embedding_history.pop(0)

        return result

    def get_latest_embedding(self):
        """Get the most recent embedding."""
        return self.latest_embedding

    def get_embedding_history(self, n=None):
        """
        Get embedding history.

        Args:
            n: Number of recent embeddings to return (None = all)
        """
        if n is None:
            return self.embedding_history
        return self.embedding_history[-n:]

    def get_embeddings_as_dataset(self, reduced=True):
        """
        Get all embeddings as a stacked numpy array for ML tasks or visualization.

        Args:
            reduced: If True, return 2D points (num_windows, 2) for visualization,
                    else return full embeddings (num_windows, 524288)

        Returns:
            np.ndarray: Stacked embeddings
        """
        if not self.embedding_history:
            return None

        if reduced:
            # Stack 2D points: (num_windows, 2) - ready for scatter plot
            embeddings_array = np.array(
                [emb["reduced"]["embeddings_2d"] for emb in self.embedding_history]
            )
        else:
            # Stack full embeddings: (num_windows, 524288)
            embeddings_array = np.vstack(
                [emb["raw"]["embeddings_flat"] for emb in self.embedding_history]
            )

        return embeddings_array

    def clear_buffer(self):
        """Clear the sample buffer."""
        with self.lock:
            self.buffer.clear()

    def reset(self):
        """Reset processor state."""
        self.clear_buffer()
        self.latest_embedding = None
        self.embedding_history = []

    def _processing_loop(self):
        """Background thread that continuously processes embeddings."""
        logger.info("[EmbeddingProcessor] Processing loop started")

        while not self._stop_event.is_set():
            if self.is_ready():
                try:
                    self.process_window(store_history=True)
                except Exception:
                    logger.exception("[EmbeddingProcessor] Error processing window")

            # Sleep briefly to avoid busy waiting
            self._stop_event.wait(0.1)

        logger.info("[EmbeddingProcessor] Processing loop stopped")

    def enable(self):
        """Start continuous embedding processing in background thread."""
        if self.enabled:
            logger.info("[EmbeddingProcessor] Already enabled")
            return

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self.enabled = True
        self.start_time = time.time()
        self._stop_event.clear()
        self.processing_thread = threading.Thread(
            target=self._processing_loop, daemon=True
        )
        self.processing_thread.start()
        logger.info("[EmbeddingProcessor] Continuous processing enabled")

    def disable(self):
        """Stop continuous embedding processing."""
        if not self.enabled:
            return

        self.enabled = False
        self.start_time = None
        self.elapsed_time = 0.0
        self._stop_event.set()

        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)

        logger.info("[EmbeddingProcessor] Continuous processing disabled")


embeddingProcessor = EmbeddingProcessor()
