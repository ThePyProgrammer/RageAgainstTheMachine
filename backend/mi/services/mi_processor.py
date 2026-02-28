import numpy as np
import logging
from typing import Optional, Callable
from scipy import signal

logger = logging.getLogger(__name__)


class MIProcessor:
    """Processes filtered EEG data for motor imagery classification.

    Accumulates filtered samples and triggers classification when enough
    data is available. Resamples epochs to match model's expected input.
    """

    def __init__(
        self,
        epoch_samples: int = 250,
        n_channels: int = 8,
        target_samples: int = 480,
        source_rate: float = 250.0,
        target_rate: float = 160.0,
        channel_indices: Optional[list[int]] = None,
    ):
        """Initialize MI processor.

        Args:
            epoch_samples: Number of samples to collect per epoch from live stream
            n_channels: Number of EEG channels
            target_samples: Number of samples expected by the model (after resampling)
            source_rate: Sampling rate of live EEG stream (Hz)
            target_rate: Sampling rate model was trained on (Hz)
            channel_indices: Optional input-channel indices used to subset/reorder
                incoming samples before buffering.
        """
        self.epoch_samples = epoch_samples
        self.n_channels = n_channels
        self.target_samples = target_samples
        self.source_rate = source_rate
        self.target_rate = target_rate
        self.channel_indices = channel_indices
        self.buffer = np.zeros((n_channels, 0))
        self.classification_callback: Optional[Callable] = None
        self.epoch_count = 0

        channel_mode = (
            f"subset/reorder={channel_indices}"
            if channel_indices is not None
            else "direct"
        )
        logger.info(
            f"[MI-Processor] Initialized - epoch_samples: {epoch_samples}, "
            f"n_channels: {n_channels}, will resample {epoch_samples}@{source_rate}Hz "
            f"to {target_samples}@{target_rate}Hz, mode: {channel_mode}"
        )

    def set_callback(self, callback: Callable):
        """Set callback function to be called when an epoch is ready.

        Args:
            callback: Function that takes (eeg_epoch: np.ndarray) as parameter
        """
        self.classification_callback = callback
        logger.info("[MI-Processor] Callback registered")

    def add_samples(self, samples: np.ndarray):
        """Add filtered EEG samples to the buffer.

        Args:
            samples: Filtered EEG data of shape (n_channels, n_samples)
        """
        working_samples = samples

        if self.channel_indices is not None:
            max_idx = max(self.channel_indices, default=-1)
            if max_idx >= samples.shape[0]:
                logger.warning(
                    "[MI-Processor] Cannot select channel indices %s from %s channels",
                    self.channel_indices,
                    samples.shape[0],
                )
                return
            working_samples = samples[self.channel_indices, :]

        if working_samples.shape[0] != self.n_channels:
            logger.warning(
                f"[MI-Processor] Channel mismatch: expected {self.n_channels}, "
                f"got {working_samples.shape[0]}"
            )
            return

        # Append to buffer
        self.buffer = np.hstack((self.buffer, working_samples))

        # Process complete epochs
        while self.buffer.shape[1] >= self.epoch_samples:
            # Extract one epoch
            epoch = self.buffer[:, : self.epoch_samples]

            # Remove from buffer
            self.buffer = self.buffer[:, self.epoch_samples :]

            # Resample epoch to match model's expected input
            try:
                resampled_epoch = signal.resample(epoch, self.target_samples, axis=1)

                logger.info(
                    f"[MI-Processor] Resampled epoch from {epoch.shape} to {resampled_epoch.shape}"
                )

                # Ensure exact size (handle rounding errors in resampling)
                if resampled_epoch.shape[1] != self.target_samples:
                    logger.warning(
                        f"[MI-Processor] Resampling produced {resampled_epoch.shape[1]} samples, "
                        f"trimming to {self.target_samples}"
                    )
                    resampled_epoch = resampled_epoch[:, : self.target_samples]
                elif resampled_epoch.shape[1] > self.target_samples:
                    # Force trim even if shape reports correct size (numerical precision issue)
                    logger.warning(
                        f"[MI-Processor] Force trimming from {resampled_epoch.shape[1]} to {self.target_samples}"
                    )
                    resampled_epoch = resampled_epoch[:, : self.target_samples]

                # Final validation
                assert resampled_epoch.shape == (
                    self.n_channels,
                    self.target_samples,
                ), (
                    f"Expected shape ({self.n_channels}, {self.target_samples}), got {resampled_epoch.shape}"
                )

                logger.info(
                    f"[MI-Processor] Final epoch shape: {resampled_epoch.shape}"
                )
            except Exception as e:
                logger.error(f"[MI-Processor] Resampling error: {e}", exc_info=True)
                continue

            # Process epoch
            if self.classification_callback:
                self.epoch_count += 1
                try:
                    self.classification_callback(resampled_epoch)
                except Exception as e:
                    logger.error(
                        f"[MI-Processor] Error in callback: {e}", exc_info=True
                    )

    def reset(self):
        """Reset the buffer and epoch count."""
        self.buffer = np.zeros((self.n_channels, 0))
        self.epoch_count = 0
        logger.info("[MI-Processor] Buffer reset")

    def get_stats(self):
        """Get processor statistics."""
        return {
            "buffer_samples": self.buffer.shape[1],
            "epochs_processed": self.epoch_count,
            "epoch_samples": self.epoch_samples,
            "n_channels": self.n_channels,
        }
