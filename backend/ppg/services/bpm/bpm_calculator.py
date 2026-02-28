"""
Signal processing module for Pulse/SpO2 detection.
Extracts BVP, BPM, and SpO2 from RGB signals.
"""

from typing import List, Optional, Tuple
import numpy as np
import csv
import datetime
import sys
import os
import torch
from scipy import signal
from ppg.services.bpm import rppg_algorithms
from ppg.services.bpm.dl_inference import DLInference
from shared.config.logging import get_logger

logger = get_logger(__name__)

# Add rPPG-Toolbox to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
TOOLBOX_PATH = os.path.join(PROJECT_ROOT, "externalrPPG-Toolbox")
sys.path.append(TOOLBOX_PATH)
from evaluation.post_process import (
    _calculate_fft_hr,
    _calculate_peak_hr,
    _detrend,
    _calculate_SNR,
)

logger.info("Successfully imported rPPG-Toolbox evaluation metrics")


class SignalProcessor:
    """
    Buffers RGB signals and processes them to extract BVP, BPM, and SpO2.
    """

    def __init__(self, bpm_limits: Tuple[int, int] = (50, 160), buffer_size: int = 300):
        """
        Initialize SignalProcessor.

        Args:
            bpm_limits: Valid heart rate range (min_bpm, max_bpm)
            buffer_size: Maximum number of samples to buffer (calibration target)
        """
        self.bpm_limits = bpm_limits
        self.buffer_size = buffer_size

        # DL Model Init
        MODEL_PATH = os.path.join(
            PROJECT_ROOT,
            "external/rPPG-Toolbox/final_model_release/UBFC-rPPG_PhysNet_DiffNormalized.pth",
        )

        # Check device availability
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        self.dl_model = DLInference(MODEL_PATH, device=device)
        if self.dl_model.model is None:
            logger.warning("DL Model failed to load, falling back to POS algorithm.")

        else:
            logger.info("DL Model loaded successfully.")

        # Log to backend/data/raw/webcam
        self.log_dir = os.path.join(PROJECT_ROOT, "data", "raw", "webcam")
        self.log_file_path = None

        # Buffers for raw RGB channels
        self.r_buffer: List[float] = []
        self.g_buffer: List[float] = []
        self.b_buffer: List[float] = []

        # Buffer for DL-inferred BVP
        self.dl_bvp_buffer: List[float] = []

        self.times: List[float] = []
        self.fps: float = 0.0

        self.bpm: float = 0.0
        self.spo2: float = 0.0
        self.rr: float = 0.0
        self.latest_bvp: float = 0.0

        self.is_calibrated = False

        self.bpm_buffer: List[float] = []
        self.bpm_buffer_size = 300  # Sync with signal buffer (10s window)

        self.spo2_buffer: List[float] = []
        self.spo2_buffer_size = 90  # Slower smoothing for SpO2 (3 sec)

    def is_calibration_complete(self) -> bool:
        """Check if calibration phase is complete."""
        return len(self.r_buffer) >= self.buffer_size

    def get_calibration_progress(self) -> Tuple[int, int]:
        """Get calibration progress (current samples, target samples)."""
        return (len(self.r_buffer), self.buffer_size)

    def get_wait_time(self) -> float:
        """Calculate estimated wait time until calibration is complete."""
        if self.fps <= 0 or len(self.r_buffer) < 2:
            return 0.0

        remaining_samples = self.buffer_size - len(self.r_buffer)
        return max(0.0, remaining_samples / self.fps)

    def add_sample(
        self,
        rgb: Tuple[float, float, float],
        timestamp: float,
        frame: Optional[np.ndarray] = None,
    ) -> None:
        """
        Add a new RGB sample to the buffer.

        Args:
            rgb: Tuple of (Red, Green, Blue) mean intensities
            timestamp: Time of sample (seconds)
            frame: Full frame image (numpy array) for DL inference
        """
        r, g, b = rgb
        self.r_buffer.append(r)
        self.g_buffer.append(g)
        self.b_buffer.append(b)
        self.times.append(timestamp)

        # Update FPS estimate
        if len(self.times) >= 2:
            time_span = self.times[-1] - self.times[0]
            if time_span > 0:
                self.fps = float(len(self.r_buffer)) / time_span

        # Maintain rolling buffer after calibration
        if self.is_calibration_complete() and len(self.r_buffer) > self.buffer_size:
            self.r_buffer = self.r_buffer[-self.buffer_size :]
            self.g_buffer = self.g_buffer[-self.buffer_size :]
            self.b_buffer = self.b_buffer[-self.buffer_size :]
            self.times = self.times[-self.buffer_size :]

        # DL Inference Step
        if frame is not None and self.dl_model.model is not None:
            dl_bvp = self.dl_model.add_frame(frame)
            if dl_bvp is not None:
                self.dl_bvp_buffer.append(dl_bvp)

            # Manage DL buffer size
            if len(self.dl_bvp_buffer) > self.buffer_size:
                self.dl_bvp_buffer = self.dl_bvp_buffer[-self.buffer_size :]

    def calculate_metrics(
        self,
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Calculate BVP, BPM, SpO2, and RR.

        Returns:
            Tuple of (current_bvp_value, bpm, spo2, rr)
        """
        L = len(self.r_buffer)
        if L < self.buffer_size:
            # Not enough data yet
            return None, None, None, None

        # Convert buffers to numpy array (samples x 3)
        # Transpose to (3 x samples) for efficiency if needed, but POS takes (samples x 3)
        rgb_array = np.array([self.r_buffer, self.g_buffer, self.b_buffer]).T

        # 1. Extract BVP
        # Use estimated FPS or default 30
        fs = self.fps if self.fps > 0 else 30.0

        bvp_signal = None

        # Try DL Model first
        if (
            self.dl_model.model is not None
            and len(self.dl_bvp_buffer) >= self.buffer_size
        ):
            # Use the DL-inferred BVP signal
            bvp_signal = np.array(self.dl_bvp_buffer)

        # FALLBACK: Use POS algorithm if DL not ready or not enabled
        if bvp_signal is None:
            bvp_signal = rppg_algorithms.POS(rgb_array, fs)

        # Using Toolbox detrend logic (Smoothness Priors Approach)
        # lambda=100 is standard for PPG
        try:
            bvp_signal = _detrend(bvp_signal, 100)
        except NameError:
            # Fallback if import failed
            bvp_signal = signal.detrend(bvp_signal)

        # Bandpass Filter [0.5 Hz - 4.0 Hz] -> [30 BPM - 240 BPM]
        # User requested to allow < 45 BPM, so we lower the cutoff.

        # Save detrended BVP for RR calculation (needs 0.1 - 0.5 Hz)
        bvp_signal.copy()

        nyquist = 0.5 * fs
        if nyquist > 0:
            low = 0.5 / nyquist
            high = 4.0 / nyquist

            # Clamp to valid range (0, 1) to avoid crash on low FPS
            if high >= 1.0:
                high = 0.99
            if low <= 0.0:
                low = 0.01

            if low < high:
                b, a = signal.butter(3, [low, high], btype="bandpass")
                bvp_signal = signal.filtfilt(b, a, bvp_signal)

        self.latest_bvp = bvp_signal[-1]

        # 2. Calculate BPM from BVP signal
        instant_bpm = self._calculate_bpm_from_signal(bvp_signal, fs)

        # Check if signal quality was too poor (returns None)
        if instant_bpm is None:
            logger.warning(
                "❌ Calibration failed due to poor signal quality - resetting"
            )
            self.reset()
            # Return special tuple indicating calibration failure
            return None, None, None, None

        # Smoothing: Average over last N estimates (stabilizes jumping)
        if instant_bpm > 0:
            self.bpm_buffer.append(instant_bpm)
            if len(self.bpm_buffer) > self.bpm_buffer_size:
                self.bpm_buffer.pop(0)
            self.bpm = float(np.mean(self.bpm_buffer))
        elif len(self.bpm_buffer) > 0:
            # If instantaneous failed but we have history, hold the average
            # Optionally decay or slowly forget, but holding is better than 0 flickering
            self.bpm = float(np.mean(self.bpm_buffer))
        else:
            self.bpm = 0.0

        # 3. Calculate SpO2
        instant_spo2 = rppg_algorithms.calculate_spo2(
            np.array(self.r_buffer), np.array(self.b_buffer), fs
        )

        if instant_spo2 > 0:
            logger.debug("SpO2: %.1f%%", instant_spo2)

        # Smooth SpO2
        if instant_spo2 > 0:
            self.spo2_buffer.append(instant_spo2)
            if len(self.spo2_buffer) > self.spo2_buffer_size:
                self.spo2_buffer.pop(0)
            self.spo2 = float(np.mean(self.spo2_buffer))
        elif len(self.spo2_buffer) > 0:
            self.spo2 = float(np.mean(self.spo2_buffer))
        else:
            self.spo2 = 0.0

        # 4. Calculate RR
        # Using RSA method which needs peak detection on the clean BVP signal.
        self.rr = rppg_algorithms.calculate_rr(bvp_signal, fs)

        # Sanitize outputs (handle NaNs/Infs that crash frontend)
        if np.isnan(self.bpm) or np.isinf(self.bpm):
            self.bpm = 0.0

        if np.isnan(self.spo2) or np.isinf(self.spo2):
            self.spo2 = 0.0

        if np.isnan(self.rr) or np.isinf(self.rr):
            self.rr = 0.0

        # Latest BVP might be NaN if POS failed
        if np.isnan(self.latest_bvp) or np.isinf(self.latest_bvp):
            self.latest_bvp = 0.0

        self.is_calibrated = True

        if self.log_file_path is None:
            try:
                os.makedirs(self.log_dir, exist_ok=True)
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                self.log_file_path = os.path.join(
                    self.log_dir, f"rppg_log_{timestamp_str}.csv"
                )

                with open(self.log_file_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Timestamp", "BVP", "BPM", "SpO2", "RR"])
                logger.info("Logging rPPG data to: %s", self.log_file_path)
            except Exception as e:
                logger.error("Failed to create log file: %s", e, exc_info=True)

        if self.log_file_path:
            with open(self.log_file_path, "a", newline="") as f:
                writer = csv.writer(f)
                ts = datetime.datetime.now().isoformat()
                writer.writerow([ts, self.latest_bvp, self.bpm, self.spo2, self.rr])

        logger.debug(
            "Metrics -> BVP: %.4f | BPM: %.1f | SpO2: %.1f%% | RR: %.1frpm",
            self.latest_bvp,
            self.bpm,
            self.spo2,
            self.rr,
        )

        return self.latest_bvp, self.bpm, self.spo2, self.rr

    def _calculate_bpm_from_signal(self, bvp_signal: np.ndarray, fs: float) -> float:
        """
        Calculates BPM using Toolbox functions (FFT + Peak Detection).
        Uses History (previous BPM) and Range Checks to resolve mismatches intelligently.
        """
        # 1. FFT Method
        fft_failed = False
        try:
            fft_bpm = _calculate_fft_hr(bvp_signal, fs=fs)
        except (NameError, ValueError, IndexError) as e:
            # ValueError/IndexError: empty sequence in argmax (poor signal quality)
            logger.warning("FFT calculation failed: %s", e, exc_info=True)
            fft_bpm = 0.0
            fft_failed = True

        # 2. Peak Detection Method
        peak_failed = False
        try:
            # Bandpass filter cleanup for Peak detection
            nyquist = 0.5 * fs
            if nyquist > 0:
                low = 0.6 / nyquist
                high = 3.0 / nyquist  # 180 BPM

                # Clamp for safety
                if high >= 1.0:
                    high = 0.99
                if low <= 0.0:
                    low = 0.01

                if low < high:
                    b, a = signal.butter(3, [low, high], btype="band")
                    filtered_bvp = signal.filtfilt(b, a, bvp_signal)
                else:
                    filtered_bvp = bvp_signal
            else:
                filtered_bvp = bvp_signal

            peak_bpm = _calculate_peak_hr(filtered_bvp, fs=fs)
        except Exception as e:
            logger.warning("Peak detection failed: %s", e, exc_info=True)
            peak_bpm = 0.0
            peak_failed = True

        # Sanity Check
        if peak_bpm > 200 or peak_bpm < 30:
            peak_bpm = 0.0
        if fft_bpm > 200 or fft_bpm < 30:
            fft_bpm = 0.0

        # Signal Quality Check: If both methods failed, signal is too poor
        if (fft_failed and peak_failed) or (fft_bpm == 0.0 and peak_bpm == 0.0):
            logger.warning("⚠️ POOR SIGNAL QUALITY: Both FFT and Peak detection failed")
            return None  # Return None to indicate calibration should fail

        # 3. Decision Logic

        # If they agree closely, trust it.
        if abs(fft_bpm - peak_bpm) < 15:
            if peak_bpm > 0:
                return float((fft_bpm + peak_bpm) / 2.0)
            return float(fft_bpm)

        logger.debug("BPM Mismatch: FFT=%.1f vs Peak=%.1f", fft_bpm, peak_bpm)

        # A. History Check (Stability)
        # If we have a stable history, prefer the value closer to it.
        if self.bpm > 30:
            dist_fft = abs(fft_bpm - self.bpm)
            dist_peak = abs(peak_bpm - self.bpm)

            # If one is significantly closer to history, pick it.
            if dist_peak < 15 and dist_fft > 30:
                logger.debug("-> Choosing Peak (closer to history %.1f)", self.bpm)
                return float(peak_bpm)
            if dist_fft < 15 and dist_peak > 30:
                logger.debug("-> Choosing FFT (closer to history %.1f)", self.bpm)
                return float(fft_bpm)

        # B. Plausibility Check (Noise Rejection)

        is_fft_low = fft_bpm < 50
        is_peak_low = peak_bpm < 50
        is_fft_normal = 50 <= fft_bpm <= 140
        is_peak_normal = 50 <= peak_bpm <= 140

        if is_fft_low and is_peak_normal:
            logger.debug("-> Choosing Peak (FFT is suspiciously low/breathing noise)")
            return float(peak_bpm)

        if is_peak_low and is_fft_normal:
            logger.debug("-> Choosing FFT (Peak is suspiciously low)")
            return float(fft_bpm)

        # C. Fallback: SNR Comparison
        try:
            # Calculate SNR for both candidates
            snr_fft = _calculate_SNR(
                bvp_signal, fft_bpm, fs, low_pass=0.6, high_pass=4.0
            )
            snr_peak = _calculate_SNR(
                bvp_signal, peak_bpm, fs, low_pass=0.6, high_pass=4.0
            )

            logger.debug(
                "-> SNR Check: FFT(%.1f)=%.2fdB vs Peak(%.1f)=%.2fdB",
                fft_bpm,
                snr_fft,
                peak_bpm,
                snr_peak,
            )

            if snr_peak > snr_fft:
                return float(peak_bpm)
            else:
                return float(fft_bpm)
        except (ValueError, IndexError) as e:
            # Empty sequence errors from poor signal quality
            logger.warning(
                "SNR Calc failed (poor signal): %s. Defaulting to FFT.",
                e,
                exc_info=True,
            )
            return float(fft_bpm) if fft_bpm > 0 else float(peak_bpm)
        except Exception as e:
            logger.warning("SNR Calc failed: %s. Defaulting to FFT.", e, exc_info=True)
            return float(fft_bpm)

    def get_latest_metrics(self) -> Tuple[float, float, float]:
        """Get the most recently calculated metrics."""
        return self.latest_bvp, self.bpm, self.spo2

    def reset(self) -> None:
        """Clear all buffers and reset state."""
        self.r_buffer.clear()
        self.g_buffer.clear()
        self.b_buffer.clear()
        self.dl_bvp_buffer.clear()
        self.times.clear()
        self.bpm_buffer.clear()
        self.spo2_buffer.clear()
        self.bpm = 0.0
        self.spo2 = 0.0
        self.latest_bvp = 0.0
        self.fps = 0.0
        self.is_calibrated = False
