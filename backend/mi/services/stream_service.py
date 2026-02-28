import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import json

from shared.config.logging import get_logger

logger = get_logger(__name__)


class MICalibrator:
    """Collect labeled calibration epochs from live EEG stream."""

    _LABEL_NAMES = {0: "left", 1: "right"}

    def __init__(
        self,
        user_id: str,
        data_dir: str = "data/calibration",
        min_epoch_std_uv: float = 0.35,
        min_epoch_peak_to_peak_uv: float = 1.0,
    ):
        """Initialize calibrator.

        Args:
            user_id: User identifier
            data_dir: Root directory for saving calibration data
            min_epoch_std_uv: Minimum epoch standard deviation to consider signal usable
            min_epoch_peak_to_peak_uv: Minimum epoch peak-to-peak to consider signal usable
        """
        self.user_id = user_id
        self.session_dir = (
            Path(data_dir) / user_id / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.is_collecting = False
        self.current_label = None
        self.current_trial_epochs = []
        self.current_trial_points = []
        self.current_trial_epoch_count = 0
        self.current_trial_rejected = 0
        self.global_epoch_index = 0
        self.trial_count = 0
        self.min_epoch_std_uv = float(min_epoch_std_uv)
        self.min_epoch_peak_to_peak_uv = float(min_epoch_peak_to_peak_uv)
        self.session_info = {
            "user_id": user_id,
            "trials": [],
            "start_time": datetime.now().isoformat(),
        }

    @classmethod
    def label_name(cls, label: int) -> str:
        if label not in cls._LABEL_NAMES:
            raise ValueError(f"Unsupported calibration label: {label}")
        return cls._LABEL_NAMES[label]

    def _epoch_signal_stats(self, eeg_epoch: np.ndarray) -> tuple[bool, float, float]:
        """Return (is_usable, std_uv, peak_to_peak_uv) for a candidate epoch."""
        centered = eeg_epoch - eeg_epoch.mean(axis=1, keepdims=True)
        std_uv = float(np.std(centered))
        peak_to_peak_uv = float(np.ptp(centered))
        is_usable = (
            np.isfinite(std_uv)
            and np.isfinite(peak_to_peak_uv)
            and std_uv >= self.min_epoch_std_uv
            and peak_to_peak_uv >= self.min_epoch_peak_to_peak_uv
        )
        return is_usable, std_uv, peak_to_peak_uv

    def start_trial(self, label: int) -> None:
        """Start collecting a trial."""
        self.label_name(label)
        self.is_collecting = True
        self.current_label = label
        self.current_trial_epochs = []
        self.current_trial_points = []
        self.current_trial_epoch_count = 0
        self.current_trial_rejected = 0
        logger.info(
            "[MICalibrator] Started trial %s (label=%s:%s)",
            self.trial_count,
            label,
            self.label_name(label),
        )

    def add_eeg_chunk(self, filtered_eeg: np.ndarray) -> None:
        """Backward-compatible alias for systems sending epoch-like chunks."""
        self.add_epoch(filtered_eeg)

    def add_epoch(self, eeg_epoch: np.ndarray) -> None:
        """Add a fixed-size filtered EEG epoch.

        Args:
            eeg_epoch: EEG epoch (n_channels, n_samples)
        """
        if not self.is_collecting:
            return
        if eeg_epoch.ndim != 2:
            logger.warning(
                "[MICalibrator] Ignoring invalid epoch shape: %s", eeg_epoch.shape
            )
            return

        self.current_trial_epoch_count += 1
        epoch = np.asarray(eeg_epoch, dtype=np.float32)
        is_usable, std_uv, peak_to_peak_uv = self._epoch_signal_stats(epoch)
        if not is_usable:
            self.current_trial_rejected += 1
            logger.warning(
                "[MICalibrator] Rejected epoch due to weak signal (std=%.4fuv, p2p=%.4fuv)",
                std_uv,
                peak_to_peak_uv,
            )
            return

        label_name = self.label_name(int(self.current_label))
        self.current_trial_epochs.append(epoch)
        self.current_trial_points.append(
            {
                "x": self.global_epoch_index,
                "y": label_name,
                "std_uv": round(std_uv, 4),
                "peak_to_peak_uv": round(peak_to_peak_uv, 4),
            }
        )
        self.global_epoch_index += 1

    def end_trial(self, quality_metrics: Optional[Dict] = None) -> Optional[Path]:
        """End trial and save."""
        if not self.is_collecting:
            return None

        self.is_collecting = False

        if not self.current_trial_epochs:
            raise ValueError(
                "No usable EEG signal detected during this trial. Check headset contact and stream health, then retry calibration."
            )

        # Save trial as stacked epochs: (n_epochs, n_channels, n_samples)
        trial_array = np.stack(self.current_trial_epochs, axis=0)

        trial_file = self.session_dir / f"trial_{self.trial_count:03d}.npy"
        np.save(trial_file, trial_array)

        quality_percent = (
            100.0
            * len(self.current_trial_epochs)
            / max(1, self.current_trial_epoch_count)
        )
        trial_info = {
            "trial_id": self.trial_count,
            "label": self.current_label,
            "label_name": self.label_name(int(self.current_label)),
            "n_epochs": int(trial_array.shape[0]),
            "epoch_shape": [int(trial_array.shape[1]), int(trial_array.shape[2])],
            "received_epochs": int(self.current_trial_epoch_count),
            "rejected_epochs": int(self.current_trial_rejected),
            "quality_percent": quality_metrics.get("quality_percent", quality_percent)
            if quality_metrics
            else quality_percent,
            "xy_points": self.current_trial_points,
        }
        self.session_info["trials"].append(trial_info)

        self.trial_count += 1
        logger.info(
            "[MICalibrator] Saved trial to %s (%s valid epochs, %s rejected)",
            trial_file,
            trial_info["n_epochs"],
            self.current_trial_rejected,
        )

        return trial_file

    def end_session(self) -> Dict:
        """End calibration session and save metadata."""
        self.session_info["end_time"] = datetime.now().isoformat()

        info_file = self.session_dir / "session_info.json"
        with open(info_file, "w") as f:
            json.dump(self.session_info, f, indent=2)

        logger.info("[MICalibrator] Session ended: %s trials saved", self.trial_count)
        return self.session_info

    def load_trials(self) -> tuple:
        """Load all saved trials as dataset.

        Returns:
            (X, y) where X is (n_epochs, n_channels, n_samples)
        """
        X_list = []
        y_list = []

        for trial_info in self.session_info["trials"]:
            trial_file = self.session_dir / f"trial_{trial_info['trial_id']:03d}.npy"
            trial_array = np.load(trial_file)
            if trial_array.ndim == 2:
                trial_array = trial_array[np.newaxis, ...]

            for epoch in trial_array:
                X_list.append(epoch)
                y_list.append(trial_info["label"])

        if not X_list:
            return np.array([]), np.array([])

        return np.array(X_list), np.array(y_list)
