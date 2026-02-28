import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import json

from shared.config.logging import get_logger

logger = get_logger(__name__)


class MICalibrator:
    """Collects labeled EEG epochs from existing EEG stream for fine-tuning."""

    def __init__(self, user_id: str, data_dir: str = "data/calibration"):
        """Initialize calibrator.

        Args:
            user_id: User identifier
            data_dir: Root directory for saving calibration data
        """
        self.user_id = user_id
        self.session_dir = (
            Path(data_dir) / user_id / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.is_collecting = False
        self.current_label = None
        self.trial_data = []
        self.trial_count = 0
        self.session_info = {
            "user_id": user_id,
            "trials": [],
            "start_time": datetime.now().isoformat(),
        }

    def start_trial(self, label: int) -> None:
        """Start collecting a trial."""
        self.is_collecting = True
        self.current_label = label
        self.trial_data = []
        logger.info(
            "[MICalibrator] Started trial %s (label=%s)", self.trial_count, label
        )

    def add_eeg_chunk(self, filtered_eeg: np.ndarray) -> None:
        """Add a chunk of filtered EEG from EEG stream.

        Args:
            filtered_eeg: Cleaned EEG chunk (n_channels, n_samples)
        """
        if self.is_collecting:
            self.trial_data.append(filtered_eeg)

    def end_trial(self, quality_metrics: Optional[Dict] = None) -> Path:
        """End trial and save."""
        if not self.is_collecting:
            return None

        self.is_collecting = False

        # Concatenate all chunks
        trial_array = np.hstack(self.trial_data)

        # Save
        trial_file = self.session_dir / f"trial_{self.trial_count:03d}.npy"
        np.save(trial_file, trial_array)

        trial_info = {
            "trial_id": self.trial_count,
            "label": self.current_label,
            "n_samples": trial_array.shape[1],
            "quality_percent": quality_metrics.get("quality_percent", 100)
            if quality_metrics
            else 100,
        }
        self.session_info["trials"].append(trial_info)

        self.trial_count += 1
        logger.info("[MICalibrator] Saved trial to %s", trial_file)

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
            (X, y) where X is (n_trials, n_channels, n_samples)
        """
        X_list = []
        y_list = []

        for trial_info in self.session_info["trials"]:
            trial_file = self.session_dir / f"trial_{trial_info['trial_id']:03d}.npy"
            X_list.append(np.load(trial_file))
            y_list.append(trial_info["label"])

        if not X_list:
            return np.array([]), np.array([])

        return np.array(X_list), np.array(y_list)
