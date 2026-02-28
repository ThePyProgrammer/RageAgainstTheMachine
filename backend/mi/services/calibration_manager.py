import numpy as np
from pathlib import Path
from typing import Tuple, Dict


class MICalibrationDataset:
    """Load calibration data from saved trials."""

    def __init__(self, user_id: str, session_dir: Path):
        """
        Args:
            user_id: User ID
            session_dir: Path to session directory with saved trials
        """
        self.user_id = user_id
        self.session_dir = Path(session_dir)
        self.trials = self._discover_trials()

    def _discover_trials(self) -> list:
        """Find all saved trial files."""
        trials = []
        for trial_file in sorted(self.session_dir.glob("trial_*.npy")):
            trial_num = int(trial_file.stem.split("_")[1])
            trials.append((trial_num, trial_file))
        return trials

    def load_as_dataset(
        self, min_quality_percent: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load all trials as (X, y).

        Args:
            min_quality_percent: Filter out trials below this quality

        Returns:
            (X, y) where X is (n_epochs, n_channels, n_samples)
        """
        X_list = []
        y_list = []

        # Load metadata to get labels and quality
        import json

        info_file = self.session_dir / "session_info.json"
        if info_file.exists():
            with open(info_file) as f:
                session_info = json.load(f)

            for trial_info in session_info["trials"]:
                quality = trial_info.get("quality_percent", 100)
                if quality < min_quality_percent:
                    continue

                trial_file = (
                    self.session_dir / f"trial_{trial_info['trial_id']:03d}.npy"
                )
                if trial_file.exists():
                    trial_array = np.load(trial_file)
                    if trial_array.ndim == 2:
                        trial_array = trial_array[np.newaxis, ...]
                    for epoch in trial_array:
                        X_list.append(epoch)
                        y_list.append(trial_info["label"])

        if not X_list:
            return np.array([]), np.array([])

        X = np.array(X_list)
        y = np.array(y_list)
        return X, y

    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        import json

        info_file = self.session_dir / "session_info.json"
        if info_file.exists():
            with open(info_file) as f:
                return json.load(f)
        return {}
