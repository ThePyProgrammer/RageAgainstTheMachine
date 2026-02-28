from typing import Dict, Optional, Tuple

import numpy as np
import torch

from mi.eeg.dataset import PhysioNetDataset, preprocess_eeg
from mi.models.eegnet import EEGClassifier
from mi.utils.config_loader import get_project_root
from shared.config.logging import get_logger

logger = get_logger(__name__)


class MotorImageryController:
    """Predict motor imagery class and map it to a command."""

    def __init__(
        self,
        classifier: EEGClassifier,
        command_mapping: Dict[int, str],
        label_mapping: Optional[Dict[int, str]] = None,
    ) -> None:
        self.classifier = classifier
        self.command_mapping = command_mapping
        self.label_mapping = label_mapping or {}
        self.last_prediction: Optional[int] = None
        self.last_confidence: Optional[float] = None

        # Get expected number of channels from model
        self.expected_channels = classifier.model.n_channels
        logger.info(f"[MI-Controller] Model expects {self.expected_channels} channels")

    def predict_and_command(self, eeg_epoch: np.ndarray) -> Tuple[str, float]:
        """Predict motor imagery class and return drone command.

        Args:
            eeg_epoch: EEG data (channels, time_samples)

        Returns:
            Tuple of (command, confidence)
        """
        n_channels, n_samples = eeg_epoch.shape

        # Pad channels if necessary (e.g., Cyton has 8 channels but model trained on 17)
        if n_channels < self.expected_channels:
            logger.info(
                f"[MI-Controller] Padding {n_channels} channels to {self.expected_channels}"
            )
            padding = np.zeros((self.expected_channels - n_channels, n_samples))
            eeg_epoch = np.vstack([eeg_epoch, padding])
        elif n_channels > self.expected_channels:
            logger.warning(
                f"[MI-Controller] Received {n_channels} channels but model expects "
                f"{self.expected_channels}, truncating"
            )
            eeg_epoch = eeg_epoch[: self.expected_channels, :]

        # Convert to tensor
        X = torch.FloatTensor(eeg_epoch)

        # Predict
        pred_class, probs = self.classifier.predict(X)
        pred_class = int(pred_class[0])
        confidence = float(probs[0, pred_class])
        command = self.command_mapping.get(pred_class, "hover")

        self.last_prediction = pred_class
        self.last_confidence = confidence

        return command, confidence

    def prediction_label(self) -> str:
        if self.last_prediction is None:
            return "Idle"
        return self.label_mapping.get(self.last_prediction, "Unknown")


def load_test_data(config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Load test EEG data for simulated streaming."""
    dataset_config = config["dataset"]
    preprocess_config = config["preprocessing"]

    # Use a subject not in training set for testing
    test_subject = 6
    data_dir = get_project_root() / "data" / "raw" / "physionet"
    dataset = PhysioNetDataset(str(data_dir))

    # Download if needed
    dataset.download_subject(test_subject, dataset_config["runs"])

    # Load data
    X, y = dataset.load_subject(
        test_subject,
        dataset_config["runs"],
        preprocess_config["channels"],
    )

    # Filter to keep only left hand (T1) and right hand (T2)
    mask = (y == 1) | (y == 2)
    X = X[mask]
    y = y[mask]

    # Remap labels
    y = y - 1

    # Apply preprocessing
    X = preprocess_eeg(
        X,
        lowcut=preprocess_config["lowcut"],
        highcut=preprocess_config["highcut"],
        fs=preprocess_config["sampling_rate"],
    )

    logger.info("Loaded %s test epochs", len(X))
    return X, y
