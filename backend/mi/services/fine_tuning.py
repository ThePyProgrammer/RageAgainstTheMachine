from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from mi.services.fine_tuner import SimpleFineTuner
from shared.config.app_config import HF_REPO_ID, HF_TOKEN
from shared.config.logging import get_logger
from shared.storage.hf_store import upload_to_hf

logger = get_logger(__name__)


@dataclass
class FineTuningSummary:
    n_samples: int
    n_left: int
    n_right: int
    n_epochs: int
    batch_size: int
    final_loss: float
    final_acc: float
    final_val_loss: float
    final_val_acc: float
    best_val_acc: float


class LightweightFineTuningService:
    """Run and persist lightweight left/right fine-tuning for MI control."""

    def __init__(self, fine_tuner: SimpleFineTuner):
        self.fine_tuner = fine_tuner
        self.last_summary: Optional[FineTuningSummary] = None

    @staticmethod
    def validate_dataset(
        X: np.ndarray, y: np.ndarray, min_samples_per_class: int = 2
    ) -> dict:
        if X.size == 0 or y.size == 0:
            raise ValueError("No calibration data available for fine-tuning.")

        if X.ndim != 3:
            raise ValueError(
                f"Expected calibration tensor shape (n_epochs, n_channels, n_samples), got {X.shape}."
            )

        class_counts = {int(label): int((y == label).sum()) for label in np.unique(y)}
        left_count = class_counts.get(0, 0)
        right_count = class_counts.get(1, 0)
        if left_count < min_samples_per_class or right_count < min_samples_per_class:
            raise ValueError(
                "Insufficient calibration data. Need at least "
                f"{min_samples_per_class} left and {min_samples_per_class} right epochs "
                f"(got left={left_count}, right={right_count})."
            )
        return {
            "left_count": left_count,
            "right_count": right_count,
            "total": int(len(y)),
        }

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_epochs: int = 12,
        batch_size: int = 8,
        val_split: float = 0.2,
    ) -> FineTuningSummary:
        counts = self.validate_dataset(X, y, min_samples_per_class=2)
        history = self.fine_tuner.train(
            X,
            y,
            n_epochs=n_epochs,
            batch_size=batch_size,
            val_split=val_split,
        )

        summary = FineTuningSummary(
            n_samples=counts["total"],
            n_left=counts["left_count"],
            n_right=counts["right_count"],
            n_epochs=n_epochs,
            batch_size=batch_size,
            final_loss=float(history["loss"][-1]),
            final_acc=float(history["acc"][-1]),
            final_val_loss=float(history["val_loss"][-1]),
            final_val_acc=float(history["val_acc"][-1]),
            best_val_acc=float(max(history["val_acc"])),
        )
        self.last_summary = summary
        return summary

    def save_and_optionally_upload(self, user_id: str) -> dict:
        checkpoint_dir = Path("mi/models/trained/user_models")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{user_id}_leftright_finetuned.pt"
        self.fine_tuner.save(str(checkpoint_path))

        upload_path = None
        if HF_REPO_ID:
            try:
                upload_path = upload_to_hf(str(checkpoint_path), HF_REPO_ID, HF_TOKEN)
            except Exception as exc:
                logger.error("[FineTuning] Model upload failed: %s", exc, exc_info=True)

        return {
            "path": str(checkpoint_path),
            "uploaded": bool(upload_path),
            "remote_path": upload_path,
        }
