import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict
import numpy as np

from shared.config.logging import get_logger

logger = get_logger(__name__)


class SimpleFineTuner:
    """Fine-tune EEGNet on user calibration data"""

    def __init__(
        self, classifier, learning_rate: float = 1e-4, freeze_early: bool = True
    ):
        """
        Args:
            classifier: Pre-trained EEGClassifier from mi.models.eegnet
            learning_rate: Fine-tuning learning rate
            freeze_early: Whether to freeze early layers
        """
        self.classifier = classifier
        self.learning_rate = learning_rate
        self.device = classifier.device
        self.criterion = nn.CrossEntropyLoss()

        if freeze_early:
            self._freeze_early_layers()

        self.optimizer = torch.optim.Adam(
            self.classifier.model.parameters(), lr=learning_rate
        )

        self.history = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}

    def _freeze_early_layers(self):
        """Freeze early layers to preserve pre-trained features."""
        model = self.classifier.model
        # Freeze conv1 and batchnorm1 (temporal features)
        for param in model.conv1.parameters():
            param.requires_grad = False
        for param in model.batchnorm1.parameters():
            param.requires_grad = False

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_epochs: int = 20,
        batch_size: int = 16,
        val_split: float = 0.2,
    ) -> Dict:
        """Fine-tune on calibration data.

        Args:
            X: Training data (n_samples, n_channels, n_time_samples)
            y: Labels (n_samples,)
            n_epochs: Number of epochs
            batch_size: Batch size
            val_split: Validation split ratio

        Returns:
            Training history
        """
        # Train/val split
        n_val = max(1, int(len(X) * val_split))
        indices = np.random.permutation(len(X))
        train_idx, val_idx = indices[n_val:], indices[:n_val]

        X_train = torch.FloatTensor(X[train_idx]).to(self.device)
        y_train = torch.LongTensor(y[train_idx]).to(self.device)
        X_val = torch.FloatTensor(X[val_idx]).to(self.device)
        y_val = torch.LongTensor(y[val_idx]).to(self.device)

        logger.info(
            "[FineTuner] Training on %s samples, validating on %s",
            len(train_idx),
            len(val_idx),
        )

        for epoch in range(n_epochs):
            # Training
            self.classifier.model.train()
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i : i + batch_size]
                y_batch = y_train[i : i + batch_size]

                self.optimizer.zero_grad()
                logits = self.classifier.model(X_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                self.optimizer.step()

            # Eval train
            self.classifier.model.eval()
            with torch.no_grad():
                train_logits = self.classifier.model(X_train)
                train_loss = self.criterion(train_logits, y_train).item()
                train_acc = (train_logits.argmax(1) == y_train).float().mean().item()

                val_logits = self.classifier.model(X_val)
                val_loss = self.criterion(val_logits, y_val).item()
                val_acc = (val_logits.argmax(1) == y_val).float().mean().item()

            self.history["loss"].append(train_loss)
            self.history["acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            if (epoch + 1) % max(1, n_epochs // 5) == 0:
                logger.info(
                    "Epoch %s/%s: Loss=%.4f Acc=%.2f%% | Val Loss=%.4f Val Acc=%.2f%%",
                    epoch + 1,
                    n_epochs,
                    train_loss,
                    train_acc * 100,
                    val_loss,
                    val_acc * 100,
                )

        return self.history

    def save(self, path: str):
        """Save fine-tuned model."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.classifier.save(path)
        logger.info("[FineTuner] Saved to %s", path)

    def load(self, path: str):
        """Load fine-tuned model."""
        self.classifier.load(path)
        logger.info("[FineTuner] Loaded from %s", path)
