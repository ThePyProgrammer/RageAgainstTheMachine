"""Train MI EEGNet/EEGNetResidual model from eeg_config.yaml."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Any

# Keep MNE quiet and avoid home-dir writes in sandboxed environments.
os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")

import numpy as np
import torch
import torch.nn as nn
import yaml

from mi.eeg.dataset import PhysioNetDataset, preprocess_eeg
from mi.models.eegnet import EEGClassifier, EEGNet
from mi.models.eegnet_residual import EEGClassifier as ResidualEEGClassifier
from mi.models.eegnet_residual import EEGNetResidual
from mi.utils.config_loader import get_project_root
from shared.config.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train motor-imagery EEGNet model")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML (default: backend/mi/config/eeg_config.yaml)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training.num_epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override training.batch_size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override training.learning_rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow downloading missing EEGBCI subject files from PhysioNet",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str | None) -> dict[str, Any]:
    backend_root = get_project_root()
    if config_path is None:
        path = backend_root / "mi" / "config" / "eeg_config.yaml"
    else:
        path = Path(config_path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_epoch_length(X: np.ndarray, target_samples: int) -> np.ndarray:
    """Trim or pad epochs to fixed sample length."""
    current_samples = X.shape[2]
    if current_samples == target_samples:
        return X
    if current_samples > target_samples:
        return X[:, :, :target_samples]
    pad_width = target_samples - current_samples
    return np.pad(X, ((0, 0), (0, 0), (0, pad_width)), mode="constant")


def load_subject_data(
    dataset: PhysioNetDataset,
    subject_id: int,
    runs: list[int],
    channels: list[str],
    lowcut: float,
    highcut: float,
    fs: float,
    target_samples: int,
    allow_download: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    subject_code = f"S{subject_id:03d}"
    base_dir = (
        Path(dataset.data_dir).parent
        / "MNE-eegbci-data"
        / "files"
        / "eegmmidb"
        / "1.0.0"
        / subject_code
    )
    expected_files = [base_dir / f"{subject_code}R{run:02d}.edf" for run in runs]
    has_all_local_files = all(path.exists() for path in expected_files)

    if not has_all_local_files:
        if not allow_download:
            logger.warning(
                "Skipping subject %s: missing local EDF files and downloads disabled",
                subject_id,
            )
            return None
        downloaded = dataset.download_subject(subject_id, runs)
        if not downloaded:
            logger.warning("Skipping subject %s: download/load unavailable", subject_id)
            return None

    try:
        X, y = dataset.load_subject(subject_id, runs, channels)
    except Exception as exc:
        logger.warning("Skipping subject %s: %s", subject_id, exc)
        return None

    X = preprocess_eeg(X, lowcut=lowcut, highcut=highcut, fs=fs)
    X = ensure_epoch_length(X, target_samples)
    return X.astype(np.float32), y.astype(np.int64)


def prepare_split(
    dataset: PhysioNetDataset,
    subject_ids: list[int],
    runs: list[int],
    channels: list[str],
    lowcut: float,
    highcut: float,
    fs: float,
    target_samples: int,
    allow_download: bool,
) -> tuple[np.ndarray, np.ndarray]:
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    loaded_subjects = 0
    for subject_id in subject_ids:
        loaded = load_subject_data(
            dataset=dataset,
            subject_id=subject_id,
            runs=runs,
            channels=channels,
            lowcut=lowcut,
            highcut=highcut,
            fs=fs,
            target_samples=target_samples,
            allow_download=allow_download,
        )
        if loaded is None:
            continue
        X_sub, y_sub = loaded
        X_parts.append(X_sub)
        y_parts.append(y_sub)
        loaded_subjects += 1
        logger.info(
            "Loaded subject %s: epochs=%s shape=%s",
            subject_id,
            X_sub.shape[0],
            tuple(X_sub.shape),
        )

    if not X_parts:
        raise RuntimeError("No subject data available for this split")

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    logger.info(
        "Prepared split from %d subjects: X=%s y=%s",
        loaded_subjects,
        tuple(X.shape),
        tuple(y.shape),
    )
    return X, y


def remap_labels(y_train: np.ndarray, y_val: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Map arbitrary event ids to contiguous class ids."""
    unique_labels = sorted(set(y_train.tolist()) | set(y_val.tolist()))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y_train_mapped = np.array([label_map[label] for label in y_train], dtype=np.int64)
    y_val_mapped = np.array([label_map[label] for label in y_val], dtype=np.int64)
    return y_train_mapped, y_val_mapped, len(unique_labels)


def build_classifier(config: dict[str, Any], n_classes: int, target_samples: int):
    model_cfg = config["model"]
    n_channels = int(model_cfg["input_channels"])
    use_residual = bool(model_cfg.get("use_residual", False))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if use_residual:
        model = EEGNetResidual(
            n_channels=n_channels,
            n_classes=n_classes,
            n_samples=target_samples,
            dropout=float(model_cfg.get("dropout", 0.5)),
            kernel_length=int(model_cfg.get("kernel_length", 64)),
            use_attention=bool(model_cfg.get("use_attention", False)),
        )
        classifier = ResidualEEGClassifier(model, device=device)
        architecture = "EEGNetResidual"
    else:
        model = EEGNet(
            n_channels=n_channels,
            n_classes=n_classes,
            n_samples=target_samples,
            dropout=float(model_cfg.get("dropout", 0.5)),
            kernel_length=int(model_cfg.get("kernel_length", 64)),
        )
        classifier = EEGClassifier(model, device=device)
        architecture = "EEGNet"

    logger.info(
        "Model: %s, channels=%s, classes=%s, samples=%s, device=%s",
        architecture,
        n_channels,
        n_classes,
        target_samples,
        device,
    )
    return classifier


def train() -> Path:
    args = parse_args()
    set_seed(args.seed)
    config = load_config(args.config)

    dataset_cfg = config["dataset"]
    prep_cfg = config["preprocessing"]
    train_cfg = config["training"]
    model_cfg = config["model"]

    runs = [int(run) for run in dataset_cfg["runs"]]
    channels = [str(channel) for channel in prep_cfg["channels"]]
    fs = float(prep_cfg["sampling_rate"])
    lowcut = float(prep_cfg["lowcut"])
    highcut = float(prep_cfg["highcut"])
    target_samples = int((config["epochs"]["tmax"] - config["epochs"]["tmin"]) * fs)

    batch_size = int(args.batch_size or train_cfg["batch_size"])
    learning_rate = float(args.learning_rate or train_cfg["learning_rate"])
    num_epochs = int(args.epochs or train_cfg["num_epochs"])
    patience = int(train_cfg.get("early_stopping_patience", 20))

    backend_root = get_project_root()
    os.environ.setdefault("MNE_DATA", str(backend_root / "data" / "raw"))
    data_dir = backend_root / "data" / "raw" / "physionet"
    dataset = PhysioNetDataset(str(data_dir))

    train_subjects = [int(subject) for subject in dataset_cfg["train_subjects"]]
    val_subjects = [int(subject) for subject in dataset_cfg["val_subjects"]]

    logger.info(
        "Loading training data: channels=%s runs=%s train_subjects=%d val_subjects=%d",
        channels,
        runs,
        len(train_subjects),
        len(val_subjects),
    )

    X_train, y_train = prepare_split(
        dataset=dataset,
        subject_ids=train_subjects,
        runs=runs,
        channels=channels,
        lowcut=lowcut,
        highcut=highcut,
        fs=fs,
        target_samples=target_samples,
        allow_download=args.allow_download,
    )
    try:
        X_val, y_val = prepare_split(
            dataset=dataset,
            subject_ids=val_subjects,
            runs=runs,
            channels=channels,
            lowcut=lowcut,
            highcut=highcut,
            fs=fs,
            target_samples=target_samples,
            allow_download=args.allow_download,
        )
    except RuntimeError:
        validation_split = float(train_cfg.get("validation_split", 0.2))
        n_samples = X_train.shape[0]
        n_val = max(1, int(round(n_samples * validation_split)))
        if n_samples - n_val < 1:
            raise RuntimeError(
                "Not enough training data to create fallback validation split"
            )
        indices = np.random.permutation(n_samples)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        X_val = X_train[val_idx]
        y_val = y_train[val_idx]
        X_train = X_train[train_idx]
        y_train = y_train[train_idx]
        logger.warning(
            "Validation subject split unavailable; using random fallback split train=%d val=%d",
            X_train.shape[0],
            X_val.shape[0],
        )

    y_train, y_val, n_classes = remap_labels(y_train, y_val)
    classifier = build_classifier(config, n_classes=n_classes, target_samples=target_samples)
    device = classifier.device

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.model.parameters(), lr=learning_rate)

    save_dir = backend_root / str(train_cfg["savedir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / str(train_cfg["savename"])

    logger.info(
        "Training start: epochs=%d batch_size=%d lr=%s train=%d val=%d save=%s",
        num_epochs,
        batch_size,
        learning_rate,
        len(X_train),
        len(X_val),
        save_path,
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        classifier.model.train()
        perm = torch.randperm(X_train_t.size(0), device=device)
        X_epoch = X_train_t[perm]
        y_epoch = y_train_t[perm]

        train_loss_sum = 0.0
        train_correct = 0
        train_total = X_epoch.size(0)

        for i in range(0, train_total, batch_size):
            X_batch = X_epoch[i : i + batch_size]
            y_batch = y_epoch[i : i + batch_size]
            loss, acc = classifier.train_step(X_batch, y_batch, optimizer, criterion)
            train_loss_sum += loss * X_batch.size(0)
            train_correct += int(round(acc * X_batch.size(0)))

        train_loss = train_loss_sum / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        classifier.model.eval()
        with torch.no_grad():
            logits = classifier.model(X_val_t)
            val_loss = criterion(logits, y_val_t).item()
            val_pred = logits.argmax(dim=1)
            val_acc = (val_pred == y_val_t).float().mean().item()

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_val_acc = val_acc
            epochs_without_improvement = 0
            classifier.save(str(save_path))
        else:
            epochs_without_improvement += 1

        logger.info(
            "Epoch %03d/%03d | train_loss=%.4f train_acc=%.2f%% | val_loss=%.4f val_acc=%.2f%% | best_val_acc=%.2f%%",
            epoch,
            num_epochs,
            train_loss,
            train_acc * 100.0,
            val_loss,
            val_acc * 100.0,
            best_val_acc * 100.0,
        )

        if epochs_without_improvement >= patience:
            logger.info(
                "Early stopping at epoch %d (patience=%d)", epoch, patience
            )
            break

    logger.info(
        "Training complete. Best model saved to %s (best_val_loss=%.4f, best_val_acc=%.2f%%)",
        save_path,
        best_val_loss,
        best_val_acc * 100.0,
    )
    return save_path


if __name__ == "__main__":
    train()
