#!/usr/bin/env python3
"""Fine-tune EEGNet on Muse calibration CSV data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import butter, filtfilt, resample
from torch.utils.data import DataLoader, TensorDataset

from models.eegnet import EEGClassifier, EEGNet
from models.eegnet_residual import EEGClassifier as ResidualEEGClassifier
from models.eegnet_residual import EEGNetResidual

TRAIN_CHANNEL_ORDER = ["AF7", "AF8", "TP9", "TP10"]
MUSE_CHANNEL_COLUMNS = {
    "TP9": "TP9_uV",
    "AF7": "AF7_uV",
    "AF8": "AF8_uV",
    "TP10": "TP10_uV",
}


def _preprocess_eeg(
    X: np.ndarray, lowcut: float, highcut: float, fs: float
) -> np.ndarray:
    """Apply the same 5th-order bandpass used in training."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(5, [low, high], btype="band")

    X_filtered = np.zeros_like(X, dtype=np.float32)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_filtered[i, j, :] = filtfilt(b, a, X[i, j, :])
    return X_filtered


def _find_column(dtype_names: tuple[str, ...], wanted: str) -> str:
    for name in dtype_names:
        if name.lower() == wanted.lower():
            return name
    raise ValueError(f"Missing required column '{wanted}'. Found: {dtype_names}")


def _infer_label_from_filename(csv_path: Path) -> int:
    stem = csv_path.stem.lower()
    if "left" in stem:
        return 0
    if "right" in stem:
        return 1
    raise ValueError(
        f"Could not infer label from '{csv_path.name}'. "
        "Filename must include 'left' or 'right'."
    )


def _estimate_sampling_rate_hz(timestamps: np.ndarray) -> float:
    diffs = np.diff(timestamps.astype(np.float64))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        raise ValueError("Cannot estimate sampling rate from timestamps.")

    median_diff = float(np.median(diffs))
    if median_diff > 1.0:
        # Timestamps are likely in milliseconds.
        return 1000.0 / median_diff
    # Timestamps are likely in seconds.
    return 1.0 / median_diff


def _epoch_signal(
    signal_2d: np.ndarray, window_samples: int, stride_samples: int
) -> np.ndarray:
    epochs = []
    total = signal_2d.shape[1]
    for start in range(0, total - window_samples + 1, stride_samples):
        epoch = signal_2d[:, start : start + window_samples]
        if np.isfinite(epoch).all():
            epochs.append(epoch)

    if not epochs:
        return np.empty((0, signal_2d.shape[0], window_samples), dtype=np.float32)
    return np.stack(epochs).astype(np.float32)


def _load_calibration_csv(csv_path: Path) -> tuple[np.ndarray, float]:
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=np.float64)
    if data.size == 0:
        raise ValueError(f"No rows found in {csv_path}")

    dtype_names = tuple(data.dtype.names or ())
    timestamp_col = _find_column(dtype_names, "timestamp")
    timestamps = np.asarray(data[timestamp_col], dtype=np.float64)
    source_fs = _estimate_sampling_rate_hz(timestamps)

    # Build channel matrix in canonical training order.
    channel_rows = []
    for channel_name in TRAIN_CHANNEL_ORDER:
        csv_col = _find_column(dtype_names, MUSE_CHANNEL_COLUMNS[channel_name])
        channel_rows.append(np.asarray(data[csv_col], dtype=np.float64))
    signal_2d = np.stack(channel_rows, axis=0)
    return signal_2d, source_fs


def _build_dataset(
    data_dir: Path,
    window_seconds: float,
    stride_seconds: float,
    lowcut: float,
    highcut: float,
    target_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    all_x = []
    all_y = []
    for csv_path in csv_files:
        label = _infer_label_from_filename(csv_path)
        signal_2d, source_fs = _load_calibration_csv(csv_path)

        window_samples = max(1, int(round(window_seconds * source_fs)))
        stride_samples = max(1, int(round(stride_seconds * source_fs)))
        epochs = _epoch_signal(signal_2d, window_samples, stride_samples)
        if len(epochs) == 0:
            print(f"Skipping {csv_path.name}: no complete epochs generated.")
            continue

        epochs = _preprocess_eeg(
            epochs, lowcut=lowcut, highcut=highcut, fs=float(source_fs)
        )
        if epochs.shape[2] != target_samples:
            epochs = resample(epochs, target_samples, axis=2)

        labels = np.full(len(epochs), label, dtype=np.int64)
        all_x.append(epochs.astype(np.float32))
        all_y.append(labels)

        print(
            f"{csv_path.name}: fs~{source_fs:.2f}Hz, epochs={len(epochs)}, "
            f"window={window_samples} samples"
        )

    if not all_x:
        raise RuntimeError("No epochs created from calibration CSVs.")

    X = np.concatenate(all_x, axis=0)
    y = np.concatenate(all_y, axis=0)
    return X, y


def _split_train_val(
    X: np.ndarray, y: np.ndarray, val_split: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(X) < 2:
        raise ValueError("Need at least 2 epochs for train/val split.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    n_val = max(1, int(round(len(X) * val_split)))
    n_val = min(n_val, len(X) - 1)

    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def _make_classifier(
    checkpoint_path: Path,
    device: str,
    dropout: float,
    kernel_length: int,
    use_attention: bool,
) -> EEGClassifier | ResidualEEGClassifier:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    model_cfg = checkpoint.get("model_config", {})

    n_channels = int(model_cfg.get("n_channels", len(TRAIN_CHANNEL_ORDER)))
    n_classes = int(model_cfg.get("n_classes", 2))
    n_samples = int(model_cfg.get("n_samples", 481))
    use_residual = "fc1.weight" in state_dict

    if use_residual:
        model = EEGNetResidual(
            n_channels=n_channels,
            n_classes=n_classes,
            n_samples=n_samples,
            dropout=dropout,
            kernel_length=kernel_length,
            use_attention=use_attention,
        )
        classifier = ResidualEEGClassifier(model, device=device)
    else:
        model = EEGNet(
            n_channels=n_channels,
            n_classes=n_classes,
            n_samples=n_samples,
            dropout=dropout,
            kernel_length=kernel_length,
        )
        classifier = EEGClassifier(model, device=device)

    classifier.load(str(checkpoint_path))
    return classifier


def _freeze_early_layers(model: nn.Module) -> None:
    for layer_name in ("conv1", "batchnorm1"):
        layer = getattr(model, layer_name, None)
        if layer is None:
            continue
        for param in layer.parameters():
            param.requires_grad = False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune EEGNet on Muse calibration CSVs."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("calibration_data"),
        help="Directory containing left/right Muse CSV files.",
    )
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=Path("models") / "baseline_2" / "eegnet_best.pth",
        help="Path to pretrained EEGNet checkpoint.",
    )
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=Path("models") / "baseline_2" / "eegnet_tuned.pth",
        help="Path to save tuned checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=3.0,
        help="Epoch window length (seconds).",
    )
    parser.add_argument(
        "--stride-seconds",
        type=float,
        default=3.0,
        help="Epoch stride (seconds). Smaller values create overlapping epochs.",
    )
    parser.add_argument("--lowcut", type=float, default=8.0)
    parser.add_argument("--highcut", type=float, default=30.0)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--kernel-length", type=int, default=64)
    parser.add_argument("--use-attention", action="store_true")
    parser.add_argument(
        "--freeze-early",
        action="store_true",
        help="Freeze conv1/batchnorm1 during fine-tuning.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    script_dir = Path(__file__).resolve().parent

    def _resolve_in(path_arg: Path) -> Path:
        if path_arg.is_absolute():
            return path_arg
        if path_arg.exists():
            return path_arg
        return script_dir / path_arg

    def _resolve_out(path_arg: Path) -> Path:
        if path_arg.is_absolute():
            return path_arg
        return script_dir / path_arg

    base_checkpoint = _resolve_in(args.base_checkpoint)
    output_checkpoint = _resolve_out(args.output_checkpoint)
    data_dir = _resolve_in(args.data_dir)
    if not base_checkpoint.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {base_checkpoint}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Calibration data dir not found: {data_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Base checkpoint: {base_checkpoint}")

    classifier = _make_classifier(
        checkpoint_path=base_checkpoint,
        device=device,
        dropout=args.dropout,
        kernel_length=args.kernel_length,
        use_attention=args.use_attention,
    )
    target_samples = int(classifier.model.n_samples)
    if int(classifier.model.n_channels) != len(TRAIN_CHANNEL_ORDER):
        raise ValueError(
            f"Checkpoint expects {classifier.model.n_channels} channels, "
            f"but this script is configured for {len(TRAIN_CHANNEL_ORDER)} Muse channels."
        )

    X, y = _build_dataset(
        data_dir=data_dir,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        lowcut=args.lowcut,
        highcut=args.highcut,
        target_samples=target_samples,
    )
    X_train, y_train, X_val, y_val = _split_train_val(
        X, y, val_split=args.val_split, seed=args.seed
    )

    print(
        f"Dataset: train={len(X_train)} val={len(X_val)} "
        f"shape={X_train.shape[1:]} classes(train)={np.bincount(y_train)}"
    )

    if args.freeze_early:
        _freeze_early_layers(classifier.model)
        print("Early layers frozen: conv1 + batchnorm1")

    train_dataset = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train).long()
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(y_val).long()
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in classifier.model.parameters() if p.requires_grad]
    optimizer = optim.Adam(
        trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay
    )

    best_val_acc = -1.0
    best_epoch = -1
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        classifier.model.train()
        train_losses = []
        train_correct = 0
        train_total = 0
        for xb, yb in train_loader:
            xb = xb.to(device).float()
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = classifier.model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))
            preds = logits.argmax(1)
            train_correct += int((preds == yb).sum().item())
            train_total += int(yb.numel())

        classifier.model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device).float()
                yb = yb.to(device)
                logits = classifier.model(xb)
                loss = criterion(logits, yb)
                val_losses.append(float(loss.item()))
                preds = logits.argmax(1)
                val_correct += int((preds == yb).sum().item())
                val_total += int(yb.numel())

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            classifier.save(str(output_checkpoint))

    print(f"\nSaved best tuned checkpoint to: {output_checkpoint}")
    print(f"Best val acc: {best_val_acc:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
