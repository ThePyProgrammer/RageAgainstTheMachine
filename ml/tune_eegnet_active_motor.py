#!/usr/bin/env python3
"""Tune EEGNet on active gameplay EEG with pre-press intent labels."""

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
CHANNEL_COLUMN_CANDIDATES = {
    "TP9": ("chTP9_raw_uv", "TP9_uV", "TP9", "tp9"),
    "AF7": ("chAF7_raw_uv", "AF7_uV", "AF7", "af7"),
    "AF8": ("chAF8_raw_uv", "AF8_uV", "AF8", "af8"),
    "TP10": ("chTP10_raw_uv", "TP10_uV", "TP10", "tp10"),
}
TIMESTAMP_COLUMN_CANDIDATES = ("timestamp_ms", "timestamp", "time_ms", "time")
KEY_COLUMN_CANDIDATES = ("key_pressed", "key", "label", "event")
INTENT_LABELS = {"left": 0, "right": 1}


def _find_column(dtype_names: tuple[str, ...], candidates: tuple[str, ...]) -> str:
    for wanted in candidates:
        for name in dtype_names:
            if name.lower() == wanted.lower():
                return name
    raise ValueError(f"Missing required column. Tried: {candidates}. Found: {dtype_names}")


def _normalize_key_label(value: str) -> str:
    token = str(value).strip().lower()
    if token in {"left", "l", "arrowleft", "a"}:
        return "left"
    if token in {"right", "r", "arrowright", "d"}:
        return "right"
    return "none"


def _estimate_sampling_rate_hz(timestamps: np.ndarray) -> float:
    diffs = np.diff(timestamps.astype(np.float64))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        raise ValueError("Cannot estimate sampling rate from timestamps.")

    median_diff = float(np.median(diffs))
    if median_diff > 1.0:
        # Milliseconds
        return 1000.0 / median_diff
    # Seconds
    return 1.0 / median_diff


def _preprocess_eeg(
    X: np.ndarray, lowcut: float, highcut: float, fs: float
) -> np.ndarray:
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(5, [low, high], btype="band")

    X_filtered = np.zeros_like(X, dtype=np.float32)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_filtered[i, j, :] = filtfilt(b, a, X[i, j, :])
    return X_filtered


def _load_active_csv(
    csv_path: Path,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, int]]:
    data = np.genfromtxt(
        csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8"
    )
    if np.size(data) == 0:
        raise ValueError(f"No rows found in {csv_path}")
    data = np.atleast_1d(data)

    dtype_names = tuple(data.dtype.names or ())
    timestamp_col = _find_column(dtype_names, TIMESTAMP_COLUMN_CANDIDATES)
    key_col = _find_column(dtype_names, KEY_COLUMN_CANDIDATES)

    timestamp = np.asarray(data[timestamp_col], dtype=np.float64)
    key_raw = np.asarray(data[key_col]).astype(str)

    channel_rows = []
    for channel_name in TRAIN_CHANNEL_ORDER:
        csv_col = _find_column(dtype_names, CHANNEL_COLUMN_CANDIDATES[channel_name])
        channel_rows.append(np.asarray(data[csv_col], dtype=np.float64))
    signal_2d = np.stack(channel_rows, axis=0)

    valid_mask = np.isfinite(timestamp) & np.isfinite(signal_2d).all(axis=0)
    if not np.any(valid_mask):
        raise ValueError(f"No valid finite rows in {csv_path}")

    timestamp = np.rint(timestamp[valid_mask]).astype(np.int64)
    signal_2d = signal_2d[:, valid_mask]
    key_state = np.asarray(
        [_normalize_key_label(v) for v in key_raw[valid_mask]], dtype=object
    )

    sort_idx = np.argsort(timestamp, kind="stable")
    timestamp = timestamp[sort_idx]
    signal_2d = signal_2d[:, sort_idx]
    key_state = key_state[sort_idx]

    unique_ts, first_idx, counts = np.unique(
        timestamp, return_index=True, return_counts=True
    )
    mean_signal = np.add.reduceat(signal_2d, first_idx, axis=1) / counts.reshape(1, -1)

    merged_key_state = np.empty(len(unique_ts), dtype=object)
    ambiguous_timestamps = 0
    for i, start in enumerate(first_idx):
        stop = start + counts[i]
        labels = set(key_state[start:stop].tolist())
        has_left = "left" in labels
        has_right = "right" in labels
        if has_left and has_right:
            merged_key_state[i] = "none"
            ambiguous_timestamps += 1
        elif has_left:
            merged_key_state[i] = "left"
        elif has_right:
            merged_key_state[i] = "right"
        else:
            merged_key_state[i] = "none"

    fs = _estimate_sampling_rate_hz(unique_ts)
    stats = {
        "raw_rows": int(np.size(data)),
        "valid_rows": int(np.sum(valid_mask)),
        "unique_timestamps": int(len(unique_ts)),
        "ambiguous_timestamps": int(ambiguous_timestamps),
    }
    return mean_signal, merged_key_state, fs, stats


def _build_offsets(
    min_offset_seconds: float, max_offset_seconds: float, stride_seconds: float
) -> np.ndarray:
    if stride_seconds <= 0:
        raise ValueError("--intent-stride-seconds must be > 0")
    if max_offset_seconds < min_offset_seconds:
        raise ValueError("--intent-max-offset-seconds must be >= --intent-min-offset-seconds")

    offsets = []
    current = max_offset_seconds
    while current + 1e-9 >= min_offset_seconds:
        offsets.append(float(current))
        current -= stride_seconds
    if not offsets:
        raise ValueError("No intent offsets generated. Check offset arguments.")
    return np.asarray(offsets, dtype=np.float64)


def _extract_intent_epochs(
    signal_2d: np.ndarray,
    key_state: np.ndarray,
    fs: float,
    window_seconds: float,
    offsets_seconds: np.ndarray,
    require_none_window: bool,
    include_press_window: bool,
    press_offset_seconds: float,
    group_id_start: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    window_samples = max(1, int(round(window_seconds * fs)))
    n_samples_total = signal_2d.shape[1]

    epochs: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[int] = []

    left_onsets = 0
    right_onsets = 0
    skipped_none_window = 0
    skipped_bounds = 0

    group_id = group_id_start
    for idx in range(n_samples_total):
        current = str(key_state[idx])
        prev = str(key_state[idx - 1]) if idx > 0 else "none"

        if current not in INTENT_LABELS:
            continue
        if current == prev:
            continue

        if current == "left":
            left_onsets += 1
        else:
            right_onsets += 1

        for end_offset_s in offsets_seconds:
            end = idx - int(round(end_offset_s * fs))
            start = end - window_samples
            if start < 0 or end > n_samples_total or end <= start:
                skipped_bounds += 1
                continue

            if require_none_window and np.any(key_state[start:end] != "none"):
                skipped_none_window += 1
                continue

            epochs.append(signal_2d[:, start:end].astype(np.float32, copy=True))
            labels.append(INTENT_LABELS[current])
            groups.append(group_id)

        if include_press_window:
            start = idx + int(round(press_offset_seconds * fs))
            end = start + window_samples
            if 0 <= start < end <= n_samples_total:
                epochs.append(signal_2d[:, start:end].astype(np.float32, copy=True))
                labels.append(INTENT_LABELS[current])
                groups.append(group_id)
            else:
                skipped_bounds += 1

        group_id += 1

    if epochs:
        X = np.stack(epochs).astype(np.float32)
        y = np.asarray(labels, dtype=np.int64)
        g = np.asarray(groups, dtype=np.int64)
    else:
        X = np.empty((0, signal_2d.shape[0], window_samples), dtype=np.float32)
        y = np.empty((0,), dtype=np.int64)
        g = np.empty((0,), dtype=np.int64)

    stats = {
        "left_onsets": left_onsets,
        "right_onsets": right_onsets,
        "epochs": int(len(X)),
        "skipped_none_window": skipped_none_window,
        "skipped_bounds": skipped_bounds,
        "next_group_id": group_id,
    }
    return X, y, g, stats


def _build_dataset(
    data_dir: Path,
    window_seconds: float,
    intent_min_offset_seconds: float,
    intent_max_offset_seconds: float,
    intent_stride_seconds: float,
    include_press_window: bool,
    press_offset_seconds: float,
    require_none_window: bool,
    lowcut: float,
    highcut: float,
    target_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    csv_files = sorted(p for p in data_dir.rglob("*.csv") if p.is_file())
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    print(f"Discovered {len(csv_files)} active calibration CSV file(s):")
    for csv_path in csv_files:
        print(f"  - {csv_path.name}")

    offsets_seconds = _build_offsets(
        min_offset_seconds=intent_min_offset_seconds,
        max_offset_seconds=intent_max_offset_seconds,
        stride_seconds=intent_stride_seconds,
    )

    all_x = []
    all_y = []
    all_groups = []
    group_id_next = 0
    skipped_files = 0

    for csv_path in csv_files:
        try:
            signal_2d, key_state, source_fs, load_stats = _load_active_csv(csv_path)
        except ValueError as exc:
            print(f"Skipping {csv_path.name}: {exc}")
            skipped_files += 1
            continue

        X_file, y_file, g_file, epoch_stats = _extract_intent_epochs(
            signal_2d=signal_2d,
            key_state=key_state,
            fs=source_fs,
            window_seconds=window_seconds,
            offsets_seconds=offsets_seconds,
            require_none_window=require_none_window,
            include_press_window=include_press_window,
            press_offset_seconds=press_offset_seconds,
            group_id_start=group_id_next,
        )
        group_id_next = epoch_stats["next_group_id"]

        if len(X_file) == 0:
            print(
                f"{csv_path.name}: fs~{source_fs:.2f}Hz, "
                f"onsets(L/R)={epoch_stats['left_onsets']}/{epoch_stats['right_onsets']}, "
                "epochs=0 (skipped)"
            )
            continue

        X_file = _preprocess_eeg(X_file, lowcut=lowcut, highcut=highcut, fs=source_fs)
        if X_file.shape[2] != target_samples:
            X_file = resample(X_file, target_samples, axis=2)

        all_x.append(X_file.astype(np.float32))
        all_y.append(y_file.astype(np.int64))
        all_groups.append(g_file.astype(np.int64))

        print(
            f"{csv_path.name}: fs~{source_fs:.2f}Hz, "
            f"rows={load_stats['raw_rows']}->valid={load_stats['valid_rows']}->unique_ts={load_stats['unique_timestamps']}, "
            f"ambiguous_ts={load_stats['ambiguous_timestamps']}, "
            f"onsets(L/R)={epoch_stats['left_onsets']}/{epoch_stats['right_onsets']}, "
            f"epochs={len(X_file)}"
        )

    if not all_x:
        raise RuntimeError("No usable intent epochs were created from active calibration CSVs.")

    X = np.concatenate(all_x, axis=0)
    y = np.concatenate(all_y, axis=0)
    groups = np.concatenate(all_groups, axis=0)

    class_counts = np.bincount(y, minlength=2)
    print(
        f"Built intent dataset: total_epochs={len(X)}, class_counts(left/right)={class_counts.tolist()}, "
        f"unique_onsets={len(np.unique(groups))}, skipped_files={skipped_files}"
    )
    return X, y, groups


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


def _split_train_val_stratified(
    X: np.ndarray, y: np.ndarray, val_split: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(X) < 2:
        raise ValueError("Need at least 2 epochs for train/val split.")

    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        cls_idx = rng.permutation(cls_idx)
        n_val_cls = max(1, int(round(len(cls_idx) * val_split)))
        if len(cls_idx) > 1:
            n_val_cls = min(n_val_cls, len(cls_idx) - 1)
        val_indices.append(cls_idx[:n_val_cls])
        train_indices.append(cls_idx[n_val_cls:])

    train_idx = rng.permutation(np.concatenate(train_indices))
    val_idx = rng.permutation(np.concatenate(val_indices))
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def _split_train_val_grouped_stratified(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    val_split: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(X) < 2:
        raise ValueError("Need at least 2 epochs for train/val split.")

    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    group_labels = {}
    for group_id in unique_groups:
        group_rows = np.where(groups == group_id)[0]
        class_ids = np.unique(y[group_rows])
        if len(class_ids) != 1:
            raise ValueError(
                f"Group {group_id} has mixed labels {class_ids.tolist()}, expected one label per onset group."
            )
        group_labels[group_id] = int(class_ids[0])

    val_group_ids = []
    for cls in sorted(np.unique(y).tolist()):
        cls_groups = np.asarray(
            [g for g in unique_groups if group_labels[g] == cls], dtype=np.int64
        )
        cls_groups = rng.permutation(cls_groups)
        if len(cls_groups) <= 1:
            n_val_cls = 0
        else:
            n_val_cls = max(1, int(round(len(cls_groups) * val_split)))
            n_val_cls = min(n_val_cls, len(cls_groups) - 1)
        if n_val_cls > 0:
            val_group_ids.append(cls_groups[:n_val_cls])

    if val_group_ids:
        val_group_ids_flat = np.concatenate(val_group_ids)
    else:
        val_group_ids_flat = np.empty((0,), dtype=np.int64)

    if len(val_group_ids_flat) == 0:
        # Fallback when grouped stratification is impossible.
        return _split_train_val_stratified(X, y, val_split=val_split, seed=seed)

    val_mask = np.isin(groups, val_group_ids_flat)
    train_mask = ~val_mask
    if not np.any(train_mask) or not np.any(val_mask):
        raise ValueError("Grouped split failed to create non-empty train/val sets.")

    return X[train_mask], y[train_mask], X[val_mask], y[val_mask]


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    recalls = []
    for cls in range(n_classes):
        mask = y_true == cls
        if np.sum(mask) == 0:
            continue
        recalls.append(float(np.mean(y_pred[mask] == cls)))
    if not recalls:
        return 0.0
    return float(np.mean(recalls))


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true.astype(int), y_pred.astype(int), strict=False):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm


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
        description=(
            "Fine-tune EEGNet on active gameplay data by labeling pre-press windows "
            "as upcoming left/right intent."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("active_calibration_data"),
        help="Directory containing active calibration CSV files.",
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
        default=Path("models") / "baseline_2" / "eegnet_tuned_active.pth",
        help="Path to save tuned checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument(
        "--use-all-data",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If enabled, train on all available epochs (no validation split). "
            "Disable with --no-use-all-data to keep a holdout set."
        ),
    )
    parser.add_argument(
        "--stratified-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use class-stratified splitting when validation is enabled.",
    )
    parser.add_argument(
        "--grouped-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep windows from the same key-press onset in the same split to reduce leakage."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--window-seconds",
        type=float,
        default=0.6,
        help="Epoch window length for intent windows.",
    )
    parser.add_argument(
        "--intent-min-offset-seconds",
        type=float,
        default=0.05,
        help="Smallest end-offset before key onset used for intent windows.",
    )
    parser.add_argument(
        "--intent-max-offset-seconds",
        type=float,
        default=0.65,
        help="Largest end-offset before key onset used for intent windows.",
    )
    parser.add_argument(
        "--intent-stride-seconds",
        type=float,
        default=0.1,
        help="Stride for multiple intent windows per onset.",
    )
    parser.add_argument(
        "--require-none-window",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require intent windows to contain only 'none' labels before onset.",
    )
    parser.add_argument(
        "--include-press-window",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also add one motor-execution window per onset.",
    )
    parser.add_argument(
        "--press-offset-seconds",
        type=float,
        default=0.0,
        help="Offset from onset for optional press window start.",
    )

    parser.add_argument("--lowcut", type=float, default=8.0)
    parser.add_argument("--highcut", type=float, default=30.0)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--kernel-length", type=int, default=64)
    parser.add_argument("--use-attention", action="store_true")
    parser.add_argument(
        "--class-weighted-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use inverse-frequency class weights in cross-entropy loss.",
    )
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
        cwd_candidate = Path.cwd() / path_arg
        if cwd_candidate.parent.exists():
            return cwd_candidate
        return script_dir / path_arg

    base_checkpoint = _resolve_in(args.base_checkpoint)
    output_checkpoint = _resolve_out(args.output_checkpoint)
    data_dir = _resolve_in(args.data_dir)
    if not base_checkpoint.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {base_checkpoint}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Active calibration data dir not found: {data_dir}")

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
    if int(classifier.model.n_classes) != 2:
        raise ValueError(
            f"Checkpoint expects {classifier.model.n_classes} classes. "
            "This script currently supports binary left/right tuning only."
        )

    X, y, groups = _build_dataset(
        data_dir=data_dir,
        window_seconds=args.window_seconds,
        intent_min_offset_seconds=args.intent_min_offset_seconds,
        intent_max_offset_seconds=args.intent_max_offset_seconds,
        intent_stride_seconds=args.intent_stride_seconds,
        include_press_window=args.include_press_window,
        press_offset_seconds=args.press_offset_seconds,
        require_none_window=args.require_none_window,
        lowcut=args.lowcut,
        highcut=args.highcut,
        target_samples=target_samples,
    )

    use_all_data = bool(args.use_all_data)
    if use_all_data:
        X_train, y_train = X, y
        X_val = np.empty((0, X.shape[1], X.shape[2]), dtype=X.dtype)
        y_val = np.empty((0,), dtype=y.dtype)
        split_mode = "all-data training (no validation split)"
    else:
        if args.grouped_split:
            X_train, y_train, X_val, y_val = _split_train_val_grouped_stratified(
                X, y, groups=groups, val_split=args.val_split, seed=args.seed
            )
            split_mode = "grouped stratified split by onset"
        elif args.stratified_split:
            X_train, y_train, X_val, y_val = _split_train_val_stratified(
                X, y, val_split=args.val_split, seed=args.seed
            )
            split_mode = "stratified split by epoch"
        else:
            X_train, y_train, X_val, y_val = _split_train_val(
                X, y, val_split=args.val_split, seed=args.seed
            )
            split_mode = "random split by epoch"

    print(f"Training mode: {split_mode}")
    print(
        f"Dataset: total={len(X)} train={len(X_train)} val={len(X_val)} "
        f"shape={X_train.shape[1:]} classes(train)={np.bincount(y_train, minlength=2).tolist()}"
    )
    if len(X_val) > 0:
        print(f"Classes(val)={np.bincount(y_val, minlength=2).tolist()}")

    if args.freeze_early:
        _freeze_early_layers(classifier.model)
        print("Early layers frozen: conv1 + batchnorm1")

    train_dataset = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train).long()
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    has_val = len(X_val) > 0
    if has_val:
        val_dataset = TensorDataset(
            torch.from_numpy(X_val), torch.from_numpy(y_val).long()
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        val_loader = None

    if args.class_weighted_loss:
        n_classes = int(classifier.model.n_classes)
        class_counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
        class_weights = (class_counts.sum() / np.maximum(class_counts, 1.0)) / float(
            n_classes
        )
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
        )
        print(f"Loss mode: class-weighted CE, weights={class_weights.tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Loss mode: standard CE")

    trainable_params = [p for p in classifier.model.parameters() if p.requires_grad]
    optimizer = optim.Adam(
        trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay
    )

    best_val_acc = -1.0
    best_val_bal_acc = -1.0
    best_train_loss = float("inf")
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

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        train_acc = train_correct / max(train_total, 1)

        if has_val and val_loader is not None:
            classifier.model.eval()
            val_losses = []
            val_correct = 0
            val_total = 0
            y_true_parts = []
            y_pred_parts = []

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
                    y_true_parts.append(yb.cpu().numpy())
                    y_pred_parts.append(preds.cpu().numpy())

            val_loss = float(np.mean(val_losses)) if val_losses else 0.0
            val_acc = val_correct / max(val_total, 1)
            y_true = np.concatenate(y_true_parts) if y_true_parts else np.empty((0,))
            y_pred = np.concatenate(y_pred_parts) if y_pred_parts else np.empty((0,))
            val_bal_acc = _balanced_accuracy(
                y_true=y_true, y_pred=y_pred, n_classes=int(classifier.model.n_classes)
            )
            print(
                f"Epoch {epoch + 1:03d}/{args.epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_bal_acc={val_bal_acc:.4f}"
            )

            better = val_acc > best_val_acc
            tie_better = np.isclose(val_acc, best_val_acc) and (
                val_bal_acc > best_val_bal_acc
            )
            if better or tie_better:
                best_val_acc = val_acc
                best_val_bal_acc = val_bal_acc
                best_epoch = epoch + 1
                classifier.save(str(output_checkpoint))
        else:
            print(
                f"Epoch {epoch + 1:03d}/{args.epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}"
            )
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                best_epoch = epoch + 1
                classifier.save(str(output_checkpoint))

    print(f"\nSaved best tuned checkpoint to: {output_checkpoint}")

    if has_val and val_loader is not None:
        classifier.load(str(output_checkpoint))
        classifier.model.eval()
        y_true_parts = []
        y_pred_parts = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device).float()
                logits = classifier.model(xb)
                preds = logits.argmax(1).cpu().numpy()
                y_pred_parts.append(preds)
                y_true_parts.append(yb.numpy())

        y_true = np.concatenate(y_true_parts)
        y_pred = np.concatenate(y_pred_parts)
        cm = _confusion_matrix(y_true, y_pred, n_classes=int(classifier.model.n_classes))
        print(f"Best val acc: {best_val_acc:.4f} at epoch {best_epoch}")
        print(f"Best val balanced acc: {best_val_bal_acc:.4f}")
        print("Val confusion matrix [rows=true(left,right), cols=pred(left,right)]:")
        print(cm)
    else:
        print(f"Best train loss: {best_train_loss:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
