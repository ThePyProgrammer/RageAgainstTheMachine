#!/usr/bin/env python3
"""Train a LaBraM embedding probe on PhysioNet Motor Imagery dataset."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import signal
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from eeg.dataset import PhysioNetDataset, preprocess_eeg
from models.labram_probe import LaBraMProbe
from utils.config_loader import get_project_root, load_config


def _resample_epochs(X: np.ndarray, original_fs: int, target_fs: int) -> np.ndarray:
    if original_fs == target_fs:
        return X
    n_samples_target = int(X.shape[-1] * target_fs / original_fs)
    return signal.resample(X, n_samples_target, axis=-1)


def _is_all_channels(channels: str | list[str]) -> bool:
    if isinstance(channels, str):
        return channels.lower() == "all"
    return len(channels) == 1 and channels[0].lower() == "all"


def prepare_data(config: dict):
    """Download and prepare dataset with subject-level train/val split."""
    print("=" * 60)
    print("PREPARING DATASET (LaBraM Probe)")
    print("=" * 60)

    dataset_config = config["dataset"]
    train_subjects = dataset_config["train_subjects"]
    val_subjects = dataset_config["val_subjects"]
    runs = dataset_config["runs"]
    event_keys = dataset_config.get("event_keys", ["T0", "T1", "T2"])

    print(f"\n  Training subjects:   {train_subjects}")
    print(f"  Validation subjects: {val_subjects}")
    print(f"  Runs per subject:    {runs}")
    print(f"  Event keys:          {event_keys}")

    data_dir = get_project_root() / "data" / "raw" / "physionet"
    dataset = PhysioNetDataset(str(data_dir))

    preprocess_config = config["preprocessing"]
    channels = preprocess_config["channels"]
    use_all_channels = _is_all_channels(channels)
    original_fs = preprocess_config["sampling_rate"]
    target_fs = preprocess_config.get("target_sampling_rate", original_fs)

    label_cfg = config["labels"]
    event_to_class = label_cfg["event_to_class"]

    channel_names: list[str] | None = None

    def load_subjects(subject_ids, label):
        nonlocal channel_names
        print(f"\n{'='*60}")
        print(f"LOADING {label.upper()} SUBJECTS")
        print("=" * 60)
        all_X, all_y = [], []

        for subject_id in tqdm(subject_ids, desc=f"{label} subjects"):
            try:
                dataset.download_subject(subject_id, runs)
                X, y, event_id, subject_channel_names = dataset.load_subject(
                    subject_id,
                    runs,
                    channels,
                    event_keys=event_keys,
                    return_event_id=True,
                    return_channel_names=True,
                )

                if X is None or len(X) == 0:
                    print(f"  Subject {subject_id}: no data, skipping")
                    continue

                if channel_names is None:
                    channel_names = subject_channel_names
                    mode = "all" if use_all_channels else "subset"
                    print(f"  Channel mode: {mode} ({len(channel_names)} channels)")
                elif subject_channel_names != channel_names:
                    print(f"  Subject {subject_id}: channel mismatch, skipping")
                    continue

                label_map = {}
                for event_key, class_id in event_to_class.items():
                    if event_key not in event_id:
                        continue
                    label_map[event_id[event_key]] = class_id

                if len(label_map) < 3:
                    print(f"  Subject {subject_id}: missing events, skipping")
                    continue

                y = np.array([label_map[v] for v in y])

                X = preprocess_eeg(
                    X,
                    lowcut=preprocess_config["lowcut"],
                    highcut=preprocess_config["highcut"],
                    fs=original_fs,
                )
                X = _resample_epochs(X, original_fs, target_fs)

                all_X.append(X)
                all_y.append(y)
                print(f"  Subject {subject_id}: OK {len(X)} epochs")

            except Exception as e:
                print(f"  Subject {subject_id}: ERROR {e}")

        return all_X, all_y

    train_X, train_y = load_subjects(train_subjects, "training")
    val_X, val_y = load_subjects(val_subjects, "validation")

    if not train_X or not val_X or channel_names is None:
        print("\nNeed data for both training and validation.")
        return None, None, None, None, None

    X_train = np.concatenate(train_X)
    y_train = np.concatenate(train_y)
    X_val = np.concatenate(val_X)
    y_val = np.concatenate(val_y)

    print(f"\n{'='*60}")
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"  Train: {len(X_train)} epochs {X_train.shape}  classes={np.bincount(y_train)}")
    print(f"  Val:   {len(X_val)} epochs {X_val.shape}  classes={np.bincount(y_val)}")
    print(f"  Channels used: {len(channel_names)}")
    print(f"  Labels: {label_cfg['class_names']}")

    return X_train, y_train, X_val, y_val, channel_names


def train_model(config: dict, X_train, y_train, X_val, y_val, channel_names: list[str]):
    """Train LaBraM probe and save the best checkpoint."""
    print("\n" + "=" * 60)
    print("TRAINING LaBraM PROBE")
    print("=" * 60)

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False
    )

    model_cfg = config["model"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")

    checkpoint_path = Path(model_cfg["checkpoint_path"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = get_project_root() / checkpoint_path

    model = LaBraMProbe(
        checkpoint_path=checkpoint_path,
        channel_names=channel_names,
        num_classes=model_cfg["num_classes"],
        freeze_encoder=model_cfg.get("freeze_encoder", True),
        pooling=model_cfg.get("pooling", "mean"),
        hidden_dim=model_cfg.get("hidden_dim", 256),
        dropout=model_cfg.get("dropout", 0.3),
    )

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    save_path = (
        get_project_root()
        / config["training"]["savedir"]
        / config["training"]["savename"]
    )
    os.makedirs(save_path.parent, exist_ok=True)

    best_val_acc = 0.0
    patience_counter = 0
    patience = config["training"]["early_stopping_patience"]
    num_epochs = config["training"]["num_epochs"]

    print(f"  Epochs: {num_epochs}  |  Early-stop patience: {patience}\n")

    for epoch in range(num_epochs):
        model.train()
        train_losses, train_accs = [], []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            acc = (logits.argmax(1) == y_batch).float().mean().item()
            train_losses.append(loss.item())
            train_accs.append(acc)

        model.eval()
        val_losses, val_accs = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                acc = (logits.argmax(1) == y_batch).float().mean().item()
                val_losses.append(loss.item())
                val_accs.append(acc)

        train_loss = np.mean(train_losses)
        train_acc = np.mean(train_accs)
        val_loss = np.mean(val_losses)
        val_acc = np.mean(val_accs)

        print(
            f"Epoch {epoch+1:03d}/{num_epochs} | "
            f"Train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"Val loss={val_loss:.4f} acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "best_val_acc": best_val_acc,
                    "channel_names": channel_names,
                },
                str(save_path),
            )
            print(f"  New best ({val_acc:.4f}) saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}.")
                break

    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")
    return model


def main():
    print("\nLaBraM EMBEDDING PROBE TRAINING\n")

    config = load_config("labram_probe")

    X_train, y_train, X_val, y_val, channel_names = prepare_data(config)
    if X_train is None:
        print("Cannot proceed without data. Exiting.")
        return

    train_model(config, X_train, y_train, X_val, y_val, channel_names)

    save_path = (
        get_project_root()
        / config["training"]["savedir"]
        / config["training"]["savename"]
    )
    print(f"\nModel saved to: {save_path}")


if __name__ == "__main__":
    main()
