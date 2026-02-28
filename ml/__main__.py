#!/usr/bin/env python3
"""Train EEGNet model on PhysioNet Motor Imagery dataset."""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from eeg.dataset import PhysioNetDataset, preprocess_eeg
from models.eegnet import EEGClassifier, EEGNet
from models.eegnet_residual import EEGNetResidual
from utils.config_loader import get_project_root, load_config


def prepare_data(config: dict):
    """Download and prepare dataset with subject-level train/val split."""
    print("=" * 60)
    print("PREPARING DATASET")
    print("=" * 60)

    dataset_config = config["dataset"]
    train_subjects = dataset_config["train_subjects"]
    val_subjects = dataset_config["val_subjects"]
    runs = dataset_config["runs"]

    print(f"\n  Training subjects:   {train_subjects}")
    print(f"  Validation subjects: {val_subjects}")
    print(f"  Runs per subject:    {runs}")

    data_dir = get_project_root() / "data" / "raw" / "physionet"
    dataset = PhysioNetDataset(str(data_dir))

    preprocess_config = config["preprocessing"]
    channels = preprocess_config["channels"]

    def load_subjects(subject_ids, label):
        print(f"\n{'='*60}")
        print(f"LOADING {label.upper()} SUBJECTS")
        print("=" * 60)
        all_X, all_y = [], []

        for subject_id in tqdm(subject_ids, desc=f"{label} subjects"):
            try:
                dataset.download_subject(subject_id, runs)
                X, y = dataset.load_subject(subject_id, runs, channels)

                if X is None or len(X) == 0:
                    print(f"  Subject {subject_id}: no data, skipping")
                    continue

                unique_events = np.unique(y)
                if len(unique_events) != 2:
                    print(
                        f"  Subject {subject_id}: expected 2 events, "
                        f"got {unique_events}, skipping"
                    )
                    continue

                # Remap event IDs to 0, 1
                label_map = {unique_events[0]: 0, unique_events[1]: 1}
                y = np.array([label_map[v] for v in y])

                X = preprocess_eeg(
                    X,
                    lowcut=preprocess_config["lowcut"],
                    highcut=preprocess_config["highcut"],
                    fs=preprocess_config["sampling_rate"],
                )

                all_X.append(X)
                all_y.append(y)
                print(f"  Subject {subject_id}: ✓ {len(X)} epochs")

            except Exception as e:
                print(f"  Subject {subject_id}: ✗ {e}")

        return all_X, all_y

    train_X, train_y = load_subjects(train_subjects, "training")
    val_X, val_y = load_subjects(val_subjects, "validation")

    if not train_X or not val_X:
        print("\n❌ Need data for both training and validation.")
        return None, None, None, None

    X_train = np.concatenate(train_X)
    y_train = np.concatenate(train_y)
    X_val = np.concatenate(val_X)
    y_val = np.concatenate(val_y)

    print(f"\n{'='*60}")
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"  Train: {len(X_train)} epochs {X_train.shape}  classes={np.bincount(y_train)}")
    print(f"  Val:   {len(X_val)} epochs {X_val.shape}  classes={np.bincount(y_val)}")
    print("  Labels: 0=left hand, 1=right hand")

    return X_train, y_train, X_val, y_val


def train_model(config: dict, X_train, y_train, X_val, y_val):
    """Train EEGNet (or EEGNetResidual) and save the best checkpoint."""
    print("\n" + "=" * 60)
    print("TRAINING MODEL")
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

    model_config = config["model"]
    n_channels = model_config["input_channels"]
    n_classes = 2  # left / right hand only
    n_samples = X_train.shape[2]
    dropout = model_config["dropout"]
    kernel_length = model_config["kernel_length"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")

    if model_config.get("use_residual", True):
        print("  Model:  EEGNetResidual")
        model = EEGNetResidual(
            n_channels=n_channels,
            n_classes=n_classes,
            n_samples=n_samples,
            dropout=dropout,
            kernel_length=kernel_length,
            use_attention=model_config.get("use_attention", False),
        )
    else:
        print("  Model:  EEGNet")
        model = EEGNet(
            n_channels=n_channels,
            n_classes=n_classes,
            n_samples=n_samples,
            dropout=dropout,
            kernel_length=kernel_length,
        )

    classifier = EEGClassifier(model, device)
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
        train_losses, train_accs = [], []
        for X_batch, y_batch in train_loader:
            loss, acc = classifier.train_step(X_batch, y_batch, optimizer, criterion)
            train_losses.append(loss)
            train_accs.append(acc)

        val_losses, val_accs = [], []
        for X_batch, y_batch in val_loader:
            loss, acc, _ = classifier.eval_step(X_batch, y_batch, criterion)
            val_losses.append(loss)
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
            classifier.save(str(save_path))
            print(f"  → New best ({val_acc:.4f}) — saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}.")
                break

    print(f"\n✓ Training complete. Best val acc: {best_val_acc:.4f}")
    return classifier


def main():
    print("\n🧠 EEG MOTOR IMAGERY CLASSIFIER TRAINING\n")

    config = load_config("eeg_config")

    X_train, y_train, X_val, y_val = prepare_data(config)
    if X_train is None:
        print("Cannot proceed without data. Exiting.")
        return

    train_model(config, X_train, y_train, X_val, y_val)

    save_path = (
        get_project_root()
        / config["training"]["savedir"]
        / config["training"]["savename"]
    )
    print(f"\nModel saved to: {save_path}")
    print("Run `uv run server --loader physionet` to use it.")


if __name__ == "__main__":
    main()
