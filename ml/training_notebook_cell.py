"""Train LaBraM probe and save the best checkpoint."""

from eeg.layers.labram_encoder import LaBraMEncoder
from torch.utils.data import WeightedRandomSampler

# from models.labram_probe import LaBraMProbe

class LaBraMProbe(nn.Module):
    def __init__(
        self,
        checkpoint_path: str | Path,
        channel_names: list[str],
        num_classes: int = 3,
        freeze_encoder: bool = True,
        unfreeze_last_n_blocks: int = 2,
        pooling: str = "mean",
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.encoder = LaBraMEncoder.from_pretrained(str(checkpoint_path))
        self.channel_names = channel_names
        self.pooling = pooling
        self.freeze_encoder = freeze_encoder
        self.unfreeze_last_n_blocks = max(0, int(unfreeze_last_n_blocks))

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if self.unfreeze_last_n_blocks > 0:
            total_blocks = len(self.encoder.model.blocks)
            n = min(self.unfreeze_last_n_blocks, total_blocks)
            for block in self.encoder.model.blocks[-n:]:
                for param in block.parameters():
                    param.requires_grad = True

            # Keep encoder output scale adaptable when any encoder blocks are trainable.
            for param in self.encoder.model.norm.parameters():
                param.requires_grad = True

        self.encoder_has_trainable_params = any(
            p.requires_grad for p in self.encoder.parameters()
        )

        embed_dim = self.encoder.model.embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, N, P, E)
        if self.pooling == "mean":
            return tokens.mean(dim=(1, 2))
        if self.pooling == "max":
            return tokens.amax(dim=(1, 2))
        if self.pooling == "cls":
            raise ValueError("CLS pooling is handled in forward() using class token.")
        raise ValueError(f"Unsupported pooling method: {self.pooling}")

    def encode(self, x: torch.Tensor, force_no_grad: bool = False) -> torch.Tensor:
        # x: (B, N, T)
        if self.encoder_has_trainable_params:
            def _encode_inner():
                if self.pooling == "cls":
                    # (B, N*P+1, E); index 0 is class token
                    tokens_all = self.encoder(
                        x, channel_names=self.channel_names, return_all_patch_tokens=True
                    )
                    return tokens_all[:, 0, :]

                patch_tokens = self.encoder(
                    x, channel_names=self.channel_names, return_patch_tokens=True
                )
                return self._pool_tokens(patch_tokens)

            if force_no_grad:
                with torch.no_grad():
                    pooled = _encode_inner()
            else:
                pooled = _encode_inner()
        else:
            with torch.no_grad():
                if self.pooling == "cls":
                    tokens_all = self.encoder(
                        x, channel_names=self.channel_names, return_all_patch_tokens=True
                    )
                    pooled = tokens_all[:, 0, :]
                else:
                    patch_tokens = self.encoder(
                        x, channel_names=self.channel_names, return_patch_tokens=True
                    )
                    pooled = self._pool_tokens(patch_tokens)

        return pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.encode(x)
        return self.classifier(pooled)


print("\n" + "=" * 60)
print("TRAINING LaBraM PROBE")
print("=" * 60)

use_binary_t1_t2 = True
use_long_context_resample = True
target_context_len = 1600

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_val = np.asarray(X_val)
y_val = np.asarray(y_val)

if use_binary_t1_t2:
    # Keep only T1/T2 classes (commonly encoded as 0/1 in this pipeline).
    # If upstream encoding differs, remap the two retained class ids to {0, 1}.
    candidate_labels = [0, 1]
    train_mask = np.isin(y_train, candidate_labels)
    val_mask = np.isin(y_val, candidate_labels)

    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    X_val = X_val[val_mask]
    y_val = y_val[val_mask]

    kept_labels = sorted(set(y_train.tolist()) | set(y_val.tolist()))
    label_remap = {old: new for new, old in enumerate(kept_labels)}
    y_train = np.array([label_remap[int(v)] for v in y_train], dtype=np.int64)
    y_val = np.array([label_remap[int(v)] for v in y_val], dtype=np.int64)

    num_classes = 2
    print(
        f"  Binary mode enabled (T1/T2 only). "
        f"Kept labels={kept_labels}, remap={label_remap}"
    )
else:
    num_classes = int(config["model"]["num_classes"])

def resample_epochs_to_length(X: np.ndarray, target_len: int) -> np.ndarray:
    # Resample each epoch/channel to fixed temporal length expected by LaBraM context.
    if X.shape[-1] == target_len:
        return X
    n_epochs, n_channels, src_len = X.shape
    x_old = np.linspace(0.0, 1.0, src_len, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    X_out = np.empty((n_epochs, n_channels, target_len), dtype=np.float32)
    for i in range(n_epochs):
        for ch in range(n_channels):
            X_out[i, ch] = np.interp(x_new, x_old, X[i, ch].astype(np.float32))
    return X_out

def zscore_epochs(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    # Per-epoch, per-channel normalization over time.
    mu = X.mean(axis=-1, keepdims=True)
    sd = X.std(axis=-1, keepdims=True)
    return (X - mu) / (sd + eps)

if use_long_context_resample:
    X_train = resample_epochs_to_length(X_train, target_context_len)
    X_val = resample_epochs_to_length(X_val, target_context_len)
    print(
        f"  Long-context resample enabled. "
        f"New sequence length={target_context_len}"
    )

X_train = zscore_epochs(X_train)
X_val = zscore_epochs(X_val)

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_val_t = torch.FloatTensor(X_val)
y_val_t = torch.LongTensor(y_val)

batch_size = config["training"]["batch_size"]
assert y_train_t.min().item() >= 0 and y_train_t.max().item() < num_classes
assert y_val_t.min().item() >= 0 and y_val_t.max().item() < num_classes

train_class_counts = torch.bincount(y_train_t, minlength=num_classes).float()
val_class_counts = torch.bincount(y_val_t, minlength=num_classes).float()
print(f"  Train class counts: {train_class_counts.tolist()}")
print(f"  Val class counts:   {val_class_counts.tolist()}")

# IMPORTANT: Do not combine sampler + weighted CE unless intentionally testing.
use_weighted_sampler = False
use_class_weighted_loss = False

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)

if use_weighted_sampler:
    class_weights_for_sampler = 1.0 / train_class_counts.clamp(min=1)
    sample_weights = class_weights_for_sampler[y_train_t]
    assert len(sample_weights) == len(train_dataset), (
        f"Sample weights length {len(sample_weights)} != "
        f"train dataset length {len(train_dataset)}"
    )
    train_sampler = WeightedRandomSampler(
        weights=sample_weights.double(),
        num_samples=len(train_dataset),
        replacement=True,
    )
else:
    train_sampler = None

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=train_sampler,
    shuffle=not use_weighted_sampler,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)

model_cfg = config["model"]
pooling_mode = "cls"
full_finetune = True
unfreeze_last_n_blocks = 0
head_lr = 2e-4
encoder_lr = 2e-5
weight_decay = 1e-4
label_smoothing = 0.05
max_grad_norm = 1.0
warmup_ratio = 0.1
print(
    f"  Overrides | pooling={pooling_mode} "
    f"full_finetune={full_finetune} "
    f"unfreeze_last_n_blocks={unfreeze_last_n_blocks} "
    f"head_lr={head_lr} encoder_lr={encoder_lr} "
    f"label_smoothing={label_smoothing}"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n  Device: {device}")

checkpoint_path = Path(model_cfg["checkpoint_path"])
if not checkpoint_path.is_absolute():
    checkpoint_path = get_project_root() / checkpoint_path

model = LaBraMProbe(
    checkpoint_path=checkpoint_path,
    channel_names=channel_names,
    num_classes=num_classes,
    freeze_encoder=not full_finetune,
    unfreeze_last_n_blocks=unfreeze_last_n_blocks,
    pooling=pooling_mode,
    hidden_dim=model_cfg.get("hidden_dim", 256),
    dropout=model_cfg.get("dropout", 0.3),
)

model.to(device)

def run_linear_probe_sanity(
    model: LaBraMProbe,
    X_train_t: torch.Tensor,
    y_train_t: torch.Tensor,
    X_val_t: torch.Tensor,
    y_val_t: torch.Tensor,
    num_classes: int,
    device: str,
    batch_size: int = 256,
    epochs: int = 25,
    lr: float = 1e-2,
):
    print("\n[Sanity] Running frozen-embedding linear probe...")

    def extract_features(X_tensor: torch.Tensor) -> torch.Tensor:
        feat_loader = DataLoader(
            TensorDataset(X_tensor), batch_size=batch_size, shuffle=False
        )
        features = []
        model.eval()
        with torch.no_grad():
            for (x_batch,) in feat_loader:
                x_batch = x_batch.to(device)
                pooled = model.encode(x_batch, force_no_grad=True)
                features.append(pooled.cpu())
        return torch.cat(features, dim=0)

    train_features = extract_features(X_train_t)
    val_features = extract_features(X_val_t)
    print(
        f"[Sanity] Feature shapes train={tuple(train_features.shape)} "
        f"val={tuple(val_features.shape)}"
    )

    linear_head = nn.Linear(train_features.shape[1], num_classes).to(device)
    sanity_optimizer = optim.AdamW(linear_head.parameters(), lr=lr)
    sanity_criterion = nn.CrossEntropyLoss()

    sanity_train_loader = DataLoader(
        TensorDataset(train_features, y_train_t), batch_size=batch_size, shuffle=True
    )
    sanity_val_loader = DataLoader(
        TensorDataset(val_features, y_val_t), batch_size=batch_size, shuffle=False
    )

    best_val_acc = 0.0
    for ep in range(1, epochs + 1):
        linear_head.train()
        train_correct = 0
        train_total = 0
        train_losses = []
        for feat_batch, y_batch in sanity_train_loader:
            feat_batch = feat_batch.to(device)
            y_batch = y_batch.to(device)
            sanity_optimizer.zero_grad()
            logits = linear_head(feat_batch)
            loss = sanity_criterion(logits, y_batch)
            loss.backward()
            sanity_optimizer.step()

            preds = logits.argmax(1)
            train_correct += (preds == y_batch).sum().item()
            train_total += y_batch.numel()
            train_losses.append(loss.item())

        linear_head.eval()
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_targets = []
        with torch.no_grad():
            for feat_batch, y_batch in sanity_val_loader:
                feat_batch = feat_batch.to(device)
                y_batch = y_batch.to(device)
                logits = linear_head(feat_batch)
                preds = logits.argmax(1)
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.numel()
                all_val_preds.append(preds.cpu())
                all_val_targets.append(y_batch.cpu())

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        best_val_acc = max(best_val_acc, val_acc)

        if ep == 1 or ep % 5 == 0 or ep == epochs:
            val_preds_flat = torch.cat(all_val_preds)
            val_targets_flat = torch.cat(all_val_targets)
            val_pred_counts = torch.bincount(val_preds_flat, minlength=num_classes)
            val_true_counts = torch.bincount(val_targets_flat, minlength=num_classes)
            print(
                f"[Sanity] Epoch {ep:02d}/{epochs} "
                f"train_loss={np.mean(train_losses):.4f} "
                f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
            )
            print(
                f"[Sanity] Val true counts={val_true_counts.tolist()} "
                f"pred counts={val_pred_counts.tolist()}"
            )

    print(f"[Sanity] Best val acc: {best_val_acc:.4f}\n")


run_linear_sanity_check = True
if run_linear_sanity_check:
    run_linear_probe_sanity(
        model=model,
        X_train_t=X_train_t,
        y_train_t=y_train_t,
        X_val_t=X_val_t,
        y_val_t=y_val_t,
        num_classes=num_classes,
        device=device,
        batch_size=batch_size,
        epochs=25,
        lr=1e-2,
    )

if use_class_weighted_loss:
    ce_weights = (len(y_train_t) / (num_classes * train_class_counts.clamp(min=1))).to(
        device=device, dtype=torch.float32
    )
    criterion = nn.CrossEntropyLoss(
        weight=ce_weights, label_smoothing=label_smoothing
    )
else:
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

head_params = [p for p in model.classifier.parameters() if p.requires_grad]
encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
if encoder_params:
    optimizer = optim.AdamW(
        [
            {"params": head_params, "lr": head_lr},
            {"params": encoder_params, "lr": encoder_lr},
        ],
        weight_decay=weight_decay,
    )
else:
    optimizer = optim.AdamW(head_params, lr=head_lr, weight_decay=weight_decay)

num_head_params = sum(p.numel() for p in head_params)
num_encoder_params = sum(p.numel() for p in encoder_params)
print(
    f"  Trainable params | head={num_head_params:,} "
    f"encoder={num_encoder_params:,} "
    f"(unfreeze_last_n_blocks={model.unfreeze_last_n_blocks})"
)
if num_encoder_params > 0:
    print(
        f"  Optimizer LRs | head={head_lr} encoder={encoder_lr} "
        f"weight_decay={weight_decay}"
    )
else:
    print(f"  Optimizer LR  | head={head_lr} weight_decay={weight_decay}")

use_amp = device == "cuda"
amp_dtype = torch.float16 if use_amp else None
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
print(f"  AMP enabled={use_amp}")





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

steps_per_epoch = max(1, len(train_loader))
total_steps = max(1, num_epochs * steps_per_epoch)
warmup_steps = int(total_steps * warmup_ratio)

def lr_lambda(current_step: int) -> float:
    if warmup_steps > 0 and current_step < warmup_steps:
        return float(current_step + 1) / float(warmup_steps)
    progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * progress))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
print(
    f"  Scheduler | total_steps={total_steps} "
    f"warmup_steps={warmup_steps} warmup_ratio={warmup_ratio}"
)

print(f"  Epochs: {num_epochs}  |  Early-stop patience: {patience}\n")

for epoch in range(num_epochs):
    model.train()
    train_losses = []
    train_correct = 0
    train_total = 0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            logits = model(X_batch)
            loss = criterion(logits, y_batch)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        preds = logits.argmax(1)
        train_correct += (preds == y_batch).sum().item()
        train_total += y_batch.numel()
        train_losses.append(loss.item())

    model.eval()
    val_losses = []
    val_correct = 0
    val_total = 0
    all_val_preds = []
    all_val_targets = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                logits = model(X_batch)
                loss = criterion(logits, y_batch)

            preds = logits.argmax(1)
            val_correct += (preds == y_batch).sum().item()
            val_total += y_batch.numel()
            all_val_preds.append(preds.cpu())
            all_val_targets.append(y_batch.cpu())

            val_losses.append(loss.item())

    train_loss = np.mean(train_losses)
    train_acc = train_correct / max(train_total, 1)
    val_loss = np.mean(val_losses)
    val_acc = val_correct / max(val_total, 1)

    val_preds_flat = torch.cat(all_val_preds) if all_val_preds else torch.empty(0, dtype=torch.long)
    val_targets_flat = torch.cat(all_val_targets) if all_val_targets else torch.empty(0, dtype=torch.long)
    val_pred_counts = torch.bincount(val_preds_flat, minlength=num_classes)
    val_true_counts = torch.bincount(val_targets_flat, minlength=num_classes)

    print(
        f"Epoch {epoch+1:03d}/{num_epochs} | "
        f"Train loss={train_loss:.4f} acc={train_acc:.4f} | "
        f"Val loss={val_loss:.4f} acc={val_acc:.4f}"
    )
    current_lrs = [pg["lr"] for pg in optimizer.param_groups]
    print(f"  LRs={current_lrs}")
    print(
        f"  Val true counts={val_true_counts.tolist()} "
        f"pred counts={val_pred_counts.tolist()}"
    )

    # Track per-class accuracy to spot class collapse early.
    per_class_acc = []
    for c in range(num_classes):
        class_mask = val_targets_flat == c
        if class_mask.any():
            class_acc = (val_preds_flat[class_mask] == c).float().mean().item()
        else:
            class_acc = float("nan")
        per_class_acc.append(class_acc)
    print(f"  Val per-class acc={per_class_acc}")

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

