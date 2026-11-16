
import copy
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from src.text_tokens import PAD_ID


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 3
    min_delta: float = 1e-4
    grad_clip: float = 1.0
    seed: int = 42
    use_class_weights: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """GPU when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- data containers
class SeqData:
    """Pre-encoded, padded token ids. Each batch is trimmed to its own longest sequence."""

    def __init__(self, ids: torch.Tensor, lengths: torch.Tensor, y: Optional[torch.Tensor] = None):
        self.ids, self.lengths, self.y = ids, lengths, y

    def __len__(self):
        return int(self.ids.size(0))

    def batch(self, idx):
        lengths = self.lengths[idx]
        x = self.ids[idx][:, : int(lengths.max())]
        return (x, lengths), (self.y[idx] if self.y is not None else None)


class TfidfData:
    """Sparse TF-IDF matrix densified one mini-batch at a time (keeps memory small)."""

    def __init__(self, X, y: Optional[torch.Tensor] = None):
        self.X, self.y = X, y

    def __len__(self):
        return int(self.X.shape[0])

    def batch(self, idx):
        rows = self.X[np.asarray(idx)]
        return (torch.from_numpy(rows.toarray()).float(),), (self.y[idx] if self.y is not None else None)


def encode_sequences(texts, vocab, max_len: int, y=None) -> SeqData:
    ids = torch.full((len(texts), max_len), PAD_ID, dtype=torch.long)
    lengths = torch.empty(len(texts), dtype=torch.long)
    for i, t in enumerate(texts):
        enc = vocab.encode(t, max_len)
        ids[i, : len(enc)] = torch.tensor(enc, dtype=torch.long)
        lengths[i] = len(enc)
    return SeqData(ids, lengths, None if y is None else torch.as_tensor(np.asarray(y), dtype=torch.long))


def class_weights_from(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """Inverse-frequency weights (n / (k * count)) from TRAINING labels only."""
    counts = np.bincount(np.asarray(y), minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    return torch.tensor(len(y) / (n_classes * counts), dtype=torch.float32)


# ---------------------------------------------------------------- evaluation helper
@torch.no_grad()
def predict_logits(model: nn.Module, data, device, batch_size: int = 512) -> torch.Tensor:
    model.eval()
    out = []
    for start in range(0, len(data), batch_size):
        idx = torch.arange(start, min(start + batch_size, len(data)))
        inputs, _ = data.batch(idx)
        out.append(model(*[t.to(device) for t in inputs]).cpu())
    return torch.cat(out) if out else torch.empty(0)


@torch.no_grad()
def evaluate_split(model, data, criterion, device, batch_size: int = 512) -> dict:
    logits = predict_logits(model, data, device, batch_size)
    y = data.y
    loss = float(criterion(logits, y).item())
    pred = logits.argmax(1).numpy()
    yn = y.numpy()
    return {"loss": loss, "acc": float((pred == yn).mean()),
            "f1_macro": float(f1_score(yn, pred, average="macro", zero_division=0))}


# ---------------------------------------------------------------- the loop
def fit(model: nn.Module, train: "SeqData | TfidfData", val: "SeqData | TfidfData", cfg: TrainConfig,
        n_classes: int, device: Optional[torch.device] = None, checkpoint_path: Optional[Path] = None,
        log: Callable[[str], None] = print) -> dict:
    """Train with Adam + CrossEntropyLoss, early stopping on validation loss, best checkpoint restored."""
    set_seed(cfg.seed)
    device = device or get_device()
    model.to(device)
    weight = class_weights_from(train.y.numpy(), n_classes).to(device) if cfg.use_class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    criterion_cpu = nn.CrossEntropyLoss(weight=weight.cpu() if weight is not None else None)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1_macro": []}
    best = {"loss": float("inf"), "epoch": 0, "state": None}
    bad_epochs, stopped_early = 0, False
    t0 = time.perf_counter()
    n = len(train)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        gen = torch.Generator().manual_seed(cfg.seed + epoch)
        perm = torch.randperm(n, generator=gen)
        loss_sum = correct = 0.0
        for start in range(0, n, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            inputs, yb = train.batch(idx)
            inputs = [t.to(device) for t in inputs]
            yb = yb.to(device)
            optim.zero_grad()
            logits = model(*inputs)
            loss = criterion(logits, yb)
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            loss_sum += float(loss.item()) * len(idx)
            correct += float((logits.argmax(1) == yb).sum().item())

        v = evaluate_split(model, val, criterion_cpu, device)
        for k, val_ in (("epoch", epoch), ("train_loss", loss_sum / n), ("train_acc", correct / n),
                        ("val_loss", v["loss"]), ("val_acc", v["acc"]), ("val_f1_macro", v["f1_macro"])):
            history[k].append(val_)
        log(f"  epoch {epoch:2d}/{cfg.epochs}  train loss {loss_sum / n:.4f} acc {correct / n:.4f} | "
            f"val loss {v['loss']:.4f} acc {v['acc']:.4f} macro-F1 {v['f1_macro']:.4f}")

        if v["loss"] < best["loss"] - cfg.min_delta:
            best.update(loss=v["loss"], epoch=epoch, state=copy.deepcopy(model.state_dict()))
            bad_epochs = 0
            if checkpoint_path is not None:
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": best["state"], "epoch": epoch, "val_loss": v["loss"]}, checkpoint_path)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                stopped_early = True
                log(f"  early stopping at epoch {epoch} (best epoch {best['epoch']})")
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.eval()
    return {"history": history, "best_epoch": best["epoch"], "best_val_loss": best["loss"],
            "stopped_early": stopped_early, "epochs_run": len(history["epoch"]),
            "train_seconds": round(time.perf_counter() - t0, 2), "device": str(device)}
