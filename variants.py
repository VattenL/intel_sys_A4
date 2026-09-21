"""The M1..M4 architecture-evolution experiment (tutorial section 45).

Kept in a module rather than inline in the notebook for two reasons:

1. `prefill_variants.py` and `04_compare.ipynb` must build *identical* models —
   sharing the code guarantees that rather than hoping two copies stay in sync.
2. Training is checkpointed per model. On this machine the GPU thermally
   throttles under sustained load (97 C, SW slowdown), which stretches a run far
   beyond its unthrottled cost, so a run that is interrupted must not lose the
   models it already finished.

Each model is the previous one plus exactly one mechanism:

    M1  Conv + ReLU + Pool                  baseline
    M2  + BatchNorm                         activations drift between layers
    M3  + Residual   Y = F(X) + X           depth stops helping
    M4  + SE attention                      channels are not equally useful
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

import ass4_utils as U

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "results", "variant_cache")

# Narrower than a "real" CIFAR model on purpose - see the notebook's honest-limits
# section. Widths 16/32/64 cost roughly a quarter of 32/64/128 in FLOPs, which is
# what makes the four-model comparison finish on a thermally throttled laptop GPU.
WIDTHS = [16, 32, 64]
EPOCHS = 10
BATCH_SIZE = 128
LR = 1e-3

VARIANTS = {
    "M1  Conv+ReLU+Pool":  dict(use_bn=False, use_residual=False, use_se=False),
    "M2  + BatchNorm":     dict(use_bn=True,  use_residual=False, use_se=False),
    "M3  + Residual":      dict(use_bn=True,  use_residual=True,  use_se=False),
    "M4  + Attention(SE)": dict(use_bn=True,  use_residual=True,  use_se=True),
}


class SEBlock(nn.Module):
    """Tutorial section 31: squeeze -> excite -> rescale. Answers "which channels matter?"."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        s = self.pool(x).view(b, c)
        s = self.fc(s).view(b, c, 1, 1)
        return x * s


class Stage(nn.Module):
    """Two 3x3 convolutions, then halve the spatial size."""

    def __init__(self, in_ch, out_ch, use_bn=False, use_residual=False, use_se=False):
        super().__init__()
        self.use_residual = use_residual

        def norm(c):
            return nn.BatchNorm2d(c) if use_bn else nn.Identity()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=not use_bn)
        self.bn1 = norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=not use_bn)
        self.bn2 = norm(out_ch)
        self.relu = nn.ReLU(inplace=True)

        # Built only when actually used, so M2's parameter count stays honest.
        if use_residual:
            self.shortcut = (nn.Identity() if in_ch == out_ch else
                             nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=not use_bn),
                                           norm(out_ch)))
        else:
            self.shortcut = None

        self.se = SEBlock(out_ch) if use_se else nn.Identity()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        identity = self.shortcut(x) if self.use_residual else None

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.use_residual:
            out = out + identity          # Y = F(X) + X
        out = self.relu(out)
        out = self.se(out)
        return self.pool(out)


class VariantCNN(nn.Module):
    def __init__(self, in_ch=3, n_classes=10, use_bn=False, use_residual=False, use_se=False):
        super().__init__()
        widths = [in_ch] + WIDTHS
        self.stages = nn.Sequential(*[
            Stage(widths[i], widths[i + 1], use_bn, use_residual, use_se)
            for i in range(len(WIDTHS))
        ])
        # Global average pooling instead of flatten: notebook 3 showed the
        # flatten-then-dense head holding 96% of that model's parameters.
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(widths[-1], n_classes),
        )

    def forward(self, x):
        return self.head(self.stages(x))


def build(name: str, in_ch=3, n_classes=10) -> VariantCNN:
    return VariantCNN(in_ch, n_classes, **VARIANTS[name])


def _key(name: str) -> str:
    return name.split()[0].lower()         # "M1  Conv+ReLU+Pool" -> "m1"


def _cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"{_key(name)}.json")


def _weights_path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"{_key(name)}.pt")


def cached(name: str):
    p = _cache_path(name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def load_model(name: str, in_ch=3, n_classes=10, device="cpu"):
    """Rebuild a trained variant from its cached weights (None if never trained)."""
    p = _weights_path(name)
    if not os.path.exists(p):
        return None
    model = build(name, in_ch, n_classes)
    model.load_state_dict(torch.load(p, map_location=device))
    return model.to(device).eval()


def train_variant(name, X_train, y_train, X_test, y_test, n_classes,
                  device, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                  seed=U.SEED, log=print, use_cache=True):
    """Train one variant, or return its cached result if it was trained before."""
    from torch.utils.data import TensorDataset, DataLoader

    if use_cache:
        hit = cached(name)
        if hit is not None:
            log(f"{name}: loaded from cache "
                f"({hit['n_params']:,} params, {hit['epochs']} epochs, "
                f"{hit['train_seconds']:.0f}s on the original run)")
            return hit

    U.set_seed(seed)
    model = build(name, X_train.shape[1], n_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # CIFAR-10 is ~614 MB as float32 and fits in this GPU's memory, so upload it
    # once and slice batches on-device. A DataLoader over a TensorDataset collates
    # 128 individual tensor slices per batch in Python, which for a model this
    # small dominates the step time completely - it measured ~140 s/epoch that way
    # versus seconds here. Falls back to the DataLoader if the upload does not fit.
    on_gpu = device.type == "cuda"
    if on_gpu:
        try:
            Xg = torch.tensor(X_train).to(device)
            yg = torch.tensor(y_train).to(device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            on_gpu = False
            log("  (dataset does not fit on GPU - falling back to DataLoader)")

    if not on_gpu:
        loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
                            batch_size=batch_size, shuffle=True)

    Xte_t = torch.tensor(X_test).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = {"loss": [], "val_acc": []}
    log(f"\n=== {name}  ({n_params:,} parameters) ===")

    n_train = len(X_train)

    with U.Timer() as t:
        for epoch in range(1, epochs + 1):
            model.train()
            running, nb = 0.0, 0

            if on_gpu:
                perm = torch.randperm(n_train, device=device)
                batches = ((Xg[perm[i:i + batch_size]], yg[perm[i:i + batch_size]])
                           for i in range(0, n_train, batch_size))
            else:
                batches = ((xb.to(device, non_blocking=True), yb.to(device, non_blocking=True))
                           for xb, yb in loader)

            for xb, yb in batches:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    # Mixed precision: the 3060 has tensor cores, and halving the
                    # arithmetic also lowers the thermal load, which matters here
                    # because the GPU is throttling.
                    with torch.autocast("cuda", dtype=torch.float16):
                        loss = criterion(model(xb), yb)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()
                running += loss.item(); nb += 1

            model.eval()
            with torch.no_grad():
                pred = np.concatenate([
                    model(Xte_t[i:i + 512]).argmax(1).cpu().numpy()
                    for i in range(0, len(Xte_t), 512)
                ])
            acc = float((pred == y_test).mean())
            history["loss"].append(running / nb)
            history["val_acc"].append(acc)
            log(f"  epoch {epoch:>2}/{epochs}  loss {running/nb:.4f}  test_acc {acc:.4f}")

    metrics = U.evaluate(y_test, pred, n_classes)
    record = {
        "framework": "PyTorch", "dataset": "CIFAR-10 (full 50k)", "model": name,
        "n_params": n_params, "epochs": epochs,
        "train_seconds": round(t.seconds, 2),
        "train_loss": round(history["loss"][-1], 4),
        "history": history, **metrics,
    }

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(_cache_path(name), "w") as f:
        json.dump(record, f, indent=2)
    # Weights too, so the notebook can inspect a trained model (e.g. the SE gates)
    # without retraining it.
    torch.save(model.state_dict(), _weights_path(name))
    log(f"  final: accuracy {metrics['test_accuracy']:.4f}  "
        f"macro-F1 {metrics['f1_macro']:.4f}  ({t.seconds:.0f}s)  [cached]")

    return record


def replay(record, log=print):
    """Print a cached run's per-epoch history, so cached output reads like a live run."""
    h = record["history"]
    log(f"=== {record['model']}  ({record['n_params']:,} parameters) ===")
    for i, (l, a) in enumerate(zip(h["loss"], h["val_acc"]), start=1):
        log(f"  epoch {i:>2}/{record['epochs']}  loss {l:.4f}  test_acc {a:.4f}")
    log(f"  final: accuracy {record['test_accuracy']:.4f}  "
        f"macro-F1 {record['f1_macro']:.4f}  ({record['train_seconds']:.0f}s)")
