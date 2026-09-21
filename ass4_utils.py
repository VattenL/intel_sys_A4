"""Shared helpers for Assignment 4.

Every framework leg (scratch NumPy / TensorFlow-Keras / PyTorch) pulls its data
from this module so that the comparison satisfies the lecture's fairness rule:

    Same Dataset + Same Split + Same Architecture + Comparable Hyperparameters

The loaders therefore return plain NumPy arrays and always use the same seed.
"""

from __future__ import annotations

import os
import json
import time
import zipfile
import urllib.request
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

SEED = 42


# --------------------------------------------------------------------------
# reproducibility
# --------------------------------------------------------------------------
def set_seed(seed: int = SEED) -> None:
    """Seed every RNG that might be active, ignoring frameworks not installed."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import random

        random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass


# --------------------------------------------------------------------------
# dataset 1 - Diabetes 130-US hospitals (tabular, 3-class)
# --------------------------------------------------------------------------
DIABETES_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)

# Dropped because they are identifiers or overwhelmingly missing ("?").
_DROP_COLS = [
    "encounter_id",
    "patient_nbr",
    "weight",           # 96.9% missing
    "payer_code",       # 39.6% missing
    "medical_specialty",  # 49.1% missing
]

# Constant in this release - every row holds the same value, so they carry no signal.
_CONSTANT_COLS = ["examide", "citoglipton"]

_NUMERIC_COLS = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

READMIT_CLASSES = ["NO", ">30", "<30"]


def _download_diabetes130() -> str:
    """Return path to diabetic_data.csv, downloading the UCI zip if needed."""
    out_dir = os.path.join(DATA, "diabetes130")
    csv = os.path.join(out_dir, "diabetic_data.csv")
    if os.path.exists(csv):
        return csv
    os.makedirs(DATA, exist_ok=True)
    zip_path = os.path.join(DATA, "diabetes130.zip")
    if not os.path.exists(zip_path):
        print("downloading Diabetes 130-US hospitals from UCI ...")
        urllib.request.urlretrieve(DIABETES_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
    return csv


def _icd9_group(code) -> str:
    """Collapse a raw ICD-9 diagnosis code into a coarse disease group.

    The three diagnosis columns hold ~700 distinct codes each; one-hot encoding
    them directly would swamp the model, so we use the grouping that is standard
    for this dataset (Strack et al., 2014).
    """
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return "missing"
    s = str(code)
    if s in ("?", "nan", ""):
        return "missing"
    if s.startswith("V") or s.startswith("E"):
        return "other"
    try:
        v = float(s)
    except ValueError:
        return "other"
    if 390 <= v <= 459 or int(v) == 785:
        return "circulatory"
    if 460 <= v <= 519 or int(v) == 786:
        return "respiratory"
    if 520 <= v <= 579 or int(v) == 787:
        return "digestive"
    if int(v) == 250:
        return "diabetes"
    if 800 <= v <= 999:
        return "injury"
    if 710 <= v <= 739:
        return "musculoskeletal"
    if 580 <= v <= 629 or int(v) == 788:
        return "genitourinary"
    if 140 <= v <= 239:
        return "neoplasms"
    return "other"


def load_diabetes130(
    n_subset: int | None = None,
    test_size: float = 0.2,
    seed: int = SEED,
    verbose: bool = True,
):
    """Load and preprocess the tabular dataset.

    Returns (X_train, y_train, X_test, y_test, meta). Features are standardised
    floats, labels are ints indexing READMIT_CLASSES.
    """
    csv = _download_diabetes130()
    df = pd.read_csv(csv)
    raw_shape = df.shape

    # One row per patient: later encounters of the same patient leak information
    # about the earlier ones, so keep only the first admission.
    df = df.drop_duplicates(subset="patient_nbr", keep="first")

    # A handful of encounters end in death or hospice; those patients cannot be
    # readmitted, so they are excluded rather than counted as "NO".
    df = df[~df["discharge_disposition_id"].isin([11, 13, 14, 19, 20, 21])]

    y = df["readmitted"].map({c: i for i, c in enumerate(READMIT_CLASSES)}).to_numpy()

    X = df.drop(columns=_DROP_COLS + _CONSTANT_COLS + ["readmitted"], errors="ignore")
    for col in ("diag_1", "diag_2", "diag_3"):
        X[col] = X[col].map(_icd9_group)

    # admission/discharge/source ids are codes, not magnitudes - treat as categorical
    for col in ("admission_type_id", "discharge_disposition_id", "admission_source_id"):
        X[col] = X[col].astype(str)

    X = X.replace("?", "missing")

    num = X[_NUMERIC_COLS].astype(np.float32)
    cat = X.drop(columns=_NUMERIC_COLS)
    cat = pd.get_dummies(cat, drop_first=False, dtype=np.float32)

    feature_names = list(num.columns) + list(cat.columns)
    Xa = np.concatenate([num.to_numpy(), cat.to_numpy()], axis=1).astype(np.float32)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Xa))
    if n_subset is not None and n_subset < len(idx):
        idx = idx[:n_subset]
    Xa, y = Xa[idx], y[idx]

    n_test = int(len(Xa) * test_size)
    X_test, y_test = Xa[:n_test], y[:n_test]
    X_train, y_train = Xa[n_test:], y[n_test:]

    # Standardise using train statistics only - test statistics would leak.
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    X_train = (X_train - mu) / sd
    X_test = (X_test - mu) / sd

    meta = {
        "name": "Diabetes 130-US hospitals",
        "raw_shape": raw_shape,
        "n_features": Xa.shape[1],
        "classes": READMIT_CLASSES,
        "feature_names": feature_names,
        "kind": "tabular",
    }
    if verbose:
        print(f"{meta['name']}: raw {raw_shape} -> {Xa.shape}, "
              f"train {X_train.shape[0]} / test {X_test.shape[0]}, "
              f"{Xa.shape[1]} features, {len(READMIT_CLASSES)} classes")
    return X_train, y_train, X_test, y_test, meta


# --------------------------------------------------------------------------
# datasets 2 and 3 - MNIST and CIFAR-10 (images)
# --------------------------------------------------------------------------
def _torchvision_to_numpy(ds):
    """Materialise a torchvision dataset as (N,C,H,W) float32 in [0,1] + int labels."""
    import torch

    arr = ds.data
    if isinstance(arr, torch.Tensor):
        arr = arr.numpy()
    arr = np.asarray(arr)
    if arr.ndim == 3:                       # MNIST: (N,H,W) grayscale
        arr = arr[:, None, :, :]
    else:                                   # CIFAR-10: (N,H,W,C)
        arr = arr.transpose(0, 3, 1, 2)
    x = arr.astype(np.float32) / 255.0
    y = np.asarray(ds.targets)
    return x, y.astype(np.int64)


def _load_image_dataset(which: str, n_train, n_test, seed, verbose):
    from torchvision import datasets

    root = os.path.join(DATA, which)
    cls = {"mnist": datasets.MNIST, "cifar10": datasets.CIFAR10}[which]
    tr = cls(root=root, train=True, download=True)
    te = cls(root=root, train=False, download=True)

    X_train, y_train = _torchvision_to_numpy(tr)
    X_test, y_test = _torchvision_to_numpy(te)

    rng = np.random.default_rng(seed)
    if n_train is not None and n_train < len(X_train):
        sel = rng.permutation(len(X_train))[:n_train]
        X_train, y_train = X_train[sel], y_train[sel]
    if n_test is not None and n_test < len(X_test):
        sel = rng.permutation(len(X_test))[:n_test]
        X_test, y_test = X_test[sel], y_test[sel]

    # Normalise with train-set channel statistics, matching the tabular leg.
    mu = X_train.mean(axis=(0, 2, 3), keepdims=True)
    sd = X_train.std(axis=(0, 2, 3), keepdims=True)
    sd[sd == 0] = 1.0
    X_train = (X_train - mu) / sd
    X_test = (X_test - mu) / sd

    names = {
        "mnist": [str(i) for i in range(10)],
        "cifar10": ["airplane", "automobile", "bird", "cat", "deer",
                    "dog", "frog", "horse", "ship", "truck"],
    }[which]
    meta = {
        "name": {"mnist": "MNIST", "cifar10": "CIFAR-10"}[which],
        "shape": X_train.shape[1:],
        "classes": names,
        "kind": "image",
        "channel_mean": mu.ravel().tolist(),
        "channel_std": sd.ravel().tolist(),
    }
    if verbose:
        print(f"{meta['name']}: train {X_train.shape} / test {X_test.shape}, "
              f"{len(names)} classes")
    return X_train, y_train, X_test, y_test, meta


def load_mnist(n_train=None, n_test=None, seed=SEED, verbose=True):
    return _load_image_dataset("mnist", n_train, n_test, seed, verbose)


def load_cifar10(n_train=None, n_test=None, seed=SEED, verbose=True):
    return _load_image_dataset("cifar10", n_train, n_test, seed, verbose)


# --------------------------------------------------------------------------
# metrics - one implementation, so all three legs are scored identically
# --------------------------------------------------------------------------
@dataclass
class RunResult:
    """One trained model, described the way slide 29 asks for."""
    framework: str
    dataset: str
    model: str
    n_params: int
    epochs: int
    train_seconds: float
    train_loss: float
    test_accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    history: dict = field(default_factory=dict)
    confusion: list = field(default_factory=list)

    def row(self) -> dict:
        d = asdict(self)
        d.pop("history")
        d.pop("confusion")
        return d


def evaluate(y_true, y_pred, n_classes: int) -> dict:
    """Accuracy, macro precision/recall/F1 and the confusion matrix."""
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix,
    )

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f1),
        "confusion": confusion_matrix(
            y_true, y_pred, labels=list(range(n_classes))
        ).tolist(),
    }


class Timer:
    """with Timer() as t: ...   then read t.seconds"""

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0
        return False


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------
def plot_history(histories: dict, title: str = "", ax=None):
    """histories: {label: {"loss": [...], "val_acc": [...]}}"""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(1, 2, figsize=(11, 4))
    for label, h in histories.items():
        if h.get("loss"):
            ax[0].plot(range(1, len(h["loss"]) + 1), h["loss"], marker="o", label=label)
        if h.get("val_acc"):
            ax[1].plot(range(1, len(h["val_acc"]) + 1), h["val_acc"], marker="o", label=label)
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("training loss"); ax[0].set_title("Loss")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("test accuracy"); ax[1].set_title("Accuracy")
    for a in ax:
        a.grid(alpha=0.3); a.legend()
    if title:
        ax[0].figure.suptitle(title)
    return ax


def plot_confusion(cm, class_names, title: str = "", ax=None, normalize: bool = True):
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=np.float64)
    if normalize:
        denom = cm.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1
        cm = cm / denom
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() if cm.max() else 1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    thresh = cm.max() / 2 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > thresh else "black")
    ax.figure.colorbar(im, ax=ax, fraction=0.046)
    return ax


def plot_samples(X, y, class_names, n: int = 10, title: str = "", preds=None):
    """Show n images; if preds is given, wrong predictions are marked in red."""
    import matplotlib.pyplot as plt

    n = min(n, len(X))
    cols = min(n, 10)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.4 * cols, 1.7 * rows))
    axes = np.atleast_1d(axes).ravel()
    for i in range(len(axes)):
        axes[i].axis("off")
        if i >= n:
            continue
        img = X[i]
        img = img - img.min()
        img = img / (img.max() if img.max() else 1)
        img = img.transpose(1, 2, 0)
        axes[i].imshow(img.squeeze(), cmap="gray" if img.shape[2] == 1 else None)
        if preds is None:
            axes[i].set_title(class_names[y[i]], fontsize=8)
        else:
            ok = preds[i] == y[i]
            axes[i].set_title(
                f"{class_names[preds[i]]}\n(true {class_names[y[i]]})",
                fontsize=7, color="green" if ok else "red",
            )
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def results_table(results) -> pd.DataFrame:
    """Turn a list of RunResult into the comparison table slide 29 asks for."""
    df = pd.DataFrame([r.row() for r in results])
    return df.round(4)


# --------------------------------------------------------------------------
# model persistence
# --------------------------------------------------------------------------
# Every notebook trains the same architecture three ways, so the weights are
# saved three ways too. One directory per notebook, one sidecar JSON per model
# recording what it is and how it scored.
#
#   results/models/<subdir>/<name>.pt      PyTorch  state_dict
#   results/models/<subdir>/<name>.keras   Keras    full model (self-contained)
#   results/models/<subdir>/<name>.npz     scratch  one array per layer param
#   results/models/<subdir>/<name>.json    metadata for all three
#
# .keras reloads on its own. The other two only hold weights: rebuild the
# architecture first, then hand the instance to load_model(). That is the same
# contract variants.py already uses for results/variant_cache/.
MODELS = os.path.join(HERE, "results", "models")

_EXT = {"torch": ".pt", "keras": ".keras", "scratch": ".npz"}


def _framework_of(model) -> str:
    """Identify a model's framework by duck-typing, importing nothing."""
    if hasattr(model, "state_dict") and hasattr(model, "parameters"):
        return "torch"
    if hasattr(model, "count_params") and hasattr(model, "save"):
        return "keras"
    if hasattr(model, "layers") and all(hasattr(l, "params") for l in model.layers):
        return "scratch"
    raise TypeError(f"unrecognised model object: {type(model).__name__}")


def _count_params(model, framework: str) -> int:
    if framework == "torch":
        return int(sum(p.numel() for p in model.parameters()))
    if framework == "keras":
        return int(model.count_params())
    return int(model.n_params())


def model_dir(subdir: str = "") -> str:
    d = os.path.join(MODELS, subdir) if subdir else MODELS
    os.makedirs(d, exist_ok=True)
    return d


def save_model(model, name: str, subdir: str = "", **meta) -> str:
    """Persist one trained model plus a metadata sidecar. Returns the weights path.

    name    file stem, e.g. "lenet5_torch_subset"
    subdir  folder under results/models/, normally the notebook stem
    meta    anything worth recording next to the weights (dataset, accuracy, ...)
    """
    framework = _framework_of(model)
    d = model_dir(subdir)
    path = os.path.join(d, name + _EXT[framework])

    if framework == "torch":
        import torch

        torch.save(model.state_dict(), path)
    elif framework == "keras":
        model.save(path)
    else:
        arrays = {
            f"L{i}__{k}": v
            for i, layer in enumerate(model.layers)
            for k, v in layer.params.items()
        }
        np.savez_compressed(path, **arrays)

    record = {
        "name": name,
        "framework": framework,
        "class": type(model).__name__,
        "n_params": _count_params(model, framework),
        "weights_file": os.path.basename(path),
        "self_contained": framework == "keras",
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    record.update(meta)
    with open(os.path.join(d, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)

    size_kb = os.path.getsize(path) / 1024
    print(f"saved {os.path.relpath(path, HERE)}  ({record['n_params']:,} params, {size_kb:.0f} KB)")
    return path


def save_all(models: dict, subdir: str = "", **meta) -> dict:
    """save_model() over a {name: model} mapping. Skips names mapped to None."""
    return {
        name: save_model(m, name, subdir, **meta)
        for name, m in models.items()
        if m is not None
    }


def load_model(name: str, subdir: str = "", model=None, device: str = "cpu"):
    """Reload a saved model.

    Keras models come back on their own. PyTorch and scratch models only hold
    weights, so pass a freshly built instance of the same architecture as
    `model` and it is filled in place and returned.
    """
    d = os.path.join(MODELS, subdir) if subdir else MODELS
    meta_path = os.path.join(d, name + ".json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"no saved model {name!r} in {d}")
    with open(meta_path, encoding="utf-8") as f:
        record = json.load(f)

    path = os.path.join(d, record["weights_file"])
    framework = record["framework"]

    if framework == "keras":
        import keras

        return keras.models.load_model(path)

    if model is None:
        raise ValueError(
            f"{name!r} is a {framework} model and stores weights only - "
            f"build a {record['class']} first and pass it as model="
        )

    if framework == "torch":
        import torch

        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        return model

    with np.load(path) as z:
        for key in z.files:
            idx, pname = key.split("__")
            layer = model.layers[int(idx[1:])]
            want = layer.params[pname].shape
            if z[key].shape != want:
                raise ValueError(f"{key}: saved {z[key].shape} != model {want}")
            layer.params[pname] = z[key]
    return model


def list_models(subdir: str = "") -> pd.DataFrame:
    """Every saved model under results/models/, read from the sidecar JSONs."""
    root = os.path.join(MODELS, subdir) if subdir else MODELS
    rows = []
    for dirpath, _, files in os.walk(root):
        for fn in sorted(files):
            if not fn.endswith(".json"):
                continue
            with open(os.path.join(dirpath, fn), encoding="utf-8") as f:
                r = json.load(f)
            r["subdir"] = os.path.relpath(dirpath, MODELS)
            rows.append(r)
    if not rows:
        return pd.DataFrame(columns=["subdir", "name", "framework", "n_params"])
    df = pd.DataFrame(rows)
    front = [c for c in ["subdir", "name", "framework", "class", "n_params"] if c in df]
    return df[front + [c for c in df.columns if c not in front]]
