"""Loaders for the three datasets that ship inside `data/`.

`ass4_utils.py` in the repository root already covers the original three
datasets (Diabetes 130-US, MNIST, CIFAR-10) and owns everything that is *not*
dataset-specific: seeding, metrics, plotting, timing and model persistence.
This module adds only the loaders for the new material and returns arrays in
exactly the same shape contract, so every notebook keeps using `ass4_utils`
for the rest:

    X_train, y_train, X_test, y_test, meta = load_*(...)

    tabular : X (N, D)       float32, standardised with *train* statistics
    image   : X (N, C, H, W) float32, standardised with *train* channel statistics
    y       : (N,)           int64, indexing meta["classes"]

Keeping the contract identical is what lets the fairness rule from lecture 04,
slide 29 hold across all three framework legs:

    Same Dataset + Same Split + Same Architecture + Comparable Hyperparameters

Decoding 75,000 JPEGs takes minutes, so every image loader caches the decoded
uint8 array under `data/cache/`. The cache stores the *unnormalised* pixels;
normalisation is redone on every call from the training split only, so a cache
hit can never leak test statistics into training.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(DATA, "cache")

SEED = 42

# --------------------------------------------------------------------------
# dataset paths, in one place so the inventory and the loaders cannot disagree
# --------------------------------------------------------------------------
DIABETES_CSV = os.path.join(DATA, "diabetes_brfss_2015_2023.csv")
RICE_DIR = os.path.join(DATA, "Rice_Image_Dataset")
INTEL_TRAIN_DIR = os.path.join(DATA, "seg_train")
INTEL_TEST_DIR = os.path.join(DATA, "seg_test")
INTEL_PRED_DIR = os.path.join(DATA, "seg_pred")

RICE_CLASSES = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]
INTEL_CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

# BRFSS `Diabetes_012`, in the order the codebook defines
BRFSS_CLASSES = ["no diabetes", "prediabetes", "diabetes"]
BRFSS_TARGET = "Diabetes_012"

# Columns that carry a magnitude or an ordered level, so standardising them is
# meaningful. Everything else in the file is already a 0/1 indicator.
BRFSS_NUMERIC = ["BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"]


# --------------------------------------------------------------------------
# where trained weights go
# --------------------------------------------------------------------------
def use_notebook_model_dir(subdir: str = "models") -> str:
    """Send `ass4_utils` model persistence into `notebook/` instead of the root.

    `ass4_utils.save_model`, `load_model` and `list_models` all resolve paths
    from the module-level `ass4_utils.MODELS`, which points at
    `<repo>/results/models/` - where the six original notebooks keep their
    committed weights. The notebooks in this folder keep theirs beside
    themselves, so new runs can never overwrite or mix with the originals.

    Call once per notebook, in the setup cell, before any `U.save_model`.
    """
    import ass4_utils as U

    U.MODELS = os.path.join(HERE, subdir)
    os.makedirs(U.MODELS, exist_ok=True)
    return U.MODELS


def notebook_results_dir(subdir: str = "results") -> str:
    """`notebook/results/`, created on demand - metrics JSON lives here."""
    d = os.path.join(HERE, subdir)
    os.makedirs(d, exist_ok=True)
    return d


# --------------------------------------------------------------------------
# inventory - what is actually on disk, checked rather than assumed
# --------------------------------------------------------------------------
def _count_images(folder: str) -> int:
    if not os.path.isdir(folder):
        return 0
    return sum(
        1
        for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )


def inventory(verbose: bool = True) -> pd.DataFrame:
    """Report what each dataset actually contributes, straight from disk.

    Nothing in this module trusts the accompanying `*_info.md` files; this walks
    the directories so a missing or partial download is visible immediately
    instead of failing halfway through a training run.
    """
    rows = []

    rows.append(
        {
            "dataset": "BRFSS diabetes",
            "part": "diabetes_brfss_2015_2023.csv",
            "present": os.path.exists(DIABETES_CSV),
            "items": (
                sum(1 for _ in open(DIABETES_CSV, encoding="utf-8")) - 1
                if os.path.exists(DIABETES_CSV)
                else 0
            ),
            "labelled": True,
            "usable_for_training": os.path.exists(DIABETES_CSV),
        }
    )

    for cls in RICE_CLASSES:
        n = _count_images(os.path.join(RICE_DIR, cls))
        rows.append(
            {
                "dataset": "Rice",
                "part": f"Rice_Image_Dataset/{cls}",
                "present": n > 0,
                "items": n,
                "labelled": True,
                "usable_for_training": n > 0,
            }
        )

    for label, base in [("seg_train", INTEL_TRAIN_DIR), ("seg_test", INTEL_TEST_DIR)]:
        # Kaggle ships this dataset double-nested: seg_train/seg_train/<class>/
        inner = os.path.join(base, label)
        root = inner if os.path.isdir(inner) else base
        n = sum(_count_images(os.path.join(root, c)) for c in INTEL_CLASSES)
        rows.append(
            {
                "dataset": "Intel scenes",
                "part": label,
                "present": n > 0,
                "items": n,
                "labelled": True,
                "usable_for_training": n > 0,
            }
        )

    inner = os.path.join(INTEL_PRED_DIR, "seg_pred")
    root = inner if os.path.isdir(inner) else INTEL_PRED_DIR
    n = _count_images(root)
    rows.append(
        {
            "dataset": "Intel scenes",
            "part": "seg_pred",
            "present": n > 0,
            "items": n,
            "labelled": False,          # no class folders -> no ground truth
            "usable_for_training": False,
        }
    )

    df = pd.DataFrame(rows)
    if verbose:
        for _, r in df.iterrows():
            flag = "OK  " if r["usable_for_training"] else ("NOLBL" if r["present"] else "MISS")
            print(f"  [{flag:5s}] {r['dataset']:15s} {r['part']:28s} {r['items']:>7,} items")
    return df


# --------------------------------------------------------------------------
# dataset A - CDC BRFSS diabetes health indicators (tabular, 3-class)
# --------------------------------------------------------------------------
def load_diabetes_brfss(
    n_subset: int | None = None,
    test_size: float = 0.2,
    seed: int = SEED,
    binary: bool = False,
    split_by_year: bool = False,
    drop_year: bool = False,
    verbose: bool = True,
):
    """Load the merged 2015+2023 BRFSS file as standardised float32 features.

    n_subset       keep only this many rows (after shuffling) - for quick runs
    binary         merge prediabetes+diabetes into one positive class
    split_by_year  train on 2015 and test on 2023 instead of a random split,
                   which measures temporal stability rather than in-sample fit
    drop_year      remove `year` from the feature matrix

    Returns (X_train, y_train, X_test, y_test, meta).
    """
    if not os.path.exists(DIABETES_CSV):
        raise FileNotFoundError(
            f"missing {DIABETES_CSV}\n"
            "See data/diabetes_info.md for how the file is rebuilt."
        )

    df = pd.read_csv(DIABETES_CSV)
    raw_shape = df.shape

    # The 67,731 duplicated rows are genuine distinct respondents who happen to
    # share all 19 coarse answers - data/diabetes_info.md is explicit that they
    # must not be dropped, so nothing here deduplicates.
    y = df[BRFSS_TARGET].to_numpy().astype(np.int64)
    classes = list(BRFSS_CLASSES)
    if binary:
        y = (y > 0).astype(np.int64)
        classes = ["no diabetes", "prediabetes or diabetes"]

    X = df.drop(columns=[BRFSS_TARGET])
    year = X["year"].to_numpy()
    if drop_year or split_by_year:
        # When the split is by year, `year` is constant within each side and
        # would be a free giveaway rather than a feature.
        X = X.drop(columns=["year"])

    feature_names = list(X.columns)
    Xa = X.to_numpy().astype(np.float32)

    if split_by_year:
        tr, te = year == 2015, year == 2023
        X_train, y_train = Xa[tr], y[tr]
        X_test, y_test = Xa[te], y[te]
        split_desc = "train 2015 / test 2023"
    else:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(Xa))
        if n_subset is not None and n_subset < len(idx):
            idx = idx[:n_subset]
        Xa, y = Xa[idx], y[idx]

        n_test = int(len(Xa) * test_size)
        X_test, y_test = Xa[:n_test], y[:n_test]
        X_train, y_train = Xa[n_test:], y[n_test:]
        split_desc = f"random {1 - test_size:.0%}/{test_size:.0%}, seed {seed}"

    # Standardise on train statistics only; test statistics would leak.
    #
    # The accumulation is float64 on purpose. `year` holds values around 2019,
    # so a float32 column sum over ~437k rows reaches ~8.8e8, where the float32
    # step size is already 64 - reducing along axis 0 does not get NumPy's
    # pairwise summation, and the mean came out ~2.1 off, leaving the column
    # standardised to mean 0.55 / std 0.84 instead of 0 / 1.
    mu = X_train.mean(axis=0, dtype=np.float64, keepdims=True)
    sd = X_train.std(axis=0, dtype=np.float64, keepdims=True)
    sd[sd == 0] = 1.0
    X_train = ((X_train - mu) / sd).astype(np.float32)
    X_test = ((X_test - mu) / sd).astype(np.float32)

    meta = {
        "name": "BRFSS diabetes 2015+2023",
        "raw_shape": raw_shape,
        "n_features": X_train.shape[1],
        "classes": classes,
        "feature_names": feature_names,
        "numeric_features": [c for c in BRFSS_NUMERIC if c in feature_names],
        "kind": "tabular",
        "split": split_desc,
    }
    if verbose:
        print(
            f"{meta['name']}: raw {raw_shape} -> train {X_train.shape} / "
            f"test {X_test.shape}, {meta['n_features']} features, "
            f"{len(classes)} classes ({split_desc})"
        )
    return X_train, y_train, X_test, y_test, meta


# --------------------------------------------------------------------------
# image helpers, shared by the two image datasets
# --------------------------------------------------------------------------
def _read_image(path: str, img_size: int) -> np.ndarray:
    """One JPEG -> (H, W, 3) uint8, forced to RGB and to a single size."""
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.size != (img_size, img_size):
            im = im.resize((img_size, img_size), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def _load_class_folders(root: str, classes: list[str], img_size: int, verbose: bool):
    """Read `root/<class>/*.jpg` into (N,H,W,3) uint8 and int64 labels."""
    xs, ys = [], []
    for label, cls in enumerate(classes):
        folder = os.path.join(root, cls)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"missing class folder {folder}")
        files = sorted(
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        t0 = time.perf_counter()
        for f in files:
            xs.append(_read_image(os.path.join(folder, f), img_size))
            ys.append(label)
        if verbose:
            print(
                f"    {cls:<12} {len(files):>6,} images "
                f"({time.perf_counter() - t0:.1f}s)"
            )
    return np.stack(xs), np.asarray(ys, dtype=np.int64)


def _rel(path: str) -> str:
    """Path relative to the repo root, or absolute if it lives elsewhere.

    On Windows `os.path.relpath` raises when the two paths sit on different
    drives, which is exactly the case if CACHE is ever pointed off the repo.
    """
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:
        return path


def _cached_folder_dataset(cache_name, build, verbose):
    """Decode once, reuse forever. The cache holds raw uint8 pixels only."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, cache_name)
    if os.path.exists(path):
        with np.load(path) as z:
            if verbose:
                print(f"  cache hit  {_rel(path)}")
            return z["X"], z["y"]

    if verbose:
        print(f"  decoding images (first run; cached to {_rel(path)})")
    X, y = build()
    np.savez_compressed(path, X=X, y=y)
    return X, y


def _finalise_images(X_train_u8, y_train, X_test_u8, y_test, name, classes, verbose, extra=None):
    """uint8 (N,H,W,C) -> normalised float32 (N,C,H,W), train statistics only.

    Mirrors `ass4_utils._load_image_dataset` so the new image datasets behave
    exactly like MNIST and CIFAR-10 did.
    """
    X_train = X_train_u8.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    X_test = X_test_u8.transpose(0, 3, 1, 2).astype(np.float32) / 255.0

    # float64 accumulation, for the same reason as the tabular loader: a
    # channel mean over 60k x 32 x 32 values is a 60-million-term sum.
    mu = X_train.mean(axis=(0, 2, 3), dtype=np.float64, keepdims=True)
    sd = X_train.std(axis=(0, 2, 3), dtype=np.float64, keepdims=True)
    sd[sd == 0] = 1.0
    X_train = ((X_train - mu) / sd).astype(np.float32)
    X_test = ((X_test - mu) / sd).astype(np.float32)

    meta = {
        "name": name,
        "shape": X_train.shape[1:],
        "classes": classes,
        "kind": "image",
        "channel_mean": mu.ravel().tolist(),
        "channel_std": sd.ravel().tolist(),
    }
    if extra:
        meta.update(extra)
    if verbose:
        print(
            f"{name}: train {X_train.shape} / test {X_test.shape}, "
            f"{len(classes)} classes"
        )
    return X_train, y_train, X_test, y_test, meta


def _subsample(X, y, n, rng):
    if n is None or n >= len(X):
        return X, y
    sel = rng.permutation(len(X))[:n]
    return X[sel], y[sel]


def _size_tag(n_train: int, subsetted: bool) -> str:
    """How a run is labelled in every results table: 'full' or 'Nk subset'."""
    if not subsetted:
        return "full"
    return f"{n_train // 1000}k subset" if n_train >= 1000 else f"{n_train} subset"


# --------------------------------------------------------------------------
# dataset B - Rice Image Dataset (75,000 images, 5 balanced classes)
# --------------------------------------------------------------------------
def load_rice(
    img_size: int = 32,
    n_train: int | None = None,
    n_test: int | None = None,
    test_size: float = 0.2,
    seed: int = SEED,
    verbose: bool = True,
):
    """Load the rice-grain images, stratified split (the dataset ships unsplit).

    img_size  square edge the 250x250 originals are resized to
    n_train   keep this many training images (None = all)
    n_test    keep this many test images (None = all)
    """
    if not os.path.isdir(RICE_DIR):
        raise FileNotFoundError(f"missing {RICE_DIR}")

    X_u8, y = _cached_folder_dataset(
        f"rice_{img_size}.npz",
        lambda: _load_class_folders(RICE_DIR, RICE_CLASSES, img_size, verbose),
        verbose,
    )

    # Stratified split: take the same fraction out of every class, so both
    # sides stay perfectly balanced the way the raw dataset is.
    rng = np.random.default_rng(seed)
    tr_idx, te_idx = [], []
    for label in range(len(RICE_CLASSES)):
        idx = np.flatnonzero(y == label)
        idx = idx[rng.permutation(len(idx))]
        cut = int(len(idx) * test_size)
        te_idx.append(idx[:cut])
        tr_idx.append(idx[cut:])
    tr_idx = rng.permutation(np.concatenate(tr_idx))
    te_idx = rng.permutation(np.concatenate(te_idx))

    X_train_u8, y_train = _subsample(X_u8[tr_idx], y[tr_idx], n_train, rng)
    X_test_u8, y_test = _subsample(X_u8[te_idx], y[te_idx], n_test, rng)

    tag = _size_tag(len(X_train_u8), n_train is not None)
    return _finalise_images(
        X_train_u8, y_train, X_test_u8, y_test,
        f"Rice ({tag})", RICE_CLASSES, verbose,
        extra={"img_size": img_size, "source_resolution": "250x250"},
    )


# --------------------------------------------------------------------------
# dataset C - Intel Image Classification (scenes, 6 classes)
# --------------------------------------------------------------------------
def _intel_root(base: str, label: str) -> str:
    """Kaggle nests this dataset as seg_train/seg_train/<class>/; accept both."""
    inner = os.path.join(base, label)
    return inner if os.path.isdir(inner) else base


def require_intel_labels() -> None:
    """Fail loudly, and with instructions, when the labelled splits are absent.

    `data/` as delivered contains only `seg_pred`, which has no class folders
    and therefore no ground truth. Training or scoring a classifier on it is
    impossible, so this is checked before anything else runs.
    """
    missing = []
    for label, base in [("seg_train", INTEL_TRAIN_DIR), ("seg_test", INTEL_TEST_DIR)]:
        root = _intel_root(base, label)
        if sum(_count_images(os.path.join(root, c)) for c in INTEL_CLASSES) == 0:
            missing.append(label)
    if missing:
        raise FileNotFoundError(
            "Intel Image Classification is incomplete in data/.\n"
            f"  missing labelled split(s): {', '.join(missing)}\n"
            f"  present: seg_pred ({_count_images(_intel_root(INTEL_PRED_DIR, 'seg_pred')):,} "
            "images, no labels)\n\n"
            "seg_pred is the competition's prediction set: flat files, no class\n"
            "folders, no ground truth. It can be used for an inference demo but\n"
            "not for training or for measuring accuracy.\n\n"
            "To complete the dataset:\n"
            "  pip install kaggle\n"
            "  kaggle datasets download -d puneet6060/intel-image-classification \\\n"
            "      -p data/ --unzip\n"
            "which restores data/seg_train/ and data/seg_test/."
        )


def load_intel_scene(
    img_size: int = 64,
    n_train: int | None = None,
    n_test: int | None = None,
    seed: int = SEED,
    verbose: bool = True,
):
    """Load the Intel scene images using the publisher's own train/test split."""
    require_intel_labels()

    X_tr_u8, y_tr = _cached_folder_dataset(
        f"intel_train_{img_size}.npz",
        lambda: _load_class_folders(
            _intel_root(INTEL_TRAIN_DIR, "seg_train"), INTEL_CLASSES, img_size, verbose
        ),
        verbose,
    )
    X_te_u8, y_te = _cached_folder_dataset(
        f"intel_test_{img_size}.npz",
        lambda: _load_class_folders(
            _intel_root(INTEL_TEST_DIR, "seg_test"), INTEL_CLASSES, img_size, verbose
        ),
        verbose,
    )

    rng = np.random.default_rng(seed)
    # The folder walk returns images grouped by class; shuffle so that a
    # subset is not just the first few classes.
    p = rng.permutation(len(X_tr_u8))
    X_tr_u8, y_tr = X_tr_u8[p], y_tr[p]
    p = rng.permutation(len(X_te_u8))
    X_te_u8, y_te = X_te_u8[p], y_te[p]

    X_tr_u8, y_tr = _subsample(X_tr_u8, y_tr, n_train, rng)
    X_te_u8, y_te = _subsample(X_te_u8, y_te, n_test, rng)

    tag = _size_tag(len(X_tr_u8), n_train is not None)
    return _finalise_images(
        X_tr_u8, y_tr, X_te_u8, y_te,
        f"Intel scenes ({tag})", INTEL_CLASSES, verbose,
        extra={"img_size": img_size, "source_resolution": "~150x150"},
    )


def load_intel_unlabeled(img_size: int = 64, n: int | None = 64, seed: int = SEED,
                         verbose: bool = True):
    """Load images from `seg_pred` for an inference demo. Returns X only.

    There are no labels here by construction, so this deliberately returns a
    single array - there is nothing to score against.
    """
    root = _intel_root(INTEL_PRED_DIR, "seg_pred")
    files = sorted(
        f for f in os.listdir(root)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not files:
        raise FileNotFoundError(f"no images under {root}")

    rng = np.random.default_rng(seed)
    if n is not None and n < len(files):
        files = [files[i] for i in sorted(rng.permutation(len(files))[:n])]

    X = np.stack([_read_image(os.path.join(root, f), img_size) for f in files])
    X = X.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    if verbose:
        print(f"seg_pred: {X.shape} (unlabelled - inference demo only)")
    return X, files


# --------------------------------------------------------------------------
# class imbalance - the technique theory_notes.md section 4.2 asks for
# --------------------------------------------------------------------------
def class_weights(y, n_classes: int) -> np.ndarray:
    """Balanced class weights, w_c = N / (K * n_c), as sklearn defines them.

    theory_notes.md section 4.2 shows why this matters on a skewed target: on
    the original diabetes data accuracy read ~61% while macro-F1 was ~0.35,
    because the model was answering with the majority class. Weighting the loss
    makes a mistake on a rare class cost proportionally more.
    """
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    return (len(y) / (n_classes * counts)).astype(np.float32)


if __name__ == "__main__":
    print("data/ inventory")
    print("-" * 70)
    inventory()
