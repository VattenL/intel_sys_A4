# `notebook/` — Assignment 4 on the new datasets

Six notebooks that redo the assignment on the datasets in `data/`: one tabular problem and two image
problems, each implemented three ways — **NumPy from scratch**, **TensorFlow/Keras** and **PyTorch** —
followed by the CNN-improvement experiment from `theory_notes.md` §3.

These are **written but never executed**. Every cell is empty of output on purpose: they are meant to be
run once, in order, on the machine that has the GPU. Nothing here depends on the six original notebooks in
the repository root, and nothing here overwrites their results.

---

## 1. What you need before starting

### 1.1 The repository

You have it. Two files in the root are imported by these notebooks and must stay where they are:

| File | Provides |
|---|---|
| `ass4_utils.py` | seeding, metrics, plots, timing, model save/load |
| `scratch_nn.py` | the from-scratch framework: `Conv2D`, `MaxPool2D`, `Dense`, `ReLU`, `Dropout`, `Adam`, `fit`, `gradient_check` |
| `variants.py` | the M1–M4 architecture ladder (used by notebook 04) |

### 1.2 The datasets

`data/` is **git-ignored**, so it does not travel with a `git clone`. Copy the whole folder across
manually. The notebooks expect exactly this layout:

```
data/
├── diabetes_brfss_2015_2023.csv        546,166 rows x 20 cols     ~26 MB
├── diabetes_info.md
├── Rice_Image_Dataset/
│   ├── Arborio/       15,000 .jpg  (250x250 RGB)
│   ├── Basmati/       15,000 .jpg
│   ├── Ipsala/        15,000 .jpg
│   ├── Jasmine/       15,000 .jpg
│   └── Karacadag/     15,000 .jpg        75,000 images total       ~267 MB
├── rice_image_dataset_info.md
├── seg_pred/           7,301 .jpg  (150x150 RGB, NO labels)        ~111 MB
└── intel_image_classification_info.md
```

Run `00_inventory.ipynb` first — it checks all of this against the disk and tells you what is missing.

> ### ⚠ The Intel scene dataset is incomplete as delivered
>
> `data/` contains only `seg_pred/`, which is the original competition's **prediction** set: 7,301 loose
> files with no class sub-folders. In this dataset the folder name *is* the label, so with no class folders
> there is no ground truth — you cannot train a classifier on it and you cannot measure an accuracy.
>
> The labelled splits `seg_train/` (~14,000 images) and `seg_test/` (~3,000) are absent. To restore them:
>
> ```bash
> pip install kaggle          # needs an API token at ~/.kaggle/kaggle.json
> kaggle datasets download -d puneet6060/intel-image-classification -p data/ --unzip
> ```
>
> The loader accepts both the flat layout and Kaggle's double-nested `seg_train/seg_train/<class>/`, so no
> path editing is needed. **Until then, notebook 03 stops at its first cell with these instructions**, and
> notebooks 00 and 05 simply report it as missing. Everything else runs normally.

### 1.3 Python environment

**Use Python 3.11 or 3.12.** TensorFlow publishes no wheels for 3.13 or 3.14, so a newer interpreter cannot
run the Keras leg at all.

```bash
conda create -n ass4 python=3.11 -y
conda activate ass4

# PyTorch with CUDA - pick the index URL matching the machine's driver from
# https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install tensorflow numpy pandas scikit-learn matplotlib pillow jupyter ipykernel

# Register the kernel the .ipynb files reference by name
python -m ipykernel install --user --name ass4 --display-name "Python 3.11 (ass4)"
```

The notebooks are saved against a kernel named **`ass4`**. If you use a different environment name, either
register it under `ass4` as above or pick your kernel from *Kernel → Change kernel* after opening each
notebook.

Verify before running anything:

```bash
python -c "import torch, tensorflow as tf, keras, numpy, pandas, sklearn, matplotlib, PIL; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); \
print('tf', tf.__version__, 'gpus', tf.config.list_physical_devices('GPU'))"
```

> **TensorFlow on Windows runs on CPU.** There has been no Windows-native GPU build since TF 2.10. On
> Windows the Keras leg is therefore CPU-bound while the PyTorch leg uses the GPU — parameter counts and
> accuracies stay comparable across those rows, but **wall-clock times do not**. Every notebook repeats
> this warning next to its timing table. On Linux both legs can use the GPU and the times become
> comparable.

---

## 2. Run order

Start Jupyter **from the `notebook/` folder** (the notebooks add `.` and `..` to `sys.path`, which assumes
that working directory):

```bash
cd notebook
jupyter lab
```

| # | Notebook | What it does | Needs |
|---|---|---|---|
| 00 | `00_inventory.ipynb` | Audits `data/` against the `*_info.md` claims. **Run this first.** | — |
| 01 | `01_diabetes_brfss.ipynb` | BRFSS diabetes, MLP × 3 frameworks, plus class weighting | — |
| 02 | `02_rice_cnn.ipynb` | Rice images, CNN × 3 frameworks, subset + full 60k | — |
| 03 | `03_intel_scene_cnn.ipynb` | Intel scenes, CNN × 3 frameworks + `seg_pred` demo | labelled splits (§1.2) |
| 04 | `04_improved_cnn.ipynb` | M1→M4 ladder + data augmentation, on rice | 02's cache helps |
| 05 | `05_compare.ipynb` | Cross-dataset comparison. **Trains nothing** — reads 01–04's JSON | 01, 02, 04 |

03 is optional; 05 detects its absence and skips that section.

### Roughly how long

Measured on the machine these were developed against (RTX 3060 laptop GPU, TF on CPU). Your numbers will
differ, but the *ratios* will not.

| Step | Cost | Note |
|---|---|---|
| First image load (rice) | **3–10 min** | decodes 75,000 JPEGs, then caches — once only |
| 01 diabetes, all 3 legs | ~2–5 min | full 437k rows, including the NumPy leg |
| 02 rice, scratch leg | ~1–2 min | 5,000 images, 5 epochs |
| 02 rice, full 60k legs | ~5–15 min | Keras on CPU is the slow one |
| 03 scenes, scratch leg | **~5–15 min** | 64×64 inputs make `im2col` four times heavier |
| 04 ladder, 4 models | ~10–20 min | cached per model; interrupting is safe |
| 05 compare | seconds | reads JSON only |

**Memory.** The rice images at 32×32 are ~0.9 GB as float32 for the full 60,000, and notebook 02 holds the
train and test arrays plus a channels-last copy for Keras at the same time. Budget ~8 GB of free RAM. If
you are tight, lower `SUBSET_TRAIN` in notebook 02 or pass a smaller `n_train=` to `D.load_rice`.

---

## 3. Where the outputs go

**Everything these notebooks produce stays inside `notebook/`.** The six original notebooks write to the
repository-root `results/`, and keeping the two separate means re-running anything here can never
overwrite or mix with the committed original results.

```
notebook/
├── models/                       <- trained weights   (git-ignored)
│   ├── n01_diabetes_brfss/         mlp_scratch.npz, mlp_keras.keras, mlp_torch.pt, + .json sidecars
│   ├── n02_rice_cnn/               5 models: 3 subset legs + 2 full-data legs
│   ├── n03_intel_scene_cnn/        same shape, when the labels exist
│   ├── n04_improved_cnn/           the augmentation pair
│   └── variant_cache_rice/         M1..M4 weights + per-model JSON (checkpointed)
└── results/                      <- metrics           (tracked in git)
    ├── n01_diabetes_brfss.json
    ├── n02_rice_cnn.json
    ├── n03_intel_scene_cnn.json
    ├── n04_improved_cnn.json
    └── all_runs_new.csv            written by notebook 05
```

The redirect is explicit, in each notebook's setup cell:

```python
RESULTS = D.notebook_results_dir()     # notebook/results/
MODELS  = D.use_notebook_model_dir()   # notebook/models/  (repoints ass4_utils.MODELS)
```

`use_notebook_model_dir()` reassigns `ass4_utils.MODELS`, which is the single global that
`save_model`, `load_model` and `list_models` all resolve paths from. Call it **before** any `U.save_model`.

### What is and is not committed

`notebook/.gitignore` excludes `models/`. Weights run to hundreds of megabytes and every one of them is
reproducible by re-running its notebook; the numbers the report quotes live in `results/*.json`, which is
tracked.

This is the opposite of the root-level convention, where `results/models/` **is** committed so that
cloning the repo is enough to load a trained network without retraining. If you want that behaviour here
too, delete the `models/` line from `notebook/.gitignore`.

Also ignored: `data/cache/*.npz`, the decoded-image caches — already covered by the root `.gitignore`,
which excludes all of `data/`.

### Reloading a saved model

```python
import ass4_newdata as D, ass4_utils as U
D.use_notebook_model_dir()

U.list_models()                                  # everything saved here
U.list_models("n02_rice_cnn")                    # one notebook

keras_model = U.load_model("rice_cnn_keras_full", "n02_rice_cnn")   # standalone
```

`.keras` files reload on their own. `.pt` and `.npz` hold **weights only**, so rebuild the architecture
first and hand the instance over:

```python
torch_model = U.load_model("rice_cnn_torch_full", "n02_rice_cnn",
                           model=RiceCNN(3, 5))   # class defined in notebook 02
```

That is deliberate: a `state_dict` is the portable half of a PyTorch model, whereas pickling the class
would tie the file to the notebook that defined it.

---

## 4. `ass4_newdata.py`

The one new module. It adds loaders for the three new datasets and nothing else — metrics, plots and
persistence all still come from `ass4_utils`. Every loader returns the same five-tuple the original
loaders did, which is what lets the fairness rule hold across the three framework legs:

```python
X_train, y_train, X_test, y_test, meta = load_*(...)

# tabular : X (N, D)       float32, standardised with TRAIN statistics
# image   : X (N, C, H, W) float32, standardised with TRAIN channel statistics
# y       : (N,)           int64, indexing meta["classes"]
```

| Function | Returns |
|---|---|
| `inventory()` | DataFrame of what is on disk, per dataset part |
| `load_diabetes_brfss(...)` | 546,166 × 19, 3 classes. Options: `binary=`, `split_by_year=`, `drop_year=`, `n_subset=` |
| `load_rice(img_size=32, ...)` | 75,000 images, 5 classes, stratified 80/20 split |
| `load_intel_scene(img_size=64, ...)` | 6 classes, the publisher's own train/test split |
| `load_intel_unlabeled(...)` | `seg_pred` images, **X only** — there are no labels to return |
| `require_intel_labels()` | raises with download instructions if the labelled splits are missing |
| `class_weights(y, k)` | balanced weights $w_c = N / (K n_c)$, for the imbalance fix in notebook 01 |
| `use_notebook_model_dir()` | repoints `ass4_utils.MODELS` at `notebook/models/` |
| `notebook_results_dir()` | creates and returns `notebook/results/` |

Run it directly for a quick audit without opening Jupyter:

```bash
cd notebook && python ass4_newdata.py
```

Three implementation details that are easy to get wrong and are handled here:

- **Caching never leaks.** Decoded images are cached to `data/cache/<name>_<size>.npz` as **unnormalised
  uint8**. Normalisation is recomputed from the training split on every call, so a cache hit cannot put
  test statistics into training.
- **Statistics accumulate in float64.** Reducing a float32 array along `axis=0` does not get NumPy's
  pairwise summation. On the BRFSS `year` column (values ~2019, ~437k rows, column sum ~8.8e8, where the
  float32 step is already 64) the mean came out ~2.1 off, leaving that column standardised to mean 0.55 /
  std 0.84 instead of 0 / 1. Notebook 01 asserts the fix held.
- **The rice split is stratified.** That dataset ships unsplit and perfectly balanced; an ordinary random
  split would not keep it that way.

---

## 5. Things worth knowing before you read the results

**The gradient check in the CNN notebooks uses `eps=1e-5`, not the `1e-3` default.** `MaxPool2D` routes the
gradient to the largest element of each window, so the loss is only piecewise differentiable: where two
elements tie, an epsilon-sized nudge changes which one wins and the numerical derivative measures a
different branch from the analytic one. Rice grains sit on a uniform black background, which makes the
convolution output constant over large regions — **about 65 % of pooling windows contain a tie**. At
`eps=1e-3` the check reports ~4e-1 and looks like a failure; at `eps=1e-5` the identical code reports
~2e-8. The MLP in notebook 01 has no pooling layer and passes at the default step.

**Accuracy means different things on the two datasets.** The rice classes are exactly balanced at 20 %
each, so accuracy is trustworthy. BRFSS is split roughly **84 / 2 / 14**, so always answering "no diabetes"
scores ~84 % while never identifying a single diabetic. Every table reports **macro F1** alongside, and
notebook 01 §7 applies class weighting and measures what it trades.

**Notebook 04 trains on a deliberately small slice.** A plain CNN nearly saturates the rice data, and an
architecture comparison run at the ceiling measures nothing. The M1→M4 ladder therefore uses 4,000 images —
the regime where normalisation and regularisation are supposed to matter.

**These are short runs by design.** Five epochs on the dataset notebooks, ten on the ladder, one seed, no
augmentation outside §4 of notebook 04, no learning-rate schedule, no tuning. The goal is a controlled
comparison in which exactly one thing changes at a time, not a competitive score.

---

## 6. If something goes wrong

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: ass4_utils` | Jupyter was started somewhere other than `notebook/`. `cd notebook` first, or fix the `sys.path` lines in the setup cell. |
| `ModuleNotFoundError: tensorflow` | Python 3.13/3.14 has no TF wheel. Rebuild the environment on 3.11 or 3.12 (§1.3). |
| `FileNotFoundError: ...diabetes_brfss_2015_2023.csv` | `data/` was not copied across — it is git-ignored (§1.2). |
| Intel notebook raises at the first cell | Expected. The labelled splits are missing; the error text has the `kaggle` command. |
| Kernel dies during notebook 02 | Out of memory on the full 60k run. Lower `n_train=` in the `D.load_rice` call. |
| First run of 02 hangs for minutes | It is decoding 75,000 JPEGs. Once only — watch for `cache hit` on the next run. |
| `torch.cuda.is_available()` is `False` | CPU-only PyTorch wheel installed. Reinstall with the CUDA index URL (§1.3). Everything still runs, just slower. |
| M1–M4 report suspiciously CIFAR-like numbers | `variants.CACHE_DIR` was not redirected. Notebook 04 does this in its setup cell; confirm it printed `variant_cache_rice`. |
| Want a clean retrain of the ladder | Delete `notebook/models/variant_cache_rice/`. |

---

## 7. Relationship to the rest of the repository

| | Original six notebooks | These six |
|---|---|---|
| Location | repository root | `notebook/` |
| Datasets | Diabetes 130-US, MNIST, CIFAR-10 | BRFSS diabetes, Rice, Intel scenes |
| Loaders | `ass4_utils.load_*` | `ass4_newdata.load_*` |
| Weights | `results/models/` (committed) | `notebook/models/` (ignored) |
| Metrics | `results/*.json`, `results/all_runs.csv` | `notebook/results/*.json`, `all_runs_new.csv` |

They share `ass4_utils.py`, `scratch_nn.py` and `variants.py`, and neither set writes into the other's
output directories. `05_compare.ipynb` reads the root `results/all_runs.csv` if it is present, to put the
framework-agreement result on six datasets rather than three — and skips that section cleanly if it is not.

Background reading, both in the repository root: `theory_notes.md` for the deep-learning and CNN theory
these notebooks apply, and `README.md` for the original assignment.
