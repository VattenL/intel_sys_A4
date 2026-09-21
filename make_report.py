"""Build README.md from the JSON each notebook wrote, then export every notebook to PDF.

Usage:
    python make_report.py            # README + PDFs
    python make_report.py --no-pdf   # README only
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

NOTEBOOKS = [
    ("01_diabetes130", "Diabetes 130-US hospitals (tabular, MLP)"),
    ("02_mnist", "MNIST (image, CNN)"),
    ("03_cifar10", "CIFAR-10 (image, CNN)"),
    ("04_compare", "Comparison + improved CNN models"),
    ("06_mnist_lenet", "MNIST (image, LeNet-5)"),
    ("07_cifar10_lenet", "CIFAR-10 (image, LeNet-5)"),
]


def _load(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _rows(payload, keys=("results", "full_results")):
    out = []
    for k in keys:
        out.extend(payload.get(k, []) if payload else [])
    return out


def _table(rows, cols, headers):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                v = f"{v:.4f}" if c != "train_seconds" else f"{v:.1f}"
            elif isinstance(v, int):
                v = f"{v:,}"
            cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_readme() -> str:
    d1, d2, d3 = _load("01_diabetes130.json"), _load("02_mnist.json"), _load("03_cifar10.json")
    d4 = _load("04_variants.json")
    d6, d7 = _load("06_mnist_lenet.json"), _load("07_cifar10_lenet.json")

    parts = []
    parts.append("""# Assignment 4 — Comparing CNN Implementations

One learning problem expressed three ways — **NumPy from scratch**, **TensorFlow/Keras**, and **PyTorch** —
across three datasets, followed by an architecture-evolution experiment.

## Deliverables

| File | Contents |
|---|---|
| `01_diabetes130.ipynb` | Diabetes 130-US hospitals (tabular, 179 features, 3 classes) — MLP in all three frameworks |
| `02_mnist.ipynb` | MNIST (1x28x28) — CNN in all three frameworks |
| `03_cifar10.ipynb` | CIFAR-10 (3x32x32) — CNN in all three frameworks |
| `04_compare.ipynb` | Cross-dataset comparison, slide-28 component table, and the M1..M4 improved-model experiment |
| `06_mnist_lenet.ipynb` | MNIST again with **LeNet-5** (LeCun et al., 1998), all three frameworks — parallel run, `02` untouched |
| `07_cifar10_lenet.ipynb` | CIFAR-10 again with the same LeNet-5, all three frameworks — `03` untouched |
| `ass4_utils.py` | Shared data loading, metrics, timing and plotting — guarantees all three legs see identical splits |
| `scratch_nn.py` | The from-scratch framework: Conv2D/MaxPool2D/Dense/ReLU/Dropout with hand-derived gradients, Adam, and a finite-difference gradient checker |
| `results/` | Per-notebook JSON plus `all_runs.csv` |
| `results/models/` | Trained weights, one folder per notebook — committed, so the networks ship with the code |
| `results/variant_cache/` | Trained weights for M1..M4 — gitignored, rebuilt by `python prefill_variants.py` |

Run the notebooks in order; `04_compare.ipynb` reads the JSON the first three write.

## Environment

- Python 3.11 (`C:\\Users\\ADMIN\\AppData\\Local\\Programs\\Python\\Python311\\python.exe`), Jupyter kernel **`ass4`**
- PyTorch with CUDA on an RTX 3060 Laptop GPU
- TensorFlow 2.21 / Keras 3.15 — **CPU only**, since Windows-native TensorFlow has shipped no GPU build since 2.10

That last point matters when reading the tables: Keras and PyTorch timings measure different hardware.
Parameter counts and accuracies are directly comparable; wall-clock times are not.

## The fairness rule

Every comparison follows the rule from lecture 04, slide 29:

> Same Dataset + Same Split + Same Architecture + Comparable Hyperparameters

All three legs pull their arrays from one loader with one seed, so they train on byte-identical data.
""")

    # ---- results ----
    parts.append("\n## Results\n")
    for payload, title in ((d1, "Diabetes 130-US hospitals — MLP"),
                           (d2, "MNIST — CNN"),
                           (d3, "CIFAR-10 — CNN")):
        if not payload:
            continue
        rows = _rows(payload)
        parts.append(f"### {title}\n")
        parts.append(_table(
            rows,
            ["framework", "dataset", "n_params", "epochs", "train_seconds",
             "test_accuracy", "f1_macro"],
            ["Framework", "Dataset", "Params", "Epochs", "Train (s)", "Accuracy", "Macro F1"],
        ))
        parts.append("")

    if d4:
        parts.append("### Improved CNN models (tutorial section 45, full CIFAR-10)\n")
        parts.append(_table(
            d4["variants"],
            ["model", "n_params", "epochs", "train_seconds", "test_accuracy", "f1_macro"],
            ["Model", "Params", "Epochs", "Train (s)", "Accuracy", "Macro F1"],
        ))
        parts.append("""
Each model is the previous one plus exactly one mechanism:

- **M1** — `Conv + ReLU + Pool` baseline
- **M2** — `+ BatchNorm`, addressing activation drift between layers and batches
- **M3** — `+ Residual` (`Y = F(X) + X`), giving gradients and information a direct path through the stack
- **M4** — `+ SE channel attention`, learning which channels matter instead of treating them as equally useful

**Adding a mechanism did not always help.** BatchNorm was the largest single win, for 224 extra parameters.
The residual connection added a smaller gain — three stages is not deep enough for the degradation problem
it exists to solve. And attention made things *worse*: M4 scored below M3 while costing the most parameters
and the most training time.

That is the useful result, not a spoiled one. "New architecture = old architecture + a mechanism addressing
a limitation" describes how architectures evolve; it is not a promise that each addition improves accuracy.
A mechanism helps when the model actually has the limitation it targets, and costs you when it does not.
Caveats: one seed, ten epochs, no tuning, and narrowed 16/32/64 stages — see the notebook for the full
reasoning and the SE-gate analysis that checks whether the attention block was inert or merely unhelpful.
""")

    if d6 or d7:
        parts.append("""### LeNet-5 (LeCun et al., 1998)

The same classic architecture applied to both image datasets, run alongside the original notebooks
rather than replacing them. Topology is LeCun's — 6 then 16 feature maps, 5x5 kernels, a 120 -> 84 -> 10
head — modernized with ReLU and max pooling in place of tanh and average pooling, which is what
`scratch_nn.py` provides and what modern implementations use.
""")
        for payload, title in ((d6, "MNIST — LeNet-5"), (d7, "CIFAR-10 — LeNet-5")):
            if not payload:
                continue
            parts.append(f"#### {title}\n")
            parts.append(_table(
                _rows(payload),
                ["framework", "dataset", "n_params", "epochs", "train_seconds",
                 "test_accuracy", "f1_macro"],
                ["Framework", "Dataset", "Params", "Epochs", "Train (s)", "Accuracy", "Macro F1"],
            ))
            parts.append("")
        parts.append("""One architecture, two datasets of the same spatial size, opposite outcomes. MNIST is the problem
LeNet-5 was designed for and it holds its own there. CIFAR-10 is 32x32 colour photographs, and the same
~62k parameters — under 3k of them in the convolutions — are not enough visual vocabulary for ten object
classes. Capacity has to match difficulty.

See `pdf/08_lenet_mnist_report.pdf` and `pdf/09_lenet_cifar10_report.pdf` for the full analysis.
""")

    # ---- parameter agreement ----
    agree = []
    for payload, name in ((d1, "Diabetes"), (d2, "MNIST"), (d3, "CIFAR-10"),
                          (d6, "MNIST (LeNet-5)"), (d7, "CIFAR-10 (LeNet-5)")):
        if not payload:
            continue
        counts = {r["n_params"] for r in _rows(payload)}
        agree.append(f"- **{name}**: {next(iter(counts)):,} parameters"
                     f" — {'all three frameworks agree' if len(counts) == 1 else 'MISMATCH: ' + str(counts)}")
    if agree:
        parts.append("\n## Parameter counts\n")
        parts.append("Computed by hand from the tutorial's section 43 formulas before building anything, "
                     "then checked against each framework:\n")
        parts.append("\n".join(agree))

    parts.append("""

## Saved models

Every notebook persists the networks it trains, not just the numbers they produced. Each leg is written in
its own framework's native format under `results/models/<notebook>/`, beside a JSON sidecar recording the
architecture, the parameter count, and the accuracy those exact weights scored:

| Leg | Format | Reloads on its own? |
|---|---|---|
| Scratch (NumPy) | `.npz` — one array per layer parameter | no |
| TensorFlow/Keras | `.keras` — graph and weights together | **yes** |
| PyTorch | `.pt` — `state_dict` | no |

```python
import ass4_utils as U

U.list_models()                                   # everything saved, from the sidecars
U.list_models("06_mnist_lenet")                   # one notebook

keras_model = U.load_model("lenet5_keras_subset", "06_mnist_lenet")       # standalone
torch_model = U.load_model("lenet5_torch_subset", "06_mnist_lenet",
                           model=LeNet5(C, N_CLASSES))                    # needs an instance
```

`.pt` and `.npz` hold weights only, so reloading them means rebuilding the architecture first and passing the
fresh instance as `model=`. That is deliberate: `state_dict` is the portable half of a PyTorch model, while
pickling the class ties the file to the notebook that defined it. `04_compare.ipynb` already followed this
contract through `results/variant_cache/`; the other notebooks now do too.

These weights are committed, so cloning the repo is enough to load any of the trained networks without
retraining. The M1..M4 cache under `results/variant_cache/` stays gitignored — rebuild it with
`python prefill_variants.py`.

## What the experiments show

**The framework does not change the model.** Across three datasets and two model families, the three
implementations produced identical parameter counts, identical tensor shapes at every stage, and accuracies
separated by less than run-to-run noise. `S.Dense`, `keras.layers.Dense` and `nn.Linear` are three names for
one function.

**What changes is visibility.** The scratch leg makes every step of
`Forward -> Loss -> Gradient -> Update` explicit and had to derive its own gradients — verified against
central finite differences in each notebook. Keras hides the loop, the gradient and the update behind
`fit()`. PyTorch keeps a compact model definition but an explicit loop, with autograd underneath.

**What changes is cost, and not in one direction.** On the tiny tabular MLP the GPU leg is the *slowest*:
each batch carries too little arithmetic to amortise moving it to the device. On the image CNNs the ordering
reverses sharply. Hardware acceleration pays off only when there is enough work per batch to pay for.

**The model should match the structure of the data.** Tabular rows have no spatial neighbourhood, so no
convolution appears in notebook 1. Images do, so local connectivity and weight sharing earn their place in
notebooks 2 and 3. A CNN is a choice justified by a property of the input, not a default.

## Honest limits

- Short training runs (5 epochs on subsets for the three-way comparison, 10 epochs on narrowed 16/32/64
  stages for M1..M4), no data augmentation, no learning-rate schedule, no hyperparameter tuning. These are
  not competitive CIFAR-10 numbers and are not meant to be.
- One seed per configuration. Notebook 04 measures the noise floor directly by re-running one model across
  five seeds: the seed-only spread was 0.053 accuracy and the framework spread on the same subset was 0.043,
  so the frameworks fall inside the noise band. The M1..M4 gaps are larger, but that experiment has no
  repeated-seed band of its own.
- The GPU in this machine thermally throttles at 96-98 C under sustained load. The M1..M4 runs therefore
  used mixed precision and on-GPU batching, and their timings are comparable with each other but not with
  the fp32 timings elsewhere.
- Hospital readmission is genuinely hard to predict from administrative fields — the modest numbers in
  notebook 1 reflect the problem, not a defect in any implementation.

## Data sources

- **Diabetes 130-US hospitals**: UCI ML Repository, dataset 296 — downloaded automatically into `data/`
- **MNIST**, **CIFAR-10**: via `torchvision.datasets`, downloaded automatically into `data/`
""")

    return "\n".join(parts)


# Lines that are pure machinery noise: dataset-download progress bars and the
# logging banners TensorFlow/ZMQ emit at import. They carry no information about
# the experiments but can run to thousands of lines in an executed notebook.
_NOISE_MARKERS = (
    "kB/s]", "MB/s]", "it/s]", "B/s]",
    "oneDNN custom operations",
    "absl::InitializeLog",
    "TF_ENABLE_ONEDNN_OPTS",
    "Proactor event loop does not implement",
    "Kernel is running over TCP without encryption",
    "WARNING: All log messages before",
    "external/local_xla",
    "computation placer already registered",
)

_NOISE_PREFIX_RE = None


def _is_noise(line: str) -> bool:
    import re

    global _NOISE_PREFIX_RE
    if _NOISE_PREFIX_RE is None:
        # e.g. "  2%|          | 2.79M/170M [00:35<34:43, 80.5kB/s]"
        _NOISE_PREFIX_RE = re.compile(r"^\s*\d+%\|")
    s = line.strip()
    if not s:
        return False
    if _NOISE_PREFIX_RE.match(line):
        return True
    return any(mark in line for mark in _NOISE_MARKERS)


def clean_notebook_outputs(path: str) -> int:
    """Strip download progress bars and import banners from an executed notebook.

    Returns the number of output lines removed. Genuine stdout - training logs,
    tables, shapes, the TensorFlow "GPU not available" notice - is untouched.
    """
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)

    removed = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        kept_outputs = []
        for out in cell.get("outputs", []):
            if out.get("output_type") != "stream":
                kept_outputs.append(out)
                continue

            text = out.get("text")
            lines = text if isinstance(text, list) else [text or ""]
            kept = [ln for ln in lines if not _is_noise(ln)]
            removed += len(lines) - len(kept)

            if not any(ln.strip() for ln in kept):
                continue            # output was nothing but noise
            out["text"] = kept
            kept_outputs.append(out)
        cell["outputs"] = kept_outputs

    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    return removed


def export_pdfs(names):
    """Notebook -> self-contained HTML -> PDF via Playwright/Chromium.

    `nbconvert --to webpdf` cannot be used here: on Windows it installs a
    SelectorEventLoop, which has no subprocess support, so launching Chromium
    dies with NotImplementedError. Driving Playwright's *sync* API ourselves
    sidesteps the event loop entirely. Neither pandoc nor LaTeX is involved.
    """
    from playwright.sync_api import sync_playwright

    html_dir = os.path.join(HERE, "_html")
    os.makedirs(html_dir, exist_ok=True)
    pdf_dir = os.path.join(HERE, "pdf")
    os.makedirs(pdf_dir, exist_ok=True)

    ok, failed = [], []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            for name in names:
                src = os.path.join(HERE, f"{name}.ipynb")
                if not os.path.exists(src):
                    print(f"skip {name}: notebook missing")
                    continue

                n_removed = clean_notebook_outputs(src)
                print(f"exporting {name}.pdf "
                      f"({n_removed:,} noise lines stripped) ...", flush=True)
                proc = subprocess.run(
                    [sys.executable, "-m", "nbconvert", "--to", "html",
                     "--embed-images", "--output-dir", html_dir, src],
                    capture_output=True, text=True,
                )
                html = os.path.join(html_dir, f"{name}.html")
                if proc.returncode != 0 or not os.path.exists(html):
                    failed.append((name, (proc.stderr or proc.stdout or "")[-400:]))
                    continue

                try:
                    page = browser.new_page()
                    page.goto("file:///" + html.replace("\\", "/"),
                              wait_until="networkidle", timeout=180_000)
                    # Notebook CSS lets long source lines run past the paper edge,
                    # where print simply clips them. Force wrapping instead, and
                    # keep figures and table rows from splitting across pages.
                    page.add_style_tag(content="""
                        pre, code, .jp-OutputArea-output, .highlight pre {
                            white-space: pre-wrap !important;
                            word-break: break-word !important;
                            overflow-wrap: anywhere !important;
                        }
                        .jp-Cell, .jp-OutputArea-child, table, tr, img {
                            break-inside: avoid;
                            page-break-inside: avoid;
                        }
                        img, svg { max-width: 100% !important; height: auto !important; }
                    """)
                    page.pdf(
                        path=os.path.join(pdf_dir, f"{name}.pdf"),
                        format="A4",
                        print_background=True,
                        margin={"top": "14mm", "bottom": "14mm",
                                "left": "10mm", "right": "10mm"},
                    )
                    page.close()
                    ok.append(name)
                except Exception as e:
                    failed.append((name, f"{type(e).__name__}: {e}"[:400]))
        finally:
            browser.close()
    return ok, failed


if __name__ == "__main__":
    readme = build_readme()
    with open(os.path.join(HERE, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)
    print(f"wrote README.md ({len(readme):,} chars)")

    if "--no-pdf" not in sys.argv:
        ok, failed = export_pdfs([n for n, _ in NOTEBOOKS])
        print(f"\nPDFs written: {', '.join(ok) if ok else 'none'}")
        for name, err in failed:
            print(f"FAILED {name}:\n{err}\n")
