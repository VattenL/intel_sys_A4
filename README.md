# Assignment 4 — Comparing CNN Implementations

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
| `report/` | LaTeX build for `pdf/01`, `03`, `04`, `05`, `06` — `python report/build_latex.py` |

Run the notebooks in order; `04_compare.ipynb` reads the JSON the first three write.

## Building the PDFs

Three commands, each owning a different part of `pdf/`:

| Command | Produces | How |
|---|---|---|
| `python report/build_latex.py` | `01`, `03`, `04`, `05_deep_learning_cnn`, `06` | nbconvert -> Markdown -> pandoc -> XeLaTeX |
| `python make_report.py` | `README.md`, `02_mnist`, `07_cifar10_lenet` | nbconvert -> HTML -> Chromium print |
| `python make_docs_pdf.py` | `08`, `09` | Markdown -> HTML -> Chromium print |

The LaTeX route needs `pypandoc-binary` (`pip install pypandoc-binary`) and MiKTeX's
`xelatex`; the fonts are Cambria, Segoe UI and Consolas, which ship with Windows.
`python report/build_latex.py 04 --keep` builds one document and leaves the
intermediate `.tex` and `.log` under `report/build/` for inspection.

## Environment

- Python 3.11 (`C:\Users\ADMIN\AppData\Local\Programs\Python\Python311\python.exe`), Jupyter kernel **`ass4`**
- PyTorch with CUDA on an RTX 3060 Laptop GPU
- TensorFlow 2.21 / Keras 3.15 — **CPU only**, since Windows-native TensorFlow has shipped no GPU build since 2.10

That last point matters when reading the tables: Keras and PyTorch timings measure different hardware.
Parameter counts and accuracies are directly comparable; wall-clock times are not.

## The fairness rule

Every comparison follows the rule from lecture 04, slide 29:

> Same Dataset + Same Split + Same Architecture + Comparable Hyperparameters

All three legs pull their arrays from one loader with one seed, so they train on byte-identical data.


## Results

### Diabetes 130-US hospitals — MLP

| Framework | Dataset | Params | Epochs | Train (s) | Accuracy | Macro F1 |
|---|---|---|---|---|---|---|
| Scratch (NumPy) | Diabetes 130-US hospitals | 31,491 | 15 | 6.0 | 0.6094 | 0.3450 |
| TensorFlow/Keras | Diabetes 130-US hospitals | 31,491 | 15 | 7.8 | 0.6077 | 0.3686 |
| PyTorch | Diabetes 130-US hospitals | 31,491 | 15 | 9.2 | 0.6105 | 0.3694 |

### MNIST — CNN

| Framework | Dataset | Params | Epochs | Train (s) | Accuracy | Macro F1 |
|---|---|---|---|---|---|---|
| Scratch (NumPy) | MNIST (10k subset) | 20,490 | 5 | 36.2 | 0.9810 | 0.9810 |
| TensorFlow/Keras | MNIST (10k subset) | 20,490 | 5 | 4.1 | 0.9755 | 0.9756 |
| PyTorch | MNIST (10k subset) | 20,490 | 5 | 1.9 | 0.9835 | 0.9836 |
| PyTorch | MNIST (full 60k) | 20,490 | 5 | 9.1 | 0.9875 | 0.9874 |
| TensorFlow/Keras | MNIST (full 60k) | 20,490 | 5 | 19.2 | 0.9855 | 0.9854 |

### CIFAR-10 — CNN

| Framework | Dataset | Params | Epochs | Train (s) | Accuracy | Macro F1 |
|---|---|---|---|---|---|---|
| Scratch (NumPy) | CIFAR-10 (5k subset) | 545,098 | 5 | 65.2 | 0.5055 | 0.5032 |
| TensorFlow/Keras | CIFAR-10 (5k subset) | 545,098 | 5 | 9.1 | 0.5485 | 0.5452 |
| PyTorch | CIFAR-10 (5k subset) | 545,098 | 5 | 2.1 | 0.5325 | 0.5288 |
| PyTorch | CIFAR-10 (full 50k) | 545,098 | 5 | 17.3 | 0.7137 | 0.7123 |
| TensorFlow/Keras | CIFAR-10 (full 50k) | 545,098 | 5 | 70.0 | 0.7043 | 0.6990 |

### Improved CNN models (tutorial section 45, full CIFAR-10)

| Model | Params | Epochs | Train (s) | Accuracy | Macro F1 |
|---|---|---|---|---|---|
| M1  Conv+ReLU+Pool | 72,730 | 10 | 552.3 | 0.6947 | 0.6956 |
| M2  + BatchNorm | 72,954 | 10 | 573.5 | 0.7524 | 0.7573 |
| M3  + Residual | 75,786 | 10 | 681.6 | 0.7639 | 0.7696 |
| M4  + Attention(SE) | 78,614 | 10 | 766.2 | 0.7367 | 0.7468 |

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

### LeNet-5 (LeCun et al., 1998)

The same classic architecture applied to both image datasets, run alongside the original notebooks
rather than replacing them. Topology is LeCun's — 6 then 16 feature maps, 5x5 kernels, a 120 -> 84 -> 10
head — modernized with ReLU and max pooling in place of tanh and average pooling, which is what
`scratch_nn.py` provides and what modern implementations use.

#### MNIST — LeNet-5

| Framework | Dataset | Params | Epochs | Train (s) | Accuracy | Macro F1 |
|---|---|---|---|---|---|---|
| Scratch (NumPy) | MNIST (10k subset) | 61,706 | 5 | 30.4 | 0.9775 | 0.9772 |
| TensorFlow/Keras | MNIST (10k subset) | 61,706 | 5 | 5.9 | 0.9760 | 0.9762 |
| PyTorch | MNIST (10k subset) | 61,706 | 5 | 3.4 | 0.9780 | 0.9780 |
| PyTorch | MNIST (full 60k) | 61,706 | 5 | 17.4 | 0.9894 | 0.9892 |
| TensorFlow/Keras | MNIST (full 60k) | 61,706 | 5 | 25.4 | 0.9859 | 0.9858 |

#### CIFAR-10 — LeNet-5

| Framework | Dataset | Params | Epochs | Train (s) | Accuracy | Macro F1 |
|---|---|---|---|---|---|---|
| Scratch (NumPy) | CIFAR-10 (5k subset) | 62,006 | 5 | 27.4 | 0.4270 | 0.4216 |
| TensorFlow/Keras | CIFAR-10 (5k subset) | 62,006 | 5 | 4.4 | 0.4545 | 0.4528 |
| PyTorch | CIFAR-10 (5k subset) | 62,006 | 5 | 2.0 | 0.4375 | 0.4273 |
| PyTorch | CIFAR-10 (full 50k) | 62,006 | 5 | 15.7 | 0.6162 | 0.6151 |
| TensorFlow/Keras | CIFAR-10 (full 50k) | 62,006 | 5 | 23.8 | 0.6032 | 0.5985 |

One architecture, two datasets of the same spatial size, opposite outcomes. MNIST is the problem
LeNet-5 was designed for and it holds its own there. CIFAR-10 is 32x32 colour photographs, and the same
~62k parameters — under 3k of them in the convolutions — are not enough visual vocabulary for ten object
classes. Capacity has to match difficulty.

See `pdf/08_lenet_mnist_report.pdf` and `pdf/09_lenet_cifar10_report.pdf` for the full analysis.


## Parameter counts

Computed by hand from the tutorial's section 43 formulas before building anything, then checked against each framework:

- **Diabetes**: 31,491 parameters — all three frameworks agree
- **MNIST**: 20,490 parameters — all three frameworks agree
- **CIFAR-10**: 545,098 parameters — all three frameworks agree
- **MNIST (LeNet-5)**: 61,706 parameters — all three frameworks agree
- **CIFAR-10 (LeNet-5)**: 62,006 parameters — all three frameworks agree


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
