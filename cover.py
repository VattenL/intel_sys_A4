"""Cover page shared by make_report.py (notebooks) and make_docs_pdf.py (prose docs).

Both pipelines end in Chromium print-to-PDF, so the cover is plain HTML: one
full-height flex block followed by a forced page break. Keeping it here means
the identity fields live in exactly one place.
"""

from __future__ import annotations

# Fill these once; every cover page reads from here.
SUBJECT = "Intelligent Systems Development"
ASSIGNMENT = "Assignment 4 — Comparing CNN Implementations"
STUDENT = "TODO — full name"
STUDENT_ID = "TODO — student ID"
CLASS_NAME = "TODO — class"
INSTRUCTOR = "TODO — instructor"
INSTITUTION = "TODO — university / faculty"
DATE = "September 2026"

# Per-document title and one-line summary shown on the cover. A document absent
# from this map gets no cover page - that is how 02_mnist stays as it was.
TITLES = {
    "01_diabetes130": (
        "Diabetes 130-US hospitals",
        "Readmission MLP built three ways — NumPy from scratch, "
        "TensorFlow/Keras, and PyTorch",
    ),
    "03_cifar10": (
        "CIFAR-10",
        "CNN over 32×32 colour images — three frameworks, one split, "
        "one set of hyperparameters",
    ),
    "04_compare": (
        "Comparison and Improved Models",
        "Cross-dataset comparison of the three implementations, and the "
        "M1–M4 architecture-evolution experiment",
    ),
    "05_deep_learning_cnn": (
        "Deep Learning and CNNs",
        "Function composition, backpropagation, and the structure of "
        "convolutional networks",
    ),
    "06_mnist_lenet": (
        "MNIST with LeNet-5",
        "LeCun et al. (1998) in three frameworks — run alongside "
        "notebook 02 rather than replacing it",
    ),
    "07_cifar10_lenet": (
        "CIFAR-10 with LeNet-5",
        "The same LeNet-5 applied to colour photographs — three "
        "frameworks, notebook 03 left untouched",
    ),
    "08_lenet_mnist_report": (
        "LeNet-5 on MNIST — Report",
        "Full analysis of the results, parameter budget, and the limits of "
        "the architecture",
    ),
    "09_lenet_cifar10_report": (
        "LeNet-5 on CIFAR-10 — Report",
        "Why one architecture produces opposite outcomes on two datasets of "
        "the same spatial size",
    ),
}

# Scoped under .a4-cover so it cannot collide with nbconvert's own stylesheet,
# which the notebook pipeline injects this markup into.
COVER_CSS = """
.a4-cover {
    height: 247mm;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    font-family: "Segoe UI", "Tahoma", "DejaVu Sans", sans-serif;
    color: #1a1a1a;
    text-align: center;
    break-after: page;
    page-break-after: always;
}
.a4-cover * { font-family: inherit; }
.a4-cover .ac-top { padding-top: 6mm; }
.a4-cover .ac-inst {
    font-size: 12pt;
    font-weight: 600;
    letter-spacing: .04em;
    text-transform: uppercase;
    color: #0b3d62;
}
.a4-cover .ac-subject {
    font-size: 10.5pt;
    color: #5a6673;
    margin-top: 5px;
}
.a4-cover .ac-mid { padding: 0 8mm; }
.a4-cover .ac-kicker {
    font-size: 10pt;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: #14538a;
    margin-bottom: 10px;
}
.a4-cover .ac-title {
    font-size: 27pt;
    font-weight: 700;
    line-height: 1.2;
    color: #0b3d62;
    margin: 0;
    padding: 14px 0;
    border-top: 3px solid #0b3d62;
    border-bottom: 3px solid #0b3d62;
}
.a4-cover .ac-sub {
    font-size: 11.5pt;
    line-height: 1.5;
    color: #3d4852;
    margin: 16px auto 0;
    max-width: 125mm;
}
.a4-cover .ac-assign {
    font-size: 10.5pt;
    color: #5a6673;
    margin-top: 22px;
}
.a4-cover .ac-meta {
    display: inline-block;
    text-align: left;
    font-size: 11pt;
    line-height: 1.85;
    border-top: 1px solid #c8d0d8;
    padding-top: 12px;
}
.a4-cover .ac-meta .k {
    display: inline-block;
    min-width: 34mm;
    color: #5a6673;
}
.a4-cover .ac-meta .v { font-weight: 600; color: #0b3d62; }
.a4-cover .ac-date { font-size: 10pt; color: #5a6673; margin-top: 14px; }
"""


def _row(key: str, value: str) -> str:
    return f"<div><span class='k'>{key}</span><span class='v'>{value}</span></div>"


def cover_html(name: str) -> str:
    """Full-page cover for `name`, or "" when the document takes no cover."""
    if name not in TITLES:
        return ""
    title, subtitle = TITLES[name]
    meta = "".join([
        _row("Student", STUDENT),
        _row("Student ID", STUDENT_ID),
        _row("Class", CLASS_NAME),
        _row("Instructor", INSTRUCTOR),
    ])
    return f"""<section class="a4-cover">
  <div class="ac-top">
    <div class="ac-inst">{INSTITUTION}</div>
    <div class="ac-subject">{SUBJECT}</div>
  </div>
  <div class="ac-mid">
    <div class="ac-kicker">Report</div>
    <h1 class="ac-title">{title}</h1>
    <p class="ac-sub">{subtitle}</p>
    <div class="ac-assign">{ASSIGNMENT}</div>
  </div>
  <div class="ac-bottom">
    <div class="ac-meta">{meta}</div>
    <div class="ac-date">{DATE}</div>
  </div>
</section>"""
