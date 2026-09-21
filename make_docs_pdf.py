"""Render the hand-written Markdown documents to pdf/.

Notebook PDFs come from make_report.py; this script covers the prose documents
that are not notebooks.

Usage:
    python make_docs_pdf.py                 # all documents
    python make_docs_pdf.py lenet_mnist     # only those whose name matches
"""

from __future__ import annotations

import os
import sys

import markdown

from cover import COVER_CSS, cover_html

HERE = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(HERE, "pdf")
HTML_DIR = os.path.join(HERE, "_html")

# (markdown source, output basename, footer caption). Footer text is ASCII-only:
# Chromium renders the header/footer templates with its own font stack, which has
# no Vietnamese coverage, so accented characters come out as tofu there.
# theory_notes.md is absent on purpose: pdf/05_deep_learning_cnn.pdf comes from
# report/build_latex.py now. Listing it here too would have the two pipelines
# overwrite each other's output depending on which ran last.
DOCS = [
    ("lenet_mnist_report.md",   "08_lenet_mnist_report",    "LeNet-5 tren MNIST"),
    ("lenet_cifar10_report.md", "09_lenet_cifar10_report",  "LeNet-5 tren CIFAR-10"),
]

# Font choice is not cosmetic here: Charter/Georgia lack the Vietnamese Extended
# block (U+1EA0-U+1EF9), so Chromium falls back per-glyph and splits the diacritics
# ("nhie`u ta`ng" instead of "nhieu tang"). Segoe UI and Consolas both carry the
# full range, and Consolas also ships the box-drawing glyphs the ASCII diagrams need.
CSS = """
@page { size: A4; margin: 16mm 14mm; }

body {
    font-family: "Segoe UI", "Tahoma", "DejaVu Sans", sans-serif;
    font-size: 10.5pt;
    line-height: 1.55;
    color: #1a1a1a;
    max-width: 100%;
}

h1, h2, h3 {
    font-family: "Segoe UI Semibold", "Segoe UI", Tahoma, sans-serif;
    line-height: 1.25;
    break-after: avoid;
    page-break-after: avoid;
}

h1 {
    font-size: 18pt;
    color: #0b3d62;
    border-bottom: 2px solid #0b3d62;
    padding-bottom: 4px;
    margin-top: 26px;
}
h2 { font-size: 13pt; color: #14538a; margin-top: 20px; }
h3 { font-size: 11pt; color: #333; margin-top: 14px; }

/* Each numbered part starts on a fresh page. The first two h1 are exempt: the
   document title, and part 1 right under it -- otherwise page 1 holds nothing
   but the title block. */
h1 { break-before: page; page-break-before: page; }
h1:nth-of-type(1),
h1:nth-of-type(2) { break-before: auto; page-break-before: auto; }

code {
    font-family: "Cascadia Mono", "Consolas", "DejaVu Sans Mono", monospace;
    font-size: 9pt;
    background: #f2f4f7;
    padding: 1px 4px;
    border-radius: 3px;
}

pre {
    font-family: "Cascadia Mono", "Consolas", "DejaVu Sans Mono", monospace;
    font-size: 8.5pt;
    line-height: 1.35;
    background: #f7f8fa;
    border: 1px solid #dde2e8;
    border-left: 3px solid #14538a;
    border-radius: 4px;
    padding: 9px 11px;
    white-space: pre-wrap;
    word-break: break-word;
    overflow-wrap: anywhere;
    break-inside: avoid;
    page-break-inside: avoid;
}
pre code { background: none; padding: 0; font-size: inherit; }

table {
    border-collapse: collapse;
    width: 100%;
    font-size: 9pt;
    margin: 10px 0;
    break-inside: avoid;
    page-break-inside: avoid;
}
th, td { border: 1px solid #c8d0d8; padding: 5px 8px; text-align: left; }
th { background: #e8eef4; font-weight: 600; }
tr:nth-child(even) td { background: #fafbfc; }

blockquote {
    margin: 10px 0;
    padding: 7px 13px;
    border-left: 3px solid #d9a441;
    background: #fdf8ec;
    break-inside: avoid;
    page-break-inside: avoid;
}
blockquote p { margin: 4px 0; }

hr { border: none; border-top: 1px solid #dde2e8; margin: 18px 0; }
ul, ol { margin: 8px 0; padding-left: 22px; }
li { margin: 3px 0; }
strong { color: #0b3d62; }
"""

MD_EXTENSIONS = ["tables", "fenced_code", "sane_lists", "attr_list"]


def build_html(src: str, name: str) -> str:
    with open(os.path.join(HERE, src), encoding="utf-8") as f:
        body = markdown.markdown(f.read(), extensions=MD_EXTENSIONS)

    os.makedirs(HTML_DIR, exist_ok=True)
    html_path = os.path.join(HTML_DIR, f"{name}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(
            "<!doctype html><html lang='vi'><head><meta charset='utf-8'>"
            f"<title>{name}</title><style>{CSS}{COVER_CSS}</style></head>"
            f"<body>{cover_html(name)}{body}</body></html>"
        )
    return html_path


def footer(caption: str) -> str:
    return (
        "<div style='width:100%;font-size:8px;color:#888;"
        "padding:0 14mm;font-family:sans-serif;'>"
        f"<span style='float:left'>Assignment 4 &mdash; {caption}</span>"
        "<span style='float:right'>"
        "<span class='pageNumber'></span>/<span class='totalPages'></span>"
        "</span></div>"
    )


def export(docs) -> None:
    from playwright.sync_api import sync_playwright

    os.makedirs(PDF_DIR, exist_ok=True)

    # One browser for every document rather than one launch each - Chromium
    # startup dominates the runtime of this script.
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        for src, name, caption in docs:
            html = build_html(src, name)
            pdf_path = os.path.join(PDF_DIR, f"{name}.pdf")

            page = browser.new_page()
            page.goto("file:///" + html.replace("\\", "/"),
                      wait_until="networkidle", timeout=120_000)
            page.pdf(
                path=pdf_path,
                format="A4",
                print_background=True,
                margin={"top": "14mm", "bottom": "14mm",
                        "left": "12mm", "right": "12mm"},
                display_header_footer=True,
                header_template="<div></div>",
                footer_template=footer(caption),
            )
            page.close()

            size = os.path.getsize(pdf_path) / 1024
            print(f"  {src:<26} -> pdf/{name}.pdf  ({size:,.1f} KB)")
        browser.close()


if __name__ == "__main__":
    wanted = sys.argv[1:]
    docs = [d for d in DOCS if not wanted or any(w in d[0] or w in d[1] for w in wanted)]

    missing = [d for d in docs if not os.path.exists(os.path.join(HERE, d[0]))]
    for src, name, _ in missing:
        print(f"  skip {src} (not written yet)")
    docs = [d for d in docs if d not in missing]

    if not docs:
        sys.exit("nothing to render")

    export(docs)
