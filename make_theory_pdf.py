"""Render theory_notes.md to pdf/05_deep_learning_cnn.pdf.

Usage:
    python make_theory_pdf.py
"""

from __future__ import annotations

import os
import sys

import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "theory_notes.md")
PDF_DIR = os.path.join(HERE, "pdf")
HTML_DIR = os.path.join(HERE, "_html")
NAME = "05_deep_learning_cnn"

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


def build_html() -> str:
    if not os.path.exists(SRC):
        sys.exit(f"missing source: {SRC}")

    with open(SRC, encoding="utf-8") as f:
        text = f.read()

    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
    )

    os.makedirs(HTML_DIR, exist_ok=True)
    html_path = os.path.join(HTML_DIR, f"{NAME}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(
            "<!doctype html><html lang='vi'><head><meta charset='utf-8'>"
            f"<title>{NAME}</title><style>{CSS}</style></head>"
            f"<body>{body}</body></html>"
        )
    return html_path


def export_pdf(html_path: str) -> str:
    from playwright.sync_api import sync_playwright

    os.makedirs(PDF_DIR, exist_ok=True)
    pdf_path = os.path.join(PDF_DIR, f"{NAME}.pdf")

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        page.goto("file:///" + html_path.replace("\\", "/"),
                  wait_until="networkidle", timeout=120_000)
        page.pdf(
            path=pdf_path,
            format="A4",
            print_background=True,
            margin={"top": "14mm", "bottom": "14mm",
                    "left": "12mm", "right": "12mm"},
            display_header_footer=True,
            header_template="<div></div>",
            footer_template=(
                "<div style='width:100%;font-size:8px;color:#888;"
                "padding:0 14mm;font-family:sans-serif;'>"
                "<span style='float:left'>Assignment 4 &mdash; Deep Learning va CNN</span>"
                "<span style='float:right'>"
                "<span class='pageNumber'></span>/<span class='totalPages'></span>"
                "</span></div>"
            ),
        )
        browser.close()
    return pdf_path


if __name__ == "__main__":
    html = build_html()
    print(f"html  -> {os.path.relpath(html, HERE)}")
    pdf = export_pdf(html)
    size = os.path.getsize(pdf) / 1024
    print(f"pdf   -> {os.path.relpath(pdf, HERE)}  ({size:,.1f} KB)")
