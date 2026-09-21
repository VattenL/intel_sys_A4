"""Render the Assignment 4 documents through LaTeX instead of Chromium.

    python report/build_latex.py                 # every document
    python report/build_latex.py 04 05           # only those whose name matches
    python report/build_latex.py --tex-only      # stop after writing the .tex
    python report/build_latex.py --keep          # leave the build directory in place

Documents:

    pdf/01_diabetes130.pdf        01_diabetes130.ipynb
    pdf/03_cifar10.pdf            03_cifar10.ipynb
    pdf/04_compare.pdf            04_compare.ipynb
    pdf/05_deep_learning_cnn.pdf  theory_notes.md
    pdf/06_mnist_lenet.pdf        06_mnist_lenet.ipynb

Pipeline per document:

    notebook  --nbconvert-->  Markdown + PNGs        (prose, source cells, cell output)
    markdown  ------------->  read as written
      -> strip the leading H1, which the title page already carries
      -> turn each pandas DataFrame's HTML repr into a pipe table
      -> rewrite pipe-table separators so column widths follow content
      -> pandoc --standalone, pandoc's own template plus preamble.tex
      -> xelatex x2 (twice, so the table of contents has page numbers)

XeLaTeX rather than pdflatex: theory_notes.md is Vietnamese and the executed cells
print box-drawing characters. Fonts are Windows system fonts chosen for coverage --
Cambria has the full Vietnamese Extended block, Consolas has the box-drawing set.

The identity fields and per-document titles come from cover.py, the same module the
Chromium pipeline reads, so the two never drift apart.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BUILD = HERE / "build"
PDF_DIR = ROOT / "pdf"
PREAMBLE = HERE / "preamble.tex"

sys.path.insert(0, str(ROOT))
import cover  # noqa: E402  -- needs ROOT on the path first

# The TeX engine. MiKTeX's xelatex on the Windows machine the documents were written
# on; otherwise whatever is on PATH -- xelatex, or tectonic, which is the same XeTeX
# engine with its own package downloader. ASS4_TEX overrides both.
def _find_engine() -> Path | None:
    override = os.environ.get("ASS4_TEX")
    if override:
        return Path(override)
    miktex = Path(r"D:/LaTeX/miktex/bin/x64/xelatex.exe")
    if miktex.exists():
        return miktex
    for exe in ("xelatex", "tectonic"):
        found = shutil.which(exe)
        if found:
            return Path(found)
    return None


ENGINE = _find_engine()

# Fonts are picked for coverage, not looks: the prose is Vietnamese (needs the
# Vietnamese Extended block, U+1EA0-U+1EF9) and the executed cells print box-drawing
# characters. Windows: Cambria + Segoe UI + Consolas, all three present by default.
# macOS: Times New Roman + Arial + Menlo, the same three properties -- Menlo is the
# only one of the mac monospaces carrying the heavy box-drawing set the Keras summary
# tables use. Override any of them with ASS4_MAIN_FONT / ASS4_SANS_FONT /
# ASS4_MONO_FONT when building somewhere else.
_FONTS = {
    "win32":  ("Cambria", "Segoe UI", "Consolas"),
    "darwin": ("Times New Roman", "Arial", "Menlo"),
}
_default_fonts = _FONTS.get(sys.platform, ("DejaVu Serif", "DejaVu Sans", "DejaVu Sans Mono"))
MAIN_FONT = os.environ.get("ASS4_MAIN_FONT", _default_fonts[0])
SANS_FONT = os.environ.get("ASS4_SANS_FONT", _default_fonts[1])
MONO_FONT = os.environ.get("ASS4_MONO_FONT", _default_fonts[2])

# name -> (source file, running head). The head is printed on every page; it is set in
# Cambria, so Vietnamese diacritics are fine there.
DOCS: dict[str, tuple[str, str]] = {
    "01_diabetes130":       ("01_diabetes130.ipynb", "Bài tập 4 — Diabetes 130-US hospitals"),
    "03_cifar10":           ("03_cifar10.ipynb",     "Bài tập 4 — CIFAR-10"),
    "04_compare":           ("04_compare.ipynb",     "Bài tập 4 — So sánh và cải tiến mô hình"),
    "05_deep_learning_cnn": ("theory_notes.md",      "Bài tập 4 — Deep Learning và CNN"),
    "06_mnist_lenet":       ("06_mnist_lenet.ipynb", "Bài tập 4 — MNIST với LeNet-5"),
}

CODE_FENCE = re.compile(r"^(```|~~~)")
SEPARATOR_ROW = re.compile(r"^\s*\|[\s:\-|]+\|\s*$")

# Placeholders, not the real \begin{landscape}: pandoc's raw-LaTeX reader swallows a
# whole environment, so writing the environment into the Markdown would take the table
# between its delimiters along with it and emit the pipe rows as literal TeX. These
# survive the conversion as ordinary paragraphs and are swapped back afterwards.
LSCAPE_OPEN = "ASS4LANDSCAPEOPEN"
LSCAPE_CLOSE = "ASS4LANDSCAPECLOSE"


# ---------------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------------

def notebook_to_markdown(name: str, src: Path, out_dir: Path) -> str:
    """Export one notebook to Markdown, its figures landing in ``<name>_files/``.

    ``display_data_priority`` is left at its default on purpose: the DataFrame HTML
    reprs it prefers are worth more than their text/plain fallbacks once
    html_tables_to_markdown() has turned them into real tables.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "nbconvert", "--to", "markdown",
         "--output-dir", str(out_dir), str(src)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    md = out_dir / f"{name}.md"
    if proc.returncode != 0 or not md.exists():
        raise RuntimeError(f"nbconvert failed for {name}:\n"
                           f"{(proc.stderr or proc.stdout or '')[-800:]}")
    return md.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------------
# Markdown fixes
# ---------------------------------------------------------------------------------

def strip_leading_title(text: str) -> str:
    """Drop the first H1. The title page already carries it, in larger type."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        if line.startswith("# "):
            del lines[i]
        break
    return "\n".join(lines)


def shallowest_heading(text: str) -> int:
    """The smallest heading level the document actually uses, ignoring code.

    Once the title H1 is gone, most of the notebooks start at ``##`` -- their H1 was
    the title and nothing else. Left alone, every heading would come out a
    \\subsection: small type, no page break, and a flat table of contents. The level
    is measured here so the conversion can shift it up to \\section.

    Lines inside a fenced block are skipped: a Python comment at the start of a line
    reads as an H1 otherwise, which made the first count report eight sections in a
    notebook that has none.
    """
    fenced = False
    levels = []
    for line in text.splitlines():
        if CODE_FENCE.match(line.strip()):
            fenced = not fenced
            continue
        if fenced or line.startswith("    "):
            continue
        match = re.match(r"^(#{1,6}) ", line)
        if match:
            levels.append(len(match.group(1)))
    return min(levels, default=1)


def html_tables_to_markdown(text: str) -> str:
    """Replace each pandas DataFrame HTML repr with an equivalent pipe table.

    nbconvert prefers a cell's ``text/html`` output, so every DataFrame arrives as a
    ``<div><style>...</style><table>...</table></div>`` block. Pandoc's LaTeX writer
    *discards* raw HTML rather than failing on it, so leaving these alone would drop
    the result tables out of the PDF without a word of warning -- which is how the
    first build lost the cross-framework comparison entirely.
    """
    import pypandoc

    def convert(match: re.Match) -> str:
        block = re.sub(r"<style.*?</style>", "", match.group(0), flags=re.S)
        if "<table" not in block:
            return match.group(0)
        table = pypandoc.convert_text(
            block, to="markdown-simple_tables-multiline_tables-grid_tables+pipe_tables",
            format="html", extra_args=["--wrap=none"],
        )
        return "\n" + table.strip() + "\n"

    return re.sub(r"<div>\s*\n.*?\n</div>", convert, text, flags=re.S)


def _cells(row: str) -> list[str]:
    return [c.strip() for c in row.strip().strip("|").split("|")]


def _longest_word(cell: str) -> int:
    """Width of the longest run with no break opportunity.

    A column narrower than this overflows however the rest is balanced: LaTeX breaks a
    cell at spaces only -- not at a slash, and not at the underscore in
    ``precision_macro`` -- so an unbreakable cell runs into the next column.
    """
    return max((len(w) for w in cell.split()), default=0)


#: Roughly how many Cambria characters fit across the text block at each size, at the
#: 11pt body size this build uses: 168 mm portrait, 255 mm landscape, average glyph
#: about 0.46 em. Tables are sized against these rather than left to overflow.
FOOTNOTESIZE_COLUMNS = 110
SCRIPTSIZE_COLUMNS = 128


def proportional_tables(text: str, total: int = 96) -> str:
    """Rewrite each pipe table's separator row so the dashes reflect content width.

    Pandoc derives LaTeX column widths from the number of dashes in the separator, not
    from the cells. A DataFrame converted from HTML comes back with a uniform
    separator, which gives ``train_seconds`` and ``Scratch (NumPy)`` the same width and
    wraps the latter to four lines. Widths are recomputed here from the content of each
    column, with a floor at the longest unbreakable token so no column is given less
    room than it can possibly use. Adapted from ass2/report/build_latex.py.

    A table whose floors do not fit even so -- ``all_runs.csv`` is thirteen columns
    wide -- is wrapped in \\scriptsize rather than allowed to run into the margin.
    """
    lines = text.splitlines()
    out: list[str] = []
    index = 0
    fenced = False

    while index < len(lines):
        line = lines[index]
        if CODE_FENCE.match(line.strip()):
            fenced = not fenced
            out.append(line)
            index += 1
            continue

        is_table_head = (
            not fenced
            and line.strip().startswith("|")
            and index + 1 < len(lines)
            and SEPARATOR_ROW.match(lines[index + 1])
        )
        if not is_table_head:
            out.append(line)
            index += 1
            continue

        header, separator = line, lines[index + 1]
        body = []
        scan = index + 2
        while scan < len(lines) and lines[scan].strip().startswith("|"):
            body.append(lines[scan])
            scan += 1

        alignments = _cells(separator)
        widths = [len(c) for c in _cells(header)]
        floors = [_longest_word(c) for c in _cells(header)]
        for row in body:
            for i, cell in enumerate(_cells(row)):
                if i < len(widths):
                    # A long cell wraps, so its full length overstates the room it
                    # needs; the exponent keeps one prose column from starving the rest.
                    widths[i] = max(widths[i], int(len(cell) ** 0.62))
                    floors[i] = max(floors[i], _longest_word(cell))

        needed = 0
        if len(alignments) == len(widths) and sum(widths):
            # Padding: \tabcolsep either side of every column.
            needed = sum(floors) + 2 * len(widths)
            widths = [max(w, f) for w, f in zip(widths, floors)]

            # The dash counts are read as proportions, so only their ratio matters;
            # `total` sets the granularity the rounding works at.
            scale = max(total, needed) / sum(widths)
            rebuilt = []
            for width, marker in zip(widths, alignments):
                left, right = marker.startswith(":"), marker.endswith(":")
                dashes = max(3, round(width * scale)) - int(left) - int(right)
                rebuilt.append((":" if left else "") + "-" * max(3, dashes) + (":" if right else ""))
            separator = "|" + "|".join(rebuilt) + "|"

        # Raw LaTeX blocks, which the reader keeps because raw_tex is on. A group
        # rather than a declaration, so the size change ends with the table. A table
        # too wide even at \scriptsize -- all_runs.csv is thirteen columns -- gets a
        # landscape page of its own rather than six points of type.
        size = r"\footnotesize" if needed <= FOOTNOTESIZE_COLUMNS else r"\scriptsize"
        turn = needed > SCRIPTSIZE_COLUMNS
        out.append("")
        if turn:
            out.append(LSCAPE_OPEN)
        out.extend(["", r"\begingroup" + size, "", header, separator, *body,
                    "", r"\endgroup", ""])
        if turn:
            out.append(LSCAPE_CLOSE)
        out.append("")
        index = scan

    return "\n".join(out)


def drop_rule_before_heading(text: str) -> str:
    """Remove a ``---`` rule that only separates one notebook part from the next.

    Every H1 already opens a fresh page (see preamble.tex), so the rule would print as
    a stray line across the top of an otherwise empty margin.
    """
    return re.sub(r"^-{3,}\s*\n(\s*\n)*(?=#)", "", text, flags=re.M)


def drop_figure_alt_text(text: str) -> str:
    """``![png](f.png)`` -> ``![](f.png)``.

    nbconvert names every image after its MIME subtype. Pandoc turns a Markdown image
    that has alt text into a *figure* and prints that alt text as the caption, so the
    plots would each arrive captioned "png".
    """
    return re.sub(r"!\[(png|jpeg|jpg|svg)\]\(", "![](", text)


def ascii_arrows_in_code(text: str) -> str:
    """Consolas has no U+21D2. It is the one character in these documents the mono
    font cannot draw, and it only appears inside fenced blocks, where the
    \\newunicodechar mapping in preamble.tex does not reach."""
    out, fenced = [], False
    for line in text.splitlines():
        if CODE_FENCE.match(line.strip()):
            fenced = not fenced
        elif fenced or line.startswith("    "):
            line = line.replace("⇒", "=>")
        out.append(line)
    return "\n".join(out)


def prepare(name: str, src: Path, work: Path) -> str:
    if src.suffix == ".ipynb":
        text = notebook_to_markdown(name, src, work)
    else:
        text = src.read_text(encoding="utf-8")

    text = strip_leading_title(text)
    text = html_tables_to_markdown(text)
    text = proportional_tables(text)
    text = drop_rule_before_heading(text)
    text = drop_figure_alt_text(text)
    text = ascii_arrows_in_code(text)
    return text


# ---------------------------------------------------------------------------------
# Title page
# ---------------------------------------------------------------------------------

def tex_escape(value: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")):
        value = value.replace(a, b)
    return value


def title_page(name: str) -> str:
    """The LaTeX counterpart of cover.cover_html(), reading the same fields.

    Identity rows still holding a ``TODO`` placeholder are left off rather than
    printed: an unfilled field is more conspicuous on a cover than a missing one.
    """
    title, subtitle = cover.TITLES.get(name, (name, ""))
    filled = lambda v: not v.strip().startswith("TODO")
    rows = [(k, v) for k, v in (
        (cover.LABELS["student"], cover.STUDENT),
        (cover.LABELS["student_id"], cover.STUDENT_ID),
        (cover.LABELS["class"], cover.CLASS_NAME),
        (cover.LABELS["instructor"], cover.INSTRUCTOR),
    ) if filled(v)]

    institution = (rf"{{\large\scshape\color{{accent}}{tex_escape(cover.INSTITUTION)}\par}}"
                   "\n\\vspace{0.3cm}" if filled(cover.INSTITUTION) else "")

    meta = ""
    if rows:
        body = r" \\[0.5em] ".join(
            rf"\textbf{{{tex_escape(k)}}} & {tex_escape(v)}" for k, v in rows)
        meta = ("\\vspace{1.4cm}\n"
                "\\begin{tabular}{@{}ll@{}}\n" + body + "\n\\end{tabular}\n")

    report_kicker = tex_escape(cover.LABELS["report"])

    return rf"""
\begin{{titlepage}}
\centering
\sffamily
\vspace*{{1.4cm}}

{institution}
{{\normalsize\color{{captiontext}}{tex_escape(cover.SUBJECT)}\par}}

\vspace{{2.2cm}}
{{\footnotesize\color{{accentmid}}\MakeUppercase{{{report_kicker}}}\par}}
\vspace{{0.5cm}}
{{\color{{accent}}\rule{{\linewidth}}{{2.2pt}}}}
\vspace{{0.8cm}}

{{\Huge\bfseries\color{{accent}}{tex_escape(title)}\par}}

\vspace{{0.8cm}}
{{\color{{accent}}\rule{{\linewidth}}{{2.2pt}}}}

\vspace{{0.9cm}}
{{\large\color{{captiontext}}\begin{{minipage}}{{0.82\linewidth}}\centering
{tex_escape(subtitle)}
\end{{minipage}}\par}}

\vspace{{1.1cm}}
{{\normalsize\color{{captiontext}}{tex_escape(cover.ASSIGNMENT)}\par}}

{meta}
\vfill
{{\footnotesize\color{{captiontext}}{tex_escape(cover.DATE)}\par}}
\vspace{{0.4cm}}
{{\color{{accent}}\rule{{0.3\linewidth}}{{0.9pt}}}}
\end{{titlepage}}
\rmfamily
"""


# ---------------------------------------------------------------------------------
# Pandoc and XeLaTeX
# ---------------------------------------------------------------------------------

def run_pandoc(markdown: str, name: str, work: Path) -> str:
    """Produce a complete LaTeX document using pandoc's own template.

    Standalone mode matters: the template emits the helper macros pandoc's table and
    image output depends on (\\real, \\tightlist, \\pandocbounded). Pasting a body
    fragment into a hand-rolled wrapper produces longtable and \\noalign errors.
    """
    import pypandoc

    source, head = DOCS[name]
    settings = ["\\renewcommand{\\runninghead}{" + head + "}"]
    # A chapter of a prose document opens a page; a section of a notebook does not --
    # see the \sectionbreak note in preamble.tex.
    if not source.endswith(".ipynb"):
        settings.append("\\renewcommand{\\sectionbreak}{\\clearpage}")

    title_tex = work / f"{name}_title.tex"
    title_tex.write_text(title_page(name) + "\n" + "\n".join(settings) + "\n",
                         encoding="utf-8", newline="\n")

    latex = pypandoc.convert_text(
        markdown,
        to="latex",
        # yaml_metadata_block off: notebook cells end on a `---` rule, and pandoc reads
        # `---` followed by prose as the start of a YAML metadata block -- which fails
        # the conversion outright ("could not find expected ':'").
        format="markdown-yaml_metadata_block"
               "+pipe_tables+backtick_code_blocks+tex_math_dollars+raw_tex",
        extra_args=[
            "--standalone",
            f"--include-in-header={PREAMBLE}",
            f"--include-before-body={title_tex}",
            "--wrap=preserve",
            "--top-level-division=section",   # '# 1. The data' -> \section
            f"--shift-heading-level-by={1 - shallowest_heading(markdown)}",
            "--toc", "--toc-depth=2",
            "--syntax-highlighting=tango",
            "-V", "documentclass=article",
            "-V", "fontsize=11pt",
            "-V", "papersize=a4",
            "-V", "geometry:margin=2.1cm",
            "-V", f"mainfont={MAIN_FONT}",
            "-V", f"sansfont={SANS_FONT}",
            "-V", f"monofont={MONO_FONT}",
            "-V", "monofontoptions=Scale=0.86",
            "-V", "colorlinks=true",
        ],
    )

    return (latex.replace(LSCAPE_OPEN, r"\begin{landscape}")
                 .replace(LSCAPE_CLOSE, r"\end{landscape}"))


_TEX_ERROR = re.compile(r"^! ", re.M)


def run_xelatex(tex: Path) -> list[str]:
    """Two passes, so \\tableofcontents has page numbers on the second.

    nonstopmode rather than halt-on-error: a missing glyph or an overfull box should
    not cost the whole document. Genuine ``!`` errors are collected and reported.
    tectonic takes neither flag -- it runs to a fixed point on its own and reports
    errors the same way in its output.
    """
    tectonic = ENGINE.name.startswith("tectonic")
    if tectonic:
        cmd = [str(ENGINE), "-X", "compile", "--keep-logs", "--outdir",
               str(tex.parent), tex.name]
        passes = 1
    else:
        cmd = [str(ENGINE), "-interaction=nonstopmode", "--enable-installer",
               f"-output-directory={tex.parent}", tex.name]
        passes = 2

    errors: list[str] = []
    for _ in range(passes):
        proc = subprocess.run(
            cmd, cwd=tex.parent, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
        out = (proc.stdout or "") + ("\n" + proc.stderr if tectonic and proc.stderr else "")
        errors = [ln.strip() for ln in out.splitlines() if _TEX_ERROR.match(ln)]
    return errors


def build(name: str, tex_only: bool) -> bool:
    src = ROOT / DOCS[name][0]
    if not src.exists():
        print(f"  skip {name}: {src.name} not found")
        return False

    work = BUILD / name
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    print(f"  {name}: reading {src.name} ...", flush=True)
    markdown = prepare(name, src, work)
    latex = run_pandoc(markdown, name, work)

    tex = work / f"{name}.tex"
    # newline="\n": pandoc emits CRLF on Windows, and a CRLF inside a Verbatim
    # environment puts a stray ^^M in the output.
    tex.write_text(latex, encoding="utf-8", newline="\n")
    if tex_only:
        print(f"      wrote {tex.relative_to(ROOT)}")
        return True

    print(f"      {ENGINE.stem} ...", flush=True)
    errors = run_xelatex(tex)

    produced = work / f"{name}.pdf"
    if not produced.exists():
        print(f"      FAILED -- no PDF. First errors:")
        for line in errors[:8]:
            print(f"        {line}")
        return False

    PDF_DIR.mkdir(exist_ok=True)
    shutil.copy2(produced, PDF_DIR / f"{name}.pdf")
    size = (PDF_DIR / f"{name}.pdf").stat().st_size / 1024
    note = f"  ({len(errors)} TeX error(s))" if errors else ""
    print(f"      -> pdf/{name}.pdf  ({size:,.1f} KB, {page_count(PDF_DIR / f'{name}.pdf')} pages){note}")
    for line in errors[:5]:
        print(f"        {line}")
    return not errors


def page_count(pdf: Path) -> str:
    """Read from the .aux-free side: xelatex reports no total, and counting
    ``/Type/Page`` in the file fails once the cross-reference stream is compressed."""
    try:
        import pypdf
        return str(len(pypdf.PdfReader(str(pdf)).pages))
    except Exception:
        return "?"


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tex_only = "--tex-only" in sys.argv
    keep = "--keep" in sys.argv

    names = [n for n in DOCS if not args or any(a in n for a in args)]
    if not names:
        print(f"no document matches {args}; known: {', '.join(DOCS)}")
        return 1

    if ENGINE is None or not ENGINE.exists():
        print("no TeX engine found -- install MiKTeX/TeX Live (xelatex) or tectonic, "
              "or point ASS4_TEX at one")
        return 1

    print(f"Assignment 4 — LaTeX build ({len(names)} document(s)), "
          f"engine {ENGINE.stem}, fonts {MAIN_FONT} / {SANS_FONT} / {MONO_FONT}")
    ok = [build(n, tex_only) for n in names]

    if not keep and not tex_only:
        shutil.rmtree(BUILD, ignore_errors=True)

    failed = [n for n, good in zip(names, ok) if not good]
    if failed:
        print(f"\n{len(failed)} document(s) need attention: {', '.join(failed)}")
        print("Re-run with --keep to inspect report/build/<name>/<name>.log")
        return 1
    print("\nAll documents built cleanly.")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
