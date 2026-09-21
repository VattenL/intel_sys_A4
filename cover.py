"""Cover page shared by make_report.py (notebooks) and make_docs_pdf.py (prose docs).

Both pipelines end in Chromium print-to-PDF, so the cover is plain HTML: one
full-height flex block followed by a forced page break. Keeping it here means
the identity fields live in exactly one place.
"""

from __future__ import annotations

# Fill these once; every cover page reads from here.
SUBJECT = "Phát triển Hệ thống Thông minh (Intelligent Systems Development)"
ASSIGNMENT = "Bài tập 4 — So sánh ba cách cài đặt CNN"
STUDENT = "Trần Xuân Thành"
STUDENT_ID = "B23DCAT280"
CLASS_NAME = "E23CNPM01 — Nhóm 02"
INSTRUCTOR = "PGS.TS. Trần Đình Quế"
INSTITUTION = "Học viện Công nghệ Bưu chính Viễn thông — Khoa Công nghệ Thông tin"
DATE = "Tháng 9 năm 2026"

# Nhãn trên trang bìa. build_latex.py đọc lại chính bộ nhãn này, nên bản LaTeX và
# bản Chromium không thể lệch chữ nhau.
LABELS = {
    "report": "Báo cáo",
    "student": "Sinh viên",
    "student_id": "Mã sinh viên",
    "class": "Lớp",
    "instructor": "Giảng viên",
}

# Per-document title and one-line summary shown on the cover. A document absent
# from this map gets no cover page; every document that ships now has an entry.
TITLES = {
    "01_diabetes130": (
        "Diabetes 130-US hospitals",
        "Mạng MLP dự đoán tái nhập viện, cài đặt ba cách — NumPy thuần, "
        "TensorFlow/Keras và PyTorch",
    ),
    "02_mnist": (
        "MNIST",
        "Mạng CNN trên ảnh chữ số viết tay — ba cách cài đặt, cùng một phép "
        "chia dữ liệu và một bộ siêu tham số",
    ),
    "03_cifar10": (
        "CIFAR-10",
        "Mạng CNN trên ảnh màu 32×32 — ba framework, một phép chia dữ liệu, "
        "một bộ siêu tham số",
    ),
    "04_compare": (
        "So sánh và cải tiến mô hình",
        "So sánh ba cách cài đặt trên cả ba bộ dữ liệu, và thí nghiệm tiến hoá "
        "kiến trúc M1–M4",
    ),
    "05_deep_learning_cnn": (
        "Deep Learning và CNN",
        "Hợp thành hàm số, lan truyền ngược, và cấu trúc của mạng tích chập",
    ),
    "06_mnist_lenet": (
        "MNIST với LeNet-5",
        "LeCun và cộng sự (1998) trên ba framework — chạy song song chứ không "
        "thay thế notebook 02",
    ),
    "07_cifar10_lenet": (
        "CIFAR-10 với LeNet-5",
        "Vẫn kiến trúc LeNet-5 ấy, nay áp lên ảnh màu — ba framework, "
        "notebook 03 giữ nguyên",
    ),
    "08_lenet_mnist_report": (
        "LeNet-5 trên MNIST — Báo cáo",
        "Phân tích đầy đủ kết quả, ngân sách tham số, và giới hạn của kiến trúc",
    ),
    "09_lenet_cifar10_report": (
        "LeNet-5 trên CIFAR-10 — Báo cáo",
        "Vì sao cùng một kiến trúc cho hai kết quả trái ngược trên hai bộ ảnh "
        "cùng kích thước",
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
    kicker = LABELS["report"]
    meta = "".join([
        _row(LABELS["student"], STUDENT),
        _row(LABELS["student_id"], STUDENT_ID),
        _row(LABELS["class"], CLASS_NAME),
        _row(LABELS["instructor"], INSTRUCTOR),
    ])
    return f"""<section class="a4-cover">
  <div class="ac-top">
    <div class="ac-inst">{INSTITUTION}</div>
    <div class="ac-subject">{SUBJECT}</div>
  </div>
  <div class="ac-mid">
    <div class="ac-kicker">{kicker}</div>
    <h1 class="ac-title">{title}</h1>
    <p class="ac-sub">{subtitle}</p>
    <div class="ac-assign">{ASSIGNMENT}</div>
  </div>
  <div class="ac-bottom">
    <div class="ac-meta">{meta}</div>
    <div class="ac-date">{DATE}</div>
  </div>
</section>"""
