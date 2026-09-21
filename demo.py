"""Demo khi bảo vệ: nạp ba mô hình đã lưu và cho chúng đoán cùng một ảnh.

Không huấn luyện lại gì cả. Trọng số đọc thẳng từ ``results/models/<notebook>/``,
đúng những tệp đã được commit cùng mã nguồn.

    python demo.py                          # LeNet-5 trên MNIST, 8 ảnh
    python demo.py --list                   # liệt kê mọi mô hình đang có
    python demo.py --notebook 03_cifar10    # đổi sang CIFAR-10
    python demo.py --samples 12 --seed 7    # số ảnh và hạt ngẫu nhiên
    python demo.py --full                   # dùng bản huấn luyện trên dữ liệu đầy đủ
    python demo.py --eval                   # chấm thêm trên toàn bộ tập test

Ba bản cài đặt nạp lại theo ba cách khác nhau, và đó chính là điều đáng xem:

    Keras  (.keras)  tự dựng lại kiến trúc — nạp một dòng là chạy được
    PyTorch(.pt)     chỉ có state_dict — phải dựng lại lớp nn.Module trước
    Scratch(.npz)    chỉ có mảng NumPy — phải dựng lại S.Sequential trước

Vì vậy tệp này giữ lại định nghĩa kiến trúc của notebook 02/03/06/07. Số tham số
được đối chiếu với sidecar JSON ngay lúc nạp, nên nếu kiến trúc ở đây lệch khỏi
kiến trúc đã huấn luyện thì demo báo lỗi chứ không đoán bừa.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ass4_utils as U  # noqa: E402
import scratch_nn as S  # noqa: E402

MODELS_DIR = os.path.join(HERE, "results", "models")


# ---------------------------------------------------------------------------------
# Kiến trúc — chép đúng từ notebook đã huấn luyện, không phải viết lại theo trí nhớ
# ---------------------------------------------------------------------------------

# n_train/n_test của bản subset phải khớp notebook: chuẩn hoá dùng trung bình và độ
# lệch chuẩn của tập train, nên lấy sai kích thước tập là lấy sai luôn thống kê.
NOTEBOOKS = {
    "02_mnist": dict(
        title="CNN 2conv+fc trên MNIST", dataset="mnist", kind="cnn",
        subset=(10_000, 2_000), ch=(16, 32), kern=3, pad=1, hidden=None,
        prefix="cnn",
    ),
    "03_cifar10": dict(
        title="CNN 2conv+fc trên CIFAR-10", dataset="cifar10", kind="cnn",
        subset=(5_000, 2_000), ch=(32, 64), kern=3, pad=1, hidden=128,
        prefix="cnn",
    ),
    "06_mnist_lenet": dict(
        title="LeNet-5 trên MNIST", dataset="mnist", kind="lenet",
        subset=(10_000, 2_000), ch=(6, 16), kern=5, pad=(2, 0), hidden=(120, 84),
        prefix="lenet5",
    ),
    "07_cifar10_lenet": dict(
        title="LeNet-5 trên CIFAR-10", dataset="cifar10", kind="lenet",
        subset=(5_000, 2_000), ch=(6, 16), kern=5, pad=(0, 0), hidden=(120, 84),
        prefix="lenet5",
    ),
}


def _shapes(cfg, C, H):
    """Kích thước sau từng tầng, tính bằng công thức (H + 2p - k)/s + 1."""
    if cfg["kind"] == "cnn":
        flat = cfg["ch"][1] * (H // 4) * (H // 4)
        return flat
    p1, p3 = cfg["pad"]
    h1 = H + 2 * p1 - cfg["kern"] + 1
    h2 = h1 // 2
    h3 = h2 + 2 * p3 - cfg["kern"] + 1
    h4 = h3 // 2
    return cfg["ch"][1] * h4 * h4


def build_scratch(cfg, C, H, n_classes):
    flat = _shapes(cfg, C, H)
    ch1, ch2 = cfg["ch"]
    if cfg["kind"] == "cnn":
        layers = [
            S.Conv2D(C, ch1, k=cfg["kern"], stride=1, pad=cfg["pad"], seed=1),
            S.ReLU(), S.MaxPool2D(2, 2),
            S.Conv2D(ch1, ch2, k=cfg["kern"], stride=1, pad=cfg["pad"], seed=2),
            S.ReLU(), S.MaxPool2D(2, 2),
            S.Flatten(),
        ]
        if cfg["hidden"]:
            layers += [S.Dense(flat, cfg["hidden"], seed=3), S.ReLU(),
                       S.Dense(cfg["hidden"], n_classes, seed=4)]
        else:
            layers += [S.Dense(flat, n_classes, seed=3)]
        return S.Sequential(layers)

    p1, p3 = cfg["pad"]
    f5, f6 = cfg["hidden"]
    return S.Sequential([
        S.Conv2D(C, ch1, k=cfg["kern"], stride=1, pad=p1, seed=1),
        S.ReLU(), S.MaxPool2D(2, 2),
        S.Conv2D(ch1, ch2, k=cfg["kern"], stride=1, pad=p3, seed=2),
        S.ReLU(), S.MaxPool2D(2, 2),
        S.Flatten(),
        S.Dense(flat, f5, seed=3), S.ReLU(),
        S.Dense(f5, f6, seed=4), S.ReLU(),
        S.Dense(f6, n_classes, seed=5),
    ])


def build_torch(cfg, C, H, n_classes):
    import torch.nn as nn

    flat = _shapes(cfg, C, H)
    ch1, ch2 = cfg["ch"]
    if cfg["kind"] == "cnn":
        head = ([nn.Flatten(), nn.Linear(flat, cfg["hidden"]), nn.ReLU(),
                 nn.Linear(cfg["hidden"], n_classes)]
                if cfg["hidden"] else [nn.Flatten(), nn.Linear(flat, n_classes)])
        pads = (cfg["pad"], cfg["pad"])
    else:
        f5, f6 = cfg["hidden"]
        head = [nn.Flatten(), nn.Linear(flat, f5), nn.ReLU(),
                nn.Linear(f5, f6), nn.ReLU(), nn.Linear(f6, n_classes)]
        pads = cfg["pad"]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(C, ch1, cfg["kern"], padding=pads[0]),
                nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(ch1, ch2, cfg["kern"], padding=pads[1]),
                nn.ReLU(), nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(*head)

        def forward(self, x):
            return self.classifier(self.features(x))

    return Net()


# ---------------------------------------------------------------------------------
# Nạp mô hình
# ---------------------------------------------------------------------------------

def sidecars(notebook: str) -> dict[str, dict]:
    """Mọi mô hình đã lưu của một notebook, đọc từ tệp JSON đi kèm trọng số."""
    d = os.path.join(MODELS_DIR, notebook)
    if not os.path.isdir(d):
        return {}
    out = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".json"):
            with open(os.path.join(d, f), encoding="utf-8") as fh:
                rec = json.load(fh)
            out[rec["name"]] = rec
    return out


def load_leg(name: str, rec: dict, cfg: dict, C: int, H: int, n_classes: int):
    """Nạp một bản cài đặt. Trả về (hàm dự đoán, số tham số) hoặc None nếu thiếu thư viện."""
    framework = rec["framework"]
    try:
        if framework == "keras":
            model = U.load_model(name, cfg["notebook"])
            n_params = int(model.count_params())

            def predict(X):
                logits = model.predict(np.transpose(X, (0, 2, 3, 1)),
                                       batch_size=256, verbose=0)
                return np.asarray(logits).argmax(axis=1)

        elif framework == "torch":
            import torch

            model = U.load_model(name, cfg["notebook"],
                                 model=build_torch(cfg, C, H, n_classes))
            n_params = sum(p.numel() for p in model.parameters())

            def predict(X):
                with torch.no_grad():
                    out = model(torch.from_numpy(X.astype(np.float32)))
                return out.numpy().argmax(axis=1)

        else:
            model = U.load_model(name, cfg["notebook"],
                                 model=build_scratch(cfg, C, H, n_classes))
            n_params = model.n_params()
            model.eval()

            def predict(X):
                return np.asarray(model(X.astype(np.float32))).argmax(axis=1)

    except ImportError as e:
        return None, f"thiếu thư viện ({e.name})"
    except (FileNotFoundError, ValueError) as e:
        return None, str(e)

    if n_params != rec["n_params"]:
        return None, (f"kiến trúc dựng lại có {n_params:,} tham số, "
                      f"trọng số đã lưu có {rec['n_params']:,}")
    return predict, n_params


# ---------------------------------------------------------------------------------
# Trình bày
# ---------------------------------------------------------------------------------

LEG_LABEL = {"scratch": "Scratch (NumPy)", "keras": "TensorFlow/Keras", "torch": "PyTorch"}


def denormalise(X, meta):
    """Đảo chuẩn hoá để vẽ: ảnh đã bị trừ trung bình và chia độ lệch chuẩn."""
    mu = np.array(meta["channel_mean"]).reshape(1, -1, 1, 1)
    sd = np.array(meta["channel_std"]).reshape(1, -1, 1, 1)
    img = X * sd + mu
    return np.clip(img, 0, 1)


def save_figure(X, y, preds, classes, meta, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = denormalise(X, meta)
    n = len(X)
    cols = min(n, 8)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(1.7 * cols, 2.5 * rows), squeeze=False)

    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        ax.axis("off")
        if i >= n:
            continue
        frame = img[i].transpose(1, 2, 0)
        ax.imshow(frame.squeeze(), cmap="gray" if frame.shape[2] == 1 else None)
        lines = [f"thật: {classes[y[i]]}"]
        for leg, p in preds.items():
            mark = "✓" if p[i] == y[i] else "✗"
            lines.append(f"{mark} {LEG_LABEL[leg].split()[0]}: {classes[p[i]]}")
        ok = all(p[i] == y[i] for p in preds.values())
        ax.set_title("\n".join(lines), fontsize=7,
                     color="#14538a" if ok else "#b3261e")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def print_table(y, preds, classes):
    legs = list(preds)
    width = max(len(LEG_LABEL[l]) for l in legs)
    header = f"{'#':>3}  {'nhãn thật':<12}" + "".join(
        f"  {LEG_LABEL[l]:<{width}}" for l in legs)
    print(header)
    print("-" * len(header))
    for i in range(len(y)):
        row = f"{i:>3}  {classes[y[i]]:<12}"
        for leg in legs:
            p = preds[leg][i]
            mark = "✓" if p == y[i] else "✗"
            row += f"  {mark} {classes[p]:<{width - 2}}"
        print(row)
    print("-" * len(header))

    agree = sum(1 for i in range(len(y))
                if len({int(preds[l][i]) for l in legs}) == 1)
    if len(legs) > 1:
        print(f"\n{len(legs)} bản cài đặt trả lời giống nhau ở {agree}/{len(y)} ảnh.")
    else:
        print()
    for leg in legs:
        right = int((preds[leg] == y).sum())
        print(f"  {LEG_LABEL[leg]:<18} đúng {right}/{len(y)}")


# ---------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Demo nạp lại mô hình đã lưu")
    ap.add_argument("--notebook", default="06_mnist_lenet", choices=list(NOTEBOOKS),
                    help="bộ mô hình muốn chạy")
    ap.add_argument("--samples", type=int, default=8, help="số ảnh đem ra đoán")
    ap.add_argument("--seed", type=int, default=U.SEED, help="hạt ngẫu nhiên chọn ảnh")
    ap.add_argument("--full", action="store_true",
                    help="dùng bản huấn luyện trên dữ liệu đầy đủ thay cho bản subset")
    ap.add_argument("--eval", action="store_true",
                    help="chấm thêm trên toàn bộ tập test (chậm hơn)")
    ap.add_argument("--no-figure", action="store_true", help="không xuất ảnh kết quả")
    ap.add_argument("--list", action="store_true", help="liệt kê mô hình đã lưu rồi thoát")
    args = ap.parse_args()

    if args.list:
        for nb, cfg in NOTEBOOKS.items():
            recs = sidecars(nb)
            print(f"\n{nb} — {cfg['title']}")
            if not recs:
                print("  (chưa có mô hình nào — chạy notebook tương ứng trước)")
            for name, r in recs.items():
                print(f"  {name:<26} {r['framework']:<8} {r['n_params']:>8,} tham số  "
                      f"acc {r['test_accuracy']:.4f}  [{r['dataset']}]")
        return 0

    cfg = dict(NOTEBOOKS[args.notebook], notebook=args.notebook)
    recs = sidecars(args.notebook)
    if not recs:
        print(f"Không có mô hình nào trong results/models/{args.notebook}/ — "
              f"chạy {args.notebook}.ipynb trước.")
        return 1

    want = "_full" if args.full else "_subset"
    chosen = {r["framework"]: (n, r) for n, r in recs.items() if n.endswith(want)}
    if not chosen:
        print(f"Không có mô hình nào kết thúc bằng '{want}' trong {args.notebook}. "
              f"Bỏ --full, hoặc xem `python demo.py --list`.")
        return 1

    print(f"\n=== DEMO · {cfg['title']} ===")
    print(f"Bộ trọng số: {'dữ liệu đầy đủ' if args.full else 'subset'}, "
          f"đọc từ results/models/{args.notebook}/\n")

    loader = U.load_mnist if cfg["dataset"] == "mnist" else U.load_cifar10
    n_train, n_test = (None, None) if args.full else cfg["subset"]
    _, _, X_test, y_test, meta = loader(n_train=n_train, n_test=n_test,
                                        seed=U.SEED, verbose=True)
    C, H, _ = meta["shape"]
    classes = meta["classes"]

    legs, failed = {}, {}
    for framework, (name, rec) in sorted(chosen.items()):
        predict, info = load_leg(name, rec, cfg, C, H, len(classes))
        if predict is None:
            failed[framework] = info
            continue
        legs[framework] = (name, rec, predict, info)

    for framework, why in failed.items():
        print(f"  bỏ qua {LEG_LABEL[framework]}: {why}")
    if not legs:
        print("Không nạp được bản cài đặt nào.")
        return 1

    print("\nMô hình đã nạp:")
    for framework, (name, rec, _, n_params) in legs.items():
        print(f"  {LEG_LABEL[framework]:<18} {name:<26} {n_params:>8,} tham số  "
              f"(độ chính xác lúc lưu: {rec['test_accuracy']:.4f})")
    counts = {n for _, _, _, n in legs.values()}
    if len(legs) > 1:
        verdict = (f"cả {len(legs)} bản cùng {counts.pop():,} tham số"
                   if len(counts) == 1 else "SỐ THAM SỐ LỆCH NHAU — kiểm tra lại kiến trúc")
        print(f"  -> {verdict}")

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(X_test))[:args.samples]
    X, y = X_test[idx], y_test[idx]

    print(f"\nĐoán {len(idx)} ảnh lấy ngẫu nhiên từ tập test (seed {args.seed}):\n")
    preds = {}
    for framework, (_, _, predict, _) in legs.items():
        t0 = time.perf_counter()
        preds[framework] = predict(X)
        print(f"  {LEG_LABEL[framework]:<18} {time.perf_counter() - t0:6.2f}s")
    print()
    print_table(y, preds, classes)

    if args.eval:
        print(f"\nChấm trên toàn bộ {len(X_test):,} ảnh test:")
        for framework, (_, rec, predict, _) in legs.items():
            t0 = time.perf_counter()
            acc = float((predict(X_test) == y_test).mean())
            print(f"  {LEG_LABEL[framework]:<18} {acc:.4f}  "
                  f"(sidecar ghi {rec['test_accuracy']:.4f})  "
                  f"{time.perf_counter() - t0:6.1f}s")

    if not args.no_figure:
        out = os.path.join(HERE, "results", f"demo_{args.notebook}.png")
        save_figure(X, y, preds, classes, meta, out, cfg["title"])
        print(f"\nẢnh kết quả: results/demo_{args.notebook}.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
