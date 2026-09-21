"""Train the M1..M4 variants ahead of running 04_compare.ipynb.

Each model is cached to results/variant_cache/ as soon as it finishes, so an
interrupted run resumes from where it stopped instead of starting over. Run it
as many times as needed:

    python prefill_variants.py            # train whatever is still missing
    python prefill_variants.py --only m1  # just one
    python prefill_variants.py --force    # retrain everything

The notebook calls the same `variants.train_variant`, so it picks these up
automatically and does not retrain.
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import torch

import ass4_utils as U
import variants as V


def main(argv):
    force = "--force" in argv
    only = None
    if "--only" in argv:
        only = argv[argv.index("--only") + 1].lower()

    log_dir = os.path.join(V.HERE, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "prefill_variants.log")
    log_file = open(log_path, "a", buffering=1, encoding="utf-8")

    def log(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        print(msg, file=log_file, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"\n===== prefill run {time.strftime('%Y-%m-%d %H:%M:%S')} =====")
    log(f"device: {device}",
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

    todo = [n for n in V.VARIANTS
            if (only is None or n.split()[0].lower() == only)
            and (force or V.cached(n) is None)]

    if not todo:
        log("nothing to do - every variant is already cached")
        return 0

    log(f"to train: {[n.split()[0] for n in todo]}")

    X_train, y_train, X_test, y_test, meta = U.load_cifar10(verbose=False)
    log(f"data: train {X_train.shape}  test {X_test.shape}")

    for name in todo:
        V.train_variant(name, X_train, y_train, X_test, y_test,
                        len(meta["classes"]), device, log=log, use_cache=not force)

    done = [n.split()[0] for n in V.VARIANTS if V.cached(n)]
    log(f"cached variants now: {done}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
