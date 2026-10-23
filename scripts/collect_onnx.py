"""Copy the deployment ONNX graphs of one run per model into models/<model>_fold<f>/ so the ARM64 latency
workflow (.github/workflows/arm_latency.yml) benchmarks them on push.

    python scripts/collect_onnx.py                      # fold 0 seed 0 of every model that has quant/
    python scripts/collect_onnx.py --model m5 --fold 1
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pcg import PROJECT_ROOT  # noqa: E402
from pcg.models import MODELS  # noqa: E402

FILES = ("backbone_fp32.onnx", "backbone_int8.onnx", "backbone_ort_int8_pct.onnx")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs-dir", default=str(PROJECT_ROOT / "runs"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "models"))
    args = ap.parse_args()
    for m in args.model:
        q = Path(args.runs_dir) / m / f"fold{args.fold}" / f"seed{args.seed}" / "quant"
        if not (q / "backbone_fp32.onnx").exists():
            print(f"[missing] {q}")
            continue
        dst = Path(args.out) / f"{m}_fold{args.fold}"
        dst.mkdir(parents=True, exist_ok=True)
        for f in FILES:
            if (q / f).exists():
                shutil.copy2(q / f, dst / f)
                print(f"  {dst / f}  {(dst / f).stat().st_size / 1024:.0f} KB")
            else:
                print(f"  [absent] {q / f}")


if __name__ == "__main__":
    main()
