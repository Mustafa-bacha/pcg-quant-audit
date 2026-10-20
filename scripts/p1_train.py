"""Phase 1 CLI: train FP32 baselines.

Examples (from the project root, with the venv active or via .venv\\Scripts\\python.exe):

  # pipeline smoke test on CPU (2 epochs, 120 recordings)
  python scripts/p1_train.py --model tiny_cnn --folds 0 --seeds 0 --smoke

  # one full run locally
  python scripts/p1_train.py --model tiny_cnn --folds 0 --seeds 0 --epochs 30

  # the whole grid (what the Kaggle notebook runs)
  python scripts/p1_train.py --model tiny_cnn m5 logmel_cnn --folds 0 1 2 3 4 --seeds 0 1 2

Finished runs are skipped, so the grid can be resumed after a session limit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcg.circor import load_metadata  # noqa: E402
from pcg.dataset import load_waveforms  # noqa: E402
from pcg.train import Config, run  # noqa: E402

DEFAULTS = {  # per-model overrides
    "tiny_cnn": {"lr": 2e-3, "mixup_alpha": 0.0},
    "m5": {"lr": 1e-3, "mixup_alpha": 0.0},
    "logmel_cnn": {"lr": 1e-3, "mixup_alpha": 0.2},
    "effnet_b0": {"lr": 3e-4, "mixup_alpha": 0.2},  # ImageNet-pretrained -> lower LR
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=["tiny_cnn"], choices=list(DEFAULTS))
    ap.add_argument("--folds", nargs="+", type=int, default=[0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--crops", type=int, default=4, help="random crops per recording per epoch")
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    pdf, rdf = load_metadata()
    waves = None if args.smoke else load_waveforms(rdf)
    results = []
    for model in args.model:
        for fold in args.folds:
            for seed in args.seeds:
                cfg = Config(model=model, fold=fold, seed=seed, epochs=args.epochs, batch_size=args.batch_size,
                             crops_per_recording=args.crops, device=args.device, num_workers=args.workers,
                             smoke=args.smoke, **DEFAULTS[model])
                if args.runs_dir:
                    cfg.runs_dir = args.runs_dir
                tm = run(cfg, pdf, rdf, waves)
                results.append({"model": model, "fold": fold, "seed": seed} | tm)
    df = pd.DataFrame(results)
    print("\n=== Summary (test folds)")
    print(df[["model", "fold", "seed", "weighted_accuracy", "auroc", "present_recall", "cost"]].to_string(index=False))
    if len(df) > 1:
        print("\n=== Mean per model")
        print(df.groupby("model")[["weighted_accuracy", "auroc", "present_recall", "cost"]].agg(["mean", "std"]).round(3))
    out = Path(args.runs_dir or (Path(__file__).resolve().parent.parent / "runs")) / "p1_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
