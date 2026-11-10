"""Task 5: quantify the ORT-MinMax operating-point shift (threshold-free vs threshold metrics).

For TinyCNN and M5 under fp32 / ort_int8 (MinMax) / ort_int8_pct (percentile):
pool the 5 test folds of each seed (each patient once), label Present=1 vs
Absent=0 (Unknown excluded), patient score = max over recordings of P(Present),
and report AUROC, AUPRC, and sensitivity at the threshold that yields
specificity >= 0.90 on the pooled set. Mean +- sd over the 3 seeds.

If ranking metrics (AUROC/AUPRC) barely move while default-threshold behaviour
moves a lot (see T3: sens 0.877->0.920, spec 0.898->0.734 on M5), the
operating-point interpretation is established quantitatively.

Usage (repo root):
    .venv/Scripts/python.exe revision_v2/scripts/operating_point.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from pcg.circor import load_metadata  # noqa: E402

MODELS = ["tiny_cnn", "m5"]
CONDS = ["fp32", "ort_int8", "ort_int8_pct"]
SEEDS = [0, 1, 2]
FOLDS = [0, 1, 2, 3, 4]


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    df = pd.DataFrame({"s": s}).groupby("s").ngroup()
    ranks = pd.Series(s).rank(method="average").to_numpy()
    n1, n0 = y.sum(), (1 - y).sum()
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def auprc(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    precision = tp / np.arange(1, len(y) + 1)
    recall = tp / y.sum()
    # average precision
    return float(np.sum(precision * y_sorted) / y.sum())


def sens_at_spec(y: np.ndarray, s: np.ndarray, spec_target: float = 0.90) -> float:
    negs = np.sort(s[y == 0])
    # threshold = smallest t such that spec = mean(neg < t)... use quantile
    thr = np.quantile(negs, spec_target, method="higher")
    return float((s[y == 1] > thr).mean())


def main() -> None:
    meta = load_metadata()[0]  # (patients_df, recordings_df)
    # patient id is the index; murmur label column is 'murmur'
    labels = {str(pid): lab for pid, lab in meta["murmur"].items()}

    rows = []
    for model in MODELS:
        for cond in CONDS:
            per_seed = {"auroc": [], "auprc": [], "sens@spec0.90": []}
            for seed in SEEDS:
                frames = []
                for fold in FOLDS:
                    p = REPO / "runs" / model / f"fold{fold}" / f"seed{seed}" / "quant" / cond / "rec_probs_test.csv"
                    if not p.exists():
                        print(f"[PENDING-RUN] missing {p}")
                        return
                    frames.append(pd.read_csv(p))
                df = pd.concat(frames)
                pat = df.groupby("patient_id")["p_present"].max().reset_index()
                pat["label"] = pat["patient_id"].astype(str).map(labels)
                pat = pat[pat["label"].isin(["Present", "Absent"])]
                y = (pat["label"] == "Present").to_numpy().astype(int)
                s = pat["p_present"].to_numpy()
                per_seed["auroc"].append(auroc(y, s))
                per_seed["auprc"].append(auprc(y, s))
                per_seed["sens@spec0.90"].append(sens_at_spec(y, s))
            rows.append(dict(model=model, cond=cond,
                             **{k: f"{np.mean(v):.3f} +- {np.std(v):.3f}" for k, v in per_seed.items()}))

    out = pd.DataFrame(rows)
    out_path = REPO / "revision_v2" / "operating_point.csv"
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
