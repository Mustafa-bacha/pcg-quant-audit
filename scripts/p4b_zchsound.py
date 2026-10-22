"""Phase 4b: zero-shot evaluation on ZCHSound (pediatric, different stethoscope, echo-confirmed CHD labels)
for every compression condition of every finished run. Companion to p4_shift.py.

    python scripts/p4b_zchsound.py --model tiny_cnn --folds 0 1 2 3 4 --seeds 0 1 2

Per run dir writes shift/zchsound.csv with, per condition: AUROC of P(Present) vs CHD (overall, clean,
noisy, per age group, per diagnosis vs normal), ECE/Brier, sensitivity/specificity at the trained decision
rule, at the CirCor-Youden threshold (from the run's own test fold) and at the oracle ZCHSound-Youden
threshold, deltas vs FP32, and the INT8 clipping/resolution probe on ZCHSound windows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcg import PROJECT_ROOT  # noqa: E402
from pcg import clipping as C  # noqa: E402
from pcg import metrics as M  # noqa: E402
from pcg import quant as Q  # noqa: E402
from pcg import zchsound as Z  # noqa: E402
from pcg.circor import load_metadata as circor_meta  # noqa: E402
from pcg.dataset import WINDOW, _pad_to  # noqa: E402
from pcg.models import MODELS  # noqa: E402
from pcg.runs import AssembledEval, load_conditions  # noqa: E402
from pcg.train import predict_recordings  # noqa: E402

CONDS = ("fp32", "int8_ptq_pc", "int8_ptq_pt", "int8_qat", "int4_wo_sim", "ort_int8")


def youden(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def run_one(run_dir: Path, zdf, zwaves, pdf, args) -> pd.DataFrame:
    out = run_dir / "shift" / "zchsound.csv"
    if out.exists() and not args.force:
        print(f"[skip] {out}")
        return pd.read_csv(out)
    out.parent.mkdir(exist_ok=True)
    fe, conds, refs, cfg = load_conditions(run_dir, want=CONDS)
    conds = {k: conds[k] for k in CONDS if k in conds}
    pct = run_dir / "quant" / "backbone_ort_int8_pct.onnx"
    if pct.exists():  # the accuracy-preserving deployment path, when p2 ran with --ort-mitigations
        conds["ort_int8_pct"] = Q.OrtBackbone(pct, threads=4)
    src_thr = {}
    for cond in conds:
        f = run_dir / "quant" / cond / "rec_probs_test.csv"
        if f.exists():
            t = M.patient_table(pd.read_csv(f, dtype={"patient_id": str, "recording_id": str}), pdf)
            src_thr[cond] = youden(t["y_present"].values, t["p_present"].values)
    y = zdf["chd"].values
    rows = []
    print(f"[{cfg['model']} fold{cfg['fold']} seed{cfg['seed']}] conditions={list(conds)}", flush=True)
    for cond, bb in conds.items():
        rec = predict_recordings(AssembledEval(fe, bb), zdf, zwaves, torch.device("cpu"), batch_windows=128)
        rec = rec.set_index("recording_id").loc[zdf["recording_id"]]
        p = rec["p_present"].values
        pred_rule = (rec[["p_present", "p_unknown", "p_absent"]].values.argmax(1) != 2).astype(int)
        r = {"cond": cond, "n": len(y), "auroc": safe_auc(y, p), "ece": M.ece(p, y), "brier": M.brier(p, y)}
        for sub in ("clean", "noisy"):
            m = (zdf["subset"] == sub).values
            r[f"auroc_{sub}"] = safe_auc(y[m], p[m])
        for ag in ("neonate", "infant", "child"):
            m = (zdf["age_group"] == ag).values
            r[f"auroc_{ag}"] = safe_auc(y[m], p[m])
        for dx in Z.CHD_LABELS:
            m = (zdf["diagnosis"] == dx).values | (y == 0)
            r[f"auroc_{dx.lower()}_vs_normal"] = safe_auc(y[m], p[m])
        thr = {"rule": None, "src_youden": src_thr.get(cond), "oracle_youden": youden(y, p)}
        for tag, th in thr.items():
            if tag != "rule" and th is None:
                continue
            pred = pred_rule if tag == "rule" else (p >= th).astype(int)
            r[f"sens_{tag}"] = float(pred[y == 1].mean())
            r[f"spec_{tag}"] = float(1 - pred[y == 0].mean())
            r[f"bal_acc_{tag}"] = 0.5 * (r[f"sens_{tag}"] + r[f"spec_{tag}"])
        rows.append(r)
        print(f"  {cond:12s} AUROC={r['auroc']:.3f} clean={r['auroc_clean']:.3f} noisy={r['auroc_noisy']:.3f} "
              f"balacc(rule)={r['bal_acc_rule']:.3f} ECE={r['ece']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    base = df[df["cond"] == "fp32"].iloc[0]
    for m in ("auroc", "auroc_clean", "auroc_noisy", "ece", "bal_acc_rule", "bal_acc_oracle_youden"):
        df[f"delta_{m}"] = df[m] - base[m]
    if "int8_ptq_pc" in refs:  # H2 probe on ZCHSound windows with the in-domain-calibrated PTQ graph
        rng = np.random.default_rng(cfg["seed"])
        ids = rng.choice(zdf["recording_id"].values, size=256, replace=False)
        w = []
        for rid in ids:
            x = _pad_to(zwaves[rid], WINDOW)
            s = int(rng.integers(0, len(x) - WINDOW + 1))
            w.append(x[s : s + WINDOW])
        w = torch.from_numpy(np.stack(w))
        summ = C.summarize(C.clip_rates(refs["int8_ptq_pc"], [fe(w[i : i + 64]) for i in range(0, len(w), 64)]))
        print(f"  clip zchsound clip={summ['mean_clip_rate']:.4f} span99={summ['mean_range_span_99']:.3f} "
              f"at_zero={summ['mean_frac_at_zero_level']:.3f}", flush=True)
        for k, v in summ.items():
            df[f"clip_{k}"] = v
    df.to_csv(out, index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=["tiny_cnn"], choices=list(MODELS))
    ap.add_argument("--folds", nargs="+", type=int, default=[0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--runs-dir", default=str(PROJECT_ROOT / "runs"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(8)
    zdf = Z.load_metadata()
    zwaves = Z.load_waveforms(zdf)
    pdf, _ = circor_meta()
    print(f"ZCHSound: {len(zdf)} recordings, CHD prevalence {zdf['chd'].mean():.2f}", flush=True)
    allrows = []
    for m in args.model:
        for f in args.folds:
            for s in args.seeds:
                rd = Path(args.runs_dir) / m / f"fold{f}" / f"seed{s}"
                if not (rd / "quant" / "summary.json").exists():
                    print(f"[missing quant] {rd}")
                    continue
                allrows.append(run_one(rd, zdf, zwaves, pdf, args).assign(model=m, fold=f, seed=s))
    if allrows:
        a = pd.concat(allrows)
        print("\n=== mean over runs")
        print(a.groupby(["model", "cond"])[["auroc", "delta_auroc", "auroc_clean", "auroc_noisy", "ece"]].mean().round(3))


if __name__ == "__main__":
    main()
