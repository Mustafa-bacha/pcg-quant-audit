"""Clinical metrics for the revision (audit F4): AUPRC as a first-class metric,
decision-curve / net-benefit analysis, and PPV at community-screening prevalence.

All computed from the per-run recording-probability CSVs already in runs/ (no re-training).
Patient score = max over the patient's recordings of P(Present); patients labelled
Present vs Absent (Unknown excluded), consistent with analysis/operating_point.py.

Outputs:
  analysis/auprc_by_condition.csv      AUPRC + AUROC per model x condition (mean +- sd over seeds)
  analysis/ppv_screening.csv           PPV / referral multiplier at screening prevalences
  results/figures/fig8_dca.{pdf,png}   net-benefit (decision) curves, M5 + EfficientNet-B0

Usage (repo root):
    .venv/Scripts/python.exe analysis/clinical_metrics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from pcg.circor import load_metadata  # noqa: E402

MODELS = ["tiny_cnn", "m5", "logmel_cnn", "effnet_b0"]
SEEDS = [0, 1, 2]
FOLDS = [0, 1, 2, 3, 4]
CONDS = ["fp32", "int8_dynamic", "int8_ptq_pc", "int8_ptq_pt", "int8_qat",
         "int4_wo_sim", "ort_int8", "ort_int8_pct", "ort_int8_pct_sym"]
DCA_CONDS = ["fp32", "int8_ptq_pc", "int8_ptq_pt", "ort_int8", "ort_int8_pct"]
DCA_LABEL = {"fp32": "FP32", "int8_ptq_pc": "INT8 PTQ-PC", "int8_ptq_pt": "INT8 PTQ-PT",
             "ort_int8": "ORT MinMax", "ort_int8_pct": "ORT percentile"}
DCA_COLOR = {"fp32": "#444444", "int8_ptq_pc": "#2a78d6", "int8_ptq_pt": "#86b6ef",
             "ort_int8": "#eb6834", "ort_int8_pct": "#e46aa7"}


def auroc(y, s):
    r = pd.Series(s).rank(method="average").to_numpy()
    n1, n0 = y.sum(), (1 - y).sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def auprc(y, s):
    order = np.argsort(-s)
    ys = y[order]
    tp = np.cumsum(ys)
    precision = tp / np.arange(1, len(y) + 1)
    return float(np.sum(precision * ys) / y.sum()) if y.sum() else np.nan


def patient_scores(model, cond, seed, labels):
    frames = []
    for fold in FOLDS:
        p = REPO / "runs" / model / f"fold{fold}" / f"seed{seed}" / "quant" / cond / "rec_probs_test.csv"
        if not p.exists():
            return None
        frames.append(pd.read_csv(p))
    df = pd.concat(frames)
    pat = df.groupby("patient_id")["p_present"].max().reset_index()
    pat["label"] = pat["patient_id"].astype(str).map(labels)
    pat = pat[pat["label"].isin(["Present", "Absent"])]
    y = (pat["label"] == "Present").to_numpy().astype(int)
    return y, pat["p_present"].to_numpy()


def net_benefit(y, s, thresholds):
    N = len(y)
    nb = []
    for pt in thresholds:
        pred = s >= pt
        tp = np.sum(pred & (y == 1))
        fp = np.sum(pred & (y == 0))
        nb.append(tp / N - fp / N * (pt / (1 - pt)))
    return np.array(nb)


def main():
    meta = load_metadata()[0]
    labels = {str(pid): lab for pid, lab in meta["murmur"].items()}

    # --- AUPRC / AUROC table ---
    rows = []
    for model in MODELS:
        for cond in CONDS:
            aps, aus = [], []
            for seed in SEEDS:
                r = patient_scores(model, cond, seed, labels)
                if r is None:
                    break
                y, s = r
                aps.append(auprc(y, s)); aus.append(auroc(y, s))
            if aps:
                rows.append(dict(model=model, cond=cond,
                                 auprc_mean=np.mean(aps), auprc_sd=np.std(aps),
                                 auroc_mean=np.mean(aus), auroc_sd=np.std(aus)))
    tab = pd.DataFrame(rows)
    tab.to_csv(REPO / "analysis" / "auprc_by_condition.csv", index=False)
    print("== AUPRC (mean +- sd over seeds), Present-vs-Absent, patient-level ==")
    piv = tab.pivot(index="cond", columns="model", values="auprc_mean").reindex(CONDS)[MODELS]
    print(piv.round(3).to_string())

    # --- PPV at screening prevalence (from T3 default-threshold sens/spec) ---
    t3 = pd.read_csv(REPO / "results" / "t3_conditions_ci_runs.csv")
    def sens_spec(model, cond):
        d = t3[(t3.model == model) & (t3.cond == cond)]
        se = d[d.metric == "sensitivity"]["point"].mean()
        sp = d[d.metric == "specificity"]["point"].mean()
        return se, sp
    ppv_rows = []
    for model in ["m5", "tiny_cnn", "effnet_b0"]:
        for cond in ["fp32", "ort_int8", "ort_int8_pct"]:
            se, sp = sens_spec(model, cond)
            for prev in (0.05, 0.10, 0.19):
                ppv = se * prev / (se * prev + (1 - sp) * (1 - prev))
                ppv_rows.append(dict(model=model, cond=cond, prevalence=prev,
                                     sensitivity=round(se, 3), specificity=round(sp, 3),
                                     ppv=round(ppv, 3)))
    ppv = pd.DataFrame(ppv_rows)
    ppv.to_csv(REPO / "analysis" / "ppv_screening.csv", index=False)
    print("\n== PPV at screening prevalence (M5) ==")
    m5p = ppv[(ppv.model == "m5") & (ppv.prevalence == 0.05)]
    for _, r in m5p.iterrows():
        print(f"  M5 {r['cond']:<12} prev 5%: sens {r.sensitivity}, spec {r.specificity}, PPV {r.ppv}")
    f = m5p[m5p.cond == "fp32"].ppv.iloc[0]; mm = m5p[m5p.cond == "ort_int8"].ppv.iloc[0]
    print(f"  -> MinMax referral multiplier at 5% prevalence: {(1/mm)/(1/f):.2f}x "
          f"(PPV {f:.3f} -> {mm:.3f})")

    # --- Decision-curve (net-benefit) figure: M5 + EfficientNet-B0 ---
    thr = np.linspace(0.05, 0.6, 40)
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0), squeeze=False)
    for ax, model, title in zip(axes[0], ["m5", "effnet_b0"], ["M5 (357k)", "EfficientNet-B0 (4.0M)"]):
        # prevalence of the pooled Present-vs-Absent set (seed 0) for treat-all line
        y0, s0 = patient_scores(model, "fp32", 0, labels)
        prev = y0.mean()
        for cond in DCA_CONDS:
            nbs = []
            for seed in SEEDS:
                r = patient_scores(model, cond, seed, labels)
                if r is None:
                    break
                nbs.append(net_benefit(r[0], r[1], thr))
            if nbs:
                ax.plot(thr, np.mean(nbs, axis=0), color=DCA_COLOR[cond], lw=1.8, label=DCA_LABEL[cond])
        ax.plot(thr, prev - (1 - prev) * thr / (1 - thr), color="#9a9995", lw=1.0, ls="--", label="Treat all")
        ax.axhline(0, color="#9a9995", lw=1.0, ls=":", label="Treat none")
        ax.set_xlabel("Threshold probability"); ax.set_title(title, fontweight="bold")
        ax.set_xlim(0.05, 0.6); ax.set_ylim(-0.05, max(0.22, prev + 0.02))
        if model == "m5":
            ax.set_ylabel("Net benefit"); ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle("Decision-curve analysis (patient-level, Present vs Absent)", fontsize=10, fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(REPO / "results" / "figures" / f"fig8_dca.{ext}", bbox_inches="tight")
    print("\nwrote results/figures/fig8_dca.{pdf,png}, analysis/auprc_by_condition.csv, analysis/ppv_screening.csv")


if __name__ == "__main__":
    main()
