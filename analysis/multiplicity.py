"""Multiplicity control (audit F3): Benjamini-Hochberg FDR across the full
subgroup x condition x model grid of pooled sensitivity contrasts vs FP32.

The family of inferences is every pooled (seed='all') subgroup-sensitivity delta
in t8_pooled_subgroups.csv for a real compression condition (fp32 self-rows and
degenerate 0.000 cells excluded). We report how many survive BH at q<0.05, and the
BH q-value for the headline contrast (EfficientNet-B0 per-tensor, Grade-I).

Usage (repo root):
    .venv/Scripts/python.exe analysis/multiplicity.py
Outputs: analysis/multiplicity_bh.csv  (every test with raw p and BH q)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values (step-up), order-preserving."""
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * m / (np.arange(1, m + 1))
    # enforce monotonicity from the largest p downward
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(m)
    out[order] = q
    return out


def main():
    df = pd.read_csv(REPO / "results" / "t8_pooled_subgroups.csv", dtype={"seed": str})
    fam = df[(df.seed == "all") & (df.metric == "sensitivity") & (df.cond != "fp32")].copy()
    # drop degenerate cells (delta exactly 0 with 0-width CI: no test performed)
    fam = fam[~((fam.delta == 0) & (fam.ci_lo == 0) & (fam.ci_hi == 0))]
    fam = fam.dropna(subset=["p_two_sided"]).reset_index(drop=True)

    fam["q_bh"] = bh_qvalues(fam["p_two_sided"].to_numpy())
    fam = fam.sort_values("p_two_sided").reset_index(drop=True)
    fam.to_csv(REPO / "analysis" / "multiplicity_bh.csv", index=False)

    m = len(fam)
    sig_raw = int((fam.p_two_sided < 0.05).sum())
    sig_bh = int((fam.q_bh < 0.05).sum())
    print(f"Family size (sensitivity contrasts, seed=all, non-degenerate): m = {m}")
    print(f"Significant at raw p<0.05:  {sig_raw}")
    print(f"Significant at BH q<0.05:   {sig_bh}")

    def show(model, cond, level="grade I (soft)"):
        r = fam[(fam.model == model) & (fam.cond == cond) & (fam.level == level)]
        if len(r):
            r = r.iloc[0]
            print(f"  {model:11} {cond:13} {level:16} delta={r.delta:+.3f} "
                  f"p={r.p_two_sided:.4g} q_BH={r.q_bh:.4g}")

    print("\nHeadline + key contrasts (BH-adjusted):")
    for m_, c_ in [("effnet_b0", "int8_ptq_pt"), ("effnet_b0", "int4_wo_sim"),
                   ("effnet_b0", "int8_qat"), ("effnet_b0", "int8_ptq_pc"),
                   ("m5", "ort_int8"), ("tiny_cnn", "ort_int8"),
                   ("logmel_cnn", "ort_int8_pct_sym")]:
        show(m_, c_)

    print("\nEfficientNet-B0 per-tensor, all subgroup levels (for Table IV):")
    e = fam[(fam.model == "effnet_b0") & (fam.cond == "int8_ptq_pt")]
    for _, r in e.sort_values("q_bh").iterrows():
        print(f"  {r.subgroup:12} {r.level:18} delta={r.delta:+.3f} "
              f"p={r.p_two_sided:.4g} q_BH={r.q_bh:.4g}")
    print(f"\n  max q_BH across the 13 EffNet per-tensor subgroup rows: "
          f"{e.q_bh.max():.4g}")
    print("wrote analysis/multiplicity_bh.csv")


if __name__ == "__main__":
    main()
