"""Phase 5: pooled (out-of-fold) analysis of the compression conditions.

    python scripts/p5_pooled.py --model tiny_cnn m5 logmel_cnn effnet_b0

The per-fold audit (p3_audit.py) gives one paired-bootstrap CI per test fold, with ~20 grade-I patients each,
so the grade-I sensitivity drop is never resolvable fold by fold. Here the five test folds of one seed are
concatenated into a single out-of-fold prediction set (every patient exactly once), and the paired bootstrap
is run on that set. Across seeds the same patient appears once per seed, so the model-level CI uses a
patient-cluster bootstrap: patients are resampled (stratified by murmur label) and all their seed rows go
along, which keeps the per-patient correlation between seeds intact.

Writes
    results/t8_pooled_conditions.csv   per model x (seed | all) x condition x metric: point, CI, paired delta vs FP32
    results/t8_pooled_subgroups.csv    per model x (seed | all) x condition x subgroup level x metric: paired delta
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="All-NaN slice")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcg import PROJECT_ROOT  # noqa: E402
from pcg import metrics as M  # noqa: E402
from pcg.circor import load_metadata  # noqa: E402
from pcg.models import MODELS  # noqa: E402

FOLDS = (0, 1, 2, 3, 4)
MIN_N = 10


def pooled_tables(model_dir: Path, seed: int, pdf: pd.DataFrame) -> dict[str, pd.DataFrame] | None:
    """Concatenate the five test folds of one seed into one patient table per condition."""
    run_dirs = [model_dir / f"fold{f}" / f"seed{seed}" for f in FOLDS]
    if not all((rd / "quant" / "summary.json").exists() for rd in run_dirs):
        return None
    conds = sorted({d.name for rd in run_dirs for d in (rd / "quant").iterdir()
                    if d.is_dir() and (d / "rec_probs_test.csv").exists()})
    conds = [c for c in conds if all((rd / "quant" / c / "rec_probs_test.csv").exists() for rd in run_dirs)]
    tables = {}
    for c in conds:
        parts = []
        for f, rd in zip(FOLDS, run_dirs):
            rec = pd.read_csv(rd / "quant" / c / "rec_probs_test.csv", dtype={"patient_id": str, "recording_id": str})
            parts.append(M.patient_table(rec, pdf).assign(fold=f))
        t = pd.concat(parts)
        assert t.index.is_unique, f"patient appears in two test folds: {model_dir} seed{seed} {c}"
        tables[c] = t
    base = tables["fp32"].index
    return {c: t.loc[base] for c, t in tables.items()}


def cluster_resamples(strata: np.ndarray, clusters: np.ndarray, n: int, seed: int) -> list[np.ndarray]:
    """Bootstrap row-index sets that resample clusters (patients) with replacement, stratified by `strata`
    (one stratum value per cluster), and carry every row of a drawn cluster."""
    rng = np.random.default_rng(seed)
    uniq, first = np.unique(clusters, return_index=True)
    rows_of = {u: np.where(clusters == u)[0] for u in uniq}
    cstrata = strata[first]
    groups = [uniq[cstrata == s] for s in np.unique(cstrata)]
    out = []
    for _ in range(n):
        drawn = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        out.append(np.concatenate([rows_of[u] for u in drawn]))
    return out


def analyse(tables: dict[str, pd.DataFrame], rs_overall, rs_sub, tag: dict) -> tuple[list[dict], list[dict]]:
    base = tables["fp32"]
    cols = {c: M.Cols(t) for c, t in tables.items()}
    crow, srow = [], []
    for cond, c in cols.items():
        for mname, fn in M.METRICS.items():
            r = tag | {"cond": cond, "metric": mname, "n_patients": int(base.index.nunique())} | M.bootstrap_ci(fn, c, resamples=rs_overall)
            if cond != "fp32":
                r |= {f"delta_{k}": v for k, v in M.paired_bootstrap_delta(fn, cols["fp32"], c, resamples=rs_overall).items()}
            crow.append(r)
        if cond == "fp32":
            continue
        for col in M.SUBGROUPS:
            for val in base[col].dropna().unique():
                m = (base[col] == val).values
                if base[m].index.nunique() < MIN_N:
                    continue
                ca, cb = M.Cols(base[m]), M.Cols(tables[cond][m])
                rs = rs_sub(m)
                for mname, fn in (("sensitivity", M.sens), ("specificity", M.spec), ("ece", M.c_ece)):
                    d = M.paired_bootstrap_delta(fn, ca, cb, resamples=rs)
                    if np.isnan(d["delta"]):
                        continue
                    srow.append(tag | {"cond": cond, "subgroup": col, "level": str(val), "n_patients": int(base[m].index.nunique()),
                                       "metric": mname, "base": float(fn(ca))} | d)
    return crow, srow


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--runs-dir", default=str(PROJECT_ROOT / "runs"))
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "results"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    pdf, _ = load_metadata()
    crows, srows = [], []
    for model in args.model:
        per_seed = {}
        for s in args.seeds:
            tabs = pooled_tables(Path(args.runs_dir) / model, s, pdf)
            if tabs is None:
                print(f"[{model} seed{s}] incomplete (need quant/ for all 5 folds)")
                continue
            per_seed[s] = tabs
            c0 = M.Cols(tabs["fp32"])
            rs = M._resamples(c0.strata, args.n_boot, args.seed, True)
            cr, sr = analyse(tabs, rs, lambda m, t=tabs: M._resamples(M.Cols(t["fp32"][m]).strata, args.n_boot, args.seed, False),
                             {"model": model, "seed": str(s)})
            crows += cr
            srows += sr
            g = [r for r in sr if r["subgroup"] == "grade_group" and r["metric"] == "sensitivity"]
            print(f"[{model} seed{s}] {len(tabs)} conditions, {c0.n} patients; grade-I sens delta: " +
                  ", ".join(f"{r['cond']} {r['delta']:+.3f} [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]"
                            for r in g if r["level"] == "grade I (soft)"), flush=True)
        if len(per_seed) > 1:
            conds = set.intersection(*(set(t) for t in per_seed.values()))
            stacked = {c: pd.concat([per_seed[s][c].assign(seed=s) for s in per_seed]) for c in conds}
            base = stacked["fp32"]
            pid = base.index.values
            rs = cluster_resamples(base["murmur"].values, pid, args.n_boot, args.seed)
            cr, sr = analyse(stacked, rs,
                             lambda m, b=base: cluster_resamples(b[m]["murmur"].values, b[m].index.values, args.n_boot, args.seed),
                             {"model": model, "seed": "all"})
            crows += cr
            srows += sr
            g = [r for r in sr if r["subgroup"] == "grade_group" and r["metric"] == "sensitivity"]
            print(f"[{model} all {len(per_seed)} seeds, patient-cluster bootstrap] grade-I sens delta: " +
                  ", ".join(f"{r['cond']} {r['delta']:+.3f} [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] p={r['p_two_sided']:.3f}"
                            for r in g if r["level"] == "grade I (soft)"), flush=True)
    if not crows:
        print("nothing to pool")
        return
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(crows).to_csv(out / "t8_pooled_conditions.csv", index=False)
    pd.DataFrame(srows).to_csv(out / "t8_pooled_subgroups.csv", index=False)
    print(f"wrote {out / 't8_pooled_conditions.csv'} and {out / 't8_pooled_subgroups.csv'}")


if __name__ == "__main__":
    main()
