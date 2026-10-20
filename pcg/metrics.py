"""Phase 3 metrics: calibration, selective prediction, subgroup performance, bootstrap CIs.

Everything works on a *patient-level* table (one row per patient) built by `patient_table()`
from a run's per-recording probabilities, so the FDA's "account for repeated recordings per
patient" requirement is met by construction: bootstrap resamples patients, not recordings.

Primary binary task for calibration/subgroups: Present vs not-Present (Unknown + Absent),
using p_present after patient aggregation. Unknown stays a separate stratum in subgroup tables.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm

from .circor import MURMUR_CLASSES
from .scoring import aggregate_recordings_to_patient, score_arrays


# --------------------------------------------------------------------------- patient table

def patient_table(rec: pd.DataFrame, pdf: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-recording probabilities to one row per patient and join metadata."""
    pids, pprob, ppred = aggregate_recordings_to_patient(
        rec["patient_id"].values, rec[["p_present", "p_unknown", "p_absent"]].values)
    t = pd.DataFrame({"patient_id": pids, "p_present": pprob[:, 0], "p_unknown": pprob[:, 1],
                      "p_absent": pprob[:, 2], "pred": ppred}).set_index("patient_id")
    meta = pdf.loc[t.index, ["murmur", "outcome", "age", "sex", "campaign", "pregnant", "most_audible",
                             "sys_timing", "sys_grading"]]
    t = t.join(meta)
    t["y_present"] = (t["murmur"] == "Present").astype(int)
    t["pred_present"] = (t["pred"] == "Present").astype(int)
    t["age_group"] = t["age"].map({"Neonate": "Neonate+Infant", "Infant": "Neonate+Infant",
                                   "Child": "Child", "Adolescent": "Adolescent"}).fillna("Unknown/Pregnant")
    t["grade_group"] = np.where(t["murmur"] != "Present", "no murmur",
                                np.where(t["sys_grading"].astype(str) == "I/VI", "grade I (soft)", "grade II-III"))
    return t


# --------------------------------------------------------------------------- calibration

def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error, equal-width bins, binary."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def nll(p: np.ndarray, y: np.ndarray, eps: float = 1e-7) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability_bins(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> list[dict]:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        out.append({"bin_lo": float(bins[b]), "bin_hi": float(bins[b + 1]), "n": int(m.sum()),
                    "conf": float(p[m].mean()) if m.any() else None, "acc": float(y[m].mean()) if m.any() else None})
    return out


def temperature_scale(logit_like: np.ndarray, y: np.ndarray, grid=np.linspace(0.25, 4.0, 76)) -> float:
    """Fit a single temperature on *binary logit* values by NLL grid search (post-hoc mitigation arm).
    Call with logit = log(p/(1-p)) from the calibration split, apply T to the test split."""
    best_t, best = 1.0, np.inf
    for t in grid:
        p = 1 / (1 + np.exp(-logit_like / t))
        v = nll(p, y)
        if v < best:
            best, best_t = v, float(t)
    return best_t


# --------------------------------------------------------------------------- selective prediction

def risk_coverage(conf: np.ndarray, correct: np.ndarray) -> dict:
    """Sort by confidence desc; risk(c) = error rate among the c most confident. AURC + risk at fixed coverage."""
    order = np.argsort(-conf)
    err = 1 - correct[order]
    n = len(err)
    cov = np.arange(1, n + 1) / n
    risk = np.cumsum(err) / np.arange(1, n + 1)
    aurc = float(np.trapezoid(risk, cov))
    def risk_at(c: float) -> float:
        k = max(int(np.ceil(c * n)), 1)
        return float(risk[k - 1])
    return {"aurc": aurc, "risk_at_cov_0.8": risk_at(0.8), "risk_at_cov_0.9": risk_at(0.9), "risk_full": float(risk[-1])}


# --------------------------------------------------------------------------- CIs

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (float(c - h), float(c + h))


def bootstrap_ci(fn: Callable[[pd.DataFrame], float], t: pd.DataFrame, n: int = 2000, seed: int = 0,
                 stratify: str | None = "murmur") -> dict:
    """Patient-level (stratified) bootstrap percentile CI of a scalar metric computed on table t."""
    rng = np.random.default_rng(seed)
    groups = [t.index.values] if stratify is None else [g.index.values for _, g in t.groupby(stratify)]
    vals = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        vals.append(fn(t.loc[idx]))
    vals = np.asarray(vals, dtype=float)
    return {"point": float(fn(t)), "ci_lo": float(np.nanpercentile(vals, 2.5)),
            "ci_hi": float(np.nanpercentile(vals, 97.5)), "n_boot": n}


def paired_bootstrap_delta(fn: Callable[[pd.DataFrame], float], ta: pd.DataFrame, tb: pd.DataFrame,
                           n: int = 2000, seed: int = 0, stratify: str | None = "murmur") -> dict:
    """CI for fn(tb) - fn(ta) with the *same* resampled patients in both (paired FP32 vs INT)."""
    assert (ta.index == tb.index).all()
    rng = np.random.default_rng(seed)
    groups = [ta.index.values] if stratify is None else [g.index.values for _, g in ta.groupby(stratify)]
    d = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        d.append(fn(tb.loc[idx]) - fn(ta.loc[idx]))
    d = np.asarray(d, dtype=float)
    return {"delta": float(fn(tb) - fn(ta)), "ci_lo": float(np.nanpercentile(d, 2.5)),
            "ci_hi": float(np.nanpercentile(d, 97.5)), "p_two_sided": float(2 * min((d <= 0).mean(), (d >= 0).mean()))}


# --------------------------------------------------------------------------- metric functions on the table

def m_wa(t: pd.DataFrame) -> float:
    return score_arrays(t["murmur"].tolist(), t["pred"].tolist(), t[["p_present", "p_unknown", "p_absent"]].values,
                        t["outcome"].tolist())["weighted_accuracy"]


def m_auroc_present(t: pd.DataFrame) -> float:
    from sklearn.metrics import roc_auc_score
    y = t["y_present"].values
    return float(roc_auc_score(y, t["p_present"].values)) if 0 < y.sum() < len(y) else float("nan")


def m_sens(t: pd.DataFrame) -> float:
    m = t["y_present"] == 1
    return float(t.loc[m, "pred_present"].mean()) if m.any() else float("nan")


def m_spec(t: pd.DataFrame) -> float:
    m = t["y_present"] == 0
    return float(1 - t.loc[m, "pred_present"].mean()) if m.any() else float("nan")


def m_ece(t: pd.DataFrame) -> float:
    return ece(t["p_present"].values, t["y_present"].values)


def m_brier(t: pd.DataFrame) -> float:
    return brier(t["p_present"].values, t["y_present"].values)


def m_nll(t: pd.DataFrame) -> float:
    return nll(t["p_present"].values, t["y_present"].values)


def m_aurc(t: pd.DataFrame) -> float:
    conf = t[["p_present", "p_unknown", "p_absent"]].values.max(axis=1)
    correct = (t["pred"] == t["murmur"]).values.astype(float)
    return risk_coverage(conf, correct)["aurc"]


METRICS = {"wa": m_wa, "auroc_present": m_auroc_present, "sensitivity": m_sens, "specificity": m_spec,
           "ece": m_ece, "brier": m_brier, "nll": m_nll, "aurc": m_aurc}


# --------------------------------------------------------------------------- subgroup table

SUBGROUPS = {"grade_group": None, "age_group": None, "sex": None, "campaign": None, "most_audible": None,
             "murmur": None}


def subgroup_table(t: pd.DataFrame, min_n: int = 5) -> pd.DataFrame:
    """Sensitivity (Present patients) and specificity (non-Present) per subgroup with Wilson 95% CIs."""
    rows = []
    for col in SUBGROUPS:
        for val, g in t.groupby(col, dropna=False):
            pos, neg = g[g["y_present"] == 1], g[g["y_present"] == 0]
            row = {"subgroup": col, "level": str(val), "n": len(g), "n_present": len(pos)}
            if len(pos) >= min_n:
                k = int(pos["pred_present"].sum())
                lo, hi = wilson(k, len(pos))
                row |= {"sensitivity": k / len(pos), "sens_lo": lo, "sens_hi": hi}
            if len(neg) >= min_n:
                k = int((neg["pred_present"] == 0).sum())
                lo, hi = wilson(k, len(neg))
                row |= {"specificity": k / len(neg), "spec_lo": lo, "spec_hi": hi}
            if len(g) >= 20 and 0 < g["y_present"].sum() < len(g):
                row["auroc_present"] = m_auroc_present(g)
                row["ece"] = m_ece(g)
            rows.append(row)
    return pd.DataFrame(rows)


def compare_conditions(tables: dict[str, pd.DataFrame], base: str = "fp32", n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Overall metrics with CIs for every condition, plus paired deltas vs. the base condition."""
    rows = []
    for cond, t in tables.items():
        for mname, fn in METRICS.items():
            r = {"cond": cond, "metric": mname} | bootstrap_ci(fn, t, n_boot, seed)
            if cond != base:
                r |= {f"delta_{k}": v for k, v in paired_bootstrap_delta(fn, tables[base], t, n_boot, seed).items()}
            rows.append(r)
    return pd.DataFrame(rows)


def subgroup_deltas(tables: dict[str, pd.DataFrame], base: str = "fp32", n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Per-subgroup paired delta of sensitivity/specificity vs. base, for every other condition."""
    rows = []
    tb = tables[base]
    for cond, t in tables.items():
        if cond == base:
            continue
        for col in SUBGROUPS:
            for val in tb[col].dropna().unique():
                ia, ib = tb[tb[col] == val], t[t[col] == val]
                if len(ia) < 10:
                    continue
                for mname, fn in (("sensitivity", m_sens), ("specificity", m_spec), ("ece", m_ece)):
                    try:
                        d = paired_bootstrap_delta(fn, ia, ib, n_boot, seed, stratify=None)
                    except Exception:  # noqa: BLE001
                        continue
                    rows.append({"cond": cond, "subgroup": col, "level": str(val), "n": len(ia), "metric": mname} | d)
    return pd.DataFrame(rows)
