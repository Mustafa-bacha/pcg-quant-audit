"""Phase 3 metrics: calibration, selective prediction, subgroup performance, bootstrap CIs.

Everything works on a *patient-level* table (one row per patient) built by `patient_table()`
from a run's per-recording probabilities, so the FDA's "account for repeated recordings per
patient" requirement is met by construction: bootstrap resamples patients, not recordings.

Primary binary task for calibration/subgroups: Present vs not-Present (Unknown + Absent),
using p_present after patient aggregation. Unknown stays a separate stratum in subgroup tables.

Implementation note: metrics are evaluated on numpy column arrays + an index vector so that a
2,000-resample bootstrap over 8 metrics x 7 conditions runs in seconds, not minutes.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .circor import MURMUR_CLASSES
from .scoring import aggregate_recordings_to_patient

W_MURMUR = np.array([5.0, 3.0, 1.0])  # Present, Unknown, Absent (official weighted-accuracy weights)


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


class Cols:
    """numpy view of a patient table for fast resampled metrics."""

    def __init__(self, t: pd.DataFrame):
        self.n = len(t)
        self.y = t["y_present"].values.astype(np.int8)
        self.p = t["p_present"].values.astype(np.float64)
        self.pred_pos = t["pred_present"].values.astype(np.int8)
        self.true_i = np.array([MURMUR_CLASSES.index(m) for m in t["murmur"]], dtype=np.int8)
        self.pred_i = np.array([MURMUR_CLASSES.index(m) for m in t["pred"]], dtype=np.int8)
        self.conf = t[["p_present", "p_unknown", "p_absent"]].values.max(axis=1)
        self.correct = (self.true_i == self.pred_i).astype(np.float64)
        self.strata = t["murmur"].values


ALL = slice(None)


# --------------------------------------------------------------------------- scalar metrics on (Cols, idx)

def wa(c: Cols, idx=ALL) -> float:
    """Official weighted accuracy: weights by true class (5 Present, 3 Unknown, 1 Absent)."""
    t, p = c.true_i[idx], c.pred_i[idx]
    w = W_MURMUR[t]
    return float((w * (t == p)).sum() / w.sum())


def auroc(c: Cols, idx=ALL) -> float:
    y, s = c.y[idx], c.p[idx]
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def sens(c: Cols, idx=ALL) -> float:
    y, q = c.y[idx], c.pred_pos[idx]
    m = y == 1
    return float(q[m].mean()) if m.any() else float("nan")


def spec(c: Cols, idx=ALL) -> float:
    y, q = c.y[idx], c.pred_pos[idx]
    m = y == 0
    return float(1 - q[m].mean()) if m.any() else float("nan")


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error, equal-width bins, binary."""
    idx = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    e = 0.0
    n = len(p)
    for b in range(n_bins):
        m = idx == b
        k = m.sum()
        if k:
            e += k / n * abs(p[m].mean() - y[m].mean())
    return float(e)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def nll(p: np.ndarray, y: np.ndarray, eps: float = 1e-7) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def c_ece(c: Cols, idx=ALL) -> float:
    return ece(c.p[idx], c.y[idx])


def c_brier(c: Cols, idx=ALL) -> float:
    return brier(c.p[idx], c.y[idx])


def c_nll(c: Cols, idx=ALL) -> float:
    return nll(c.p[idx], c.y[idx])


def risk_coverage(conf: np.ndarray, correct: np.ndarray) -> dict:
    """Sort by confidence desc; risk(c) = error rate among the c most confident. AURC + risk at fixed coverage."""
    order = np.argsort(-conf)
    err = 1 - correct[order]
    n = len(err)
    cov = np.arange(1, n + 1) / n
    risk = np.cumsum(err) / np.arange(1, n + 1)
    aurc = float(np.trapezoid(risk, cov))

    def risk_at(cv: float) -> float:
        k = max(int(np.ceil(cv * n)), 1)
        return float(risk[k - 1])
    return {"aurc": aurc, "risk_at_cov_0.8": risk_at(0.8), "risk_at_cov_0.9": risk_at(0.9), "risk_full": float(risk[-1])}


def c_aurc(c: Cols, idx=ALL) -> float:
    return risk_coverage(c.conf[idx], c.correct[idx])["aurc"]


METRICS: dict[str, Callable[[Cols, object], float]] = {
    "wa": wa, "auroc_present": auroc, "sensitivity": sens, "specificity": spec,
    "ece": c_ece, "brier": c_brier, "nll": c_nll, "aurc": c_aurc}


# pandas-facing convenience wrappers (used by p4_shift and subgroup tables)
def m_wa(t): return wa(Cols(t))
def m_auroc_present(t): return auroc(Cols(t))
def m_sens(t): return sens(Cols(t))
def m_spec(t): return spec(Cols(t))
def m_ece(t): return c_ece(Cols(t))
def m_brier(t): return c_brier(Cols(t))
def m_nll(t): return c_nll(Cols(t))
def m_aurc(t): return c_aurc(Cols(t))


# --------------------------------------------------------------------------- calibration extras

def reliability_bins(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> list[dict]:
    idx = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        out.append({"bin_lo": b / n_bins, "bin_hi": (b + 1) / n_bins, "n": int(m.sum()),
                    "conf": float(p[m].mean()) if m.any() else None, "acc": float(y[m].mean()) if m.any() else None})
    return out


def temperature_scale(logit_like: np.ndarray, y: np.ndarray, grid=np.linspace(0.25, 4.0, 76)) -> float:
    """Fit one temperature on binary logits by NLL grid search (post-hoc mitigation arm)."""
    best_t, best = 1.0, np.inf
    for t in grid:
        v = nll(1 / (1 + np.exp(-logit_like / t)), y)
        if v < best:
            best, best_t = v, float(t)
    return best_t


# --------------------------------------------------------------------------- CIs

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (float(c - h), float(c + h))


def _resamples(strata: np.ndarray, n: int, seed: int, stratify: bool) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    groups = [np.where(strata == s)[0] for s in np.unique(strata)] if stratify else [np.arange(len(strata))]
    return [np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups]) for _ in range(n)]


def bootstrap_ci(fn: Callable, c: Cols, n: int = 2000, seed: int = 0, stratify: bool = True,
                 resamples: list[np.ndarray] | None = None) -> dict:
    """Patient-level (stratified) percentile CI of fn(c, idx)."""
    rs = resamples or _resamples(c.strata, n, seed, stratify)
    vals = np.array([fn(c, idx) for idx in rs], dtype=float)
    return {"point": float(fn(c)), "ci_lo": float(np.nanpercentile(vals, 2.5)),
            "ci_hi": float(np.nanpercentile(vals, 97.5)), "n_boot": len(rs)}


def paired_bootstrap_delta(fn: Callable, ca: Cols, cb: Cols, n: int = 2000, seed: int = 0, stratify: bool = True,
                           resamples: list[np.ndarray] | None = None) -> dict:
    """CI for fn(cb) - fn(ca) with the same resampled patients in both (paired FP32 vs INT)."""
    assert ca.n == cb.n
    rs = resamples or _resamples(ca.strata, n, seed, stratify)
    d = np.array([fn(cb, idx) - fn(ca, idx) for idx in rs], dtype=float)
    return {"delta": float(fn(cb) - fn(ca)), "ci_lo": float(np.nanpercentile(d, 2.5)),
            "ci_hi": float(np.nanpercentile(d, 97.5)),
            "p_two_sided": float(min(1.0, 2 * min(np.nanmean(d <= 0), np.nanmean(d >= 0))))}


# --------------------------------------------------------------------------- tables

SUBGROUPS = ["grade_group", "age_group", "sex", "campaign", "most_audible", "murmur"]


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
    """Overall metrics with CIs for every condition, plus paired deltas vs. the base condition.
    All conditions share the same patient order (same test fold), so one set of resamples is reused."""
    cols = {k: Cols(v.loc[tables[base].index]) for k, v in tables.items()}
    rs = _resamples(cols[base].strata, n_boot, seed, True)
    rows = []
    for cond, c in cols.items():
        for mname, fn in METRICS.items():
            r = {"cond": cond, "metric": mname} | bootstrap_ci(fn, c, resamples=rs)
            if cond != base:
                r |= {f"delta_{k}": v for k, v in paired_bootstrap_delta(fn, cols[base], c, resamples=rs).items()}
            rows.append(r)
    return pd.DataFrame(rows)


def subgroup_deltas(tables: dict[str, pd.DataFrame], base: str = "fp32", n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Per-subgroup paired delta of sensitivity/specificity/ECE vs. base, for every other condition."""
    rows = []
    tb = tables[base]
    for cond, t in tables.items():
        if cond == base:
            continue
        t = t.loc[tb.index]
        for col in SUBGROUPS:
            for val in tb[col].dropna().unique():
                m = (tb[col] == val).values
                if m.sum() < 10:
                    continue
                ca, cb = Cols(tb[m]), Cols(t[m])
                rs = _resamples(ca.strata, n_boot, seed, False)
                for mname, fn in (("sensitivity", sens), ("specificity", spec), ("ece", c_ece)):
                    d = paired_bootstrap_delta(fn, ca, cb, resamples=rs)
                    if np.isnan(d["delta"]):
                        continue
                    rows.append({"cond": cond, "subgroup": col, "level": str(val), "n": int(m.sum()), "metric": mname} | d)
    return pd.DataFrame(rows)
