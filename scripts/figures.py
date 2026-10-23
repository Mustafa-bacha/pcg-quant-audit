"""Paper figures from the CSVs written by scripts/report.py (results/t*_runs.csv, results/t8_*.csv) and the
per-run reliability bins in runs/<model>/fold*/seed*/audit/.

    python scripts/figures.py                      # all figures -> results/figures/*.pdf + .png
    python scripts/figures.py --only compression grade

Figures
    fig1_compression   WA / AUROC minus FP32 per condition (mean ± sd over runs, one panel per model)
    fig2_grade         H1 forest: pooled sensitivity delta vs FP32 for grade-I vs grade-II/III murmurs
    fig3_reliability   reliability diagrams pooled over runs, raw and temperature-scaled arms
    fig4_explanations  IG phase relevance per condition, drift vs FP32, deletion faithfulness
    fig5_shift         SNR sweep (WA vs SNR) and zero-shot AUROC deltas on PhysioNet 2016 and ZCHSound
    fig6_latency       model size vs batch-1 latency per device
    fig7_range         INT8 range utilisation of the PTQ graph per input set (the H2 probe)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pcg import PROJECT_ROOT  # noqa: E402

matplotlib.use("Agg")

# Fixed condition order and hues (colour follows the condition everywhere; FP32 is the neutral reference).
CONDS = {
    "fp32": ("FP32", "#52514e"),
    "int8_ptq_pc": ("INT8 PTQ", "#2a78d6"),
    "int8_ptq_pt": ("INT8 PTQ (per-tensor)", "#86b6ef"),
    "int8_qat": ("INT8 QAT", "#1baf7a"),
    "int4_wo_sim": ("INT4 weight-only", "#6250d6"),
    "int8_dynamic": ("INT8 dynamic", "#eda100"),
    "ort_int8": ("ORT INT8 (MinMax)", "#eb6834"),
    "ort_int8_pct": ("ORT INT8 (percentile)", "#e87ba4"),
}
MAIN = ["int8_ptq_pc", "int8_qat", "int4_wo_sim", "ort_int8", "ort_int8_pct"]
MODELS = ["tiny_cnn", "m5", "logmel_cnn", "effnet_b0"]
MODEL_LABEL = {"tiny_cnn": "TinyCNN (27k)", "m5": "M5 (357k)", "logmel_cnn": "LogMel-CNN (308k)", "effnet_b0": "EfficientNet-B0 (4.0M)"}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#9a9995", "axes.linewidth": 0.6,
    "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b", "grid.color": "#e6e5e1", "grid.linewidth": 0.5,
    "axes.grid": True, "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
    "lines.linewidth": 1.6, "lines.markersize": 4, "figure.constrained_layout.use": True,
})


def label(c: str) -> str:
    return CONDS.get(c, (c, "#999999"))[0]


def color(c: str) -> str:
    return CONDS.get(c, (c, "#999999"))[1]


def models_in(df: pd.DataFrame) -> list[str]:
    return [m for m in MODELS if m in set(df["model"])] + sorted(set(df["model"]) - set(MODELS))


def save(fig, out: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {out / name}.pdf/.png")


# --------------------------------------------------------------------------- fig 1

def fig_compression(res: Path, out: Path) -> None:
    f = res / "t2_compression_runs.csv"
    if not f.exists():
        return print("  [skip] fig1: no t2_compression_runs.csv")
    df = pd.read_csv(f)
    n_runs = df[df["cond"] == "fp32"].groupby("model").size()
    df = df[df["cond"].isin(MAIN)]
    models = models_in(df)
    metrics = [("d_weighted_accuracy", "Weighted accuracy − FP32"), ("d_auroc", "AUROC − FP32")]
    fig, axes = plt.subplots(len(metrics), len(models), figsize=(2.6 * len(models) + 0.6, 2.2 * len(metrics)), sharey="row", squeeze=False)
    rng = np.random.default_rng(0)
    for j, m in enumerate(models):
        d = df[df["model"] == m]
        for i, (met, ttl) in enumerate(metrics):
            ax = axes[i, j]
            ax.axhline(0, color="#9a9995", lw=0.8, zorder=1)
            for k, c in enumerate(MAIN):
                v = d.loc[d["cond"] == c, met].dropna().values
                if len(v) == 0:
                    continue
                ax.scatter(k + rng.uniform(-0.18, 0.18, len(v)), v, s=7, color=color(c), alpha=0.45, lw=0, zorder=2)
                ax.errorbar(k, v.mean(), yerr=v.std(ddof=1) if len(v) > 1 else 0, fmt="o", color=color(c), ms=4.5,
                            capsize=2.5, lw=1.2, zorder=3, markeredgecolor="white", markeredgewidth=0.6)
            ax.set_xticks(range(len(MAIN)))
            ax.set_xticklabels([label(c).replace(" (", "\n(").replace("weight-only", "w-only") for c in MAIN], rotation=25, ha="right", fontsize=6)
            if i == 0:
                ax.set_title(f"{MODEL_LABEL.get(m, m)}  (n={int(n_runs.get(m, 0))} runs)")
            if j == 0:
                ax.set_ylabel(ttl)
            ax.grid(axis="x", visible=False)
    fig.suptitle("Compression cost per condition (dots = runs, bar = mean ± sd over folds × seeds)", fontsize=9)
    save(fig, out, "fig1_compression")


# --------------------------------------------------------------------------- fig 2

def fig_grade(res: Path, out: Path) -> None:
    f = res / "t8_pooled_subgroups.csv"
    if not f.exists():
        return print("  [skip] fig2: no t8_pooled_subgroups.csv (run scripts/p5_pooled.py)")
    df = pd.read_csv(f, dtype={"seed": str})
    df = df[(df["subgroup"] == "grade_group") & (df["metric"] == "sensitivity") & df["cond"].isin(MAIN)]
    use_all = (df["seed"] == "all").any()
    df = df[df["seed"] == "all"] if use_all else df
    models = models_in(df)
    levels = [("grade I (soft)", "Grade I/VI (soft)", "o"), ("grade II-III", "Grade II–III", "s")]
    fig, axes = plt.subplots(1, len(models), figsize=(2.3 * len(models) + 0.6, 0.42 * len(MAIN) * 2 + 0.9), sharex=True, squeeze=False)
    for j, m in enumerate(models):
        ax = axes[0, j]
        d = df[df["model"] == m]
        ax.axvline(0, color="#9a9995", lw=0.8)
        yt, yl = [], []
        for k, c in enumerate(MAIN):
            for li, (lev, lname, mk) in enumerate(levels):
                r = d[(d["cond"] == c) & (d["level"] == lev)]
                y = k * 2.6 + li
                yt.append(y)
                yl.append(f"{label(c)}\n{lname}" if li == 0 else lname)
                if r.empty:
                    continue
                if use_all:
                    r = r.iloc[0]
                    ax.plot([r["ci_lo"], r["ci_hi"]], [y, y], color=color(c), lw=1.6, alpha=0.55 if li else 1.0)
                    ax.plot(r["delta"], y, marker=mk, color=color(c), ms=5, markeredgecolor="white", markeredgewidth=0.6)
                else:  # one row per seed: draw each seed thin, mean as the marker
                    for _, rr in r.iterrows():
                        ax.plot([rr["ci_lo"], rr["ci_hi"]], [y, y], color=color(c), lw=0.8, alpha=0.4)
                    ax.plot(r["delta"].mean(), y, marker=mk, color=color(c), ms=5, markeredgecolor="white", markeredgewidth=0.6)
        ax.set_yticks(yt)
        ax.set_yticklabels(yl, fontsize=6)
        ax.invert_yaxis()
        ax.set_title(MODEL_LABEL.get(m, m))
        ax.set_xlabel("Sensitivity − FP32 (pooled test folds)")
        ax.grid(axis="y", visible=False)
    src = "patient-cluster bootstrap over seeds" if use_all else "per-seed pooled bootstrap"
    fig.suptitle(f"H1: does quantization lose the soft murmurs?  (95% paired-bootstrap CI, {src})", fontsize=9)
    save(fig, out, "fig2_grade")


# --------------------------------------------------------------------------- fig 3

def pooled_bins(files: list[Path]) -> pd.DataFrame | None:
    rows = []
    for f in files:
        for b in json.loads(f.read_text()):
            rows.append(b)
    if not rows:
        return None
    d = pd.DataFrame(rows)
    g = d.groupby(["bin_lo", "bin_hi"]).apply(lambda x: pd.Series({"n": x["n"].sum(), "conf": np.average(x["conf"], weights=x["n"]) if x["n"].sum() else np.nan,
                                                                   "acc": np.average(x["acc"], weights=x["n"]) if x["n"].sum() else np.nan}), include_groups=False)
    return g.reset_index()


def fig_reliability(runs: Path, out: Path) -> None:
    conds = ["fp32", "int8_ptq_pc", "int8_qat", "ort_int8", "ort_int8_pct"]
    run_dirs = sorted(p.parent for p in runs.glob("*/fold*/seed*/audit/reliability_fp32.json"))
    if not run_dirs:
        return print("  [skip] fig3: no audit/reliability_*.json")
    models = models_in(pd.DataFrame({"model": [rd.parts[-4] for rd in run_dirs]}))
    fig, axes = plt.subplots(2, len(models), figsize=(2.2 * len(models) + 0.6, 4.4), squeeze=False, sharex=True, sharey=True)
    for j, m in enumerate(models):
        dirs = [rd for rd in run_dirs if rd.parts[-4] == m]
        for i, suffix in enumerate(("", "_ts")):
            ax = axes[i, j]
            ax.plot([0, 1], [0, 1], color="#9a9995", lw=0.8, ls="--")
            for c in conds:
                files = [rd / f"reliability_{c}{suffix}.json" for rd in dirs]
                files = [f for f in files if f.exists()]
                if not files:
                    continue
                b = pooled_bins(files)
                b = b[b["n"] > 0]
                ece = float(np.sum(b["n"] * np.abs(b["acc"] - b["conf"])) / b["n"].sum())
                ax.plot(b["conf"], b["acc"], marker="o", color=color(c), label=f"{label(c)}  ECE {ece:.3f}", ms=3.2, lw=1.3,
                        markeredgecolor="white", markeredgewidth=0.5)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            if i == 0:
                ax.set_title(MODEL_LABEL.get(m, m))
            if i == 1:
                ax.set_xlabel("Mean P(Present) in bin")
            if j == 0:
                ax.set_ylabel("Observed Present rate" + ("\n(temperature-scaled on val fold)" if suffix else "\n(raw)"))
            ax.legend(loc="upper left", frameon=False, handlelength=1.2, fontsize=5.5)
    fig.suptitle("Reliability of P(Present), test patients pooled over folds × seeds", fontsize=9)
    save(fig, out, "fig3_reliability")


# --------------------------------------------------------------------------- fig 4

def fig_explanations(res: Path, out: Path) -> None:
    f = res / "t5_explanations_windows.csv"
    if not f.exists():
        return print("  [skip] fig4: no t5_explanations_windows.csv")
    df = pd.read_csv(f)
    models = models_in(df)
    fig, axes = plt.subplots(len(models), 3, figsize=(9.6, 2.5 * len(models)), squeeze=False)
    for i, m in enumerate(models):
        d = df[df["model"] == m]
        # (a) phase relevance of IG per condition
        ax = axes[i, 0]
        ig = d[d["method"] == "ig"]
        conds = [c for c in ("fp32", "int8_ptq_pc", "int8_qat", "int4_wo_sim") if c in set(ig["cond"])]
        phases = [("prr_s1s2", "S1+S2"), ("prr_sys", "systole"), ("prr_dia", "diastole")]
        w = 0.8 / max(len(conds), 1)
        for k, c in enumerate(conds):
            g = ig[ig["cond"] == c]
            means = [g[p].mean() for p, _ in phases]
            errs = [1.96 * g[p].std(ddof=1) / np.sqrt(g[p].notna().sum()) for p, _ in phases]
            ax.bar(np.arange(len(phases)) + (k - (len(conds) - 1) / 2) * w, means, w * 0.92, yerr=errs, color=color(c), label=label(c),
                   error_kw={"lw": 0.8, "capsize": 1.5}, lw=0)
        uni = ig["prr_sys_uniform"].mean()
        ax.axhline(uni, color="#0b0b0b", lw=0.8, ls=":", label=f"systole share of frames ({uni:.2f})")
        ax.set_xticks(range(len(phases))); ax.set_xticklabels([p for _, p in phases])
        ax.set_ylabel("Share of IG attribution mass")
        ax.set_title(f"{MODEL_LABEL.get(m, m)}: IG phase relevance")
        ax.legend(frameon=False, fontsize=5.5, loc="upper right")
        ax.grid(axis="x", visible=False)
        # (b) drift vs FP32 (cosine) per condition x method
        ax = axes[i, 1]
        dd = d[d["cond"] != "fp32"].dropna(subset=["drift_cosine"])
        methods = [mm for mm in ("ig", "gradxinput", "occlusion") if mm in set(dd["method"])]
        conds = [c for c in ("int8_ptq_pc", "int8_qat", "int4_wo_sim", "ort_int8", "ort_int8_pct") if c in set(dd["cond"])]
        pos, ticks, tl = 0, [], []
        for mm in methods:
            start = pos
            for c in conds:
                v = dd[(dd["method"] == mm) & (dd["cond"] == c)]["drift_cosine"].values
                if len(v) == 0:
                    continue
                bp = ax.boxplot(v, positions=[pos], widths=0.7, patch_artist=True, showfliers=False,
                                medianprops={"color": "white", "lw": 1.2}, boxprops={"facecolor": color(c), "lw": 0}, whiskerprops={"color": color(c), "lw": 0.8}, capprops={"color": color(c), "lw": 0.8})
                pos += 1
            ticks.append((start + pos - 1) / 2); tl.append({"ig": "IG", "gradxinput": "Grad×Input", "occlusion": "Occlusion"}[mm])
            pos += 0.8
        ax.set_xticks(ticks); ax.set_xticklabels(tl)
        ax.set_ylabel("Cosine similarity to FP32 attribution")
        ax.set_ylim(min(0, ax.get_ylim()[0]), 1.02)
        ax.set_title("Drift vs FP32 (colours as in the left panel)")
        ax.grid(axis="x", visible=False)
        # (c) deletion faithfulness
        ax = axes[i, 2]
        dl = d[(d["cond"] == "fp32") & (d["method"] == "ig")].dropna(subset=["deletion_auc_norm"])
        if not dl.empty:
            vals = [dl["deletion_auc_norm"].values, dl["deletion_auc_norm_random"].values]
            ax.boxplot(vals, positions=[0, 1], widths=0.6, patch_artist=True, showfliers=False,
                       medianprops={"color": "white", "lw": 1.2}, boxprops={"facecolor": "#2a78d6", "lw": 0})
            ax.set_xticks([0, 1]); ax.set_xticklabels(["IG order", "random order"])
            ax.set_ylabel("Deletion AUC (lower = more faithful)")
            ax.set_title(f"FP32 IG faithfulness (n={len(dl)} windows)")
            ax.grid(axis="x", visible=False)
    save(fig, out, "fig4_explanations")


# --------------------------------------------------------------------------- fig 5

def fig_shift(res: Path, out: Path) -> None:
    fn, fp, fz = res / "t6_noise_runs.csv", res / "t6_physionet2016_runs.csv", res / "t6_zchsound_runs.csv"
    if not fn.exists():
        return print("  [skip] fig5: no t6_noise_runs.csv")
    noise = pd.read_csv(fn)
    noise = noise[noise["cond"].isin(["fp32"] + MAIN)]
    models = models_in(noise)
    ncol = 2 + (fp.exists() or fz.exists())
    fig, axes = plt.subplots(len(models), ncol, figsize=(3.2 * ncol + 0.4, 2.6 * len(models)), squeeze=False)
    order = ["clean", "20.0", "10.0", "5.0", "0.0"]
    for i, m in enumerate(models):
        d = noise[noise["model"] == m].copy()
        d["snr_db"] = d["snr_db"].astype(str)
        d["snr_db"] = d["snr_db"].replace({"20": "20.0", "10": "10.0", "5": "5.0", "0": "0.0"})
        xs = [o for o in order if o in set(d["snr_db"])]
        # (a) WA vs SNR
        ax = axes[i, 0]
        for c in ["fp32"] + MAIN:
            g = d[d["cond"] == c].groupby("snr_db")["wa"].agg(["mean", "std"]).reindex(xs)
            if g["mean"].isna().all():
                continue
            ax.errorbar(range(len(xs)), g["mean"], yerr=g["std"].fillna(0), color=color(c), label=label(c), marker="o", capsize=1.5, lw=1.2, ms=3,
                        markeredgecolor="white", markeredgewidth=0.5)
        ax.set_xticks(range(len(xs))); ax.set_xticklabels([x.replace(".0", " dB") if x != "clean" else "clean" for x in xs])
        ax.set_ylabel("Weighted accuracy"); ax.set_xlabel("Additive white noise SNR")
        ax.set_title(f"{MODEL_LABEL.get(m, m)}: noise sweep")
        if i == 0:
            ax.legend(frameon=False, fontsize=5.5, ncol=2, loc="lower left")
        # (b) interaction: WA − FP32 at the same SNR
        ax = axes[i, 1]
        ax.axhline(0, color="#9a9995", lw=0.8)
        for c in MAIN:
            g = d[d["cond"] == c].groupby("snr_db")["delta_wa"].agg(["mean", "std"]).reindex(xs)
            if g["mean"].isna().all():
                continue
            ax.errorbar(range(len(xs)), g["mean"], yerr=g["std"].fillna(0), color=color(c), marker="o", capsize=1.5, lw=1.2, ms=3,
                        markeredgecolor="white", markeredgewidth=0.5)
        ax.set_xticks(range(len(xs))); ax.set_xticklabels([x.replace(".0", " dB") if x != "clean" else "clean" for x in xs])
        ax.set_ylabel("WA − FP32, same SNR"); ax.set_xlabel("Additive white noise SNR")
        ax.set_title("Quantization × noise interaction")
        # (c) zero-shot external sets: AUROC per condition
        if ncol == 3:
            ax = axes[i, 2]
            sets = []
            if fp.exists():
                p16 = pd.read_csv(fp); p16 = p16[p16["model"] == m]
                sets.append(("PhysioNet\n2016", p16, "auroc"))
            if fz.exists():
                z = pd.read_csv(fz); z = z[z["model"] == m]
                sets.append(("ZCHSound\nclean", z, "auroc_clean"))
                sets.append(("ZCHSound\nnoisy", z, "auroc_noisy"))
            conds = ["fp32"] + MAIN
            w = 0.8 / len(conds)
            for k, c in enumerate(conds):
                means = [s[1].loc[s[1]["cond"] == c, s[2]].mean() for s in sets]
                sds = [s[1].loc[s[1]["cond"] == c, s[2]].std(ddof=1) for s in sets]
                ax.bar(np.arange(len(sets)) + (k - (len(conds) - 1) / 2) * w, means, w * 0.92, yerr=np.nan_to_num(sds), color=color(c), lw=0,
                       error_kw={"lw": 0.8, "capsize": 1.2}, label=label(c))
            ax.axhline(0.5, color="#9a9995", lw=0.8, ls="--")
            ax.set_xticks(range(len(sets))); ax.set_xticklabels([s[0] for s in sets], fontsize=6)
            ax.set_ylim(0.3, 1.0)
            ax.set_ylabel("Zero-shot AUROC")
            ax.set_title("External sets (no fine-tuning)")
            ax.grid(axis="x", visible=False)
    save(fig, out, "fig5_shift")


# --------------------------------------------------------------------------- fig 6

def fig_latency(res: Path, out: Path) -> None:
    f = res / "t7b_device_latency.csv"
    if not f.exists():
        return print("  [skip] fig6: no t7b_device_latency.csv")
    d = pd.read_csv(f)
    d = d[d["threads"] == 1] if (d["threads"] == 1).any() else d
    d["cond"] = d["onnx"].str.replace("backbone_", "").str.replace(".onnx", "")
    d["cond"] = d["cond"].replace({"int8": "ort_int8"})
    d = d[d["cond"].isin(["fp32", "ort_int8", "ort_int8_pct"])]
    devices = sorted(d["device"].unique())
    fig, axes = plt.subplots(1, len(devices), figsize=(3.0 * len(devices) + 0.5, 2.6), squeeze=False, sharey=True)
    markers = {m: mk for m, mk in zip(MODELS, ["o", "s", "^", "D"])}
    for j, dev in enumerate(devices):
        ax = axes[0, j]
        g = d[d["device"] == dev].groupby(["model", "cond"])[["median_ms", "size_kb"]].median().reset_index()
        for _, r in g.iterrows():
            ax.scatter(r["size_kb"], r["median_ms"], color=color(r["cond"]), marker=markers.get(r["model"], "o"), s=28, edgecolor="white", lw=0.5, zorder=3)
        for m in g["model"].unique():
            gm = g[g["model"] == m].sort_values("size_kb")
            ax.plot(gm["size_kb"], gm["median_ms"], color="#c3c2b7", lw=0.8, zorder=2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("ONNX size (KB)")
        if j == 0:
            ax.set_ylabel("Batch-1 median latency (ms, 1 thread)")
        cpu = d[d["device"] == dev]["cpu"].dropna().astype(str)
        cpu = (cpu.iloc[0][:34] if len(cpu) else {"github_arm64": "Neoverse-N2 (GitHub ARM64 runner)"}.get(dev, ""))
        ax.set_title(f"{dev}\n{cpu}", fontsize=7)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=color(c), marker="o", ls="", label=label(c)) for c in ("fp32", "ort_int8", "ort_int8_pct")]
    h += [Line2D([], [], color="white", markeredgecolor="#52514e", marker=markers[m], ls="", label=MODEL_LABEL[m]) for m in MODELS if m in set(d["model"])]
    fig.legend(handles=h, loc="outside lower center", ncol=4, frameon=False, fontsize=6)
    save(fig, out, "fig6_latency")


# --------------------------------------------------------------------------- fig 7

def fig_range(res: Path, out: Path) -> None:
    f = res / "t6_clipping_runs.csv"
    if not f.exists():
        return print("  [skip] fig7: no t6_clipping_runs.csv")
    d = pd.read_csv(f)
    cols = [c for c in ("mean_range_span_99", "mean_frac_at_zero_level", "mean_clip_rate") if c in d.columns]
    if not cols:
        return print("  [skip] fig7: no range columns")
    models = models_in(d)
    names = {"mean_range_span_99": "99th-pct span / INT8 range", "mean_frac_at_zero_level": "Fraction at the zero level", "mean_clip_rate": "Clipping rate"}
    fig, axes = plt.subplots(1, len(cols), figsize=(3.2 * len(cols) + 0.4, 2.8), squeeze=False)
    sets = [s for s in ("circor_clean", "circor_snr20", "circor_snr10", "circor_snr5", "circor_snr0", "physionet2016", "zchsound") if s in set(d["input_set"])]
    sets += [s for s in d["input_set"].unique() if s not in sets]
    w = 0.8 / max(len(models), 1)
    mc = dict(zip(MODELS, ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]))
    for j, c in enumerate(cols):
        ax = axes[0, j]
        for k, m in enumerate(models):
            g = d[d["model"] == m].groupby("input_set")[c].agg(["mean", "std"]).reindex(sets)
            ax.bar(np.arange(len(sets)) + (k - (len(models) - 1) / 2) * w, g["mean"], w * 0.92, yerr=g["std"].fillna(0), color=mc.get(m, "#999"), lw=0,
                   error_kw={"lw": 0.8, "capsize": 1.2}, label=MODEL_LABEL.get(m, m))
        ax.set_xticks(range(len(sets))); ax.set_xticklabels([s.replace("circor_", "CirCor ").replace("snr", "SNR ") for s in sets], rotation=30, ha="right", fontsize=6)
        ax.set_ylabel(names[c])
        ax.grid(axis="x", visible=False)
        if j == 0:
            ax.legend(frameon=False, fontsize=6)
    fig.suptitle("H2 probe: in-domain-calibrated INT8 PTQ graph on shifted inputs", fontsize=9)
    save(fig, out, "fig7_range")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(PROJECT_ROOT / "results"))
    ap.add_argument("--runs-dir", default=str(PROJECT_ROOT / "runs"))
    ap.add_argument("--out", default=None, help="default <results>/figures")
    ap.add_argument("--only", nargs="*", default=None, choices=["compression", "grade", "reliability", "explanations", "shift", "latency", "range"])
    args = ap.parse_args()
    res, runs = Path(args.results), Path(args.runs_dir)
    out = Path(args.out) if args.out else res / "figures"
    out.mkdir(parents=True, exist_ok=True)
    todo = {"compression": lambda: fig_compression(res, out), "grade": lambda: fig_grade(res, out), "reliability": lambda: fig_reliability(runs, out),
            "explanations": lambda: fig_explanations(res, out), "shift": lambda: fig_shift(res, out), "latency": lambda: fig_latency(res, out),
            "range": lambda: fig_range(res, out)}
    for k, fn in todo.items():
        if args.only is None or k in args.only:
            print(f"[{k}]")
            fn()


if __name__ == "__main__":
    main()
