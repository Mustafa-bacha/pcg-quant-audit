"""Aggregate every run folder into the paper's result tables (Markdown + CSV).

    python scripts/report.py [--runs-dir runs] [--out results/RESULTS.md]

Tables
  T1  FP32 baselines: WA / AUROC / Present recall / cost per model, mean +- sd over folds x seeds
  T2  Compression: per model x condition, mean over folds x seeds, plus delta vs FP32
  T3  Calibration & selective prediction: ECE / Brier / NLL / AURC per model x condition (from audit/)
  T4  Subgroups (H1): sensitivity per murmur grade x condition, and grade-I minus grade-II/III gap
  T5  Explanations: drift vs FP32 and phase relevance per model x condition x method (from audit/)
  T6  Shift (H2): PhysioNet 2016 AUROC, SNR sweep deltas, clipping rates (from shift/)
  T7  Size and latency per condition
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def mean_sd(df: pd.DataFrame, by: list[str], cols: list[str]) -> pd.DataFrame:
    g = df.groupby(by)[cols]
    m, s, n = g.mean(), g.std(), g.size()
    out = m.copy()
    for c in cols:
        out[c] = [f"{a:.3f} ± {b:.3f}" if not np.isnan(b) else f"{a:.3f}" for a, b in zip(m[c], s[c])]
    out["n_runs"] = n
    return out.reset_index()


def load_runs(runs: Path) -> list[Path]:
    return sorted(p.parent for p in runs.glob("*/fold*/seed*/patient_test.json"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out", default="results/RESULTS.md")
    args = ap.parse_args()
    runs = Path(args.runs_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    md = [f"# Results (auto-generated from `{runs}/`)\n"]
    csvdir = out.parent

    def meta(rd: Path) -> dict:
        return {"model": rd.parts[-3], "fold": int(rd.parts[-2][4:]), "seed": int(rd.parts[-1][4:])}

    run_dirs = load_runs(runs)
    if not run_dirs:
        print("no finished runs")
        return

    # T1
    rows = [meta(rd) | json.loads((rd / "patient_test.json").read_text()) for rd in run_dirs]
    t1 = pd.DataFrame(rows)
    t1.to_csv(csvdir / "t1_fp32_runs.csv", index=False)
    md += ["## T1. FP32 baselines (patient-level, held-out test folds)\n",
           mean_sd(t1, ["model"], ["weighted_accuracy", "auroc", "present_recall", "cost"]).to_markdown(index=False), ""]

    # T2 + T7
    rows, lat = [], []
    for rd in run_dirs:
        s = rd / "quant" / "summary.json"
        if not s.exists():
            continue
        summ = json.loads(s.read_text())
        for cond, e in summ["conditions"].items():
            rows.append(meta(rd) | {"cond": cond} | e["metrics"] | {k: v for k, v in e.get("vs_fp32", {}).items()})
            lat.append(meta(rd) | {"cond": cond, "size_kb": e.get("size_bytes", np.nan) / 1024 if e.get("size_bytes") else np.nan,
                                   "latency_b1_ms": e.get("latency_b1", {}).get("median_ms", np.nan)})
    if rows:
        t2 = pd.DataFrame(rows)
        base = t2[t2["cond"] == "fp32"].set_index(["model", "fold", "seed"])
        for m in ("weighted_accuracy", "auroc", "present_recall", "cost"):
            t2[f"d_{m}"] = t2[m].values - base.loc[list(zip(t2["model"], t2["fold"], t2["seed"])), m].values
        t2.to_csv(csvdir / "t2_compression_runs.csv", index=False)
        md += ["## T2. Compression conditions (mean ± sd over folds × seeds; d_ = minus FP32 of the same run)\n",
               mean_sd(t2, ["model", "cond"], ["weighted_accuracy", "d_weighted_accuracy", "auroc", "d_auroc",
                                               "present_recall", "d_present_recall", "rec_top1_agreement"]).to_markdown(index=False), ""]
        t7 = pd.DataFrame(lat)
        md += ["## T7. Size and batch-1 latency (laptop x86 CPU, 4 threads)\n",
               mean_sd(t7, ["model", "cond"], ["size_kb", "latency_b1_ms"]).to_markdown(index=False), ""]

    # T7b: device latency JSONs written by scripts/pi_benchmark.py (laptop_latency.json, pi_latency*.json, arm64_latency*.json)
    dev_rows = []
    for rd in run_dirs:
        for f in list((rd / "quant").glob("*latency*.json")):
            d = json.loads(f.read_text())
            tag = f.stem.replace("_latency", "")
            for r in d.get("results", []):
                dev_rows.append(meta(rd) | {"device": tag, "cpu": d.get("device", {}).get("cpu_model", d.get("device", {}).get("processor", ""))} | {k: v for k, v in r.items() if k != "model"} | {"onnx": r["model"]})
    for f in Path(".").glob("arm64_latency_*.json"):
        d = json.loads(f.read_text())
        for r in d.get("results", []):
            dev_rows.append({"model": f.stem.replace("arm64_latency_", ""), "fold": -1, "seed": -1, "device": "github_arm64",
                             "cpu": d.get("device", {}).get("cpu_model", "")} | {k: v for k, v in r.items() if k != "model"} | {"onnx": r["model"]})
    if dev_rows:
        t7b = pd.DataFrame(dev_rows)
        t7b.to_csv(csvdir / "t7b_device_latency.csv", index=False)
        md += ["## T7b. Device latency from scripts/pi_benchmark.py (batch 1, ONNX Runtime; onnx = file)\n",
               t7b.groupby(["device", "cpu", "model", "onnx", "threads"])[["median_ms", "p90_ms", "size_kb"]].mean().round(3).reset_index().to_markdown(index=False), ""]

    # T3 + T4 from audit/
    ci_rows, sg_rows, sd_rows = [], [], []
    for rd in run_dirs:
        a = rd / "audit"
        if (a / "conditions_ci.csv").exists():
            ci_rows.append(pd.read_csv(a / "conditions_ci.csv").assign(**meta(rd)))
        for f in a.glob("subgroups_*.csv"):
            sg_rows.append(pd.read_csv(f).assign(cond=f.stem.replace("subgroups_", ""), **meta(rd)))
        if (a / "subgroup_deltas.csv").exists():
            sd_rows.append(pd.read_csv(a / "subgroup_deltas.csv").assign(**meta(rd)))
    if ci_rows:
        ci = pd.concat(ci_rows)
        ci.to_csv(csvdir / "t3_conditions_ci_runs.csv", index=False)
        piv = ci.pivot_table(index=["model", "cond"], columns="metric", values="point", aggfunc="mean").round(3)
        md += ["## T3. Calibration and selective prediction (mean of per-run point estimates; CIs per run in CSV)\n",
               piv[[c for c in ("wa", "auroc_present", "sensitivity", "specificity", "ece", "brier", "nll", "aurc") if c in piv.columns]].reset_index().to_markdown(index=False), ""]
    if sg_rows:
        sg = pd.concat(sg_rows)
        sg.to_csv(csvdir / "t4_subgroups_runs.csv", index=False)
        gr = sg[sg["subgroup"] == "grade_group"].pivot_table(index=["model", "cond"], columns="level", values="sensitivity", aggfunc="mean").round(3)
        if {"grade I (soft)", "grade II-III"} <= set(gr.columns):
            gr["gap (II-III minus I)"] = (gr["grade II-III"] - gr["grade I (soft)"]).round(3)
        md += ["## T4. H1: sensitivity by murmur grade (Present patients; mean over runs)\n", gr.reset_index().to_markdown(index=False), ""]
        other = sg[sg["subgroup"].isin(["age_group", "sex", "campaign", "most_audible"])]
        piv = other.pivot_table(index=["model", "subgroup", "level"], columns="cond", values="sensitivity", aggfunc="mean").round(3)
        md += ["### T4b. Sensitivity by other subgroups\n", piv.reset_index().to_markdown(index=False), ""]
    if sd_rows:
        pd.concat(sd_rows).to_csv(csvdir / "t4_subgroup_deltas_runs.csv", index=False)

    # T5
    ex_rows = []
    for rd in run_dirs:
        f = rd / "audit" / "explanations.csv"
        if f.exists():
            ex_rows.append(pd.read_csv(f).assign(**meta(rd)))
    if ex_rows:
        ex = pd.concat(ex_rows)
        ex.to_csv(csvdir / "t5_explanations_windows.csv", index=False)
        cols = [c for c in ("drift_cosine", "drift_spearman", "drift_top10_overlap", "drift_top20_iou", "prr_sys",
                            "prr_sys_control", "prr_sys_uniform", "delta_prr_sys", "deletion_auc_norm", "deletion_auc_norm_random") if c in ex.columns]
        piv = ex.groupby(["model", "cond", "method"])[cols].mean().round(3)
        md += ["## T5. Explanations: drift vs FP32, phase relevance (PRR_sys; control = systole/diastole swapped), deletion faithfulness\n",
               piv.reset_index().to_markdown(index=False), ""]

    # T6
    p16_rows, noise_rows, clip_rows = [], [], []
    for rd in run_dirs:
        s = rd / "shift"
        if (s / "physionet2016.csv").exists():
            p16_rows.append(pd.read_csv(s / "physionet2016.csv").assign(**meta(rd)))
        if (s / "noise.csv").exists():
            noise_rows.append(pd.read_csv(s / "noise.csv").assign(**meta(rd)))
        if (s / "clipping.csv").exists():
            clip_rows.append(pd.read_csv(s / "clipping.csv", index_col=0).rename_axis("input_set").reset_index().assign(**meta(rd)))
    if p16_rows:
        p16 = pd.concat(p16_rows)
        p16.to_csv(csvdir / "t6_physionet2016_runs.csv", index=False)
        md += ["## T6a. Zero-shot PhysioNet 2016 (P(Present) vs abnormal)\n",
               mean_sd(p16, ["model", "cond"], ["auroc", "delta_auroc", "bal_acc_rule", "bal_acc_oracle_youden", "ece", "delta_ece"]).to_markdown(index=False), ""]
    zch_rows = [pd.read_csv(rd / "shift" / "zchsound.csv").assign(**meta(rd)) for rd in run_dirs if (rd / "shift" / "zchsound.csv").exists()]
    if zch_rows:
        zc = pd.concat(zch_rows)
        zc.to_csv(csvdir / "t6_zchsound_runs.csv", index=False)
        cols = [c for c in ("auroc", "delta_auroc", "auroc_clean", "auroc_noisy", "auroc_neonate", "auroc_child",
                            "bal_acc_rule", "bal_acc_oracle_youden", "ece") if c in zc.columns]
        md += ["## T6d. Zero-shot ZCHSound (pediatric, other stethoscope; P(Present) vs echo-confirmed CHD)\n",
               mean_sd(zc, ["model", "cond"], cols).to_markdown(index=False), ""]
        dx = [c for c in zc.columns if c.endswith("_vs_normal")]
        if dx:
            md += ["### T6d'. ZCHSound AUROC per diagnosis vs normal (mean over runs)\n",
                   zc.groupby(["model", "cond"])[dx].mean().round(3).reset_index().to_markdown(index=False), ""]
    if noise_rows:
        nz = pd.concat(noise_rows)
        nz.to_csv(csvdir / "t6_noise_runs.csv", index=False)
        order = ["clean"] + sorted([c for c in nz["snr_db"].astype(str).unique() if c != "clean"], key=lambda s: -float(s))
        nz["snr_db"] = nz["snr_db"].astype(str)
        piv = nz.pivot_table(index=["model", "cond"], columns="snr_db", values="wa", aggfunc="mean").round(3)[order]
        dpiv = nz.pivot_table(index=["model", "cond"], columns="snr_db", values="delta_wa", aggfunc="mean").round(3)[order]
        md += ["## T6b. SNR sweep: WA per condition (columns = SNR dB or clean)\n", piv.reset_index().to_markdown(index=False), "",
               "### T6b'. SNR sweep: WA minus FP32 at the same SNR (the interaction)\n", dpiv.reset_index().to_markdown(index=False), ""]
    if clip_rows:
        cl = pd.concat(clip_rows)
        cl.to_csv(csvdir / "t6_clipping_runs.csv", index=False)
        md += ["## T6c. H2: activation clipping rate of the in-domain-calibrated INT8 PTQ graph\n",
               cl.groupby(["model", "input_set"])[[c for c in ("mean_clip_rate", "max_clip_rate", "nodes_over_1pct", "mean_range_span_99", "mean_frac_at_zero_level") if c in cl.columns]].mean().round(4).reset_index().to_markdown(index=False), ""]

    out.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {out} and CSVs in {csvdir}/")
    print("\n".join(md[:6]))


if __name__ == "__main__":
    main()
