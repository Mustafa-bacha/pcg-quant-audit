"""Phase 4 CLI: robustness of every compression condition under shift (RQ4, hypothesis H2).

    python scripts/p4_shift.py --model tiny_cnn --folds 0 1 2 3 4 --seeds 0

Three shifts, each reported as the FP32 number and the *interaction* (condition minus FP32):
  A. campaign shift   CirCor test fold split by CC2014 vs CC2015 (same device, different year/sites)
  B. noise sweep      CirCor test fold with additive white noise at SNR 20/10/5/0 dB
                      (plus the H2 probe: PTQ calibrated on noisy windows, see p2_quantize --calib noise)
  C. dataset/device   zero-shot on PhysioNet/CinC 2016 (multi-device adult set): AUROC of P(Present)
                      vs 'abnormal', sensitivity/specificity at the trained rule, at the CirCor-Youden
                      threshold and at the oracle target-Youden threshold (JCM 2026 protocol)

Outputs: runs/<model>/fold<k>/seed<s>/shift/{campaign.csv, noise.csv, physionet2016.csv, summary.json}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcg import PROJECT_ROOT  # noqa: E402
from pcg import metrics as M  # noqa: E402
from pcg import physionet2016 as P16  # noqa: E402
from pcg.circor import load_metadata  # noqa: E402
from pcg.dataset import load_waveforms  # noqa: E402
from pcg.models import MODELS  # noqa: E402
from pcg.runs import AssembledEval, load_conditions  # noqa: E402
from pcg.splits import load_folds  # noqa: E402
from pcg.train import patient_metrics, predict_recordings  # noqa: E402

CONDS = ("fp32", "int8_ptq_pc", "int8_ptq_pt", "int8_qat", "int4_wo_sim", "ort_int8")


def youden(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def add_noise(waves: dict, snr_db: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for k, x in waves.items():
        p_sig = float(np.mean(x**2))
        out[k] = (x + rng.normal(0, np.sqrt(p_sig / 10 ** (snr_db / 10)), size=x.shape)).astype(np.float32)
    return out


def run_one(run_dir: Path, pdf, rdf, waves, p16_df, p16_waves, args) -> dict:
    sdir = run_dir / "shift"
    if (sdir / "summary.json").exists() and not args.force:
        print(f"[skip] {sdir}")
        return json.loads((sdir / "summary.json").read_text())
    sdir.mkdir(exist_ok=True)
    fe, conds, refs, cfg = load_conditions(run_dir, want=CONDS)
    conds = {k: conds[k] for k in CONDS if k in conds}
    folds = load_folds(Path(cfg["split_file"]))
    te = rdf[rdf["patient_id"].map(folds) == cfg["test_fold"]]
    dev = torch.device("cpu")
    print(f"[{cfg['model']} fold{cfg['fold']} seed{cfg['seed']}] conditions={list(conds)}")

    # A + B: CirCor test fold, clean and noisy
    rows_c, rows_n, src_thr = [], [], {}
    for snr in [None] + args.snr:
        w = waves if snr is None else add_noise({k: waves[k] for k in te["recording_id"]}, snr, cfg["seed"])
        for cond, bb in conds.items():
            rec = predict_recordings(AssembledEval(fe, bb), te, w, dev, batch_windows=128)
            t = M.patient_table(rec, pdf)
            pm = patient_metrics(rec, pdf)
            base = {"cond": cond, "snr_db": snr if snr is not None else "clean", "wa": pm["weighted_accuracy"],
                    "auroc": pm["auroc"], "present_recall": pm["present_recall"], "ece": M.m_ece(t), "brier": M.m_brier(t)}
            rows_n.append(base)
            if snr is None:
                src_thr[cond] = youden(t["y_present"].values, t["p_present"].values)
                for camp, g in t.groupby("campaign"):
                    rows_c.append({"cond": cond, "campaign": camp, "n": len(g), "sensitivity": M.m_sens(g),
                                   "specificity": M.m_spec(g), "auroc_present": M.m_auroc_present(g), "ece": M.m_ece(g)})
        print(f"  snr={snr}: " + " ".join(f"{r['cond']}={r['wa']:.3f}" for r in rows_n if r['snr_db'] == (snr if snr is not None else 'clean')), flush=True)
    camp, noise = pd.DataFrame(rows_c), pd.DataFrame(rows_n)
    for df in (camp, noise):
        base = df[df["cond"] == "fp32"].set_index([c for c in ("campaign", "snr_db") if c in df.columns])
        key = [c for c in ("campaign", "snr_db") if c in df.columns]
        for m in ("wa", "auroc", "sensitivity", "specificity", "ece"):
            if m in df.columns:
                df[f"delta_{m}"] = df[m].values - base.loc[df.set_index(key).index, m].values
    camp.to_csv(sdir / "campaign.csv", index=False)
    noise.to_csv(sdir / "noise.csv", index=False)

    # C: PhysioNet 2016 zero-shot
    rows_p = []
    if p16_df is not None:
        y = p16_df["abnormal"].values
        for cond, bb in conds.items():
            rec = predict_recordings(AssembledEval(fe, bb), p16_df, p16_waves, dev, batch_windows=128)
            p = rec.set_index("recording_id").loc[p16_df["recording_id"], "p_present"].values
            pred_rule = (rec.set_index("recording_id").loc[p16_df["recording_id"], ["p_present", "p_unknown", "p_absent"]]
                         .values.argmax(1) != 2).astype(int)  # Present or Unknown -> refer (positive)
            thr_src, thr_orc = src_thr[cond], youden(y, p)
            r = {"cond": cond, "n": len(y), "auroc": float(roc_auc_score(y, p)), "ece": M.ece(p, y), "brier": M.brier(p, y)}
            for tag, pred in (("rule", pred_rule), ("src_youden", (p >= thr_src).astype(int)), ("oracle_youden", (p >= thr_orc).astype(int))):
                r[f"sens_{tag}"] = float(pred[y == 1].mean())
                r[f"spec_{tag}"] = float(1 - pred[y == 0].mean())
                r[f"bal_acc_{tag}"] = 0.5 * (r[f"sens_{tag}"] + r[f"spec_{tag}"])
            for s, g in p16_df.groupby("subset"):
                idx = g.index.values
                if 0 < y[idx].sum() < len(idx):
                    r[f"auroc_subset_{s}"] = float(roc_auc_score(y[idx], p[idx]))
            rows_p.append(r)
            print(f"  p16 {cond:12s} AUROC={r['auroc']:.3f} balacc(rule)={r['bal_acc_rule']:.3f} ECE={r['ece']:.3f}", flush=True)
        p16 = pd.DataFrame(rows_p)
        base = p16[p16["cond"] == "fp32"].iloc[0]
        for m in ("auroc", "ece", "bal_acc_rule", "bal_acc_oracle_youden"):
            p16[f"delta_{m}"] = p16[m] - base[m]
        p16.to_csv(sdir / "physionet2016.csv", index=False)

    # H2: clipping rates of the in-domain-calibrated PTQ graph on clean / noisy / foreign-device windows
    clip = {}
    if "int8_ptq_pc" in refs:
        from pcg import clipping as C
        from pcg.dataset import WINDOW, _pad_to
        rng = np.random.default_rng(cfg["seed"])

        def windows_from(wdict: dict, ids, n=256):
            out = []
            for rid in rng.choice(list(ids), size=min(n, len(ids)), replace=False):
                x = _pad_to(wdict[rid], WINDOW)
                s = int(rng.integers(0, len(x) - WINDOW + 1))
                out.append(x[s : s + WINDOW])
            return torch.from_numpy(np.stack(out))

        sets = {"circor_clean": windows_from(waves, te["recording_id"])}
        for snr in args.snr:
            sets[f"circor_snr{int(snr)}"] = windows_from(add_noise({k: waves[k] for k in te["recording_id"]}, snr, cfg["seed"]), te["recording_id"])
        if p16_waves is not None:
            sets["physionet2016"] = windows_from(p16_waves, p16_df["recording_id"])
        for tag, w in sets.items():
            feats = [fe(w[i : i + 64]) for i in range(0, len(w), 64)]
            rates = C.clip_rates(refs["int8_ptq_pc"], feats)
            clip[tag] = C.summarize(rates) | {"per_node": rates}
            print(f"  clip {tag:16s} clip={clip[tag]['mean_clip_rate']:.4f} max={clip[tag]['max_clip_rate']:.4f} "
                  f"span99={clip[tag]['mean_range_span_99']:.3f} at_zero={clip[tag]['mean_frac_at_zero_level']:.3f}", flush=True)
        pd.DataFrame({t: {k: v for k, v in d.items() if k != "per_node"} for t, d in clip.items()}).T.to_csv(sdir / "clipping.csv")

    summ = {"model": cfg["model"], "fold": cfg["fold"], "seed": cfg["seed"],
            "noise": noise.to_dict(orient="records"), "campaign": camp.to_dict(orient="records"),
            "physionet2016": rows_p, "clipping": clip}
    (sdir / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    return summ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=["tiny_cnn"], choices=list(MODELS))
    ap.add_argument("--folds", nargs="+", type=int, default=[0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--runs-dir", default=str(PROJECT_ROOT / "runs"))
    ap.add_argument("--snr", nargs="+", type=float, default=[20, 10, 5, 0])
    ap.add_argument("--p16-subsets", default="a,b,c,d,e,f", help="PhysioNet 2016 training subsets; '' to skip")
    ap.add_argument("--p16-max", type=int, default=0, help="cap on PhysioNet 2016 recordings (0 = all)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(8)

    pdf, rdf = load_metadata()
    waves = load_waveforms(rdf)
    p16_df = p16_waves = None
    if args.p16_subsets and P16.P16_DIR.exists():
        p16_df = P16.load_metadata(subsets=tuple(args.p16_subsets.split(",")))
        if args.p16_max:
            p16_df = p16_df.groupby("subset", group_keys=False).apply(lambda g: g.sample(min(len(g), args.p16_max // 6), random_state=0)).reset_index(drop=True)
        p16_waves = P16.load_waveforms(p16_df)
        print(f"PhysioNet 2016: {len(p16_df)} recordings, abnormal={p16_df['abnormal'].mean():.2f}")
    for m in args.model:
        for f in args.folds:
            for s in args.seeds:
                rd = Path(args.runs_dir) / m / f"fold{f}" / f"seed{s}"
                if not (rd / "quant" / "summary.json").exists():
                    print(f"[missing quant] {rd}")
                    continue
                run_one(rd, pdf, rdf, waves, p16_df, p16_waves, args)


if __name__ == "__main__":
    main()
