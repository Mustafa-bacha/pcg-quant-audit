"""Phase 3 CLI: trustworthiness audit of every compression condition produced by p2_quantize.py.

    python scripts/p3_audit.py --model tiny_cnn --folds 0 1 2 3 4 --seeds 0
    python scripts/p3_audit.py --model tiny_cnn --folds 0 --seeds 0 --n-windows 60 --n-boot 500   # quick

Part A (RQ1-RQ2, from the saved per-recording probabilities; no model needed):
    audit/conditions_ci.csv        every metric (WA, AUROC, sens, spec, ECE, Brier, NLL, AURC) with
                                   patient-level stratified bootstrap CIs, plus paired deltas vs FP32
    audit/subgroups_<cond>.csv     sensitivity/specificity per grade, age, sex, campaign, site with Wilson CIs
    audit/subgroup_deltas.csv      paired deltas vs FP32 per subgroup (the H1 table: grade x precision)
    audit/reliability_<cond>.json  reliability-diagram bins

Part B (RQ3, needs the models): on N test windows from recordings with an audible murmur and phase
annotations, for conditions fp32 / int8_ptq_pc / int8_qat / int4_wo_sim (gradient methods on the
fake-quant graphs) and ort_int8 (occlusion only):
    audit/explanations.csv         per window x condition x method: drift vs FP32 (cosine, Spearman,
                                   top-10% overlap, top-20% IoU), phase relevance (PRR_sys, control), deletion AUC
    audit/explanations_summary.json
    audit/attr_profiles.npz        the raw frame profiles for figures
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

print = functools.partial(print, flush=True)
import warnings  # noqa: E402

warnings.filterwarnings("ignore", message="All-NaN slice")

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcg import PROJECT_ROOT  # noqa: E402
from pcg import explain as X  # noqa: E402
from pcg import metrics as M  # noqa: E402
from pcg import quant as Q  # noqa: E402
from pcg.circor import load_metadata, load_segmentation, segmentation_to_frame_labels  # noqa: E402
from pcg.dataset import WINDOW, _pad_to, load_waveforms  # noqa: E402
from pcg.features import FS  # noqa: E402
from pcg.models import MODELS  # noqa: E402
from pcg.splits import load_folds  # noqa: E402

GRAD_CONDS = ["fp32", "int8_ptq_pc", "int8_qat", "int4_wo_sim"]


class AssembledEval(Q.Assembled):
    def train(self, mode: bool = True):
        self.frontend.train(mode)
        return self

    def eval(self):
        return self.train(False)


# --------------------------------------------------------------------------- part A

def part_a(run_dir: Path, pdf, args) -> None:
    qdir, adir = run_dir / "quant", run_dir / "audit"
    adir.mkdir(exist_ok=True)
    tables = {}
    for cdir in sorted(qdir.iterdir()):
        f = cdir / "rec_probs_test.csv"
        if cdir.is_dir() and f.exists():
            tables[cdir.name] = M.patient_table(pd.read_csv(f, dtype={"patient_id": str, "recording_id": str}), pdf)
    if "fp32" not in tables:
        print("  [part A] no quant/fp32 -> run p2_quantize.py first")
        return
    tables = {"fp32": tables.pop("fp32")} | tables
    ci = M.compare_conditions(tables, "fp32", args.n_boot, args.seed)
    ci.to_csv(adir / "conditions_ci.csv", index=False)
    for cond, t in tables.items():
        M.subgroup_table(t).to_csv(adir / f"subgroups_{cond}.csv", index=False)
        (adir / f"reliability_{cond}.json").write_text(json.dumps(
            M.reliability_bins(t["p_present"].values, t["y_present"].values), indent=1))
    M.subgroup_deltas(tables, "fp32", max(args.n_boot // 2, 200), args.seed).to_csv(adir / "subgroup_deltas.csv", index=False)
    piv = ci.pivot(index="metric", columns="cond", values="point").round(3)
    print("  [part A] point estimates:\n" + piv.to_string())
    g = pd.concat([M.subgroup_table(t).assign(cond=c) for c, t in tables.items()])
    gr = g[g["subgroup"] == "grade_group"].pivot(index="level", columns="cond", values="sensitivity").round(3)
    print("  [part A] sensitivity by grade:\n" + gr.to_string())


# --------------------------------------------------------------------------- part B

def pick_windows(rdf: pd.DataFrame, waves: dict, test: pd.DataFrame, n: int, seed: int):
    """Windows from Present-at-location recordings in the test fold with >=50% annotated frames."""
    rng = np.random.default_rng(seed)
    cand = test[(test["murmur"] == "Present") & test["murmur_at_location"]].sample(frac=1, random_state=seed)
    out = []
    for r in cand.itertuples():
        x = _pad_to(waves[r.recording_id], WINDOW)
        seg = load_segmentation(r.tsv)
        lab = segmentation_to_frame_labels(seg, len(x), FS)
        for _ in range(3):
            s = int(rng.integers(0, len(x) - WINDOW + 1))
            fl = X.phase_frames(lab[s : s + WINDOW])
            if (fl > 0).mean() >= 0.5:
                out.append({"recording_id": r.recording_id, "patient_id": r.patient_id, "start": s,
                            "wav": torch.from_numpy(x[s : s + WINDOW].copy())[None], "frames": fl,
                            "grade": str(r.sys_grading), "timing": str(r.sys_timing)})
                break
        if len(out) >= n:
            break
    return out


def build_backbones(run_dir: Path, name: str, calib: torch.Tensor):
    kw = {"pretrained": False} if name == "effnet_b0" else {}
    model = MODELS[name](**kw)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
    model.eval()
    fe, bb = Q.split(model)
    fe.eval()
    ex = Q.backbone_inputs(fe, [calib[:8]])[0]
    qdir = run_dir / "quant"
    grad_bb, int8_bb = {"fp32": X.disable_inplace(Q.fold_bn(bb))}, {}
    for cond, is_qat in (("int8_ptq_pc", False), ("int8_qat", True)):
        sd = qdir / cond / "prepared_sd.pt"
        if sd.exists():
            _, conv = Q.rebuild_from_state(bb, ex, sd, is_qat=is_qat, per_channel=True)
            # gradient methods need the STE rewrite of the converted (QDQ) graph: same numbers, STE gradient
            grad_bb[cond], int8_bb[cond] = Q.ste_graph(conv), conv
    grad_bb["int4_wo_sim"] = X.disable_inplace(Q.int4_weight_only_sim(Q.fold_bn(bb)))
    if (qdir / "backbone_int8.onnx").exists():
        int8_bb["ort_int8"] = Q.OrtBackbone(qdir / "backbone_int8.onnx", threads=4)
    return fe, grad_bb, int8_bb


def part_b(run_dir: Path, rdf, waves, args) -> None:
    cfg = json.loads((run_dir / "config.json").read_text())
    name = cfg["model"]
    adir = run_dir / "audit"
    adir.mkdir(exist_ok=True)
    if (adir / "explanations.csv").exists() and not args.force:
        print("  [part B] exists, skip")
        return
    folds = load_folds(Path(cfg["split_file"]))
    test = rdf[rdf["patient_id"].map(folds) == cfg["test_fold"]]
    calib = torch.load(run_dir / "quant" / "calib_windows.pt")
    fe, grad_bb, int8_bb = build_backbones(run_dir, name, calib)
    wins = pick_windows(rdf, waves, test, args.n_windows, args.seed)
    print(f"  [part B] {len(wins)} windows; grad conds={list(grad_bb)}; int8 conds={list(int8_bb)}")

    rows, profiles = [], {}
    fp32_asm = AssembledEval(fe, grad_bb["fp32"])
    for wi, w in enumerate(wins):
        wav, fl = w["wav"], w["frames"]
        with torch.no_grad():
            feats = fe(wav)
        sys_mask = fl == 2
        prof = {}
        # gradient methods on eager / fake-quant backbones
        for cond, bb in grad_bb.items():
            prof[(cond, "ig")] = X.to_frames(X.attr_ig(bb, feats, steps=args.ig_steps)[0])
            prof[(cond, "gradxinput")] = X.to_frames(X.attr_gradxinput(bb, feats)[0])
        # Grad-CAM FP32 only
        prof[("fp32", "gradcam")] = X.attr_gradcam(grad_bb["fp32"], feats, X.last_conv(grad_bb["fp32"]))
        # occlusion on everything incl. real INT8
        occ_targets = {"fp32": grad_bb["fp32"], "int4_wo_sim": grad_bb["int4_wo_sim"]} | int8_bb
        for cond, bb in occ_targets.items():
            prof[(cond, "occlusion")] = X.to_frames(X.attr_occlusion(AssembledEval(fe, bb), wav))
        # deletion faithfulness for fp32 IG, and for int8 (ptq) occlusion if present
        del_fp = X.deletion_curve(fp32_asm, wav, prof[("fp32", "ig")], restrict_frames=sys_mask if sys_mask.sum() >= 10 else None)
        del_rand = X.random_deletion_curve(fp32_asm, wav, seed=wi, restrict_frames=sys_mask if sys_mask.sum() >= 10 else None)
        for (cond, method), p in prof.items():
            r = {"window": wi, "recording_id": w["recording_id"], "patient_id": w["patient_id"], "grade": w["grade"],
                 "timing": w["timing"], "cond": cond, "method": method}
            r |= X.phase_relevance(p, fl)
            if cond != "fp32" and ("fp32", method) in prof:
                r |= {f"drift_{k}": v for k, v in X.drift(prof[("fp32", method)], p).items()}
                ref = X.phase_relevance(prof[("fp32", method)], fl)
                r["delta_prr_sys"] = r["prr_sys"] - ref["prr_sys"]
            if cond == "fp32" and method == "ig":
                r["deletion_auc_norm"] = del_fp["auc_norm"]
                r["deletion_auc_norm_random"] = del_rand["auc_norm"]
            rows.append(r)
            profiles[f"{wi}|{cond}|{method}"] = p
        if (wi + 1) % 10 == 0:
            print(f"    {wi + 1}/{len(wins)} windows", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(adir / "explanations.csv", index=False)
    np.savez_compressed(adir / "attr_profiles.npz", **profiles,
                        frames=np.stack([w["frames"] for w in wins]), starts=np.array([w["start"] for w in wins]),
                        recording_ids=np.array([w["recording_id"] for w in wins]))
    summ = (df.groupby(["cond", "method"])[[c for c in df.columns if c.startswith("drift_") or c.startswith("prr_") or c.startswith("delta_") or c.startswith("deletion")]]
            .mean(numeric_only=True).round(3))
    (adir / "explanations_summary.json").write_text(summ.reset_index().to_json(orient="records", indent=1))
    print("  [part B] mean per condition x method:\n" + summ.to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=["tiny_cnn"], choices=list(MODELS))
    ap.add_argument("--folds", nargs="+", type=int, default=[0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--runs-dir", default=str(PROJECT_ROOT / "runs"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-windows", type=int, default=100)
    ap.add_argument("--ig-steps", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-b", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(4)

    pdf, rdf = load_metadata()
    waves = None
    for m in args.model:
        for f in args.folds:
            for s in args.seeds:
                rd = Path(args.runs_dir) / m / f"fold{f}" / f"seed{s}"
                if not (rd / "quant" / "summary.json").exists():
                    print(f"[missing quant] {rd}")
                    continue
                print(f"[{m} fold{f} seed{s}]")
                part_a(rd, pdf, args)
                if not args.skip_b:
                    if waves is None:
                        waves = load_waveforms(rdf)
                    part_b(rd, rdf, waves, args)


if __name__ == "__main__":
    main()
