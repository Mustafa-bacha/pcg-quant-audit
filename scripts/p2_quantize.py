"""Phase 2 CLI: apply every compression condition to trained FP32 checkpoints and score them
with the official metrics on the same test fold.

    python scripts/p2_quantize.py --model tiny_cnn --folds 0 --seeds 0
    python scripts/p2_quantize.py --model tiny_cnn m5 logmel_cnn effnet_b0 --folds 0 1 2 3 4 --seeds 0 1 2 --qat-epochs 5

Per run dir  runs/<model>/fold<k>/seed<s>/  it writes
    quant/<condition>/rec_probs_test.csv   per-recording probabilities (same format as the FP32 run)
    quant/<condition>/patient_test.json    official patient-level metrics
    quant/summary.json                     all conditions: metrics, agreement with FP32, size, latency

Conditions: fp32, int8_ptq_pc, int8_ptq_pt, int8_qat, int4_wo_sim, int8_dynamic, ort_int8.
Calibration set (--calib): in_domain = random 5 s windows from the training folds;
                            noise     = the same windows with additive Gaussian noise at --calib-snr dB (H2 probe).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcg import PROJECT_ROOT  # noqa: E402
from pcg import quant as Q  # noqa: E402
from pcg.circor import load_metadata  # noqa: E402
from pcg.dataset import WINDOW, TrainCrops, _pad_to, load_waveforms  # noqa: E402
from pcg.models import MODELS  # noqa: E402
from pcg.train import patient_metrics, predict_recordings  # noqa: E402
from pcg.splits import load_folds  # noqa: E402


class AssembledEval(Q.Assembled):
    """Assembled whose train()/eval() only touch the float front-end (PT2E graphs forbid mode switches)."""

    def train(self, mode: bool = True):
        self.frontend.train(mode)
        return self

    def eval(self):
        return self.train(False)


def calib_windows(rdf: pd.DataFrame, waves: dict, n: int, seed: int, snr_db: float | None) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    ids = rdf["recording_id"].sample(min(n, len(rdf)), random_state=seed).tolist()
    out = []
    for rid in ids:
        x = _pad_to(waves[rid], WINDOW)
        s = int(rng.integers(0, len(x) - WINDOW + 1))
        seg = x[s : s + WINDOW].copy()
        if snr_db is not None:
            p_sig = float(np.mean(seg**2))
            seg = seg + rng.normal(0, np.sqrt(p_sig / 10 ** (snr_db / 10)), size=seg.shape).astype(np.float32)
        out.append(seg)
    return torch.from_numpy(np.stack(out))


def qat_finetune(prepared, fe: nn.Module, tr: pd.DataFrame, waves: dict, epochs: int, lr: float, seed: int,
                 device: torch.device, class_weights=(5.0, 3.0, 1.0), batch_size: int = 64) -> None:
    ds = TrainCrops(tr, waves, crops_per_recording=2, augment=True, seed=seed)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    params = [p for p in prepared.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    cw = torch.tensor(class_weights, device=device)
    cw = cw / cw.sum() * len(cw)
    for ep in range(epochs):
        losses = []
        for x, y in loader:
            with torch.no_grad():
                feats = fe(x.to(device))
            loss = F.cross_entropy(prepared(feats), y.to(device), weight=cw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"    qat ep{ep} loss={np.mean(losses):.3f}", flush=True)


def run_one(run_dir: Path, pdf, rdf, waves, args) -> dict:
    cfg = json.loads((run_dir / "config.json").read_text())
    name, fold, seed = cfg["model"], cfg["fold"], cfg["seed"]
    qdir = run_dir / "quant"
    only = set(args.only_cond or [])
    if (qdir / "summary.json").exists() and not args.force:
        old = json.loads((qdir / "summary.json").read_text())
        if not only:
            print(f"[skip] {qdir} done")
            return old
        if "env" in old:  # the env stamp marks a summary (re)computed under a verified PT2E environment
            print(f"[skip] {qdir} already redone under {old['env']}")
            return old
    wanted = lambda c: not only or c in only
    qdir.mkdir(exist_ok=True)
    device = torch.device("cpu")  # quantized kernels are CPU kernels; everything here runs on CPU
    torch.manual_seed(seed)

    folds = load_folds(Path(cfg["split_file"]))
    rdf = rdf.assign(f=rdf["patient_id"].map(folds))
    te = rdf[rdf["f"] == cfg["test_fold"]]
    tr = rdf[~rdf["f"].isin([cfg["test_fold"], cfg["val_fold"]])]

    kw = {"pretrained": False} if name == "effnet_b0" else {}
    model = MODELS[name](**kw)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
    model.eval()
    fe, bb = Q.split(model)
    fe.eval()

    if only and (qdir / "calib_windows.pt").exists():
        calib = torch.load(qdir / "calib_windows.pt")  # the exact windows every other arm was calibrated with
    else:
        calib = calib_windows(tr, waves, args.n_calib, seed, args.calib_snr if args.calib == "noise" else None)
    calib_in = Q.backbone_inputs(fe, [calib[i : i + 64] for i in range(0, len(calib), 64)])
    x1 = calib_in[0][:1]
    print(f"[{name} fold{fold} seed{seed}] calib={args.calib} n={len(calib)} backbone_in={tuple(calib_in[0].shape)}"
          + (f" only={sorted(only)}" if only else ""))

    conditions: dict[str, nn.Module] = {}
    refs: dict[str, nn.Module] = {}
    if wanted("fp32"):
        conditions["fp32"] = bb
    t0 = time.time()
    if wanted("int8_ptq_pc"):
        ref, conv = Q.ptq_static(bb, calib_in, per_channel=True)
        conditions["int8_ptq_pc"], refs["int8_ptq_pc"] = conv, ref
    if wanted("int8_ptq_pt"):
        ref, conv = Q.ptq_static(bb, calib_in, per_channel=False)
        conditions["int8_ptq_pt"], refs["int8_ptq_pt"] = conv, ref
    if wanted("int4_wo_sim"):
        conditions["int4_wo_sim"] = Q.int4_weight_only_sim(bb)
    if wanted("int8_dynamic"):
        conditions["int8_dynamic"] = Q.dynamic_int8(bb)
    ort_tags = [("ort_int8", "minmax", False)]
    if args.ort_mitigations or (only and only & {"ort_int8_pct", "ort_int8_ent", "ort_int8_pct_sym"}):
        # P5 arms: better range calibration for the deployment path
        ort_tags += [("ort_int8_pct", "percentile", False), ("ort_int8_ent", "entropy", False),
                     ("ort_int8_pct_sym", "percentile", True)]
    ort_tags = [t for t in ort_tags if wanted(t[0])]
    onnx_fp = None
    if ort_tags:
        onnx_fp = Q.export_onnx(bb, calib_in[0][:8], qdir / "backbone_fp32.onnx")
        for tag, cm, sym in ort_tags:
            out = qdir / ("backbone_int8.onnx" if tag == "ort_int8" else f"backbone_{tag}.onnx")
            p = Q.ort_quantize_static(onnx_fp, calib_in, out, calibrate_method=cm, symmetric_activations=sym)
            conditions[tag] = Q.OrtBackbone(p, threads=args.threads)
    print(f"  ptq/int4/dynamic/ort built in {time.time() - t0:.0f}s", flush=True)
    if args.qat_epochs > 0 and wanted("int8_qat"):
        t0 = time.time()
        prepared = Q.prepare_qat(bb, calib_in[0][:8], per_channel=True)
        qat_finetune(prepared, fe, tr, waves, args.qat_epochs, args.qat_lr, seed, device)
        ref, conv = Q.convert_qat(prepared)
        conditions["int8_qat"], refs["int8_qat"] = conv, ref
        print(f"  qat {args.qat_epochs} epochs in {time.time() - t0:.0f}s", flush=True)

    # persist fake-quant states so Phase 3 can rebuild the exact graphs (PT2E graphs are not picklable)
    for cond, ref in refs.items():
        (qdir / cond).mkdir(exist_ok=True)
        torch.save(ref.state_dict(), qdir / cond / "prepared_sd.pt")
    torch.save(calib, qdir / "calib_windows.pt")

    summary = {"model": name, "fold": fold, "seed": seed, "calib": args.calib, "n_calib": len(calib),
               "calib_snr": args.calib_snr if args.calib == "noise" else None, "env": Q.env_versions(),
               "conditions": {}}
    fp32_rec = None
    if "fp32" not in conditions and (qdir / "fp32" / "rec_probs_test.csv").exists():
        fp32_rec = pd.read_csv(qdir / "fp32" / "rec_probs_test.csv")  # reference for vs_fp32 agreement
    for cond, backbone in conditions.items():
        cdir = qdir / cond
        cdir.mkdir(exist_ok=True)
        asm = AssembledEval(fe, backbone)
        rec = predict_recordings(asm, te, waves, device, batch_windows=128)
        rec.to_csv(cdir / "rec_probs_test.csv", index=False)
        if args.predict_val:  # validation-fold predictions: needed to fit post-hoc temperature scaling (P5)
            va = rdf[rdf["f"] == cfg["val_fold"]]
            predict_recordings(asm, va, waves, device, batch_windows=128).to_csv(cdir / "rec_probs_val.csv", index=False)
        pm = patient_metrics(rec, pdf)
        (cdir / "patient_test.json").write_text(json.dumps(pm, indent=1))
        entry = {"metrics": pm}
        if cond == "fp32":
            fp32_rec = rec
            entry["size_bytes"] = Q.state_dict_bytes(bb)
        elif fp32_rec is not None:
            d = (rec[["p_present", "p_unknown", "p_absent"]].values - fp32_rec[["p_present", "p_unknown", "p_absent"]].values)
            entry["vs_fp32"] = {
                "rec_top1_agreement": float((rec[["p_present", "p_unknown", "p_absent"]].values.argmax(1)
                                             == fp32_rec[["p_present", "p_unknown", "p_absent"]].values.argmax(1)).mean()),
                "rec_mean_abs_dp_present": float(np.abs(d[:, 0]).mean()),
                "rec_max_abs_dp": float(np.abs(d).max()),
            }
        if cond in refs:
            # sanity: the differentiable STE rewrite must reproduce the converted QDQ graph exactly
            entry["ste_vs_qdq"] = Q.agreement(Q.ste_graph(backbone), backbone, calib_in[:4])
        if cond.startswith("ort_int8"):
            entry["size_bytes"] = (qdir / f"backbone_{cond if cond != 'ort_int8' else 'int8'}.onnx").stat().st_size
            entry["size_bytes_onnx_fp32"] = onnx_fp.stat().st_size
        if cond != "int8_dynamic":
            entry["latency_b1"] = Q.latency_ms(backbone, x1, n_iter=args.lat_iter, threads=args.threads)
        summary["conditions"][cond] = entry
        print(f"  {cond:12s} WA={pm['weighted_accuracy']:.3f} AUROC={pm['auroc']:.3f} presRec={pm['present_recall']:.2f}"
              + (f"  agree={entry['vs_fp32']['rec_top1_agreement']:.3f}" if "vs_fp32" in entry else ""), flush=True)
    if only and (qdir / "summary.json").exists():
        # surgical redo: keep every condition this invocation did not recompute
        old = json.loads((qdir / "summary.json").read_text())
        merged = old.get("conditions", {})
        merged.update(summary["conditions"])
        summary["conditions"] = merged
    (qdir / "summary.json").write_text(json.dumps(summary, indent=1))
    # new quantized graphs invalidate the audit; drop p3's skip-gate files so it recomputes this run
    for gate in ("subgroup_deltas.csv", "explanations.csv"):
        g = run_dir / "audit" / gate
        if g.exists():
            g.unlink()
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=["tiny_cnn"], choices=list(MODELS))
    ap.add_argument("--folds", nargs="+", type=int, default=[0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--runs-dir", default=str(PROJECT_ROOT / "runs"))
    ap.add_argument("--calib", choices=["in_domain", "noise"], default="in_domain")
    ap.add_argument("--calib-snr", type=float, default=10.0)
    ap.add_argument("--n-calib", type=int, default=512)
    ap.add_argument("--qat-epochs", type=int, default=5)
    ap.add_argument("--qat-lr", type=float, default=1e-4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--lat-iter", type=int, default=30)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only-cond", nargs="+", default=None,
                    choices=["fp32", "int8_ptq_pc", "int8_ptq_pt", "int8_qat", "int4_wo_sim", "int8_dynamic",
                             "ort_int8", "ort_int8_pct", "ort_int8_ent", "ort_int8_pct_sym"],
                    help="recompute only these conditions and merge them into the existing quant/summary.json "
                         "(the others' outputs are kept); implies redoing runs whose summary already exists")
    ap.add_argument("--predict-val", action="store_true", help="also save validation-fold probabilities per condition (for temperature scaling)")
    ap.add_argument("--ort-mitigations", action="store_true", help="add ORT percentile/entropy/symmetric calibration conditions (P5)")
    args = ap.parse_args()

    pdf, rdf = load_metadata()
    waves = load_waveforms(rdf)
    rows = []
    for m in args.model:
        for f in args.folds:
            for s in args.seeds:
                rd = Path(args.runs_dir) / m / f"fold{f}" / f"seed{s}"
                if not (rd / "best.pt").exists():
                    print(f"[missing] {rd}")
                    continue
                summ = run_one(rd, pdf, rdf, waves, args)
                for cond, e in summ["conditions"].items():
                    rows.append({"model": m, "fold": f, "seed": s, "cond": cond} | e["metrics"])
    if rows:
        df = pd.DataFrame(rows)
        print("\n=== Mean over folds/seeds per model x condition")
        print(df.groupby(["model", "cond"])[["weighted_accuracy", "auroc", "present_recall", "cost"]].mean().round(3))
        out = Path(args.runs_dir) / "p2_summary.csv"
        if out.exists():  # keep rows of runs this invocation did not touch (other models/folds/seeds)
            prev = pd.read_csv(out)
            done = set(zip(df["model"], df["fold"], df["seed"]))
            prev = prev[[(m, f, s) not in done for m, f, s in zip(prev["model"], prev["fold"], prev["seed"])]]
            df = pd.concat([prev, df], ignore_index=True)
        df.to_csv(out, index=False)


if __name__ == "__main__":
    main()
