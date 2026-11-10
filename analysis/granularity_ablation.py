"""Task 2(c): granularity-localization ablation on EfficientNet-B0.

Two mixed-granularity PTQ variants on the same checkpoints, mirroring the exact
pipeline of scripts/p2_quantize.py (same calibration windows, same evaluation):

  depthwise_pt : per-TENSOR weight scales ONLY in depthwise convs, per-channel
                 everywhere else.
  pointwise_pt : per-tensor ONLY in the non-depthwise convs/linear, per-channel
                 in depthwise.

If depthwise_pt alone reproduces the Grade-I collapse (reference points from
t8_pooled_subgroups.csv: int8_ptq_pt -0.407, int8_ptq_pc +0.003) and
pointwise_pt does not, the mechanism (depthwise weight-range disparity under a
single scale) is established causally.

Usage (repo root; validate with one run first):
    .venv/Scripts/python.exe revision_v2/scripts/granularity_ablation.py --folds 0 --seeds 0
    .venv/Scripts/python.exe revision_v2/scripts/granularity_ablation.py            # all 15 runs

Outputs: revision_v2/granularity_ablation/<variant>/fold<k>_seed<s>_rec_probs.csv
and a pooled Grade-I / Grade-II-III sensitivity summary per variant.
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from pcg import quant as Q  # noqa: E402
from pcg.circor import load_metadata  # noqa: E402
from pcg.dataset import load_waveforms  # noqa: E402
from pcg.models import MODELS  # noqa: E402
from pcg.splits import load_folds  # noqa: E402
from pcg.train import predict_recordings  # noqa: E402
from p2_quantize import AssembledEval  # noqa: E402

VARIANTS = ("depthwise_pt", "pointwise_pt")


def depthwise_names(model: nn.Module) -> list[str]:
    return [n for n, m in model.named_modules()
            if isinstance(m, nn.Conv2d) and m.groups > 1 and m.groups == m.in_channels]


def mixed_quantizer(bb: nn.Module, variant: str):
    """Global per-channel config + per-tensor weight override on the chosen module names."""
    from pcg.pt2e_compat import (MinMaxObserver, QuantizationConfig, QuantizationSpec,
                                 X86InductorQuantizer,
                                 get_default_x86_inductor_quantization_config)
    base = get_default_x86_inductor_quantization_config(is_qat=False, is_dynamic=False)
    w_pt = QuantizationSpec(dtype=torch.int8, quant_min=-128, quant_max=127,
                            qscheme=torch.per_tensor_symmetric, is_dynamic=False,
                            observer_or_fake_quant_ctr=MinMaxObserver.with_args(eps=2**-12))
    pt_cfg = QuantizationConfig(base.input_activation, base.output_activation, w_pt, base.bias, False)

    dw = set(depthwise_names(bb))
    if variant == "depthwise_pt":
        pt_names = sorted(dw)
    else:
        pt_names = sorted(n for n, m in bb.named_modules()
                          if isinstance(m, (nn.Conv2d, nn.Linear)) and n not in dw)
    q = X86InductorQuantizer()
    q.set_global(base)  # per-channel everywhere ...
    for n in pt_names:  # ... except the named modules
        q.set_module_name_qconfig(n, pt_cfg)
    return q, len(pt_names)


def ptq_mixed(bb: nn.Module, calib_in: list[torch.Tensor], variant: str):
    """Mirror of pcg.quant.ptq_static with the mixed quantizer."""
    from pcg.pt2e_compat import convert_pt2e, move_exported_model_to_eval, prepare_pt2e
    quantizer, n_pt = mixed_quantizer(Q.fold_bn(bb), variant)
    ex = Q.export_backbone(bb, calib_in[0])
    prepared = prepare_pt2e(ex, quantizer)
    Q._assert_prepared(prepared, f"ablation:{variant}")
    with torch.no_grad():
        for b in calib_in:
            prepared(b)
    move_exported_model_to_eval(prepared)
    converted = convert_pt2e(prepared)
    Q._assert_converted(converted, f"ablation:{variant}")
    return converted, n_pt


def grade_sensitivity(rec_frames: list[pd.DataFrame], pdf: pd.DataFrame) -> dict:
    """Pooled sensitivity by murmur grade, patient decision = max P(Present) > others."""
    df = pd.concat(rec_frames)
    pat = df.groupby("patient_id")[["p_present", "p_unknown", "p_absent"]].max()
    pred_present = pat[["p_present", "p_unknown", "p_absent"]].idxmax(axis=1) == "p_present"
    out = {}
    meta = pdf.loc[pdf.index.intersection(pat.index)]
    present = meta[meta["murmur"] == "Present"]
    for name, grades in (("grade I (soft)", {"I/VI"}), ("grade II-III", {"II/VI", "III/VI"})):
        ids = present[present["sys_grading"].isin(grades)].index
        ids = ids.intersection(pat.index)
        if len(ids):
            out[name] = (float(pred_present.loc[ids].mean()), int(len(ids)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS, nargs="+", default=list(VARIANTS))
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    pdf, rdf = load_metadata()
    waves = load_waveforms(rdf)
    folds = load_folds(REPO / "splits" / "folds_v1.json")
    rdf = rdf.assign(f=rdf["patient_id"].map(folds))
    device = torch.device("cpu")

    for variant in args.variant:
        out_root = REPO / "revision_v2" / "granularity_ablation" / variant
        out_root.mkdir(parents=True, exist_ok=True)
        frames = []
        for fold in args.folds:
            for seed in args.seeds:
                out_csv = out_root / f"fold{fold}_seed{seed}_rec_probs.csv"
                if out_csv.exists():
                    print(f"[skip] {out_csv}")
                    frames.append(pd.read_csv(out_csv))
                    continue
                rd = REPO / "runs" / "effnet_b0" / f"fold{fold}" / f"seed{seed}"
                ckpt, calib_p = rd / "best.pt", rd / "quant" / "calib_windows.pt"
                if not ckpt.exists() or not calib_p.exists():
                    print(f"[PENDING-RUN] missing {ckpt} or {calib_p}")
                    continue
                t0 = time.time()
                model = MODELS["effnet_b0"](pretrained=False)
                model.load_state_dict(torch.load(ckpt, map_location="cpu"))
                model.eval()
                fe, bb = Q.split(model)
                fe.eval()
                calib = torch.load(calib_p)
                calib_in = Q.backbone_inputs(fe, [calib[i:i + 64] for i in range(0, len(calib), 64)])
                converted, n_pt = ptq_mixed(bb, calib_in, variant)
                te = rdf[rdf["f"] == fold]
                rec = predict_recordings(AssembledEval(fe, converted), te, waves, device, batch_windows=128)
                rec.to_csv(out_csv, index=False)
                frames.append(rec)
                print(f"[{variant} fold{fold} seed{seed}] {n_pt} per-tensor modules; "
                      f"done in {time.time()-t0:.0f}s", flush=True)
        if frames:
            sens = grade_sensitivity(frames, pdf)
            print(f"\n== {variant}: pooled sensitivity {sens}")
            print("   reference (t8_pooled): fp32 grade-I 0.817; int8_ptq_pc 0.820; int8_ptq_pt 0.410\n")


if __name__ == "__main__":
    main()
