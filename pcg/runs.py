"""Rebuild every compression condition of a finished run folder (after p2_quantize.py).

    fe, conds, refs = load_conditions(run_dir)
    conds: {"fp32": eager BN-folded backbone, "int8_ptq_pc": converted INT8 graph, "int8_ptq_pt": ...,
            "int8_qat": ..., "int4_wo_sim": eager, "int8_dynamic": eager, "ort_int8": OrtBackbone}
    refs:  {"int8_ptq_pc": fake-quant graph, "int8_ptq_pt": ..., "int8_qat": ...}   (autograd-capable)

Shared by the Phase 3 audit and the Phase 4 shift scripts.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from . import quant as Q
from .models import MODELS


class AssembledEval(Q.Assembled):
    """Assembled whose train()/eval() only touch the float front-end (PT2E graphs forbid mode switches)."""

    def train(self, mode: bool = True):
        self.frontend.train(mode)
        return self

    def eval(self):
        return self.train(False)


def load_fp32(run_dir: Path) -> tuple[nn.Module, nn.Module, nn.Module, dict]:
    """(model, frontend, backbone, config) from best.pt."""
    cfg = json.loads((run_dir / "config.json").read_text())
    name = cfg["model"]
    kw = {"pretrained": False} if name == "effnet_b0" else {}
    model = MODELS[name](**kw)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
    model.eval()
    fe, bb = Q.split(model)
    fe.eval()
    return model, fe, bb, cfg


def load_conditions(run_dir: Path, want: tuple[str, ...] | None = None, ort_threads: int = 4):
    run_dir = Path(run_dir)
    qdir = run_dir / "quant"
    _, fe, bb, cfg = load_fp32(run_dir)
    calib = torch.load(qdir / "calib_windows.pt")
    ex = Q.backbone_inputs(fe, [calib[:8]])[0]
    conds: dict[str, nn.Module] = {"fp32": Q.fold_bn(bb)}
    refs: dict[str, nn.Module] = {}
    spec = {"int8_ptq_pc": (False, True), "int8_ptq_pt": (False, False), "int8_qat": (True, True)}
    for cond, (is_qat, pc) in spec.items():
        if want and cond not in want:
            continue
        sd = qdir / cond / "prepared_sd.pt"
        if sd.exists():
            refs[cond], conds[cond] = Q.rebuild_from_state(bb, ex, sd, is_qat=is_qat, per_channel=pc)
    if not want or "int4_wo_sim" in want:
        conds["int4_wo_sim"] = Q.int4_weight_only_sim(Q.fold_bn(bb))
    if not want or "int8_dynamic" in want:
        conds["int8_dynamic"] = Q.dynamic_int8(Q.fold_bn(bb))
    if (not want or "ort_int8" in want) and (qdir / "backbone_int8.onnx").exists():
        conds["ort_int8"] = Q.OrtBackbone(qdir / "backbone_int8.onnx", threads=ort_threads)
    return fe, conds, refs, cfg
