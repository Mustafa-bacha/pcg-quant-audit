"""Phase 2: compression conditions (FINAL_research_problem.md, sections 7-8).

Conditions implemented
  fp32          reference
  int8_dynamic  torch.ao dynamic quantization (Linear layers only; weak condition, kept for completeness)
  int8_ptq      static PTQ via torchao PT2E + X86InductorQuantizer, per-tensor or per-channel weights,
                calibration set chosen by the caller (in-domain vs. noise-augmented, hypothesis H2)
  int8_qat      PT2E quantization-aware training; `prepare_qat()` returns a trainable module,
                `convert()` finishes it
  int4_wo_sim   weight-only INT4, symmetric per-channel, *simulated* (fake-quant) -- no CPU INT4 conv kernels exist
  ort_int8      ONNX Runtime static QDQ S8S8 quantization of the FP32 ONNX export: the deployment path
                (x86 and Raspberry Pi ARM); the only condition that runs real integer kernels here

Important facts (verified 2026-10-20 on torch 2.14 / torchao 0.18):
  * A `convert_pt2e` graph is a QDQ *simulation* (quantize->dequantize in float, INT8 weights as
    constants). It is numerically what integer kernels compute, but it is not faster on CPU; real INT8
    speed needs inductor lowering (no MSVC here) or ONNX Runtime. Latency claims use `ort_int8`.
  * The *prepared* PTQ graph only has pass-through observers: it is FP32. Gradient-based explanations
    of a PTQ model must use `ste_graph(converted)`, which replaces q/dq pairs by
    torch.fake_quantize_per_tensor_affine (same values, straight-through gradient).
  * The prepared QAT graph does use FakeQuantize (STE) modules, but `ste_graph(converted)` is used
    for both conditions so that explanations always come from the exact deployed numerics.

Quantization unit: for raw-waveform models the whole network; for logmel_cnn only `.backbone`
(the log-mel front-end stays float, as it would on a device). `split()` handles both.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn

from .models import LogMelCNN

INT8_CONDITIONS = ["fp32", "int8_dynamic", "int8_ptq_pt", "int8_ptq_pc", "int8_qat", "int4_wo_sim", "ort_int8"]


# --------------------------------------------------------------------------- split

class Identity(nn.Module):
    def forward(self, x):
        return x


def split(model: nn.Module) -> tuple[nn.Module, nn.Module]:
    """(frontend, backbone). frontend maps waveform -> backbone input; it is never quantized."""
    if isinstance(model, LogMelCNN):
        fe = copy.deepcopy(model.frontend).eval()
        return fe, model.backbone
    return Identity(), model


class Assembled(nn.Module):
    """frontend + (possibly quantized / exported) backbone, waveform in, logits out."""

    def __init__(self, frontend: nn.Module, backbone: nn.Module):
        super().__init__()
        self.frontend, self.backbone = frontend, backbone

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.frontend(wav)
        return self.backbone(feats)


def backbone_inputs(frontend: nn.Module, wav_batches: Iterable[torch.Tensor]) -> list[torch.Tensor]:
    frontend.eval()
    with torch.no_grad():
        return [frontend(w) for w in wav_batches]


# --------------------------------------------------------------------------- BN folding

def _fold_rec(mod: nn.Module) -> None:
    from torch.nn.utils.fusion import fuse_conv_bn_eval
    from .models import BasicBlock2d
    convs, bns = (nn.Conv1d, nn.Conv2d), (nn.BatchNorm1d, nn.BatchNorm2d)
    if isinstance(mod, nn.Sequential):
        children, new, i = list(mod.children()), [], 0
        while i < len(children):
            c = children[i]
            if isinstance(c, convs) and i + 1 < len(children) and isinstance(children[i + 1], bns):
                new.append(fuse_conv_bn_eval(c, children[i + 1]))
                i += 2
            else:
                _fold_rec(c)
                new.append(c)
                i += 1
        for k in list(mod._modules):
            del mod._modules[k]
        for j, c in enumerate(new):
            mod.add_module(str(j), c)
    elif isinstance(mod, BasicBlock2d):
        mod.c1, mod.b1 = fuse_conv_bn_eval(mod.c1, mod.b1), nn.Identity()
        mod.c2, mod.b2 = fuse_conv_bn_eval(mod.c2, mod.b2), nn.Identity()
        if mod.short is not None:
            _fold_rec(mod.short)
    else:
        for c in mod.children():
            _fold_rec(c)


def fold_bn(backbone: nn.Module) -> nn.Module:
    """Return an eval-mode deep copy with every Conv+BatchNorm pair fused into the conv.
    Required before PT2E export: the x86 quantizer's conv-add-relu pattern matcher fails on
    unfused BN in residual blocks (torchao 0.18), and deployment graphs are BN-free anyway."""
    m = copy.deepcopy(backbone).eval()
    _fold_rec(m)
    return m


# --------------------------------------------------------------------------- PT2E static PTQ / QAT

def _quantizer(per_channel: bool, is_qat: bool = False):
    from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
        X86InductorQuantizer, get_default_x86_inductor_quantization_config)
    q = X86InductorQuantizer()
    q.set_global(get_default_x86_inductor_quantization_config(is_qat=is_qat, is_dynamic=False))
    if not per_channel:
        # the default x86 config is per-channel for weights; build a per-tensor-symmetric variant
        from torchao.quantization.pt2e import FusedMovingAvgObsFakeQuantize, MinMaxObserver, MovingAverageMinMaxObserver
        from torchao.quantization.pt2e.quantizer import QuantizationSpec
        from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import QuantizationConfig
        cfg = get_default_x86_inductor_quantization_config(is_qat=is_qat, is_dynamic=False)
        if is_qat:
            ctr = FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver, eps=2**-12)
        else:
            ctr = MinMaxObserver.with_args(eps=2**-12)
        w_pt = QuantizationSpec(dtype=torch.int8, quant_min=-128, quant_max=127,
                                qscheme=torch.per_tensor_symmetric, is_dynamic=False,
                                observer_or_fake_quant_ctr=ctr)
        q.set_global(QuantizationConfig(cfg.input_activation, cfg.output_activation, w_pt, cfg.bias, is_qat))
    return q


def _export(m: nn.Module, example: torch.Tensor) -> torch.fx.GraphModule:
    """torch.export with a dynamic batch dimension so batch-1 latency and large eval batches both run."""
    batch = torch.export.Dim("batch", min=1, max=65536)
    return torch.export.export(m, (example,), dynamic_shapes=({0: batch},)).module()


def export_backbone(backbone: nn.Module, example: torch.Tensor) -> torch.fx.GraphModule:
    return _export(fold_bn(backbone), example)


def ptq_static(backbone: nn.Module, calib: list[torch.Tensor], per_channel: bool = True
               ) -> tuple[torch.fx.GraphModule, torch.fx.GraphModule]:
    """Returns (prepared_fake_quant, converted_int8). Both take backbone inputs."""
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
    ex = export_backbone(backbone, calib[0])
    prepared = prepare_pt2e(ex, _quantizer(per_channel))
    with torch.no_grad():
        for b in calib:
            prepared(b)
    from torchao.quantization.pt2e import move_exported_model_to_eval
    move_exported_model_to_eval(prepared)
    reference = copy.deepcopy(prepared)  # frozen observers, fake-quant forward with autograd
    converted = convert_pt2e(prepared)
    return reference, converted


def prepare_qat(backbone: nn.Module, example: torch.Tensor, per_channel: bool = True) -> torch.fx.GraphModule:
    """Trainable fake-quant graph. Train it with the normal loop (it has parameters + autograd)."""
    from torchao.quantization.pt2e.quantize_pt2e import prepare_qat_pt2e
    # BN is folded first (see fold_bn); QAT then fine-tunes a BN-free graph from the trained weights.
    backbone = fold_bn(backbone).train()
    ex = _export(backbone, example)
    from torchao.quantization.pt2e import move_exported_model_to_train
    prepared = prepare_qat_pt2e(ex, _quantizer(per_channel, is_qat=True))
    move_exported_model_to_train(prepared)
    return prepared


def convert_qat(prepared: torch.fx.GraphModule) -> tuple[torch.fx.GraphModule, torch.fx.GraphModule]:
    from torchao.quantization.pt2e import move_exported_model_to_eval
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e
    move_exported_model_to_eval(prepared)
    reference = copy.deepcopy(prepared)
    return reference, convert_pt2e(prepared)


def ste_graph(converted: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Turn a converted PT2E (QDQ) graph into a *differentiable* fake-quant graph.

    `convert_pt2e` emits quantize_per_tensor -> dequantize_per_tensor pairs around every quantized
    activation. Numerically that is exactly INT8 arithmetic simulated in float, but the rounding has a
    zero gradient, and the *prepared* PTQ graph only contains pass-through observers (no rounding at
    all). This pass replaces each activation q/dq pair with torch.fake_quantize_per_tensor_affine,
    which computes the same values and carries a straight-through-estimator gradient, so
    Integrated Gradients / Gradient x Input can be computed on the quantized model.
    Weight dequantize nodes (int8 constants) are left untouched. Verified bit-identical outputs.
    """
    from torch.fx import Node
    g = copy.deepcopy(converted)
    qd = torch.ops.quantized_decomposed
    q_ops = {qd.quantize_per_tensor.default, qd.quantize_per_tensor.tensor}
    dq_ops = {qd.dequantize_per_tensor.default, qd.dequantize_per_tensor.tensor}
    n_rewritten = 0
    for node in list(g.graph.nodes):
        if node.op != "call_function" or node.target not in dq_ops:
            continue
        q = node.args[0]
        if not (isinstance(q, Node) and q.op == "call_function" and q.target in q_ops):
            continue
        x, scale, zp, qmin, qmax = q.args[:5]
        with g.graph.inserting_before(node):
            new = g.graph.call_function(torch.fake_quantize_per_tensor_affine, (x, scale, zp, int(qmin), int(qmax)))
        node.replace_all_uses_with(new)
        g.graph.erase_node(node)
        if not q.users:
            g.graph.erase_node(q)
        n_rewritten += 1
    g.graph.lint()
    g.recompile()
    g.meta["n_ste_rewrites"] = n_rewritten
    return g


def rebuild_from_state(backbone: nn.Module, example: torch.Tensor, state_path: Path, is_qat: bool,
                       per_channel: bool = True) -> tuple[torch.fx.GraphModule, torch.fx.GraphModule]:
    """Recreate (reference_fake_quant, converted_int8) from a saved prepared-graph state_dict.
    PT2E graphs cannot be pickled, but prepare + load_state_dict reproduces them exactly (verified)."""
    from torchao.quantization.pt2e import move_exported_model_to_eval
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
    if is_qat:
        prepared = prepare_qat(backbone, example, per_channel)
    else:
        prepared = prepare_pt2e(export_backbone(backbone, example), _quantizer(per_channel))
    res = prepared.load_state_dict(torch.load(state_path, map_location="cpu"), strict=False)
    assert not res.missing_keys and not res.unexpected_keys, (res.missing_keys, res.unexpected_keys)
    move_exported_model_to_eval(prepared)
    reference = copy.deepcopy(prepared)
    return reference, convert_pt2e(prepared)


# --------------------------------------------------------------------------- dynamic INT8 and INT4 simulation

def dynamic_int8(backbone: nn.Module) -> nn.Module:
    return torch.ao.quantization.quantize_dynamic(copy.deepcopy(backbone).eval(), {nn.Linear}, dtype=torch.qint8)


def fake_quant_weight(w: torch.Tensor, bits: int = 4, per_channel: bool = True) -> torch.Tensor:
    """Symmetric uniform fake quantization of a weight tensor (out-channel axis 0)."""
    qmax = 2 ** (bits - 1) - 1
    if per_channel:
        amax = w.detach().abs().flatten(1).max(dim=1).values.clamp_min(1e-8)
        scale = (amax / qmax).view(-1, *([1] * (w.dim() - 1)))
    else:
        scale = (w.detach().abs().max().clamp_min(1e-8) / qmax)
    return torch.clamp(torch.round(w / scale), -qmax - 1, qmax) * scale


def int4_weight_only_sim(backbone: nn.Module, bits: int = 4, per_channel: bool = True) -> nn.Module:
    """Weight-only INTk simulation on every Conv/Linear weight; activations stay float."""
    m = copy.deepcopy(backbone).eval()
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                mod.weight.copy_(fake_quant_weight(mod.weight, bits, per_channel))
    return m


# --------------------------------------------------------------------------- ONNX Runtime path

def export_onnx(backbone: nn.Module, example: torch.Tensor, path: Path) -> Path:
    backbone = copy.deepcopy(backbone).eval()
    torch.onnx.export(backbone, (example,), str(path), input_names=["x"], output_names=["logits"],
                      dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
    return path


def ort_quantize_static(fp32_onnx: Path, calib: list[torch.Tensor], out: Path, per_channel: bool = True) -> Path:
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre = out.with_suffix(".pre.onnx")
    quant_pre_process(str(fp32_onnx), str(pre))

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(calib)

        def get_next(self):
            b = next(self.it, None)
            return None if b is None else {"x": b.numpy().astype(np.float32)}

    quantize_static(str(pre), str(out), Reader(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, per_channel=per_channel)
    pre.unlink(missing_ok=True)
    return out


class OrtBackbone(nn.Module):
    """Run an ONNX file as a torch-callable backbone (CPU)."""

    def __init__(self, path: Path, threads: int | None = None):
        super().__init__()
        import onnxruntime as ort
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.sess.run(["logits"], {"x": x.detach().cpu().numpy().astype(np.float32)})[0]
        return torch.from_numpy(y)


# --------------------------------------------------------------------------- utilities

@torch.no_grad()
def agreement(a: Callable, b: Callable, inputs: list[torch.Tensor]) -> dict[str, float]:
    """Max |logit difference| and top-1 agreement between two backbones on the same inputs."""
    diffs, agree, n = [], 0, 0
    for x in inputs:
        la, lb = a(x), b(x)
        diffs.append((la - lb).abs().max().item())
        agree += (la.argmax(1) == lb.argmax(1)).sum().item()
        n += x.shape[0]
    return {"max_abs_logit_diff": float(max(diffs)), "top1_agreement": agree / n}


@torch.no_grad()
def latency_ms(fn: Callable, x: torch.Tensor, n_warm: int = 5, n_iter: int = 30, threads: int | None = None) -> dict:
    import time
    if threads:
        torch.set_num_threads(threads)
    for _ in range(n_warm):
        fn(x)
    t = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn(x)
        t.append((time.perf_counter() - t0) * 1e3)
    return {"median_ms": float(np.median(t)), "p90_ms": float(np.percentile(t, 90)), "batch": int(x.shape[0])}


def state_dict_bytes(m: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in m.state_dict().values() if torch.is_tensor(p))
