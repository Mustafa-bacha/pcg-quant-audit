"""H2 probe: activation clipping (range saturation) in a calibrated fake-quant graph.

For every activation observer / fake-quantize node in a *prepared* PT2E graph, record the fraction
of values that fall outside the calibrated [qmin, qmax] range when the graph sees new inputs.
If shifted inputs (noise, other stethoscope) saturate more often than in-domain inputs, that is
direct evidence for H2 ("shifted inputs saturate ranges calibrated in-domain").
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _qparams(mod: nn.Module):
    """(scale, zero_point, qmin, qmax) from an observer or fake-quant module, or None."""
    try:
        scale, zp = mod.calculate_qparams()
    except Exception:  # noqa: BLE001
        return None
    qmin = getattr(mod, "quant_min", None)
    qmax = getattr(mod, "quant_max", None)
    if qmin is None and hasattr(mod, "activation_post_process"):
        qmin, qmax = mod.activation_post_process.quant_min, mod.activation_post_process.quant_max
    if qmin is None:
        return None
    return scale.detach().float(), zp.detach().float(), float(qmin), float(qmax)


@torch.no_grad()
def clip_rates(prepared: torch.fx.GraphModule, inputs: list[torch.Tensor], activations_only: bool = True) -> dict[str, float]:
    """Mean fraction of clipped values per observer/fake-quant node over `inputs` (backbone inputs)."""
    stats: dict[str, list[float]] = {}
    hooks = []
    for name, mod in prepared.named_modules():
        qp = _qparams(mod)
        if qp is None:
            continue
        scale, zp, qmin, qmax = qp
        if activations_only and scale.numel() > 1:  # per-channel == weight observer
            continue

        def hook(m, inp, out, name=name, scale=scale, zp=zp, qmin=qmin, qmax=qmax):
            x = inp[0].detach().float()
            q = x / scale + zp
            rate = ((q < qmin) | (q > qmax)).float().mean().item()
            stats.setdefault(name, []).append(rate)

        hooks.append(mod.register_forward_hook(hook))
    try:
        for x in inputs:
            prepared(x)
    finally:
        for h in hooks:
            h.remove()
    return {k: float(np.mean(v)) for k, v in stats.items()}


def summarize(rates: dict[str, float]) -> dict[str, float]:
    v = np.array(list(rates.values())) if rates else np.array([0.0])
    return {"n_nodes": len(rates), "mean_clip_rate": float(v.mean()), "max_clip_rate": float(v.max()),
            "nodes_over_1pct": int((v > 0.01).sum())}
