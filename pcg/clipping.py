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
            clip = ((q < qmin) | (q > qmax)).float().mean().item()
            qr = torch.clamp(torch.round(q), qmin, qmax)
            # resolution actually used: fraction of the integer range spanned by 99% of values, and
            # fraction of values that collapse onto the single level nearest zero (lost small signals)
            lo, hi = torch.quantile(qr.flatten()[: 2_000_000], torch.tensor([0.005, 0.995]))
            span = float((hi - lo) / (qmax - qmin))
            zero_level = torch.round(torch.as_tensor(zp, dtype=torch.float32))
            at_zero = float((qr == zero_level).float().mean())
            stats.setdefault(name, []).append((clip, span, at_zero))

        hooks.append(mod.register_forward_hook(hook))
    try:
        for x in inputs:
            prepared(x)
    finally:
        for h in hooks:
            h.remove()
    return {k: {"clip_rate": float(np.mean([s[0] for s in v])), "range_span_99": float(np.mean([s[1] for s in v])),
                "frac_at_zero_level": float(np.mean([s[2] for s in v]))} for k, v in stats.items()}


def summarize(rates: dict[str, dict]) -> dict[str, float]:
    if not rates:
        return {"n_nodes": 0, "mean_clip_rate": 0.0, "max_clip_rate": 0.0, "nodes_over_1pct": 0,
                "mean_range_span_99": float("nan"), "mean_frac_at_zero_level": float("nan")}
    c = np.array([r["clip_rate"] for r in rates.values()])
    return {"n_nodes": len(rates), "mean_clip_rate": float(c.mean()), "max_clip_rate": float(c.max()),
            "nodes_over_1pct": int((c > 0.01).sum()),
            "mean_range_span_99": float(np.mean([r["range_span_99"] for r in rates.values()])),
            "mean_frac_at_zero_level": float(np.mean([r["frac_at_zero_level"] for r in rates.values()]))}
