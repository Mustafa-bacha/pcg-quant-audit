"""Import shim for the PT2E quantization API, which moved from torch.ao to torchao.

torch >= 2.12 / torchao >= 0.12: everything lives under torchao.quantization.pt2e (dev laptop).
Kaggle's image (torch 2.11, older torchao): the same names live under torch.ao.quantization.
All project code imports from here so both environments work unchanged.

PCG_PT2E_BACKEND=torch.ao|torchao forces one branch (e.g. when the torchao installed next to an
older torch imports fine but its quantizer annotations do not match that torch's exported graphs,
which makes prepare_pt2e a silent no-op - see pcg.quant's prepared/converted-graph guards).
"""

from __future__ import annotations

import os

_FORCE = os.environ.get("PCG_PT2E_BACKEND", "").strip().lower()
if _FORCE not in ("", "torchao", "torch.ao"):
    raise ValueError(f"PCG_PT2E_BACKEND must be 'torchao' or 'torch.ao', got {_FORCE!r}")

try:
    if _FORCE == "torch.ao":
        raise ImportError("PCG_PT2E_BACKEND=torch.ao")
    from torchao.quantization.pt2e import (  # noqa: F401
        FusedMovingAvgObsFakeQuantize, MinMaxObserver, MovingAverageMinMaxObserver,
        move_exported_model_to_eval, move_exported_model_to_train)
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e, prepare_qat_pt2e  # noqa: F401
    from torchao.quantization.pt2e.quantizer import QuantizationSpec  # noqa: F401
    from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (  # noqa: F401
        QuantizationConfig, X86InductorQuantizer, get_default_x86_inductor_quantization_config)
    BACKEND = "torchao"
except ImportError:
    if _FORCE == "torchao":
        raise
    from torch.ao.quantization import (  # noqa: F401
        FusedMovingAvgObsFakeQuantize, MinMaxObserver, MovingAverageMinMaxObserver,
        move_exported_model_to_eval, move_exported_model_to_train)
    from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e, prepare_qat_pt2e  # noqa: F401
    from torch.ao.quantization.quantizer import QuantizationSpec  # noqa: F401
    from torch.ao.quantization.quantizer.x86_inductor_quantizer import (  # noqa: F401
        QuantizationConfig, X86InductorQuantizer, get_default_x86_inductor_quantization_config)
    BACKEND = "torch.ao"


def versions() -> dict:
    """Provenance record for summaries: which torch/torchao produced a quantized artifact."""
    import torch
    try:
        import torchao
        ta = getattr(torchao, "__version__", "unknown")
    except Exception:
        ta = None
    return {"torch": torch.__version__, "torchao": ta, "pt2e_backend": BACKEND}
