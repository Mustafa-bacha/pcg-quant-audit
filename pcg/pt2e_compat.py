"""Import shim for the PT2E quantization API, which moved from torch.ao to torchao.

torch >= 2.12 / torchao >= 0.12: everything lives under torchao.quantization.pt2e (dev laptop).
Kaggle's image (torch 2.11, older torchao): the same names live under torch.ao.quantization.
All project code imports from here so both environments work unchanged.
"""

from __future__ import annotations

try:
    from torchao.quantization.pt2e import (  # noqa: F401
        FusedMovingAvgObsFakeQuantize, MinMaxObserver, MovingAverageMinMaxObserver,
        move_exported_model_to_eval, move_exported_model_to_train)
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e, prepare_qat_pt2e  # noqa: F401
    from torchao.quantization.pt2e.quantizer import QuantizationSpec  # noqa: F401
    from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (  # noqa: F401
        QuantizationConfig, X86InductorQuantizer, get_default_x86_inductor_quantization_config)
    BACKEND = "torchao"
except ImportError:  # older torchao / torch <= 2.11
    from torch.ao.quantization import (  # noqa: F401
        FusedMovingAvgObsFakeQuantize, MinMaxObserver, MovingAverageMinMaxObserver,
        move_exported_model_to_eval, move_exported_model_to_train)
    from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e, prepare_qat_pt2e  # noqa: F401
    from torch.ao.quantization.quantizer import QuantizationSpec  # noqa: F401
    from torch.ao.quantization.quantizer.x86_inductor_quantizer import (  # noqa: F401
        QuantizationConfig, X86InductorQuantizer, get_default_x86_inductor_quantization_config)
    BACKEND = "torch.ao"
