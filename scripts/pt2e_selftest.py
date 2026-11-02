"""Fail-fast check that PT2E quantization actually quantizes in this environment.

    python scripts/pt2e_selftest.py

Builds a tiny conv net, runs the project's PTQ path (prepare -> calibrate -> convert) and verifies
that observers were inserted, the converted graph contains q/dq nodes, and its output differs from
FP32 by more than float noise. Exits non-zero otherwise. Run this at the top of every Kaggle
session BEFORE any GPU work: on torch/torchao pairs that do not match (e.g. Kaggle's torch 2.11
image with a newer torchao), prepare_pt2e used to be a silent no-op and every PT2E condition
silently evaluated the FP32 model. If it fails, try PCG_PT2E_BACKEND=torch.ao (or torchao), or
install a matching torchao for the image's torch.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from pcg import quant as Q  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    net = nn.Sequential(
        nn.Conv1d(1, 8, 9, padding=4), nn.ReLU(),
        nn.Conv1d(8, 8, 9, padding=4), nn.ReLU(),
        nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(8, 3),
    ).eval()
    calib = [torch.randn(16, 1, 400) for _ in range(4)]
    print("pt2e selftest:", Q.env_versions(), flush=True)
    ref, conv = Q.ptq_static(net, calib, per_channel=True)  # raises on a silent no-op
    x = torch.randn(32, 1, 400)
    with torch.no_grad():
        d = float((conv(x) - net(x)).abs().max())
    print(f"observers={Q.n_quant_modules(ref)} qdq_nodes={Q.n_qdq_nodes(conv)} max|dy| vs fp32={d:.3g}")
    assert d > 1e-6, "converted model is numerically identical to FP32 - quantization is a no-op"
    print("OK: PT2E quantization is real in this environment")


if __name__ == "__main__":
    main()
