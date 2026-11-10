"""Task 2(b): per-layer weight quantization error, per-tensor vs per-channel INT8 and INT4.

For every conv layer (post BN-fold), quantize the weight tensor symmetrically to
INT8 / INT4 with (i) one scale per tensor and (ii) one scale per output channel,
then report SQNR (dB) and relative Frobenius error. Lists the worst layers.

Usage (repo root):
    .venv/Scripts/python.exe revision_v2/scripts/weight_quant_error.py \
        [--fold 0] [--seed 0] [--out revision_v2/weight_quant_error.csv]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from pcg.models import build_model  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from weight_range_disparity import conv_bn_pairs, fold_bn, layer_type  # noqa: E402

MODELS = ["tiny_cnn", "m5", "logmel_cnn", "effnet_b0"]


def quantize(w: torch.Tensor, n_bits: int, per_channel: bool) -> torch.Tensor:
    qmax = 2 ** (n_bits - 1) - 1
    if per_channel:
        s = w.abs().flatten(1).amax(dim=1).clamp(min=1e-12) / qmax
        s = s.view(-1, *([1] * (w.dim() - 1)))
    else:
        s = w.abs().max().clamp(min=1e-12) / qmax
    return (w / s).round().clamp(-qmax - 1, qmax) * s


def sqnr_db(w: torch.Tensor, wq: torch.Tensor) -> float:
    noise = (w - wq).pow(2).sum().item()
    sig = w.pow(2).sum().item()
    if noise == 0:
        return float("inf")
    return 10.0 * torch.log10(torch.tensor(sig / noise)).item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs-dir", default=str(REPO / "runs"))
    ap.add_argument("--out", default=str(REPO / "revision_v2" / "weight_quant_error.csv"))
    args = ap.parse_args()

    rows = []
    for model_name in MODELS:
        ckpt = Path(args.runs_dir) / model_name / f"fold{args.fold}" / f"seed{args.seed}" / "best.pt"
        if not ckpt.exists():
            print(f"[PENDING-RUN] missing checkpoint: {ckpt}")
            continue
        m = build_model(model_name)
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd) if isinstance(sd, dict) else sd
        m.load_state_dict(sd)
        m.eval()
        for name, conv, bn in conv_bn_pairs(m):
            w = fold_bn(conv, bn)
            row = dict(model=model_name, layer=name, type=layer_type(conv))
            for bits in (8, 4):
                for pc in (False, True):
                    wq = quantize(w, bits, pc)
                    tag = f"int{bits}_{'pc' if pc else 'pt'}"
                    row[f"sqnr_db_{tag}"] = round(sqnr_db(w, wq), 2)
                    row[f"rel_frob_{tag}"] = round(((w - wq).norm() / w.norm().clamp(min=1e-12)).item(), 5)
            rows.append(row)

    with open(args.out, "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(rows)
    print(f"wrote {args.out}\n")

    import statistics
    print(f"{'model':<12}{'type':<11}{'n':>3} | median SQNR dB: int8_pt / int8_pc / int4_pt / int4_pc")
    for mn in MODELS:
        for t in sorted({r['type'] for r in rows if r['model'] == mn}):
            rs = [r for r in rows if r['model'] == mn and r['type'] == t]
            med = lambda k: statistics.median(x[k] for x in rs)  # noqa: E731
            print(f"{mn:<12}{t:<11}{len(rs):>3} |  {med('sqnr_db_int8_pt'):6.1f} / {med('sqnr_db_int8_pc'):6.1f}"
                  f" / {med('sqnr_db_int4_pt'):6.1f} / {med('sqnr_db_int4_pc'):6.1f}")
    print("\nWorst 8 layers by int8 per-tensor SQNR:")
    for r in sorted(rows, key=lambda x: x["sqnr_db_int8_pt"])[:8]:
        print(f"  {r['model']:<11} {r['type']:<10} {r['layer']:<55} {r['sqnr_db_int8_pt']:6.1f} dB"
              f" (pc: {r['sqnr_db_int8_pc']:6.1f} dB)")


if __name__ == "__main__":
    main()
