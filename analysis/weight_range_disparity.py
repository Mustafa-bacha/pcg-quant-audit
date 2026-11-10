"""Task 2(a): per-convolution post-BN-fold per-output-channel weight range disparity.

For every conv layer (after folding the batch norm that immediately follows it),
compute r_c = max_c |W_c| per output channel and report the disparity ratio
max_c r_c / median_c r_c, grouped by layer type (stem / depthwise / pointwise /
regular / head-linear), for all four architectures.

High ratios mean a single per-tensor weight scale starves most channels of
integer levels (Krishnamoorthi 2018; Sheng et al. 2018; Nagel et al. 2019).

Usage (repo root):
    .venv/Scripts/python.exe revision_v2/scripts/weight_range_disparity.py \
        [--fold 0] [--seed 0] [--runs-dir runs] [--out revision_v2/weight_range_disparity.csv]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from pcg.models import build_model  # noqa: E402

MODELS = ["tiny_cnn", "m5", "logmel_cnn", "effnet_b0"]


def conv_bn_pairs(root: nn.Module):
    """Yield (name, conv, bn_or_None) for convs, pairing each conv with the BN that
    immediately follows it inside the same parent container."""
    for parent_name, parent in root.named_modules():
        children = list(parent.named_children())
        for i, (cname, child) in enumerate(children):
            if isinstance(child, (nn.Conv1d, nn.Conv2d)):
                bn = None
                if i + 1 < len(children) and isinstance(
                    children[i + 1][1], (nn.BatchNorm1d, nn.BatchNorm2d)
                ):
                    bn = children[i + 1][1]
                full = f"{parent_name}.{cname}" if parent_name else cname
                yield full, child, bn


def fold_bn(conv: nn.Conv2d | nn.Conv1d, bn) -> torch.Tensor:
    """Return the BN-folded weight tensor (out_ch, ...)."""
    w = conv.weight.detach()
    if bn is None:
        return w
    gamma = bn.weight.detach()
    var = bn.running_var.detach()
    eps = bn.eps
    scale = gamma / torch.sqrt(var + eps)  # (out_ch,)
    return w * scale.view(-1, *([1] * (w.dim() - 1)))


def layer_type(conv) -> str:
    if isinstance(conv, nn.Conv1d):
        return "conv1d"
    k = conv.kernel_size
    if conv.groups > 1 and conv.groups == conv.in_channels:
        return "depthwise"
    if k == (1, 1):
        return "pointwise"
    return "regular2d"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs-dir", default=str(REPO / "runs"))
    ap.add_argument("--out", default=str(REPO / "revision_v2" / "weight_range_disparity.csv"))
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
            r_c = w.abs().flatten(1).amax(dim=1)  # per-output-channel abs-max
            med = r_c.median().item()
            ratio = (r_c.max().item() / med) if med > 0 else float("inf")
            rows.append(
                dict(model=model_name, layer=name, type=layer_type(conv),
                     out_channels=w.shape[0], bn_folded=bn is not None,
                     r_max=round(r_c.max().item(), 6), r_median=round(med, 6),
                     ratio_max_over_median=round(ratio, 2))
            )
        print(f"{model_name}: {sum(r['model']==model_name for r in rows)} conv layers analysed")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")

    # summary by model x type
    import statistics
    print("\nSummary: ratio max/median of per-channel |w|max (post BN-fold)")
    print(f"{'model':<12} {'type':<10} {'n':>3}  {'median_ratio':>12} {'max_ratio':>10}")
    for mn in MODELS:
        for t in sorted({r["type"] for r in rows if r["model"] == mn}):
            rs = [r["ratio_max_over_median"] for r in rows if r["model"] == mn and r["type"] == t]
            if rs:
                print(f"{mn:<12} {t:<10} {len(rs):>3}  {statistics.median(rs):>12.1f} {max(rs):>10.1f}")


if __name__ == "__main__":
    main()
