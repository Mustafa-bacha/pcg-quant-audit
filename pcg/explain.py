"""Phase 3 explanations: attributions on waveform windows, drift between precisions, and
heart-phase grounding using CirCor's S1/systole/S2/diastole annotations.

Every attribution is reduced to a per-frame time profile (25 ms frames, FRAME samples each) so
that raw-waveform and log-mel models are compared on the same axis:
  * raw-waveform backbones: attribution per sample -> |.| summed within frames
  * log-mel backbones:     attribution per (mel, frame) -> |.| summed over mel bins

Methods
  ig          Integrated Gradients w.r.t. the backbone input (Captum), zero baseline, 32 steps
  gradxinput  Gradient x Input w.r.t. the backbone input
  occlusion   model-agnostic: zero out one 250 ms block at a time, drop in P(Present) is the attribution;
              the only method that runs on the real INT8 / ONNX Runtime backbones
  gradcam     Grad-CAM on the last conv layer; FP32 eager modules only (needs a named layer)

Gradient methods run on *fake-quant* graphs for the quantized conditions (autograd through the
straight-through estimator); occlusion runs on the converted INT8 backbones. Report both.

Phase metrics (frames whose majority segmentation label is known):
  PRR_sys  = attribution mass in systole frames / mass in all annotated frames
  PRR_s1s2 = mass in S1+S2 frames / mass in annotated frames
  control  = same quantity with systole and diastole labels swapped (Rohr et al. 2024 style control)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

from .dataset import WINDOW
from .features import FS

FRAME = 100  # 25 ms at 4 kHz, = the log-mel hop
N_FRAMES = WINDOW // FRAME  # 200
PRESENT = 0


# --------------------------------------------------------------------------- helpers

def to_frames(attr: torch.Tensor | np.ndarray) -> np.ndarray:
    """Reduce an attribution over backbone input to a (N_FRAMES,) non-negative profile."""
    a = np.abs(np.asarray(attr.detach().cpu().numpy() if torch.is_tensor(attr) else attr, dtype=np.float64))
    if a.ndim == 1:  # waveform (T,)
        n = (len(a) // FRAME) * FRAME
        return a[:n].reshape(-1, FRAME).sum(axis=1)
    if a.ndim == 3:  # (1, mel, frames)
        a = a[0]
    if a.ndim == 2:  # (mel, frames)
        prof = a.sum(axis=0)
        return np.interp(np.linspace(0, 1, N_FRAMES), np.linspace(0, 1, len(prof)), prof)
    raise ValueError(f"unexpected attribution shape {a.shape}")


def phase_frames(sample_labels: np.ndarray) -> np.ndarray:
    """Per-sample segmentation labels (0..4) for one window -> per-frame majority label (N_FRAMES,)."""
    n = (len(sample_labels) // FRAME) * FRAME
    fr = sample_labels[:n].reshape(-1, FRAME)
    out = np.zeros(fr.shape[0], dtype=np.int8)
    for i, row in enumerate(fr):
        counts = np.bincount(row, minlength=5)
        out[i] = int(np.argmax(counts))
    return out


# --------------------------------------------------------------------------- attribution methods

def _present_logit(backbone: nn.Module, feats: torch.Tensor) -> torch.Tensor:
    return backbone(feats)[:, PRESENT]


def attr_gradxinput(backbone: nn.Module, feats: torch.Tensor) -> torch.Tensor:
    x = feats.clone().requires_grad_(True)
    _present_logit(backbone, x).sum().backward()
    return (x.grad * x).detach()


def attr_ig(backbone: nn.Module, feats: torch.Tensor, steps: int = 32) -> torch.Tensor:
    """Integrated Gradients, zero baseline, Riemann-trapezoid in `steps` chunks (no Captum dependency
    needed here, but numerically equivalent to captum.IntegratedGradients with method='riemann_trapezoid')."""
    base = torch.zeros_like(feats)
    alphas = torch.linspace(0, 1, steps + 1)
    total = torch.zeros_like(feats)
    for i in range(steps):
        a0, a1 = alphas[i], alphas[i + 1]
        for a, w in ((a0, 0.5), (a1, 0.5)):
            x = (base + a * (feats - base)).requires_grad_(True)
            _present_logit(backbone, x).sum().backward()
            total += w * x.grad / steps
    return ((feats - base) * total).detach()


@torch.no_grad()
def attr_occlusion(assembled: nn.Module, wav: torch.Tensor, block: int = 1000, stride: int = 500,
                   batch: int = 64) -> np.ndarray:
    """Zero-mask occlusion on the waveform; returns per-sample attribution = drop in P(Present).
    Works for any backbone, including converted INT8 and ONNX Runtime."""
    assert wav.ndim == 2 and wav.shape[0] == 1
    T = wav.shape[1]
    p0 = F.softmax(assembled(wav), dim=1)[0, PRESENT].item()
    starts = list(range(0, T - block + 1, stride))
    attr = np.zeros(T)
    count = np.zeros(T)
    for i in range(0, len(starts), batch):
        chunk = starts[i : i + batch]
        xs = wav.repeat(len(chunk), 1)
        for j, s in enumerate(chunk):
            xs[j, s : s + block] = 0
        p = F.softmax(assembled(xs), dim=1)[:, PRESENT].numpy()
        for j, s in enumerate(chunk):
            attr[s : s + block] += p0 - p[j]
            count[s : s + block] += 1
    return attr / np.maximum(count, 1)


def attr_gradcam(backbone: nn.Module, feats: torch.Tensor, layer: nn.Module) -> np.ndarray:
    """Grad-CAM on `layer` (last conv). Returns (N_FRAMES,) profile. FP32 eager modules only."""
    acts, grads = {}, {}
    h1 = layer.register_forward_hook(lambda m, i, o: acts.setdefault("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.setdefault("g", go[0]))
    try:
        _present_logit(backbone, feats).sum().backward()
    finally:
        h1.remove()
        h2.remove()
    a, g = acts["a"].detach(), grads["g"].detach()
    w = g.mean(dim=tuple(range(2, g.ndim)), keepdim=True)
    cam = F.relu((w * a).sum(dim=1))  # (1, T') or (1, F', T')
    cam = cam[0].cpu().numpy()
    prof = cam.sum(axis=0) if cam.ndim == 2 else cam
    return np.interp(np.linspace(0, 1, N_FRAMES), np.linspace(0, 1, len(prof)), prof)


def last_conv(backbone: nn.Module) -> nn.Module:
    convs = [m for m in backbone.modules() if isinstance(m, (nn.Conv1d, nn.Conv2d))]
    return convs[-1]


# --------------------------------------------------------------------------- drift metrics

def drift(a: np.ndarray, b: np.ndarray, top_frac: float = 0.1) -> dict[str, float]:
    """Similarity of two frame profiles (reference a vs. quantized b)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    eps = 1e-12
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))
    sp = float(spearmanr(a, b).correlation) if a.std() > 0 and b.std() > 0 else float("nan")
    pc = float(pearsonr(a, b)[0]) if a.std() > 0 and b.std() > 0 else float("nan")
    k = max(int(round(top_frac * len(a))), 1)
    ta, tb = set(np.argsort(-a)[:k]), set(np.argsort(-b)[:k])
    k2 = max(int(round(0.2 * len(a))), 1)
    ta2, tb2 = set(np.argsort(-a)[:k2]), set(np.argsort(-b)[:k2])
    return {"cosine": cos, "spearman": sp, "pearson": pc,
            "top10_overlap": len(ta & tb) / k, "top20_iou": len(ta2 & tb2) / max(len(ta2 | tb2), 1)}


# --------------------------------------------------------------------------- phase grounding

def phase_relevance(profile: np.ndarray, frame_labels: np.ndarray) -> dict[str, float]:
    """Share of attribution mass in each cardiac phase, over annotated frames only; plus the
    systole/diastole-swapped control. Returns NaN when fewer than 20% of frames are annotated."""
    p = np.asarray(profile, float)
    lab = np.asarray(frame_labels)
    ann = lab > 0
    if ann.mean() < 0.2 or p[ann].sum() <= 0:
        return {k: float("nan") for k in ("prr_sys", "prr_dia", "prr_s1s2", "prr_sys_control", "frac_annotated")}
    mass = p[ann].sum()
    sys_, dia, s1s2 = p[lab == 2].sum() / mass, p[lab == 4].sum() / mass, p[(lab == 1) | (lab == 3)].sum() / mass
    # control: pretend systole is where diastole is (and vice versa)
    swapped = lab.copy(); swapped[lab == 2] = 4; swapped[lab == 4] = 2
    ctrl = p[swapped == 2].sum() / mass
    # time-share baselines (what a uniform attribution would score)
    share_sys = float((lab == 2).sum() / ann.sum())
    return {"prr_sys": float(sys_), "prr_dia": float(dia), "prr_s1s2": float(s1s2), "prr_sys_control": float(ctrl),
            "prr_sys_uniform": share_sys, "frac_annotated": float(ann.mean())}


# --------------------------------------------------------------------------- faithfulness

@torch.no_grad()
def deletion_curve(assembled: nn.Module, wav: torch.Tensor, profile: np.ndarray, steps: int = 10,
                   restrict_frames: np.ndarray | None = None) -> dict:
    """Remove (zero) the top-attributed frames in `steps` increments and track P(Present).
    `restrict_frames` (bool mask over frames) limits deletion to e.g. systole frames only.
    Returns the curve and its normalised AUC (lower = attribution found the evidence)."""
    order = np.argsort(-np.asarray(profile, float))
    if restrict_frames is not None:
        order = np.array([i for i in order if restrict_frames[i]], dtype=int)
    n = len(order)
    p0 = F.softmax(assembled(wav), dim=1)[0, PRESENT].item()
    ys = [p0]
    xs = torch.empty((steps, wav.shape[1]))
    for s in range(1, steps + 1):
        k = int(round(n * s / steps))
        x = wav.clone()
        for i in order[:k]:
            x[0, i * FRAME : (i + 1) * FRAME] = 0
        xs[s - 1] = x[0]
    ps = F.softmax(assembled(xs), dim=1)[:, PRESENT].numpy()
    ys.extend(ps.tolist())
    ys = np.asarray(ys)
    return {"p0": p0, "curve": ys.tolist(), "auc_norm": float(np.trapezoid(ys, dx=1 / steps) / max(p0, 1e-6)),
            "n_frames_deleted_total": n}


def random_deletion_curve(assembled: nn.Module, wav: torch.Tensor, steps: int = 10, seed: int = 0,
                          restrict_frames: np.ndarray | None = None) -> dict:
    rng = np.random.default_rng(seed)
    prof = rng.random(N_FRAMES)
    return deletion_curve(assembled, wav, prof, steps, restrict_frames)
