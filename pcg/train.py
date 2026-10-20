"""Train one (model, fold, seed) and evaluate on the held-out test fold.

Fold protocol (no selection on the test fold):
    test  = fold k
    val   = fold (k + 1) % 5         -> early stopping / checkpoint selection on WA
    train = the remaining 3 folds

Outputs in <runs>/<model>/fold<k>/seed<s>/:
    config.json          everything needed to reproduce
    best.pt              state_dict of the best-val-WA epoch
    history.csv          per-epoch train loss, val WA, val AUROC
    rec_probs_test.csv   per-recording probabilities on the test fold (Present, Unknown, Absent)
    patient_test.json    official metrics on the test fold after patient aggregation
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import PROJECT_ROOT
from .circor import MURMUR_CLASSES, load_metadata
from .dataset import TrainCrops, eval_windows, load_waveforms
from .models import build_model, n_params
from .scoring import aggregate_recordings_to_patient, score_arrays
from .splits import DEFAULT_SPLIT_FILE, load_folds


@dataclass
class Config:
    model: str = "tiny_cnn"
    fold: int = 0
    seed: int = 0
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    crops_per_recording: int = 4
    class_weights: tuple[float, float, float] = (5.0, 3.0, 1.0)  # Present, Unknown, Absent = metric weights
    mixup_alpha: float = 0.0  # set 0.2 for logmel_cnn
    label_smoothing: float = 0.0
    num_workers: int = 0
    runs_dir: str = str(PROJECT_ROOT / "runs")
    split_file: str = str(DEFAULT_SPLIT_FILE)
    smoke: bool = False  # tiny subset + 2 epochs, for pipeline checks
    device: str = "auto"
    notes: str = ""
    extra: dict = field(default_factory=dict)


def pick_device(spec: str) -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_lr(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(total - warmup, 1)
    return base * 0.5 * (1 + math.cos(math.pi * p))


@torch.no_grad()
def predict_recordings(model: nn.Module, rdf: pd.DataFrame, waves: dict, device: torch.device,
                       batch_windows: int = 256) -> pd.DataFrame:
    """Sliding-window inference, mean of softmax over windows per recording."""
    model.eval()
    rows = []
    for r in rdf.itertuples():
        w = torch.from_numpy(eval_windows(waves[r.recording_id]))
        probs = []
        for i in range(0, len(w), batch_windows):
            logits = model(w[i : i + batch_windows].to(device))
            probs.append(F.softmax(logits, dim=1).cpu())
        p = torch.cat(probs).mean(dim=0).numpy()
        rows.append({"recording_id": r.recording_id, "patient_id": r.patient_id,
                     "p_present": p[0], "p_unknown": p[1], "p_absent": p[2]})
    return pd.DataFrame(rows)


def patient_metrics(rec: pd.DataFrame, pdf: pd.DataFrame) -> dict:
    pids, pprob, ppred = aggregate_recordings_to_patient(
        rec["patient_id"].values, rec[["p_present", "p_unknown", "p_absent"]].values)
    truth = pdf.loc[pids, "murmur"].tolist()
    outcome = pdf.loc[pids, "outcome"].tolist()
    return score_arrays(truth, ppred, pprob, outcome)


def run(cfg: Config, pdf: pd.DataFrame | None = None, rdf: pd.DataFrame | None = None,
        waves: dict | None = None) -> dict:
    out_dir = Path(cfg.runs_dir) / cfg.model / f"fold{cfg.fold}" / f"seed{cfg.seed}"
    if (out_dir / "patient_test.json").exists():
        print(f"[skip] {out_dir} already done")
        return json.loads((out_dir / "patient_test.json").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(cfg.device)
    set_seed(cfg.seed)

    if pdf is None:
        pdf, rdf = load_metadata()
    folds = load_folds(Path(cfg.split_file))
    n_folds = max(folds.values()) + 1
    test_f, val_f = cfg.fold, (cfg.fold + 1) % n_folds
    rdf = rdf.assign(fold=rdf["patient_id"].map(folds))
    tr = rdf[~rdf["fold"].isin([test_f, val_f])]
    va = rdf[rdf["fold"] == val_f]
    te = rdf[rdf["fold"] == test_f]
    if cfg.smoke:
        tr, va, te = tr.sample(120, random_state=0), va.sample(40, random_state=0), te.sample(40, random_state=0)
        cfg.epochs = 2
    if waves is None:
        waves = load_waveforms(pd.concat([tr, va, te]))

    train_ds = TrainCrops(tr, waves, cfg.crops_per_recording, augment=True, seed=cfg.seed)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                        drop_last=True, pin_memory=device.type == "cuda")
    model = build_model(cfg.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    cw = torch.tensor(cfg.class_weights, dtype=torch.float32, device=device)
    cw = cw / cw.sum() * len(cw)
    steps_per_epoch = len(loader)
    total_steps, warmup_steps = cfg.epochs * steps_per_epoch, cfg.warmup_epochs * steps_per_epoch

    cfg_dump = asdict(cfg) | {"n_params": n_params(model), "device": str(device),
                              "n_train_rec": len(tr), "n_val_rec": len(va), "n_test_rec": len(te),
                              "train_class_counts": train_ds.class_counts().tolist(),
                              "classes": MURMUR_CLASSES, "test_fold": test_f, "val_fold": val_f}
    (out_dir / "config.json").write_text(json.dumps(cfg_dump, indent=1))
    print(f"[{cfg.model} fold{cfg.fold} seed{cfg.seed}] {device} params={n_params(model):,} "
          f"train={len(tr)} val={len(va)} test={len(te)} rec; {steps_per_epoch} steps/epoch")

    history, best_wa, best_state, step = [], -1.0, None, 0
    for epoch in range(cfg.epochs):
        model.train()
        t0, losses = time.time(), []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            lr = cosine_lr(step, total_steps, warmup_steps, cfg.lr)
            for g in opt.param_groups:
                g["lr"] = lr
            if cfg.mixup_alpha > 0:
                lam = float(np.random.beta(cfg.mixup_alpha, cfg.mixup_alpha))
                perm = torch.randperm(x.size(0), device=device)
                x = lam * x + (1 - lam) * x[perm]
                logits = model(x)
                loss = lam * F.cross_entropy(logits, y, weight=cw, label_smoothing=cfg.label_smoothing) + \
                       (1 - lam) * F.cross_entropy(logits, y[perm], weight=cw, label_smoothing=cfg.label_smoothing)
            else:
                loss = F.cross_entropy(model(x), y, weight=cw, label_smoothing=cfg.label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())
            step += 1
        val_rec = predict_recordings(model, va, waves, device)
        vm = patient_metrics(val_rec, pdf)
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_wa": vm["weighted_accuracy"],
               "val_auroc": vm["auroc"], "val_present_recall": vm["present_recall"], "sec": time.time() - t0}
        history.append(rec)
        improved = vm["weighted_accuracy"] > best_wa
        if improved:
            best_wa = vm["weighted_accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, out_dir / "best.pt")
        print(f"  ep{epoch:02d} loss={rec['train_loss']:.3f} valWA={rec['val_wa']:.3f} "
              f"valAUROC={rec['val_auroc']:.3f} presRec={rec['val_present_recall']:.2f} "
              f"{rec['sec']:.0f}s{' *' if improved else ''}")
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    model.load_state_dict(best_state)
    test_rec = predict_recordings(model, te, waves, device)
    test_rec.to_csv(out_dir / "rec_probs_test.csv", index=False)
    tm = patient_metrics(test_rec, pdf) | {"best_val_wa": best_wa, "epochs_run": cfg.epochs}
    (out_dir / "patient_test.json").write_text(json.dumps(tm, indent=1))
    print(f"  TEST fold{cfg.fold}: WA={tm['weighted_accuracy']:.3f} AUROC={tm['auroc']:.3f} "
          f"presRec={tm['present_recall']:.2f} cost={tm['cost']:.0f}")
    return tm
