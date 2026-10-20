"""Recording-level dataset for CirCor.

Design choices (documented here because reviewers will ask):

* Recording label. A patient marked Present has the murmur audible only at the sites
  listed in `Murmur locations`. Recordings at the other sites are labelled Absent for
  training ("location-aware labels"). Unknown patients give Unknown for every site.
  The patient-level decision is recovered later by max-aggregation (pcg.scoring).

* Windows. Models see fixed 4 s windows (16,000 samples at 4 kHz). Training draws
  `crops_per_recording` random crops per recording per epoch; evaluation slides a
  window with 2 s hop over the whole recording and averages probabilities.

* Preprocessing (pcg.features.preprocess) is applied once at load time and cached in
  RAM as float32 (~1.2 GB for the full training set).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .circor import MURMUR_CLASSES, load_wav
from .features import FS, preprocess

WINDOW_S = 4.0
WINDOW = int(WINDOW_S * FS)
EVAL_HOP = int(2.0 * FS)


def recording_label(row: pd.Series) -> str:
    if row["murmur"] == "Present":
        return "Present" if row["murmur_at_location"] else "Absent"
    return row["murmur"]


def load_waveforms(rdf: pd.DataFrame, verbose: bool = True) -> dict[str, np.ndarray]:
    """recording_id -> preprocessed float32 waveform."""
    out = {}
    it = rdf.itertuples()
    if verbose:
        from tqdm import tqdm
        it = tqdm(it, total=len(rdf), desc="loading wav", unit="rec")
    for r in it:
        x, fs = load_wav(r.wav)
        assert fs == FS, f"{r.recording_id}: fs={fs}"
        out[r.recording_id] = preprocess(x, fs)
    return out


def _pad_to(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x
    reps = int(np.ceil(n / len(x)))
    return np.tile(x, reps)[:n]


class TrainCrops(Dataset):
    """Random fixed-length crops with light waveform augmentation."""

    def __init__(self, rdf: pd.DataFrame, waves: dict[str, np.ndarray], crops_per_recording: int = 4,
                 augment: bool = True, seed: int = 0):
        self.ids = rdf["recording_id"].tolist()
        self.labels = [MURMUR_CLASSES.index(recording_label(r)) for _, r in rdf.iterrows()]
        self.waves = waves
        self.k = crops_per_recording
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.ids) * self.k

    def __getitem__(self, i: int):
        j = i // self.k
        x = _pad_to(self.waves[self.ids[j]], WINDOW)
        start = int(self.rng.integers(0, len(x) - WINDOW + 1))
        seg = x[start : start + WINDOW].copy()
        if self.augment:
            gain_db = self.rng.uniform(-6, 6)
            seg *= 10 ** (gain_db / 20)
            if self.rng.random() < 0.5:
                seg = -seg  # polarity flip is physically harmless for a stethoscope
        return torch.from_numpy(seg), self.labels[j]

    def class_counts(self) -> np.ndarray:
        return np.bincount(self.labels, minlength=len(MURMUR_CLASSES))


def eval_windows(x: np.ndarray) -> np.ndarray:
    """All sliding windows of one recording, shape (n_windows, WINDOW)."""
    x = _pad_to(x, WINDOW)
    starts = list(range(0, len(x) - WINDOW + 1, EVAL_HOP))
    if starts[-1] != len(x) - WINDOW:
        starts.append(len(x) - WINDOW)
    return np.stack([x[s : s + WINDOW] for s in starts])
