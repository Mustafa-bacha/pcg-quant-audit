"""PhysioNet/CinC 2016 heart-sound challenge: the multi-device external test set (Phase 4).

Layout (S3 mirror `challenge-2016/1.0.0/`, downloaded with
`scripts/download_circor_s3.py --prefix challenge-2016/1.0.0/ --dest physionet-2016 --only training-,validation/,annotations/`):
    training-a/a0001.wav ... + REFERENCE.csv  (name,label)  label: -1 normal, 1 abnormal
    training-b ... training-f
    validation/*.wav + REFERENCE.csv  (300 recordings, drawn from the training sets)
Recordings are 2,000 Hz; resampled here to 4,000 Hz so the CirCor pipeline applies unchanged.
Label semantics differ from CirCor (abnormal != murmur): we score P(Present) against "abnormal"
and say so explicitly (docs/lit_update_2026-10-20.md: JCM 2026 did the same).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

from . import RAW_DIR
from .features import FS, preprocess

P16_DIR = RAW_DIR / "physionet-2016"


def load_metadata(root: Path = P16_DIR, subsets: tuple[str, ...] = ("a", "b", "c", "d", "e", "f")) -> pd.DataFrame:
    rows = []
    for s in subsets:
        d = root / f"training-{s}"
        ref = pd.read_csv(d / "REFERENCE.csv", header=None, names=["name", "label"])
        for r in ref.itertuples():
            rows.append({"recording_id": r.name, "subset": s, "wav": d / f"{r.name}.wav",
                         "abnormal": int(r.label == 1)})
    df = pd.DataFrame(rows)
    df["patient_id"] = df["recording_id"]  # one recording per subject in the public set
    return df


def load_wave_4k(path: Path) -> np.ndarray:
    x, fs = sf.read(str(path), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if fs != FS:
        g = np.gcd(FS, fs)
        x = resample_poly(x, FS // g, fs // g).astype(np.float32)
    return preprocess(x, FS)


def load_waveforms(df: pd.DataFrame, verbose: bool = True) -> dict[str, np.ndarray]:
    it = df.itertuples()
    if verbose:
        from tqdm import tqdm
        it = tqdm(it, total=len(df), desc="loading PhysioNet 2016", unit="rec")
    return {r.recording_id: load_wave_4k(r.wav) for r in it}
