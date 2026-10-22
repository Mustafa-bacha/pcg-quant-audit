"""ZCHSound (Jia et al., IEEE TBME 2024): pediatric CHD heart sounds, the second external test set (Phase 4).

Obtained 2026-10-22 from the first author's GitHub repository (github.com/WeiJieOvO/ZCHSound-Dataset,
Git LFS archive ZCHSound.zip, sha256 112432cb...58b7) after the official site zchsound.ncrcch.org.cn went offline.
Layout after extraction (data/raw/zchsound/extracted/ZCHSound/):
    clean Heartsound Data/ZCH0001.wav ...            941 files + "Clean Heartsound Data Details.csv" (comma-separated)
    Noise Heartsound Data Details/ZCH0942.wav ...    318 files + "Noise Heartsound Data Details.csv" (semicolon-separated)
Labels: NORMAL / ASD / VSD / PDA / PFO (echo-confirmed). 8 kHz, 16-bit; a few noisy files are stereo.
Device: ChildCare G-100 digital stethoscope, one site (left sternal border), 11-30 s per child.
We score P(Present) against "CHD" (any non-NORMAL diagnosis) and keep the clean/noisy split as a factor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

from . import RAW_DIR
from .features import FS, preprocess

ZCH_DIR = RAW_DIR / "zchsound" / "extracted" / "ZCHSound"
CHD_LABELS = ("ASD", "VSD", "PDA", "PFO")


def _age_group(days: float) -> str:
    if days <= 28:
        return "neonate"
    if days <= 365:
        return "infant"
    if days <= 12 * 365:
        return "child"
    return "adolescent"


def load_metadata(root: Path = ZCH_DIR) -> pd.DataFrame:
    clean = pd.read_csv(root / "Clean Heartsound Data Details.csv", encoding="utf-8")
    clean = clean[["fileName", "gender", "age(days)", "diagnosis"]].assign(subset="clean")
    noisy = pd.read_csv(root / "Noise Heartsound Data Details.csv", sep=";", encoding="utf-8").assign(subset="noisy")
    df = pd.concat([clean, noisy], ignore_index=True)
    df = df.rename(columns={"fileName": "recording_id", "gender": "sex", "age(days)": "age_days"})
    df["recording_id"] = df["recording_id"].str.replace(".wav", "", regex=False).str.strip()
    df["sex"] = df["sex"].str.strip().str.upper()
    df["diagnosis"] = df["diagnosis"].str.strip().str.upper()
    df["age_days"] = pd.to_numeric(df["age_days"], errors="coerce")
    df["age_group"] = df["age_days"].map(_age_group)
    df["chd"] = df["diagnosis"].isin(CHD_LABELS).astype(int)
    folder = {"clean": "clean Heartsound Data", "noisy": "Noise Heartsound Data Details"}
    df["wav"] = [root / folder[s] / f"{r}.wav" for s, r in zip(df["subset"], df["recording_id"])]
    df["patient_id"] = df["recording_id"]  # one recording per child
    missing = [p for p in df["wav"] if not p.exists()]
    assert not missing, f"{len(missing)} wav files listed in the CSVs are missing, e.g. {missing[:3]}"
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
        it = tqdm(it, total=len(df), desc="loading ZCHSound", unit="rec")
    return {r.recording_id: load_wave_4k(r.wav) for r in it}
