"""CirCor DigiScope v1.0.3 loading: patient metadata, recordings, segmentations.

Patient .txt format (one per patient):

    2530 2 4000
    AV 2530_AV.hea 2530_AV.wav 2530_AV.tsv
    PV 2530_PV.hea 2530_PV.wav 2530_PV.tsv
    #Age: Child
    #Sex: Female
    ...
    #Murmur: Present
    #Outcome: Abnormal
    #Campaign: CC2014

Segmentation .tsv (one per recording): start_s, end_s, label
    0 = unannotated, 1 = S1, 2 = systole, 3 = S2, 4 = diastole
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from . import RAW_DIR

MURMUR_CLASSES = ["Present", "Unknown", "Absent"]
OUTCOME_CLASSES = ["Abnormal", "Normal"]
LOCATIONS = ["AV", "PV", "TV", "MV", "Phc"]
SEG_LABELS = {0: "unannotated", 1: "S1", 2: "systole", 3: "S2", 4: "diastole"}

# Fields written as "#Key: value" in each patient file -> column name
_PATIENT_FIELDS = {
    "Age": "age",
    "Sex": "sex",
    "Height": "height",
    "Weight": "weight",
    "Pregnancy status": "pregnant",
    "Murmur": "murmur",
    "Murmur locations": "murmur_locations",
    "Most audible location": "most_audible",
    "Systolic murmur timing": "sys_timing",
    "Systolic murmur shape": "sys_shape",
    "Systolic murmur grading": "sys_grading",
    "Systolic murmur pitch": "sys_pitch",
    "Systolic murmur quality": "sys_quality",
    "Diastolic murmur timing": "dia_timing",
    "Diastolic murmur shape": "dia_shape",
    "Diastolic murmur grading": "dia_grading",
    "Diastolic murmur pitch": "dia_pitch",
    "Diastolic murmur quality": "dia_quality",
    "Outcome": "outcome",
    "Campaign": "campaign",
    "Additional ID": "additional_id",
}


def find_training_dir(raw_dir: Path = RAW_DIR) -> Path:
    """Locate the extracted `training_data` folder under data/raw."""
    # Walk with symlinks followed: Path.rglob does not descend into symlinked dirs (Python 3.13),
    # and on Kaggle the dataset is attached as a symlink into /kaggle/input.
    import os
    candidates = []
    for root, dirs, _files in os.walk(raw_dir, followlinks=True):
        if "training_data" in dirs:
            candidates.append(Path(root) / "training_data")
    candidates = [c for c in candidates if c.is_dir() and any(c.glob("*.txt"))]
    if not candidates:
        raise FileNotFoundError(
            f"No extracted CirCor training_data folder under {raw_dir}. "
            "Run scripts/p0_verify_data.py to extract the zip."
        )
    return candidates[0]


def parse_patient_file(path: Path) -> tuple[dict, list[dict]]:
    """Return (patient_dict, [recording_dict, ...]) for one patient .txt file."""
    lines = path.read_text().splitlines()
    pid, n_loc, fs = lines[0].split()
    n_loc, fs = int(n_loc), int(fs)
    patient = {"patient_id": pid, "n_recordings": n_loc, "fs": fs}
    recordings = []
    for line in lines[1 : 1 + n_loc]:
        loc, hea, wav, tsv = line.split()
        recordings.append(
            {
                "patient_id": pid,
                "location": loc,
                "wav": path.parent / wav,
                "tsv": path.parent / tsv,
                "fs": fs,
            }
        )
    for line in lines[1 + n_loc :]:
        if not line.startswith("#") or ": " not in line:
            continue
        key, val = line[1:].split(": ", 1)
        col = _PATIENT_FIELDS.get(key.strip())
        if col is None:
            continue
        val = val.strip()
        patient[col] = np.nan if val.lower() == "nan" else val
    return patient, recordings


def load_metadata(training_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all patient files into (patients_df, recordings_df).

    patients_df: one row per patient, indexed by patient_id (str).
    recordings_df: one row per recording with patient-level labels joined in,
                   plus `recording_id` = f"{patient_id}_{location}" and
                   `murmur_at_location` (True if this site is in murmur_locations).
    """
    training_dir = training_dir or find_training_dir()
    patients, recordings = [], []
    for txt in sorted(training_dir.glob("*.txt"), key=lambda p: int(p.stem)):
        p, recs = parse_patient_file(txt)
        patients.append(p)
        recordings.extend(recs)

    pdf = pd.DataFrame(patients).set_index("patient_id")
    for c in ("height", "weight"):
        pdf[c] = pd.to_numeric(pdf[c], errors="coerce")
    pdf["pregnant"] = pdf["pregnant"].map({"True": True, "False": False})
    pdf["age"] = pdf["age"].astype("string")  # keep <NA> for missing

    rdf = pd.DataFrame(recordings)
    # A location can repeat within a patient (e.g. two AV recordings), so use the
    # wav stem, which is unique, as the recording id.
    rdf["recording_id"] = rdf["wav"].map(lambda p: Path(p).stem)
    rdf = rdf.join(
        pdf[["murmur", "outcome", "age", "sex", "campaign", "pregnant",
             "murmur_locations", "most_audible", "sys_timing", "sys_grading"]],
        on="patient_id",
    )
    rdf["murmur_at_location"] = [
        (loc in str(ml).split("+")) if isinstance(ml, str) else False
        for loc, ml in zip(rdf["location"], rdf["murmur_locations"])
    ]
    rdf["has_tsv"] = rdf["tsv"].map(lambda p: Path(p).exists())
    return pdf, rdf


def load_wav(path: Path | str) -> tuple[np.ndarray, int]:
    """Load a recording as float32 in [-1, 1] plus sample rate."""
    x, fs = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, fs


def load_segmentation(path: Path | str) -> np.ndarray:
    """Return array of shape (n, 3): start_s, end_s, label (int)."""
    arr = np.loadtxt(str(path), dtype=np.float64, ndmin=2)
    return arr


def segmentation_to_frame_labels(seg: np.ndarray, n_samples: int, fs: int) -> np.ndarray:
    """Expand a segmentation table to a per-sample label vector (int8)."""
    lab = np.zeros(n_samples, dtype=np.int8)
    for start, end, label in seg:
        a, b = int(round(start * fs)), int(round(end * fs))
        lab[max(a, 0) : min(b, n_samples)] = int(label)
    return lab


def summarize(pdf: pd.DataFrame, rdf: pd.DataFrame) -> dict:
    """Counts used to verify the dataset against the numbers in FINAL_research_problem.md."""
    return {
        "patients": int(len(pdf)),
        "recordings": int(len(rdf)),
        "recordings_with_tsv": int(rdf["has_tsv"].sum()),
        "murmur": pdf["murmur"].value_counts().to_dict(),
        "outcome": pdf["outcome"].value_counts().to_dict(),
        "age": pdf["age"].value_counts(dropna=False).rename(lambda k: str(k)).to_dict(),
        "sex": pdf["sex"].value_counts().to_dict(),
        "pregnant": int(pdf["pregnant"].fillna(False).sum()),
        "campaign": pdf["campaign"].value_counts().to_dict(),
        "location": rdf["location"].value_counts().to_dict(),
        "patients_with_all_4_sites": int(
            rdf.groupby("patient_id")["location"].apply(lambda s: {"AV", "PV", "TV", "MV"} <= set(s)).sum()
        ),
        "sys_timing": pdf["sys_timing"].dropna().value_counts().to_dict(),
        "sys_grading": pdf["sys_grading"].dropna().value_counts().to_dict(),
        "dia_timing": pdf["dia_timing"].dropna().value_counts().to_dict(),
        "fs_unique": sorted(rdf["fs"].unique().tolist()),
    }
