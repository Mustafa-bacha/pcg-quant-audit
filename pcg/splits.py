"""Patient-level stratified K-fold splits for CirCor.

Rules (FINAL_research_problem.md, section 9):
- split by PATIENT, never by recording -> no leakage of the same child across folds
- stratify by murmur label x campaign (the two factors we later analyse for shift)
- fixed seed, written to splits/folds_v1.json so every experiment uses identical folds
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import SPLITS_DIR

DEFAULT_SPLIT_FILE = SPLITS_DIR / "folds_v1.json"


def make_folds(pdf: pd.DataFrame, n_folds: int = 5, seed: int = 0) -> dict[str, int]:
    """Return {patient_id: fold_index} with stratification on murmur x campaign."""
    strata = pdf["murmur"].astype(str) + "|" + pdf["campaign"].astype(str)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    ids = np.asarray(pdf.index, dtype=str)
    assign: dict[str, int] = {}
    for k, (_, test_idx) in enumerate(skf.split(ids, strata.values)):
        for i in test_idx:
            assign[ids[i]] = k
    return assign


def save_folds(assign: dict[str, int], pdf: pd.DataFrame, path: Path = DEFAULT_SPLIT_FILE,
               n_folds: int = 5, seed: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    per_fold = []
    for k in range(n_folds):
        sub = pdf.loc[[p for p, f in assign.items() if f == k]]
        per_fold.append(
            {
                "fold": k,
                "n_patients": int(len(sub)),
                "murmur": sub["murmur"].value_counts().to_dict(),
                "campaign": sub["campaign"].value_counts().to_dict(),
            }
        )
    doc = {
        "version": "v1",
        "n_folds": n_folds,
        "seed": seed,
        "stratify": "murmur|campaign",
        "level": "patient",
        "summary": per_fold,
        "assignment": assign,
    }
    path.write_text(json.dumps(doc, indent=1))


def load_folds(path: Path = DEFAULT_SPLIT_FILE) -> dict[str, int]:
    return json.loads(path.read_text())["assignment"]


def fold_patients(assign: dict[str, int], fold: int) -> tuple[list[str], list[str]]:
    """(train_ids, test_ids) for one fold."""
    test = sorted(p for p, f in assign.items() if f == fold)
    train = sorted(p for p, f in assign.items() if f != fold)
    return train, test


def check_no_leakage(assign: dict[str, int], rdf: pd.DataFrame) -> None:
    """Every recording of a patient must land in exactly one fold."""
    folds = rdf["patient_id"].map(assign)
    assert folds.notna().all(), "some recordings belong to patients without a fold"
    n_per_patient = rdf.assign(f=folds).groupby("patient_id")["f"].nunique()
    assert (n_per_patient == 1).all(), "a patient appears in more than one fold"
