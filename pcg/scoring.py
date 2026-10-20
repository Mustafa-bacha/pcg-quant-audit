"""Official PhysioNet/CinC 2022 scoring, wired in two ways.

1. `score_arrays(...)`: call the *unmodified* functions from
   third_party/evaluation-2022-master/evaluate_model.py on in-memory arrays.
   This is what training/eval loops use.

2. `write_official_outputs(...)` + `run_official_scorer(...)`: write files in the
   Challenge output format and run `evaluate_model.py` as a subprocess, exactly as
   the organisers did. Use this once per final result table as the ground truth.

Patient-level aggregation rule (Challenge convention, also used here):
    Present > Unknown > Absent  -- if any recording says Present, the patient is Present.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from . import THIRD_PARTY_DIR
from .circor import MURMUR_CLASSES, OUTCOME_CLASSES

EVAL_DIR = THIRD_PARTY_DIR / "evaluation-2022-master"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import evaluate_model as _em  # noqa: E402  (official, unmodified)
import helper_code as _hc  # noqa: E402


def one_hot(labels: list[str] | np.ndarray, classes: list[str]) -> np.ndarray:
    out = np.zeros((len(labels), len(classes)), dtype=bool)
    for i, lab in enumerate(labels):
        out[i, classes.index(str(lab))] = True
    return out


def score_arrays(
    murmur_true: list[str] | np.ndarray,
    murmur_pred: list[str] | np.ndarray,
    murmur_prob: np.ndarray,
    outcome_true: list[str] | np.ndarray | None = None,
) -> dict[str, float]:
    """Compute the official murmur metrics from per-patient arrays.

    murmur_prob: (n_patients, 3) scores ordered as MURMUR_CLASSES = Present, Unknown, Absent.
    outcome_true: needed for the Challenge cost metric (uses outcomes to score murmur outputs).
    """
    yt = _em.enforce_positives(one_hot(murmur_true, MURMUR_CLASSES), MURMUR_CLASSES, "Present")
    yp = _em.enforce_positives(one_hot(murmur_pred, MURMUR_CLASSES), MURMUR_CLASSES, "Present")
    prob = np.asarray(murmur_prob, dtype=np.float64)
    assert prob.shape == yt.shape, f"prob shape {prob.shape} != {yt.shape}"

    auroc, auprc, auroc_c, auprc_c = _em.compute_auc(yt, prob)
    f1, f1_c = _em.compute_f_measure(yt, yp)
    acc, acc_c = _em.compute_accuracy(yt, yp)
    wa = _em.compute_weighted_accuracy(yt, yp, MURMUR_CLASSES)
    res = {
        "weighted_accuracy": float(wa),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "f1_macro": float(f1),
        "accuracy": float(acc),
        "present_recall": float(acc_c[0]),  # per-class accuracy == recall of that class
        "unknown_recall": float(acc_c[1]),
        "absent_recall": float(acc_c[2]),
        "auroc_present": float(auroc_c[0]),
    }
    if outcome_true is not None:
        ot = _em.enforce_positives(one_hot(outcome_true, OUTCOME_CLASSES), OUTCOME_CLASSES, "Abnormal")
        res["cost"] = float(_em.compute_cost(ot, yp, OUTCOME_CLASSES, MURMUR_CLASSES))
    return res


def aggregate_recordings_to_patient(
    rec_patient_ids: np.ndarray, rec_probs: np.ndarray, rule: str = "max_present"
) -> tuple[list[str], np.ndarray, list[str]]:
    """Combine per-recording class probabilities into one decision per patient.

    rule="max_present": patient score for Present = max over recordings, Unknown = max,
    Absent = min; decision = argmax of the aggregated vector with the Challenge priority
    Present > Unknown > Absent on ties.
    Returns (patient_ids, patient_probs (n,3), patient_pred_labels).
    """
    rec_patient_ids = np.asarray(rec_patient_ids, dtype=str)
    rec_probs = np.asarray(rec_probs, dtype=np.float64)
    pids = sorted(set(rec_patient_ids.tolist()), key=lambda s: int(s) if s.isdigit() else s)
    out = np.zeros((len(pids), 3))
    for i, pid in enumerate(pids):
        m = rec_probs[rec_patient_ids == pid]
        out[i] = [m[:, 0].max(), m[:, 1].max(), m[:, 2].min()]
    # normalise so rows sum to 1 (keeps ranking, makes AUROC well-defined)
    out = out / np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
    pred = [MURMUR_CLASSES[int(np.argmax(row))] for row in out]
    return pids, out, pred


def write_official_outputs(
    out_dir: Path,
    patient_ids: list[str],
    murmur_pred: list[str],
    murmur_prob: np.ndarray,
    outcome_pred: list[str] | None = None,
    outcome_prob: np.ndarray | None = None,
) -> None:
    """Write one <patient_id>.csv per patient in the Challenge output format."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(patient_ids)
    if outcome_pred is None:
        # No outcome model yet: emit a constant "Abnormal" so the file is well-formed.
        outcome_pred = ["Abnormal"] * n
        outcome_prob = np.tile([1.0, 0.0], (n, 1))
    for i, pid in enumerate(patient_ids):
        classes = MURMUR_CLASSES + OUTCOME_CLASSES
        labels = one_hot([murmur_pred[i]], MURMUR_CLASSES)[0].astype(int).tolist() + \
                 one_hot([outcome_pred[i]], OUTCOME_CLASSES)[0].astype(int).tolist()
        probs = list(map(float, murmur_prob[i])) + list(map(float, outcome_prob[i]))
        _hc.save_challenge_outputs(str(out_dir / f"{pid}.csv"), pid, classes, labels, probs)


def run_official_scorer(label_dir: Path, output_dir: Path, scores_csv: Path | None = None) -> str:
    """Run the organisers' evaluate_model.py unchanged and return its stdout / file text."""
    cmd = [sys.executable, str(EVAL_DIR / "evaluate_model.py"), str(label_dir), str(output_dir)]
    if scores_csv is not None:
        cmd.append(str(scores_csv))
    r = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=str(EVAL_DIR))
    return scores_csv.read_text() if scores_csv is not None else r.stdout
