"""Phase 0: extract CirCor, verify counts, build patient-level folds, prove the official
scorer is wired in. Run from the project root:

    .venv\\Scripts\\python.exe scripts\\p0_verify_data.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcg import RAW_DIR, SPLITS_DIR  # noqa: E402
from pcg.circor import find_training_dir, load_metadata, load_segmentation, load_wav, summarize  # noqa: E402
from pcg.scoring import run_official_scorer, score_arrays, write_official_outputs  # noqa: E402
from pcg.splits import DEFAULT_SPLIT_FILE, check_no_leakage, make_folds, save_folds  # noqa: E402

ZIP = RAW_DIR / "circor-1.0.3.zip"

# Numbers recomputed in the 2026-10-20 feasibility pass (FINAL_research_problem.md, section 6.1)
EXPECTED = {
    "patients": 942,
    "recordings": 3163,
    "murmur": {"Absent": 695, "Present": 179, "Unknown": 68},
    "outcome": {"Normal": 486, "Abnormal": 456},
    "location": {"MV": 861, "AV": 800, "PV": 766, "TV": 732, "Phc": 4},
    # 586 patients have exactly one recording per site (AV+PV+TV+MV in training_data.csv);
    # 12 more have all four sites plus a repeat recording at one site -> 598 cover all four.
    "patients_with_all_4_sites": 598,
    "sex": {"Female": 486, "Male": 456},
    "pregnant": 70,
    "sys_grading": {"I/VI": 104, "II/VI": 28, "III/VI": 46},
    "sys_timing": {"Holosystolic": 101, "Early-systolic": 59, "Mid-systolic": 17, "Late-systolic": 1},
}


def step(msg: str) -> None:
    print(f"\n=== {msg}")


def extract_if_needed() -> Path:
    try:
        return find_training_dir()
    except FileNotFoundError:
        pass
    step(f"Extracting {ZIP.name} ({ZIP.stat().st_size / 1e6:.0f} MB)")
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(RAW_DIR)
    return find_training_dir()


def compare(actual: dict, expected: dict) -> list[str]:
    problems = []
    for k, exp in expected.items():
        act = actual.get(k)
        if isinstance(exp, dict):
            for kk, vv in exp.items():
                if act.get(kk) != vv:
                    problems.append(f"{k}[{kk}]: expected {vv}, got {act.get(kk)}")
        elif act != exp:
            problems.append(f"{k}: expected {exp}, got {act}")
    return problems


def main() -> None:
    training_dir = extract_if_needed()
    step(f"Training data at {training_dir}")

    pdf, rdf = load_metadata(training_dir)
    summ = summarize(pdf, rdf)
    print(json.dumps(summ, indent=1, default=str))

    step("Comparing with the counts from the feasibility pass")
    problems = compare(summ, EXPECTED)
    if problems:
        print("MISMATCHES:")
        for p in problems:
            print("  -", p)
    else:
        print("All expected counts match.")

    step("Checking audio and segmentation files")
    missing_wav = [p for p in rdf["wav"] if not Path(p).exists()]
    print(f"missing wav files: {len(missing_wav)}")
    print(f"recordings with .tsv: {summ['recordings_with_tsv']} / {summ['recordings']}")
    x, fs = load_wav(rdf["wav"].iloc[0])
    print(f"example {rdf['recording_id'].iloc[0]}: {len(x)} samples @ {fs} Hz = {len(x) / fs:.1f} s, "
          f"peak {np.abs(x).max():.3f}")
    seg = load_segmentation(rdf["tsv"].iloc[0])
    print(f"example segmentation rows: {len(seg)}, labels present: {sorted(set(seg[:, 2].astype(int)))}")
    # Fraction of annotated (non-zero) time over all recordings, cheap estimate on 200 files
    cover = []
    for tsv in rdf["tsv"].iloc[:200]:
        s = load_segmentation(tsv)
        dur = s[:, 1].max() if len(s) else 0
        ann = s[s[:, 2] > 0]
        cover.append((ann[:, 1] - ann[:, 0]).sum() / dur if dur else 0)
    print(f"annotated time fraction (first 200 recordings): mean {np.mean(cover):.2f}, "
          f"min {np.min(cover):.2f}, zero-annotation files {sum(c == 0 for c in cover)}")

    step("Building patient-level 5-fold splits (seed 0, stratified by murmur x campaign)")
    assign = make_folds(pdf, n_folds=5, seed=0)
    check_no_leakage(assign, rdf)
    save_folds(assign, pdf, DEFAULT_SPLIT_FILE)
    doc = json.loads(DEFAULT_SPLIT_FILE.read_text())
    for f in doc["summary"]:
        print(f"  fold {f['fold']}: {f['n_patients']} patients, murmur={f['murmur']}, campaign={f['campaign']}")
    print(f"saved {DEFAULT_SPLIT_FILE}")

    step("Scorer check 1: in-memory wrapper vs. known-answer cases")
    yt = pdf["murmur"].tolist()
    ot = pdf["outcome"].tolist()
    # oracle: perfect predictions -> WA must be 1.0
    prob_oracle = np.zeros((len(yt), 3))
    for i, lab in enumerate(yt):
        prob_oracle[i, ["Present", "Unknown", "Absent"].index(lab)] = 1.0
    r = score_arrays(yt, yt, prob_oracle, ot)
    print(f"oracle:        WA={r['weighted_accuracy']:.3f} AUROC={r['auroc']:.3f} cost={r['cost']:.0f}")
    assert abs(r["weighted_accuracy"] - 1.0) < 1e-9
    # all-Absent: WA = 695 / (5*179 + 3*68 + 695) = 695/1794 = 0.3874
    all_absent = ["Absent"] * len(yt)
    prob_abs = np.tile([0.0, 0.0, 1.0], (len(yt), 1))
    r = score_arrays(yt, all_absent, prob_abs, ot)
    print(f"all-Absent:    WA={r['weighted_accuracy']:.3f} (expected 0.387) cost={r['cost']:.0f}")
    assert abs(r["weighted_accuracy"] - 695 / 1794) < 1e-6
    all_present = ["Present"] * len(yt)
    prob_pre = np.tile([1.0, 0.0, 0.0], (len(yt), 1))
    r = score_arrays(yt, all_present, prob_pre, ot)
    print(f"all-Present:   WA={r['weighted_accuracy']:.3f} (expected 0.499) cost={r['cost']:.0f}")

    step("Scorer check 2: organisers' evaluate_model.py as a subprocess, same oracle case")
    tmp = Path(tempfile.mkdtemp(prefix="circor_score_"))
    try:
        write_official_outputs(tmp / "out", pdf.index.tolist(), yt, prob_oracle,
                               outcome_pred=ot, outcome_prob=np.array([[1, 0] if o == "Abnormal" else [0, 1] for o in ot], dtype=float))
        text = run_official_scorer(training_dir, tmp / "out", tmp / "scores.csv")
        print(text)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    step("Phase 0 verification complete")


if __name__ == "__main__":
    main()
