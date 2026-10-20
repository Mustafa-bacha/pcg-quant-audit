# Does Quantization Forget the Quiet Murmurs?

Trustworthiness audit of INT8/INT4-quantized pediatric heart-sound (PCG) classifiers on CirCor.
Research definition: `FINAL_research_problem.md`. Machine constraints: `SYSTEM_SPEC.md`.

## Layout

```
pcg/                 project package
  circor.py          CirCor metadata / wav / segmentation loading
  splits.py          patient-level stratified K-fold (splits/folds_v1.json)
  scoring.py         official CinC 2022 scorer, wrapped (unmodified code in third_party/)
scripts/
  p0_verify_data.py  Phase 0: extract, verify counts, build folds, test scorer
splits/              frozen fold assignments (committed)
data/raw/            CirCor zip + extracted files (not committed)
third_party/         physionetchallenges/evaluation-2022 and python-classifier-2022 (BSD-2)
runs/                experiment outputs (not committed)
```

## Setup (Windows, CPU)

```powershell
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts\p0_verify_data.py
```

## Phase 1: training

```powershell
# smoke test (CPU, ~1 min)
.venv\Scripts\python.exe scripts\p1_train.py --model tiny_cnn m5 logmel_cnn --folds 0 --seeds 0 --smoke --runs-dir runs_smoke
# one real run locally (tiny_cnn ~70 s/epoch on the laptop CPU => ~35 min for 30 epochs)
.venv\Scripts\python.exe scripts\p1_train.py --model tiny_cnn --folds 0 --seeds 0
# full grid on Kaggle/Colab GPU: build the code bundle, upload dist/pcg_code.zip, run the notebook
.venv\Scripts\python.exe scripts\make_code_bundle.py
```

Protocol per run: test = fold k, validation = fold (k+1) mod 5 for checkpoint selection, train = the other three.
Outputs land in `runs/<model>/fold<k>/seed<s>/` (config, best checkpoint, history, per-recording test
probabilities, official patient-level metrics). Finished runs are skipped on re-run, so grids are resumable.

Modules: `pcg/features.py` (20-800 Hz band-pass, per-recording standardisation, torch log-mel front-end),
`pcg/dataset.py` (location-aware recording labels, 4 s crops, sliding-window eval), `pcg/models.py`
(tiny_cnn 27k, m5 357k, logmel_cnn 308k params), `pcg/train.py`.

## Phases (see FINAL_research_problem.md section 10)

| Phase | Status |
|---|---|
| P0 Setup: data, folds, scorer | done locally 2026-10-20 (`scripts/p0_verify_data.py` passes). Still open: ZCHSound access request, reading the JCM 2026 benchmark and qZACH-ViT in full |
| P1 FP32 baselines | code done 2026-10-20; smoke test passes for all 3 models; full 45-run grid runs on Kaggle (`notebooks/kaggle_p1_train.ipynb`) |
| P2 Compression sweep | |
| P3 Trust audit | |
| P4 Shift | |
| P5 Mitigation | |
| P6 Writing | |
