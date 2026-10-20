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

## Phases (see FINAL_research_problem.md section 10)

| Phase | Status |
|---|---|
| P0 Setup: data, folds, scorer | done locally 2026-10-20 (`scripts/p0_verify_data.py` passes). Still open: ZCHSound access request, reading the JCM 2026 benchmark and qZACH-ViT in full |
| P1 FP32 baselines | |
| P2 Compression sweep | |
| P3 Trust audit | |
| P4 Shift | |
| P5 Mitigation | |
| P6 Writing | |
