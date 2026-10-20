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

Modules: `pcg/features.py` (25-400 Hz band-pass, per-recording standardisation, torch log-mel front-end),
`pcg/dataset.py` (location-aware recording labels, 5 s crops, sliding-window eval with 2.5 s hop),
`pcg/models.py` (tiny_cnn 27k, m5 357k, logmel_cnn 308k, effnet_b0 4.0M params), `pcg/train.py`.
Preprocessing choices follow the two 2026 PCG papers summarised in `docs/lit_update_2026-10-20.md`.

## Phase 2: compression

```powershell
.venv\Scripts\python.exe scripts\p2_quantize.py --model tiny_cnn --folds 0 1 2 3 4 --seeds 0 --qat-epochs 5
```

Writes `runs/<model>/fold<k>/seed<s>/quant/<condition>/` plus `quant/summary.json` (official metrics per
condition, agreement with FP32, fake-quant vs INT8 agreement, model size, batch-1 latency). Calibration windows
come from the training folds (`--calib in_domain`) or the same windows with additive noise (`--calib noise`, H2 probe).

## Phase 3: trust audit

```powershell
.venv\Scripts\python.exe scripts\p3_audit.py --model tiny_cnn --folds 0 1 2 3 4 --seeds 0
```

Part A (needs only Phase 2's per-recording probabilities): every metric with patient-level stratified bootstrap
CIs and paired deltas vs FP32 (`audit/conditions_ci.csv`), subgroup sensitivity/specificity with Wilson CIs per
murmur grade, age group, sex, campaign and site (`audit/subgroups_<cond>.csv`, `audit/subgroup_deltas.csv`),
reliability bins. Part B (rebuilds the fake-quant graphs from `quant/<cond>/prepared_sd.pt`): Integrated Gradients,
Gradient x Input, occlusion and FP32 Grad-CAM on test windows with phase annotations; drift vs FP32 (cosine,
Spearman, top-10% overlap, top-20% IoU), phase relevance (share of attribution in systole, with a
systole/diastole-swapped control), deletion faithfulness vs random deletion (`audit/explanations.csv`).

## Phases (see FINAL_research_problem.md section 10)

| Phase | Status |
|---|---|
| P0 Setup: data, folds, scorer | done locally 2026-10-20 (`scripts/p0_verify_data.py` passes). Still open: ZCHSound access request, reading the JCM 2026 benchmark and qZACH-ViT in full |
| P1 FP32 baselines | code done 2026-10-20. tiny_cnn 5-fold seed 0 on the laptop: WA 0.783 ± 0.020, AUROC 0.862. The other models and seeds run on Kaggle (`notebooks/kaggle_p1_train.ipynb`) |
| P2 Compression sweep | code done 2026-10-20 (`pcg/quant.py`, `scripts/p2_quantize.py`): INT8 PTQ per-channel/per-tensor (torchao PT2E), INT8 QAT, INT4 weight-only sim, dynamic INT8, ONNX Runtime INT8; verified on random weights and a smoke checkpoint |
| P3 Trust audit | done for tiny_cnn seed 0 (5 folds). Key numbers in `results/INTERIM_FINDINGS.md` |
| P4 Shift | done for tiny_cnn fold 0 (full PhysioNet 2016), folds 1-4 with a 900-recording subsample. H2 clipping not supported; quantization x dataset-shift interaction is large |
| P5 Mitigation | |
| P6 Writing | |
