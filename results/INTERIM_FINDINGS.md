# Interim findings, 2026-10-20 (tiny_cnn only, seed 0, 5 patient-level folds, laptop CPU)

These are first-pass numbers from `scripts/report.py` (tables in `RESULTS.md`). One model, one seed:
treat every statement below as a hypothesis to confirm with the other three models and two more seeds on Kaggle.

## FP32 baseline (T1)
- tiny_cnn (27k params, raw waveform, 25–400 Hz, 5 s windows): **WA 0.783 ± 0.020, AUROC 0.862 ± 0.029,
  Present recall 0.84 ± 0.05** over folds. In the target band; not comparable with the hidden-test 0.780.

## Compression (T2)
| condition | WA | Δ vs FP32 | recording-level agreement |
|---|---|---|---|
| int8_dynamic | 0.784 | +0.001 | 1.000 (Linear only: no effect on a conv net) |
| int4_wo_sim | 0.779 | −0.004 | 0.982 |
| int8_qat (5 ep) | 0.779 | −0.004 | 0.963 |
| int8_ptq_pc | 0.768 | −0.015 | 0.983 |
| int8_ptq_pt | 0.767 | −0.016 | 0.982 |
| **ort_int8** | **0.720** | **−0.063 ± 0.058** | **0.875 ± 0.119** (fold 2: 0.593, 0.67) |

- The PyTorch QDQ-simulated INT8 costs ~1.5 WA points; the **ONNX Runtime deployment path costs 6 points on
  average and collapses on one fold**, with sensitivity up (0.90) and specificity down (0.69): the model drifts
  toward predicting "Present". Same qualitative behaviour as the MICCAI 2026 collapse, but here it is confined
  to the ORT calibration (MinMax, uint8 activations) and does not happen with torchao's histogram calibration on
  the same checkpoint. → P5 mitigation arm: ORT percentile/entropy calibration and symmetric activations.

## Calibration and selective prediction (T3)
- FP32 ECE 0.131 (already poorly calibrated; weighted CE loss over-confident on Present).
- PTQ leaves ECE unchanged (0.130) but **QAT worsens it (0.170) and ORT INT8 doubles it (0.223)**; AURC
  0.068 → 0.095 (QAT) → 0.198 (ORT). QAT buys sensitivity (0.871) at the price of calibration: a
  sensitivity/calibration trade-off worth a figure.
- INT4 weight-only *improves* ECE (0.106) and AURC: weight rounding acted as a regulariser on this tiny model.

## H1: soft murmurs (T4)
| condition | grade I sens. | grade II–III sens. | gap |
|---|---|---|---|
| fp32 | 0.748 | 0.971 | 0.223 |
| int8_ptq_pc | 0.719 | 0.986 | 0.267 |
| int8_ptq_pt | 0.728 | 0.986 | 0.258 |
| int4_wo_sim | 0.701 | 0.957 | 0.256 |
| int8_qat | 0.777 | 1.000 | 0.223 |
| ort_int8 | 0.831 | 0.986 | 0.155 (but specificity 0.69) |

- **Direction consistent with H1**: PTQ and INT4 lose 3–5 points of grade-I sensitivity while grade II–III is
  unchanged or better, so the grade gap widens. The ORT "improvement" on grade I is an artefact of its shift
  toward Present. Needs the paired bootstrap CIs (`audit/subgroup_deltas.csv`, n ≈ 104 grade-I patients
  pooled) and the log-mel models before it is a claim.

## Explanations (T5)
- **Faithfulness**: deleting the top IG frames drives P(Present) down far more than random deletion
  (normalised deletion AUC 0.80 vs 0.94), so the attributions are informative.
- **Drift vs FP32** (IG, 60 windows/fold): PTQ cosine 0.989, Spearman 0.971, top-10% overlap 0.867;
  INT4 0.996 / 0.990 / 0.915; **QAT 0.977 / 0.942 / 0.809**; ORT occlusion overlap 0.64. Quantization-aware
  *training* moves the explanation more than post-training quantization, because it changes the weights.
- **Phase grounding**: the share of attribution in systole (PRR_sys ≈ 0.21–0.25) is at the uniform
  time-share baseline (0.227), while **S1+S2 carry ~0.5 of the mass** for gradient methods. The tiny model
  listens mostly to the heart sounds, not to the murmur interval. Quantization barely changes this
  (ΔPRR_sys within ±0.01 for PTQ/INT4, +0.025 for QAT occlusion). This is a faithfulness finding about the
  model itself and needs the phase-restricted deletion test before interpretation.

## Shift (T6, fold 0 only so far)
- **Zero-shot PhysioNet 2016** (adult, multi-device; P(Present) vs "abnormal"): FP32 AUROC **0.557**, i.e.
  near chance for this 27k-param model (the JCM 2026 benchmark reaches 0.70–0.74 with 1–4M-param models).
  Quantization makes the transfer *worse*: INT8 PTQ 0.488 (Δ −0.07), QAT 0.490, **ORT INT8 0.343 (Δ −0.21,
  below chance = systematically inverted)**, while INT4 weight-only is unchanged (0.550). The
  quantization × shift interaction is therefore large relative to the in-domain cost (−0.015 WA).
- **SNR sweep** (additive white noise on the test fold): FP32 WA 0.773 → 0.720 (20 dB) → 0.645 (10 dB) →
  0.576 (5 dB) → 0.385 (0 dB = the all-Absent score). The INT8 conditions track FP32 within ±0.01 at every
  SNR, i.e. **no extra fragility to white noise from quantization**; QAT is slightly *more* robust (0.482 at
  0 dB, 0.679 at 10 dB).
- **H2 (range saturation) is not supported** for this model: the in-domain-calibrated INT8 graph clips
  < 0.02% of activations on clean CirCor and 0% on noisy and PhysioNet inputs. The transfer loss under
  quantization must come from lost resolution, not clipping; `pcg/clipping.py` now also reports the 99%
  range span and the fraction of activations collapsing onto the zero level, to test that next.

## Open items before any of this is a result
1. Repeat for m5, logmel_cnn and effnet_b0 × 3 seeds on Kaggle (`notebooks/kaggle_p1_train.ipynb`).
2. Phase 4 (shift) on all folds; fold 0 running now.
3. P5 mitigation: ORT calibration method, temperature scaling after PTQ, noise-augmented calibration.
4. Raspberry Pi latency for the ORT INT8 models.
