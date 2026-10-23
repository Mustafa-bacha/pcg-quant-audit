# Interim findings, 2026-10-20 (tiny_cnn only, seed 0, 5 patient-level folds, laptop CPU)

## Update 2026-10-23: tiny_cnn with 3 seeds (15 runs), pooled out-of-fold bootstrap (T8)

Seeds 1 and 2 were run through P2-P4b on the laptop (P4 on the full 3,240-recording PhysioNet set for every fold);
`scripts/p5_pooled.py` concatenates the 5 test folds of a seed (942 patients, each once) and resamples patients as
clusters across seeds. Headline changes vs the seed-0 picture below:

- **Compression cost is small and stable** (T2, mean +- sd over 15 runs, WA minus FP32): INT8 PTQ -0.005 +- 0.017,
  INT8 QAT +0.001 +- 0.021, INT4 weight-only -0.003 +- 0.013, ORT INT8 MinMax -0.044 +- 0.042 (collapse in some
  folds), ORT INT8 percentile +0.001 +- 0.009. FP32: WA 0.780 +- 0.028, AUROC 0.864 +- 0.023.
- **H1 does not replicate across seeds.** The seed-0 grade-I sensitivity drop (INT4 -0.048 [-0.096, -0.010],
  PTQ -0.029 [-0.067, 0.000]) is absent in seeds 1 and 2 (PTQ 0.000, INT4 +0.019 / -0.010). Pooled over the 3 seeds
  with a patient-cluster bootstrap (T8b): INT4 -0.013 [-0.038, +0.010], INT8 PTQ -0.006 [-0.022, +0.010],
  QAT +0.026 [-0.006, +0.058]; grade II-III unchanged throughout. The only condition that moves grade-I
  sensitivity with a CI excluding zero is ORT MinMax (+0.071 [+0.022, +0.119]), and it does so by collapsing toward
  "Present" (specificity -0.170 [-0.191, -0.151], ECE +0.075). Conclusion for the paper: for this model family,
  INT8/INT4 do not selectively forget the quiet murmurs; a seed-0-only analysis would have claimed they do.
  The remaining H1 question is whether the larger models (m5, logmel_cnn, effnet_b0; Kaggle runs) behave the same.
- **Calibration** (T8, pooled): ECE change INT8 PTQ +0.004 [-0.000, +0.009], QAT +0.016 [+0.012, +0.023],
  ORT MinMax +0.075 [+0.069, +0.084], ORT percentile -0.004 [-0.007, 0.000]. Temperature scaling on the validation
  fold still halves ECE for every condition (T3 `_ts` arms).
- **Shift** (T6a, 15 runs, full PhysioNet 2016): zero-shot AUROC is at chance for FP32 (0.499 +- 0.094), so the
  adult multi-device transfer is a failure of the FP32 model first; quantization adds INT8 PTQ -0.037 +- 0.032,
  QAT -0.060 +- 0.032, ORT MinMax -0.127 +- 0.069 on top. The SNR sweep shows no quantization x noise interaction
  (T6b'). H2 clipping stays refuted (clip rate ~0 on every input set; activations use 23-34% of the INT8 range).
- **ZCHSound** (T6d, 15 runs): FP32 AUROC 0.728 +- 0.027 (clean 0.789, noisy 0.547); INT8 PTQ +0.013 +- 0.011,
  QAT +0.015 +- 0.016, INT4 -0.003 +- 0.011, ORT MinMax -0.027 +- 0.029, ORT percentile +0.012 +- 0.007. The
  pediatric, different-stethoscope transfer works and quantization does not hurt it; the MinMax collapse does.
- Figures for all of the above are in `results/figures/` (`scripts/figures.py`).

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
  unchanged or better, so the grade gap widens. The ORT-MinMax "improvement" on grade I is an artefact of its
  shift toward Present; with percentile calibration the grade-I loss is −0.011.
- **Not yet significant**: per-fold paired bootstrap deltas for grade I (n ≈ 21 patients per fold) are
  −0.029 [−0.087, 0.000] for PTQ and −0.048 [−0.125, 0.000] for INT4; every per-fold CI touches zero.
  A pooled analysis over the 104 grade-I patients (and the 3-seed, 4-model grid) is required before this is
  a claim. The effect is one-sided in every fold, which is encouraging but not evidence.

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

## Shift (T6, 5 folds; fold 0 on all 3,240 PhysioNet-2016 recordings, folds 1–4 on a 900-recording subsample)
- **Zero-shot PhysioNet 2016** (adult, multi-device; P(Present) vs "abnormal"): FP32 AUROC **0.545 ± 0.044**,
  near chance for this 27k-param model (the JCM 2026 benchmark reaches 0.70–0.74 with 1–4M-param models).
  Quantization makes the transfer *worse* in every fold: INT8 PTQ **−0.030 ± 0.022**, QAT −0.040 ± 0.018,
  **ORT INT8 −0.099 ± 0.069** (fold 0: 0.343, below chance), while INT4 weight-only is unchanged (+0.002).
  ECE under shift rises from 0.26 to 0.29 (PTQ) and 0.36 (ORT). The quantization × dataset-shift
  interaction is therefore 2–6× the in-domain cost (−0.015 WA).
- **SNR sweep** (additive white noise on the test fold, 5-fold means): FP32 WA 0.783 → 0.754 (20 dB) →
  0.675 (10 dB) → 0.574 (5 dB) → 0.387 (0 dB = the all-Absent score). The INT8 PTQ conditions track FP32
  within ±0.003 at every SNR: **no extra fragility to white noise from post-training quantization**. QAT
  is *more* robust at low SNR (+0.022 at 5 dB, +0.042 at 0 dB), presumably because its 5 fine-tuning epochs
  with gain/polarity augmentation act as extra regularisation.
- **H2 as "range saturation" is refuted** for this model: the in-domain-calibrated INT8 graph clips ≤ 0.1%
  of activations on every input set (max 0.6% on one node at 0 dB). The mechanism is **lost resolution,
  not clipping**: 99% of activation values span only **25–30% of the INT8 range** (the histogram observer
  widens the range to fit rare S1/S2 peaks) and **27–35% of all activations collapse onto the single
  zero level**, rising with noise (0.29 clean → 0.35 at 0 dB). Quiet signal content therefore lives in a
  handful of integer levels, which is the quantitative form of the "quiet murmurs fall below the
  quantization step" hypothesis (H1). To be tested next: per-channel activation ranges / percentile
  calibration as a mitigation, and whether the log-mel models (whose inputs are already compressed) show
  smaller zero-level collapse.

## Mitigations (P5 arms, 5 folds)
| arm | effect |
|---|---|
| **ORT percentile calibration** (99.99th percentile instead of MinMax) | WA **0.781** (vs 0.720 default, FP32 0.783); recording-level agreement 0.99; specificity back to 0.914; ECE 0.130 = FP32. The deployment collapse was entirely a range-calibration artefact. Symmetric activations change nothing further (0.780). |
| ORT entropy calibration | identical scales to MinMax on every tensor and fold. Verified 2026-10-21 by inspecting the QuantizeLinear initializers and re-running `quantize_static` with `CalibrationMethod.Entropy` (128 and 2048 bins): the KL-optimal threshold coincides with the full MinMax range for these sparse, peaky activations, so entropy offers no mitigation here. `Distribution` behaves the same (input scale 0.235 vs 0.229). Only percentile narrows the range (input scale 0.112 vs 0.229 on 128 windows; MinMax grows to 0.463 on 512 windows because rare S1/S2 peaks set the range). |
| **Temperature scaling** fitted on the validation fold | ECE roughly halves for every condition: FP32 0.131 → 0.066, PTQ 0.130 → 0.073, QAT 0.170 → 0.097, ORT-MinMax 0.223 → 0.178. Decisions unchanged (WA identical). AURC slightly worse (0.068 → 0.080) because the max-probability confidence ordering changes. |
| QAT (5 epochs) | recovers WA (0.779) and grade-I sensitivity (0.777) and is the most noise-robust condition, but doubles AURC and raises ECE unless followed by temperature scaling |

Takeaway for the paper: on this model the *deployment* INT8 path, not quantization per se, caused the
accuracy and calibration damage, and two cheap fixes (percentile range calibration + post-hoc temperature)
remove nearly all of it. The residual effects that survive the fixes are the ones worth reporting:
grade-I sensitivity loss under PTQ/INT4 and the quantization × dataset-shift interaction.

## Edge latency (T7b, batch 1, ONNX Runtime, tiny_cnn fold 0)
| device | FP32 1 thr | INT8 1 thr | speed-up | FP32 4 thr | INT8 4 thr | size |
|---|---|---|---|---|---|---|
| x86 laptop (Core Ultra 7 265U) | 1.45 ms | 0.96 ms | 1.5x | 0.61 ms | 0.49 ms | 107 -> 40 KB |
| **ARM64 Neoverse-N2** (GitHub Actions `ubuntu-24.04-arm`, real INT8 dot-product kernels) | 0.59 ms | 0.25 ms | **2.4x** | 0.19 ms | 0.13 ms | 107 -> 40 KB |

- The percentile-calibrated INT8 model (the accuracy-preserving one) has the same latency as the MinMax one, so
  the mitigation is free at inference time. A 5 s window therefore costs ~0.25 ms on an ARM64 core: the
  model is nowhere near the latency budget; the paper's edge argument rests on size and trustworthiness, not speed.
- Raspberry Pi / Orange Pi numbers are still to be measured on real boards (`docs/raspberry_pi_setup.md`);
  the ARM64 server core is a defensible proxy and must be labelled as such.

## Open items before any of this is a result
1. Repeat for m5, logmel_cnn and effnet_b0 × 3 seeds on Kaggle (`notebooks/kaggle_p1_train.ipynb`).
2. Phase 4 (shift) on all folds; fold 0 running now.
3. P5 mitigation: ORT calibration method, temperature scaling after PTQ, noise-augmented calibration.
4. Raspberry Pi latency for the ORT INT8 models.
