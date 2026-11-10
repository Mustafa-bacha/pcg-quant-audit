# Interim findings, 2026-11-05 (full grid: 4 models x 3 seeds x 5 patient-level folds)

**Provenance.** On 2026-11-02 every PT2E arm (int8_ptq_pc, int8_ptq_pt, int8_qat) of the earlier Kaggle runs was
found to be a silent no-op (mismatched torch/torchao made `prepare_pt2e` insert zero observers, so the "quantized"
models were bit-identical FP32). All PT2E arms for m5, logmel_cnn and effnet_b0 were recomputed on Kaggle
(sessions v14-v16) in a guard-verified environment (torch 2.15.0+cu132 + torchao 0.18.0; `scripts/pt2e_selftest.py`
passed; every summary carries an `env` stamp; `pcg/quant.py` now raises rather than silently passing FP32 through).
FP32, ORT, int8_dynamic and int4_wo_sim arms were never affected; the redone FP32 reference rows reproduce the old
ones to ~1e-9 (same checkpoints). tiny_cnn (laptop, always a matched pair) was never affected.
ZCHSound (p4b) was recomputed locally for all 45 Kaggle-model runs on 2026-11-05.
Everything below is from `scripts/report.py` / `scripts/p5_pooled.py` over the verified `runs/` tree
(tables T1-T8 in `RESULTS.md` and `results/*.csv`; figures in `results/figures/`).

## Headline answers to "does quantization forget the quiet murmurs?"

1. **Yes - catastrophically - if you quantize the deep model per-tensor.** On effnet_b0, pooled over 942
   out-of-fold patients x 3 seeds (patient-cluster bootstrap, T8b): int8_ptq_pt grade-I (soft murmur) sensitivity
   delta **-0.407 [-0.494, -0.327], p<0.001** (grade II-III -0.182), from an FP32 base of 0.817. INT4 weight-only
   is just as bad on this model (**-0.417 [-0.500, -0.337]**; overall WA -0.251 +- 0.049, T2). Aggregate WA hides
   half of this: the per-tensor WA delta is "only" -0.179.
2. **No - when the same model is quantized per-channel.** effnet int8_ptq_pc: grade-I delta +0.003 (p=0.935),
   WA -0.014 +- 0.011, prediction agreement 0.969. The *granularity* choice, not INT8 itself, decides whether the
   soft murmurs survive. QAT lands in between (grade-I -0.061 [-0.096, -0.029], p<0.001; WA -0.034).
3. **Small models do not forget - but the deployment path can shift their operating point.** For tiny_cnn, m5 and
   logmel_cnn every PT2E arm is flat on grade-I sensitivity (|delta| <= 0.013, all n.s.). The only significant
   movers are ORT arms: default MinMax calibration *raises* grade-I sensitivity (m5 +0.074, p<0.001; tiny_cnn
   +0.071, p=0.005) by collapsing toward "Present" (specificity and ECE pay for it), and the symmetric-percentile
   variant *lowers* it on logmel_cnn (-0.080, p<0.001) and effnet (-0.042, p=0.019). Quantization moves the
   clinical operating point in either direction; it has to be audited, not assumed benign.

## FP32 baselines (T1)
WA / AUROC over 15 runs per model: effnet_b0 0.796 +- 0.021 / 0.854 +- 0.028; m5 0.794 +- 0.029 / 0.863 +- 0.024;
logmel_cnn 0.783 +- 0.031 / 0.870 +- 0.017; tiny_cnn 0.780 +- 0.028 / 0.864 +- 0.023. Grade-I sensitivity is
0.75-0.82 vs 0.97-0.99 for grade II-III: the soft-murmur gap already exists at FP32; the question is whether
compression widens it.

## Compression cost (T2, WA delta vs same-run FP32)
- **m5 / logmel_cnn / tiny_cnn**: every PT2E arm and INT4 within -0.015; int8_dynamic exactly 0 (conv nets);
  ORT MinMax is the only costly arm (m5 -0.052, tiny -0.044, logmel 0.000) and percentile calibration repairs it
  (m5 -0.002, tiny +0.001).
- **effnet_b0**: ptq_pc -0.014; ptq_pt **-0.179 +- 0.099** (agreement 0.821; fold4 runs drop to WA 0.38-0.66);
  int4_wo_sim **-0.251 +- 0.049** (present recall -0.352); qat -0.034 +- 0.049; ort -0.061 -> pct -0.015.
  The deep separable-conv model is fragile to *any* coarse weight quantization; the compact models are not.

## Calibration (T3)
Temperature scaling on the validation fold still roughly halves ECE for every surviving arm (e.g. effnet
int8_ptq_pt 0.198 -> 0.105, int8_qat 0.143 -> 0.080) without changing decisions. ORT MinMax remains the
calibration-damaging arm on the small models; percentile calibration restores FP32-level ECE.

## Explanations (T5)
Where accuracy survives, explanations survive: effnet int8_ptq_pc keeps IG drift cosine 0.973, Spearman 0.785,
top-10% overlap 0.61 vs FP32. Where accuracy collapses, explanations scramble: effnet int4_wo_sim IG cosine 0.919
but rank correlation 0.37 and occlusion cosine 0.75 / Spearman 0.01. Faithfulness (deletion AUC vs random) and the
phase-grounding picture from the tiny_cnn analysis carry over qualitatively.

## Distribution shift
- **PhysioNet 2016 zero-shot (T6a)**: weak for every FP32 model (AUROC 0.499 tiny_cnn - 0.616 effnet); this
  transfer is a failure of the models first. Quantization adds on top: effnet ptq_pt -0.037 +- 0.026 while
  ptq_pc is +0.008 +- 0.005; tiny_cnn ORT MinMax -0.127 +- 0.069. Same per-tensor-vs-per-channel pattern as
  in-domain.
- **ZCHSound pediatric zero-shot (T6d, echo-confirmed CHD, different stethoscope)**: the transfer works
  (FP32 AUROC 0.728-0.772; clean subsets 0.79-0.84, noisy 0.55-0.64) and per-channel INT8 never hurts it
  (deltas -0.001 to +0.013 across all four models). The effnet fragilities follow the model across datasets:
  int4 -0.101 +- 0.067, ptq_pt -0.048 +- 0.087. QAT is mildly positive on m5/tiny (+0.014/+0.015).
- **SNR sweep (T6b')**: no quantization x white-noise interaction for the surviving arms (within +-0.003 of
  FP32 at every SNR); QAT remains the most noise-robust arm at low SNR.
- **H2 (clipping) stays refuted**: activation clip rates ~0 on every logged input set (max per-node rates
  <0.1%; ZCHSound spot checks 0.0001-0.0002). The resolution story (activations spanning 25-30% of the INT8
  range, 27-35% of values on the zero level, tiny_cnn analysis) is the surviving mechanism candidate.

## Mitigations (unchanged conclusions, now grid-wide)
ORT percentile calibration repairs the MinMax collapse wherever it occurs (m5 -0.052 -> -0.002; tiny -0.044 ->
+0.001; effnet -0.061 -> -0.015) at zero latency cost; entropy calibration is still a no-op (KL threshold = MinMax
range on these peaky activations). Temperature scaling halves ECE everywhere. **New grid-level mitigation
message: use per-channel PTQ.** It is the only arm that is simultaneously near-lossless on WA, grade-I
sensitivity, calibration, explanations and both shift datasets, on all four architectures.

## Edge latency (T7b; tiny_cnn fold0 measured, unchanged)
x86 laptop: FP32 1.45 ms -> INT8 0.96 ms (1 thread); ARM64 Neoverse-N2 (GitHub Actions): 0.59 -> 0.25 ms (2.4x),
107 -> 40 KB. Percentile-calibrated INT8 has identical latency to MinMax, so the accuracy-preserving deployment
path is free. The m5/logmel_cnn/effnet_b0 ONNX graphs were re-collected from the verified runs
(`scripts/collect_onnx.py`, `models/`) and need one ARM64 workflow run after the next push.

## Open items
1. ARM64 latency for the three Kaggle models' refreshed ONNX graphs (push -> `.github/workflows` run).
2. Paper draft (P6): headline = per-tensor vs per-channel grade-I result; secondary = ORT operating-point shifts
   + cheap mitigations; methods cautionary tale = the PT2E silent no-op and the guard/selftest recipe.
3. Optional if time: Raspberry Pi on-device numbers; phase-restricted deletion test for the explanation story.
