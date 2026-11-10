# Mechanism analyses

Supporting analyses for the paper's mechanism section (why per-tensor INT8 collapses on
EfficientNet-B0 but not on the compact models). Each script runs against the trained
checkpoints under `runs/` (regenerate those with `scripts/p1_train.py`).

| Script | Question | Output |
|--------|----------|--------|
| `weight_range_disparity.py` | per-output-channel weight-range ratio (post-BN-fold), by layer type | `weight_range_disparity.csv` |
| `weight_quant_error.py` | per-layer SQNR / relative Frobenius error, per-tensor vs per-channel INT8/INT4 | `weight_quant_error.csv` |
| `granularity_ablation.py` | per-tensor scales in depthwise-only vs everywhere-else; Grade-I sensitivity | `granularity_ablation/` |
| `operating_point.py` | threshold-free AUROC/AUPRC vs sensitivity@fixed-specificity for the MinMax operating-point shift | `operating_point.csv` |
| `clinical_metrics.py` | AUPRC per model x condition, PPV at screening prevalence, decision-curve / net-benefit analysis | `auprc_by_condition.csv`, `ppv_screening.csv`, `../results/figures/fig8_dca.*` |
| `multiplicity.py` | Benjamini-Hochberg FDR across the full subgroup x condition grid of sensitivity contrasts | `multiplicity_bh.csv` |

**Headline numbers (fold0/seed0 for weight stats; pooled for the ablation):**

- EfficientNet-B0 depthwise channel-range ratio: median **27.3**, max **59.3** (compact models 1.4–2.8).
- EfficientNet-B0 depthwise weight SQNR: **27.9 dB** per-tensor vs **45.3 dB** per-channel at INT8.
- Localization ablation: depthwise-only per-tensor → Grade-I sensitivity **0.346**; everywhere-else per-tensor → **0.673** (FP32 0.817).
- Operating point (M5): AUROC 0.942→0.928 under MinMax while sensitivity@spec0.90 drops 0.877→0.838; percentile calibration restores both.
