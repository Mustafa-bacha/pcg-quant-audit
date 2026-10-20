# Results (auto-generated from `runs/`)

## T1. FP32 baselines (patient-level, held-out test folds)

| model    | weighted_accuracy   | auroc         | present_recall   | cost                 |   n_runs |
|:---------|:--------------------|:--------------|:-----------------|:---------------------|---------:|
| tiny_cnn | 0.783 ± 0.020       | 0.862 ± 0.029 | 0.843 ± 0.049    | 16380.011 ± 2485.671 |        5 |

## T2. Compression conditions (mean ± sd over folds × seeds; d_ = minus FP32 of the same run)

| model    | cond         | weighted_accuracy   | d_weighted_accuracy   | auroc         | d_auroc        | present_recall   | d_present_recall   | rec_top1_agreement   |   n_runs |
|:---------|:-------------|:--------------------|:----------------------|:--------------|:---------------|:-----------------|:-------------------|:---------------------|---------:|
| tiny_cnn | fp32         | 0.783 ± 0.020       | 0.000 ± 0.000         | 0.862 ± 0.029 | 0.000 ± 0.000  | 0.843 ± 0.049    | 0.000 ± 0.000      | nan                  |        5 |
| tiny_cnn | int4_wo_sim  | 0.779 ± 0.028       | -0.004 ± 0.014        | 0.861 ± 0.029 | -0.001 ± 0.005 | 0.810 ± 0.079    | -0.034 ± 0.036     | 0.982 ± 0.010        |        5 |
| tiny_cnn | int8_dynamic | 0.784 ± 0.020       | 0.001 ± 0.001         | 0.862 ± 0.029 | -0.000 ± 0.000 | 0.843 ± 0.049    | 0.000 ± 0.000      | 1.000 ± 0.000        |        5 |
| tiny_cnn | int8_ptq_pc  | 0.768 ± 0.022       | -0.015 ± 0.008        | 0.858 ± 0.026 | -0.004 ± 0.004 | 0.832 ± 0.061    | -0.011 ± 0.015     | 0.983 ± 0.011        |        5 |
| tiny_cnn | int8_ptq_pt  | 0.767 ± 0.023       | -0.016 ± 0.016        | 0.859 ± 0.026 | -0.003 ± 0.005 | 0.837 ± 0.063    | -0.006 ± 0.023     | 0.982 ± 0.013        |        5 |
| tiny_cnn | int8_qat     | 0.779 ± 0.031       | -0.004 ± 0.014        | 0.862 ± 0.032 | -0.001 ± 0.009 | 0.871 ± 0.033    | 0.028 ± 0.020      | 0.963 ± 0.007        |        5 |
| tiny_cnn | ort_int8     | 0.720 ± 0.077       | -0.063 ± 0.058        | 0.851 ± 0.033 | -0.011 ± 0.021 | 0.900 ± 0.095    | 0.057 ± 0.094      | 0.875 ± 0.119        |        5 |

## T7. Size and batch-1 latency (laptop x86 CPU, 4 threads)

| model    | cond         | size_kb         | latency_b1_ms   |   n_runs |
|:---------|:-------------|:----------------|:----------------|---------:|
| tiny_cnn | fp32         | 106.496 ± 0.000 | 1.007 ± 0.173   |        5 |
| tiny_cnn | int4_wo_sim  | nan             | 1.002 ± 0.188   |        5 |
| tiny_cnn | int8_dynamic | nan             | nan             |        5 |
| tiny_cnn | int8_ptq_pc  | nan             | 2.141 ± 0.260   |        5 |
| tiny_cnn | int8_ptq_pt  | nan             | 1.711 ± 0.190   |        5 |
| tiny_cnn | int8_qat     | nan             | 2.154 ± 0.539   |        5 |
| tiny_cnn | ort_int8     | 39.574 ± 0.000  | 0.283 ± 0.121   |        5 |

## T3. Calibration and selective prediction (mean of per-run point estimates; CIs per run in CSV)

| model    | cond         |    wa |   auroc_present |   sensitivity |   specificity |   ece |   brier |   nll |   aurc |
|:---------|:-------------|------:|----------------:|--------------:|--------------:|------:|--------:|------:|-------:|
| tiny_cnn | fp32         | 0.783 |           0.935 |         0.843 |         0.914 | 0.131 |   0.074 | 0.272 |  0.068 |
| tiny_cnn | int4_wo_sim  | 0.779 |           0.934 |         0.81  |         0.936 | 0.106 |   0.066 | 0.249 |  0.059 |
| tiny_cnn | int8_dynamic | 0.784 |           0.935 |         0.843 |         0.914 | 0.131 |   0.074 | 0.272 |  0.068 |
| tiny_cnn | int8_ptq_pc  | 0.768 |           0.931 |         0.832 |         0.895 | 0.13  |   0.077 | 0.281 |  0.075 |
| tiny_cnn | int8_ptq_pt  | 0.767 |           0.933 |         0.837 |         0.886 | 0.132 |   0.078 | 0.283 |  0.076 |
| tiny_cnn | int8_qat     | 0.779 |           0.932 |         0.871 |         0.869 | 0.17  |   0.092 | 0.323 |  0.095 |
| tiny_cnn | ort_int8     | 0.72  |           0.926 |         0.9   |         0.691 | 0.223 |   0.131 | 0.413 |  0.198 |

## T4. H1: sensitivity by murmur grade (Present patients; mean over runs)

| model    | cond         |   grade I (soft) |   grade II-III |   gap (II-III minus I) |
|:---------|:-------------|-----------------:|---------------:|-----------------------:|
| tiny_cnn | fp32         |            0.748 |          0.971 |                  0.223 |
| tiny_cnn | int4_wo_sim  |            0.701 |          0.957 |                  0.256 |
| tiny_cnn | int8_dynamic |            0.748 |          0.971 |                  0.223 |
| tiny_cnn | int8_ptq_pc  |            0.719 |          0.986 |                  0.267 |
| tiny_cnn | int8_ptq_pt  |            0.728 |          0.986 |                  0.258 |
| tiny_cnn | int8_qat     |            0.777 |          1     |                  0.223 |
| tiny_cnn | ort_int8     |            0.831 |          0.986 |                  0.155 |

### T4b. Sensitivity by other subgroups

| model    | subgroup     | level          |   fp32 |   int4_wo_sim |   int8_dynamic |   int8_ptq_pc |   int8_ptq_pt |   int8_qat |   ort_int8 |
|:---------|:-------------|:---------------|-------:|--------------:|---------------:|--------------:|--------------:|-----------:|-----------:|
| tiny_cnn | age_group    | Adolescent     |  0.7   |         0.6   |          0.7   |         0.6   |         0.6   |      0.8   |      0.8   |
| tiny_cnn | age_group    | Child          |  0.86  |         0.828 |          0.86  |         0.851 |         0.857 |      0.876 |      0.909 |
| tiny_cnn | age_group    | Neonate+Infant |  0.796 |         0.796 |          0.796 |         0.796 |         0.796 |      0.844 |      0.844 |
| tiny_cnn | campaign     | CC2014         |  0.808 |         0.774 |          0.808 |         0.823 |         0.823 |      0.872 |      0.903 |
| tiny_cnn | campaign     | CC2015         |  0.862 |         0.828 |          0.862 |         0.837 |         0.845 |      0.871 |      0.897 |
| tiny_cnn | most_audible | AV             |  0.929 |         0.929 |          0.929 |         0.929 |         0.929 |      0.929 |      0.929 |
| tiny_cnn | most_audible | MV             |  0.849 |         0.849 |          0.849 |         0.829 |         0.829 |      0.869 |      0.849 |
| tiny_cnn | most_audible | PV             |  0.9   |         0.813 |          0.9   |         0.864 |         0.88  |      0.9   |      0.914 |
| tiny_cnn | most_audible | TV             |  0.758 |         0.741 |          0.758 |         0.775 |         0.775 |      0.832 |      0.911 |
| tiny_cnn | sex          | Female         |  0.846 |         0.812 |          0.846 |         0.823 |         0.823 |      0.856 |      0.886 |
| tiny_cnn | sex          | Male           |  0.837 |         0.804 |          0.837 |         0.837 |         0.848 |      0.885 |      0.914 |

## T5. Explanations: drift vs FP32, phase relevance (PRR_sys; control = systole/diastole swapped), deletion faithfulness

| model    | cond        | method     |   drift_cosine |   drift_spearman |   drift_top10_overlap |   drift_top20_iou |   prr_sys |   prr_sys_control |   prr_sys_uniform |   delta_prr_sys |   deletion_auc_norm |   deletion_auc_norm_random |
|:---------|:------------|:-----------|---------------:|-----------------:|----------------------:|------------------:|----------:|------------------:|------------------:|----------------:|--------------------:|---------------------------:|
| tiny_cnn | fp32        | gradcam    |        nan     |          nan     |               nan     |           nan     |     0.144 |             0.319 |             0.227 |         nan     |             nan     |                    nan     |
| tiny_cnn | fp32        | gradxinput |        nan     |          nan     |               nan     |           nan     |     0.21  |             0.274 |             0.227 |         nan     |             nan     |                    nan     |
| tiny_cnn | fp32        | ig         |        nan     |          nan     |               nan     |           nan     |     0.232 |             0.227 |             0.227 |         nan     |               0.799 |                      0.935 |
| tiny_cnn | fp32        | occlusion  |        nan     |          nan     |               nan     |           nan     |     0.246 |             0.369 |             0.227 |         nan     |             nan     |                    nan     |
| tiny_cnn | int4_wo_sim | gradxinput |          0.989 |            0.972 |                 0.857 |             0.814 |     0.211 |             0.274 |             0.227 |           0.001 |             nan     |                    nan     |
| tiny_cnn | int4_wo_sim | ig         |          0.996 |            0.99  |                 0.915 |             0.882 |     0.231 |             0.227 |             0.227 |          -0.001 |             nan     |                    nan     |
| tiny_cnn | int4_wo_sim | occlusion  |          0.996 |            0.978 |                 0.894 |             0.874 |     0.246 |             0.371 |             0.227 |          -0.001 |             nan     |                    nan     |
| tiny_cnn | int8_ptq_pc | gradxinput |          0.99  |            0.966 |                 0.871 |             0.817 |     0.21  |             0.271 |             0.227 |           0     |             nan     |                    nan     |
| tiny_cnn | int8_ptq_pc | ig         |          0.989 |            0.971 |                 0.867 |             0.827 |     0.242 |             0.199 |             0.227 |           0.01  |             nan     |                    nan     |
| tiny_cnn | int8_ptq_pc | occlusion  |          0.974 |            0.888 |                 0.819 |             0.745 |     0.241 |             0.375 |             0.227 |          -0.005 |             nan     |                    nan     |
| tiny_cnn | int8_qat    | gradxinput |          0.973 |            0.915 |                 0.792 |             0.708 |     0.212 |             0.253 |             0.227 |           0.002 |             nan     |                    nan     |
| tiny_cnn | int8_qat    | ig         |          0.977 |            0.942 |                 0.809 |             0.751 |     0.239 |             0.195 |             0.227 |           0.007 |             nan     |                    nan     |
| tiny_cnn | int8_qat    | occlusion  |          0.927 |            0.722 |                 0.658 |             0.597 |     0.272 |             0.334 |             0.227 |           0.025 |             nan     |                    nan     |
| tiny_cnn | ort_int8    | occlusion  |          0.922 |            0.709 |                 0.623 |             0.556 |     0.256 |             0.356 |             0.227 |           0.01  |             nan     |                    nan     |

## T6a. Zero-shot PhysioNet 2016 (P(Present) vs abnormal)

| model    | cond        |   auroc |   delta_auroc |   bal_acc_rule |   bal_acc_oracle_youden |   ece |   delta_ece |   n_runs |
|:---------|:------------|--------:|--------------:|---------------:|------------------------:|------:|------------:|---------:|
| tiny_cnn | fp32        |   0.557 |         0     |          0.507 |                   0.565 | 0.22  |       0     |        1 |
| tiny_cnn | int4_wo_sim |   0.55  |        -0.006 |          0.503 |                   0.563 | 0.213 |      -0.008 |        1 |
| tiny_cnn | int8_ptq_pc |   0.488 |        -0.068 |          0.465 |                   0.538 | 0.262 |       0.042 |        1 |
| tiny_cnn | int8_ptq_pt |   0.487 |        -0.07  |          0.464 |                   0.536 | 0.265 |       0.045 |        1 |
| tiny_cnn | int8_qat    |   0.49  |        -0.067 |          0.479 |                   0.531 | 0.248 |       0.028 |        1 |
| tiny_cnn | ort_int8    |   0.343 |        -0.213 |          0.363 |                   0.517 | 0.404 |       0.183 |        1 |

## T6b. SNR sweep: WA per condition (columns = SNR dB or clean)

| model    | cond        |     0 |    10 |    20 |     5 |   clean |
|:---------|:------------|------:|------:|------:|------:|--------:|
| tiny_cnn | fp32        | 0.385 | 0.645 | 0.72  | 0.576 |   0.773 |
| tiny_cnn | int4_wo_sim | 0.399 | 0.634 | 0.726 | 0.576 |   0.748 |
| tiny_cnn | int8_ptq_pc | 0.385 | 0.645 | 0.731 | 0.576 |   0.762 |
| tiny_cnn | int8_ptq_pt | 0.385 | 0.643 | 0.731 | 0.576 |   0.77  |
| tiny_cnn | int8_qat    | 0.482 | 0.679 | 0.756 | 0.59  |   0.751 |
| tiny_cnn | ort_int8    | 0.385 | 0.623 | 0.778 | 0.576 |   0.709 |

### T6b'. SNR sweep: WA minus FP32 at the same SNR (the interaction)

| model    | cond        |     0 |     10 |    20 |     5 |   clean |
|:---------|:------------|------:|-------:|------:|------:|--------:|
| tiny_cnn | fp32        | 0     |  0     | 0     | 0     |   0     |
| tiny_cnn | int4_wo_sim | 0.014 | -0.011 | 0.006 | 0     |  -0.025 |
| tiny_cnn | int8_ptq_pc | 0     |  0     | 0.011 | 0     |  -0.011 |
| tiny_cnn | int8_ptq_pt | 0     | -0.003 | 0.011 | 0     |  -0.003 |
| tiny_cnn | int8_qat    | 0.097 |  0.033 | 0.036 | 0.014 |  -0.022 |
| tiny_cnn | ort_int8    | 0     | -0.022 | 0.058 | 0     |  -0.064 |

## T6c. H2: activation clipping rate of the in-domain-calibrated INT8 PTQ graph

| model    | input_set     |   mean_clip_rate |   max_clip_rate |   nodes_over_1pct |
|:---------|:--------------|-----------------:|----------------:|------------------:|
| tiny_cnn | circor_clean  |           0.0001 |          0.0002 |                 0 |
| tiny_cnn | circor_snr0   |           0      |          0      |                 0 |
| tiny_cnn | circor_snr10  |           0      |          0      |                 0 |
| tiny_cnn | circor_snr20  |           0      |          0      |                 0 |
| tiny_cnn | circor_snr5   |           0      |          0      |                 0 |
| tiny_cnn | physionet2016 |           0      |          0      |                 0 |
