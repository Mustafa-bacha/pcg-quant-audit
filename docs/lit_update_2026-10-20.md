# Literature update, 2026-10-20

Full read of the two papers left open in Phase 0, plus a July–October 2026 novelty sweep. Produced by a research agent with live web access; where a source could not be opened it says so. Numbers below were taken from the papers' text or code and should be re-checked against the PDFs before they go into the manuscript.

## Access notes
- **MDPI JCM 15(20):8058**: mdpi.com returned 403; full text obtained via a reader proxy of the /htm page and the mdpi-res.com PDF mirror. Supplementary Tables S1, S3–S5 were not accessible; log-mel/CWT settings were taken from the authors' GitHub code (`03_preprocess.py`, `36_cwt_cache.py`).
- **qZACH-ViT (arXiv 2607.15421)**: fully read via arxiv.org/html.
- **MICCAI 2026 MIRASOL_023** (found during the sweep; closest overlap to us): main PDF + supplement read. Code repo (github.com/domludera/mirasol2026) 404s as of today.

---

## Paper 1 — Kutlu & Çoban, *J. Clin. Med.* 2026, 15(20), 8058 (17 Oct 2026)
"Cross-Dataset Generalization in Phonocardiogram Murmur Classification: A Benchmark Study of Domain-Conditional Performance and the Role of Multi-Representation Fusion." Code: github.com/huseyinKutlu/PCG-CrossDataset (MIT), Zenodo 10.5281/zenodo.22957181.

### What they did
- **Source**: CirCor, *2,531 recordings / 753 patients / 19,460 five-second segments* (smaller than the public 3,163 / 942 set; not explained). Labels: Present + Unknown → abnormal, Absent → normal.
- **Targets (zero-shot)**: PhysioNet/CinC 2016 (3,240 rec) and Yaseen 2018 (1,000 rec, 5-class → binary).
- **Preprocessing (from code)**: resample **2,000 Hz**; **4th-order zero-phase Butterworth 25–400 Hz**; spike clipping at 3×IQR; **per-recording z-score**; **5 s windows, 2.5 s hop**. Log-mel: librosa `n_fft=1024, win_length=100, hop_length=50, n_mels=64, fmin=25, fmax=400, power=2, power_to_db` → 64×201.
- **Models (11)**: SVM-RBF, RF; 1D-CNN (raw), 2D-CNN (mel), ResNet1D, CNN-BiLSTM (~1M), Hybrid-MLP, Hybrid-KAN, EfficientNet-B0, MaxViT-tiny (30M), proposed PCG-MambaConformer.
- **Protocol**: 5-fold patient-disjoint CV stratified by murmur class, seed 42; 5-fold ensemble for external predictions; class-weighted CE with label smoothing.
- **Metrics**: AUROC with 2,000-resample stratified bootstrap CIs, AP, balanced accuracy at three thresholds, DeLong + Holm–Bonferroni. ECE only inside a UQ side-benchmark (no numbers in main text).

### Numbers (AUROC; internal / CinC 2016 / Yaseen 2018)
| Model | Internal | CinC 2016 | Yaseen |
|---|---|---|---|
| 1D-CNN | 0.829 | 0.713 | 0.851 |
| 2D-CNN | 0.869 | 0.693 | 0.686 |
| ResNet1D | 0.850 | 0.703 | 0.814 |
| CNN-BiLSTM | **0.903** | 0.727 | 0.806 |
| EfficientNet-B0 | 0.865 | **0.738** | 0.783 |
| MaxViT | 0.865 | 0.729 | 0.846 |
| PCG-MambaConformer | 0.862 | 0.601 | **0.985** |

- On CinC no model reached sens ≥0.80 and spec ≥0.80 at any threshold; oracle recalibration gained only +0.022 BA → discrimination failure, not threshold failure.
- Dropping Unknown raised internal AUROC (0.903→0.965) but hurt transfer (Yaseen 0.806→0.483).
- Disease-specific on Yaseen: AS 0.982, MR 0.945, MS 0.780, MVP 0.518 — "softer or click-dominated findings were not" detected across datasets.

### Quantization / calibration / XAI / subgroups
- **None of quantization, pruning, edge, latency, model size.**
- Calibration: ECE only in a 4-method UQ comparison, qualitative.
- Explanations: Grad-CAM on 4 hand-picked Yaseen recordings, visual only.
- Subgroups: none by grade, age, sex, site, quality.
- Future work: multi-source training, target-aware adaptation, prospective evaluation. Nothing on compression or interpretability.

### What it changes for us
- No overlap with our four audit axes. Cite for (i) CirCor→PhysioNet-2016 FP32 baselines, (ii) "soft murmurs don't transfer" (MS/MVP), (iii) the Unknown-label sensitivity caveat.
- Their CinC ceiling (~0.70–0.74 AUROC) means FP32 shift degradation is already large; report the *interaction* (Δ from quantization at fixed shift), not raw numbers.

---

## Paper 2 — Angelakis, *qZACH-ViT* (arXiv 2607.15421, 16 Jul 2026)
- Zero-token ViT (~250k params) with per-patch evidence heads; intrinsic attribution reconstructs the logits exactly.
- Loss: ℒ_attr = ½·JS(A_q⁽¹⁾, A_q⁽²⁾) + ½·JS(A_q^rec, A_fp^rec); RASO gradient-surgery optimizer (norm-match, project out conflicts, weight 0.1).
- **QAT, W8A8 signed INT8**, per-channel symmetric weights, per-tensor activations; mixed-precision ONNX INT8. No PTQ, no INT4.
- **Seven 2D MedMNIST sets only.** Nothing 1D/audio/ECG.
- Explanation metrics: deletion/insertion AUC, comprehensiveness, sufficiency, SaCo, Spearman vs occlusion; **noise stability** (cosine, Spearman, top-10% overlap, JS); **deployment retention FP32→INT8**: cosine 0.999955, Spearman 0.9944, top-10% overlap 0.9692. Parameter-randomization sanity check.
- INT8 prediction agreement 99.98%. No calibration, fairness or subgroups.
- Limitations (their §7): mixed precision not integer-only; no ground-truth localization; single benchmark family; no clinical claim.
- No "first" claims that threaten us.

### What it changes for us
- Cite as the nearest quantization-aware XAI *method* paper (imaging, intrinsic, QAT). Does not make "first quantization × XAI on 1D physiological signals" unsafe.
- **Copy their deployment-retention trio** (cosine, Spearman ρ, top-k overlap of FP32 vs INT8 attributions) and the noise-stability protocol.
- Their near-perfect retention under attribution-preserving QAT is the natural upper bound to contrast with post-hoc PTQ drift.

---

## Novelty-risk sweep (Jul–Oct 2026)

### High relevance — must cite, affects claims
1. **Rabbi, Farabe, Rahman, arXiv 2607.22872 (24 Jul 2026)** — PTQ weight-only INT8/INT4 (GPFQ), five CNNs, Grad-CAM + LIME; PCC, SSIM, top-20% IoU, deletion/insertion. Images only. **EfficientNet-B0 most fragile** (Grad-CAM PCC 0.670 INT8 / 0.593 INT4; LIME PCC 0.228 / 0.109), DenseNet161 ≈ 0.98. Our novelty must be *1D/physiological + phase-grounded faithfulness + joint calibration/subgroup*, not "quantization affects explanations".
2. **Ludera & Kersten-Oertel, "A Low-Cost AI Stethoscope for Resource-Constrained Heart Murmur Screening," MICCAI 2026 MIRASOL workshop.** **CirCor (3,163 rec / 942 pt), 3-class, INT8 on an Orange Pi Zero 2 W (Cortex-A53), ONNX Runtime.** Six models (MobileNetV3-S 1.52M, FastViT 3.26M, EfficientNet-B0 4.01M, TinyViT 5.07M, ResNet-18 11.17M, HTS-AT 27.57M); FP32 Challenge Score 0.875–0.896. **Static INT8 PTQ: EfficientNet-B0 0.889→0.784, TinyViT 0.894→0.784, FastViT 0.888→0.311**; EfficientNet-B0 and TinyViT INT8 "collapse to a degenerate near-constant prediction"; **QAT (10 epochs, lr 1e-5) made it worse**; dynamic INT8 3× slower than FP32 on A53. Latency: ENet-B0 INT8 210 ms / 4.3 MB vs FP32 456 ms / 15.3 MB. **Supplement Table S5 (FP32 only): sensitivity by murmur grade: grade I/VI (n=104) 0.385–0.577, grade II (n=28) 0.714–0.893, ≥III (n=46) 0.957–1.000.** Preprocessing: 4 kHz, 25–400 Hz 4th-order Butterworth, z-score, **30 s windows**, 64 mel, win 128, hop 120; AdamW 150 ep, SpecAugment, Mixup 0.5, class weights [4,10,1]. Ablation: 30 s > 15 s > 10 s; 25–400 Hz > none ≈ 25–1000 Hz. **No XAI, no ECE/Brier, no quantized-subgroup analysis, no noise/device robustness.** → **Closest overlap.** They show INT8 on CirCor can be catastrophic and grade-I is the weakest stratum at FP32, but never cross the two.
3. **Ozdemir & Li, CHIL 2026 (PMLR 333:952–965)** — FP32/FP16/INT8-PTQ/INT8-QAT, 8 backbones, 3 imaging datasets, ECE + Brier. PTQ causes architecture-dependent calibration shifts; QAT recovers. Imaging only.

### Medium relevance
4. **Sun et al., Frontiers in AI, 18 May 2026** — Gradient×Input in 4 frequency bands with KL alignment to clinical priors on PhysioNet 2016 + CirCor. No quantization. Ours grounds in **time-phase S1/systole/S2/diastole** instead of frequency bands.
5. **Al Tawil et al., Frontiers in Physiology, 26 Aug 2026** — PhysioNet 2016; 8-bit TFLite size only; zero-shot to CirCor at chance.
6. **Elangovan & Ting, ICML 2026 (arXiv 2607.16652)** — explanation stability is a property of the model–method pair → justify three attribution methods, per-method drift.
7. Future Internet 18(6):286 (ECG TinyML INT8, accuracy only); Roodi et al. arXiv 2606.12742 (EEG 8-bit, accuracy only); arXiv 2411.05858 (saliency under quantization, images); Rasmussen & Qin arXiv 2608.03854 (biomedical LLM calibration under INT8/INT4, text).

### Not found
No paper from Jul–Oct 2026 measuring explanation drift or probabilistic calibration under quantization on ECG, PCG, EEG, audio or time series.

---

## Novelty verdict update
- **Safe**: "First audit of explanation faithfulness, calibration and subgroup performance of INT8/INT4-quantized PCG (1D physiological-signal) classifiers."
- **Unsafe / soften**: bare "first quantization × XAI" (Rabbi 2026), bare "first INT8 on CirCor" (MICCAI MIRASOL_023), bare "first calibration-under-quantization in medical AI" (CHIL 2026).
- **Recommended wording**: *"To our knowledge this is the first trustworthiness audit of quantized models on 1D physiological signals; prior quantization-XAI work [Rabbi 2026; Angelakis 2026] and quantization-calibration work [Ozdemir 2026] are confined to 2D images, and prior PCG edge work [Ludera 2026] reports accuracy only."*
- **New mandatory experiment**: the MICCAI INT8 PTQ collapse (EfficientNet-B0/TinyViT). Either show properly calibrated PTQ does *not* collapse (graded-degradation story holds, corrects their finding) or show collapse and pivot the mechanism analysis. Both are publishable.
- **Strengthened**: FP32 grade-I sensitivity is documented at 0.39–0.58 (n=104); grade I is large enough for bootstrap CIs; grade II (n=28) and diastolic (n=5) are not — pool II with ≥III.

---

## Design decisions adopted (2026-10-20)
**Copy**
- Band-pass **25–400 Hz** 4th-order Butterworth (both PCG papers; MICCAI ablation), 64 mel bins, per-recording z-score, patient-disjoint 5-fold CV.
- **5 s windows / 2.5 s hop** (MDPI) as the primary pipeline so CirCor→PhysioNet-2016 numbers are comparable; keep 4 kHz native (MICCAI) rather than resampling to 2 kHz. 30 s windows as an optional ablation.
- Add **EfficientNet-B0 on log-mel** (in both PCG papers; most quantization-fragile in Rabbi 2026; collapsed in MICCAI) alongside the 1D raw-waveform models.
- Quantization conditions: FP32 / INT8 static PTQ (per-channel, calibration ≥500 segments) / INT8 QAT (longer than 10 epochs) / INT4 weight-only sim; ONNX Runtime for deployment; report prediction-agreement rate and max|Δp|.
- Explanation drift: cosine, Spearman, top-10% overlap (qZACH-ViT) + PCC, SSIM, top-20% IoU, deletion/insertion (Rabbi) + phase-grounded metrics; per attribution method.
- Calibration: ECE + Brier + NLL; temperature scaling of the quantized model as a mitigation arm.
- Statistics: 2,000-resample stratified bootstrap CIs, DeLong + Holm; Wilson intervals for grade strata; three operating points when reporting shift.
- Treat segment→patient aggregation (mean vs max) as an explicit factor.

**Avoid**
- Dynamic INT8 on Cortex-A53-class boards; qualitative Grad-CAM on a handful of examples; single-fold ablations; grade-stratified claims for diastolic (n=5) or grade II (n=28) alone.
