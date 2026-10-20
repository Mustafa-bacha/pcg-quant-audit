# Final Research Problem Definition

## Does Quantization Forget the Quiet Murmurs?
### A trustworthiness audit of edge-deployed pediatric heart-sound classifiers under grade, age and device shift

*Version 1.0, 2026-10-20. This replaces the earlier `research_gap_analysis.md` (keep it for the venue table).*
*Sources were checked live on 2026-10-20 by four separate research passes:*
- *the closest papers, read in full*
- *the clinical evidence*
- *a map of the heart-sound (PCG) field*
- *the CirCor dataset and tooling (including counts computed from the official CSV and a local PyTorch test)*

*Markers:*
- *[V] = verified at the source*
- *[S] = seen only as a search excerpt; read before citing*
- *[C] = counted directly from `training_data.csv`*

---

## 1. The problem in one paragraph

Children in Pakistan with congenital or rheumatic heart disease are diagnosed late, because echocardiography and pediatric cardiologists are concentrated in a few cities. AI on a low-cost digital stethoscope is the obvious way to triage them. In low-connectivity settings that AI has to run **on the device**, which means a **compressed (quantized) model**.

Deployed auscultation AI already fails in three known places:
- **Poor recordings:** crying, rubbing and noise.
- **Soft grade-I murmurs and very young children.**
- **New devices and populations.**

Every published edge heart-sound model reports accuracy only. Nobody has asked whether compression makes these three failure modes worse. Compression could:
- silently drop the quiet murmurs,
- make the model overconfident,
- shift its explanation away from the right part of the heartbeat,
- or degrade more on a different stethoscope, while overall accuracy looks unchanged.

Regulators now explicitly require those properties: subgroup results by age, site and equipment; output uncertainty; testing at independent sites; and validated (not decorative) explanations.

**This project measures whether edge compression breaks clinical trustworthiness for pediatric murmur screening, explains why, and tests cheap fixes.**

---

## 2. Why this is a real problem (clinical grounding)

### 2.1 Burden and access in Pakistan
| Fact | Value | Source |
|---|---|---|
| CHD deaths in Pakistan (GBD 2019) | ~11,000 (range 7,000–19,000) | Malik et al., Indian J Thorac Cardiovasc Surg 2024 [V] |
| Congenital cardiac programs | < 20 nationally; **none in Balochistan**; more than half in Punjab | same [V] |
| CHD in a district hospital (Buner, 2013–2023) | 9.1 / 1000 deliveries; **34.3% diagnosed after day 20**; home births are usually not screened | Haq et al., Health Sci Rep 2026 [V] |
| RHD | Pakistan is among the top-3 countries for RHD deaths (~18,900/yr). Pooled prevalence is 2.9/1000 by auscultation vs. 12.9/1000 by echo, so **most cases are missed by the human ear** | Ali et al., BMJ Open 2021, a *Karachi digital-stethoscope deep-learning protocol* [V; the 12.9 figure is FLAGGED, verify] |
| Specialist density (LMIC proxy) | 0.17 pediatric cardiologists per million in Africa vs. 2 per million recommended | Aldersley et al., WJPCHS 2025 [V] |

A national count of Pakistani pediatric cardiologists or echo machines was **not found**. Do not invent one; cite the program counts above.

### 2.2 The screening gap: human auscultation is weak
- **Pediatricians:** sensitivity 93%, specificity 59%. Up to **90% of first cardiology referrals are innocent murmurs** (Papunen 2024, citing Sackey 2016) [V].
- **Students and residents:** sensitivity 82%, specificity 56% [V].
- **Large neonatal screen (China, n = 41,320):** manual auscultation missed about two-thirds of cases (sensitivity 32.8%); AI-assisted auscultation reached 65.3% [S].

### 2.3 Where deployed AI fails today (the three failure modes)
| Failure mode | Evidence |
|---|---|
| **F1. Poor-quality recordings, with no calibrated way to abstain** | Eko (FDA K213794) rejects **15.7%** of recordings, and 34.8% of patients have at least one inadequate recording (Prince et al., JAHA 2023) [V]. Lv et al. 2021 excluded crying children [V]. Papunen 2024 enrolled only cooperative children [V]. In CirCor, 7.6% of patients are low quality and 119 are labelled "Unknown" [V]. |
| **F2. Soft murmurs and young children** | Eko's sensitivity is **50.8% at grade ≤ 1**, against 99.7% at grade 2–3 [V]. In the CirCor training set, **104 of 178 graded systolic-murmur patients are grade I** [C]. Neonates are almost absent from every study (CirCor training set: 6) [C]. |
| **F3. Device and population shift** | Zero-shot transfer from PhysioNet 2016 to CirCor scores AUC **0.49–0.55**, chance level (Al Tawil, Front Physiol 2026) [V]. CirCor → PhysioNet 2016 reaches a best AUROC of 0.738, and no model achieves both sensitivity and specificity ≥ 0.80 (JCM 2026, 17 Oct) [S]. Device identity can be predicted from recordings with 95% accuracy [S]. The 2022 Challenge test set used **the same device and campaigns** as training, so cross-device generalization was never tested [V]. |

### 2.4 What regulators and clinicians require
**FDA draft guidance on AI-enabled device software (Jan 2025)** [V] asks for:
- subgroup analysis by **age, sex, disease severity, site and data-collection equipment**;
- 95% confidence intervals, with repeated recordings from the same patient accounted for;
- **output uncertainty**;
- test data from **completely different clinical sites**.

It also warns that explanation visualizations "could also significantly mislead users" unless validated.

**EU AI Act (MDCG 2025-6):** high-risk medical AI must let users "interpret outputs correctly" (Art. 13) and use bias-examined data (Art. 10) [V]. The date the obligations apply is uncertain (2027 or 2028) [FLAG].

**WHO 2021:** transparency, explainability and equity [S].

**This is the core argument:** a compressed model is a *different device* from the FP32 model it came from. Nobody has checked whether compression preserves the very properties these guidelines require.

---

## 3. The research gap, confirmed against full papers

### 3.1 The closest work and what each leaves out
| Paper | What it did | What it did *not* do |
|---|---|---|
| Rabbi et al., arXiv 2607.22872 (Jul 2026) | INT8/INT4 weight-only PTQ on 5 CNNs. Grad-CAM/LIME drift. EfficientNet-B0 LIME correlation falls to 0.11 at INT4 | Images only (pizza/steak, bikes); no QAT, calibration or subgroups |
| qZACH-ViT, Angelakis, arXiv 2607.15421 (Jul 2026) | QAT with a JS-divergence loss aligning FP and quantized attribution maps; MedMNIST | 2D images; a single architecture with built-in explanation maps; no calibration or fairness |
| Ozdemir & Li, CHIL 2026 (PMLR 333) | 8 backbones on medical images. **PTQ wrecks calibration** (BrainMRI ECE 0.007 → 0.25) and QAT restores it | No explanations; *their stated future work is "whether quantization causes models to shift attention… via Grad-CAM"*; no signals or subgroups |
| Khalid et al., arXiv 2609.07803 (Sep 2026) | Pruning makes rare classes degrade first; Grad-CAM collapses with gradient collapse | Pruning only, images only, no calibration |
| FairQuant, arXiv 2602.23192 (Feb 2026) | Group-aware mixed-precision QAT on dermatology data | No explanations or calibration; images only |
| Park et al., NeurIPS 2020 | Training with CE + KL + teacher–student Grad-CAM matching (for distillation and pruning) | No quantization experiments; natural images |
| Quant-XAI, Research Square 2026 | "XAI-preserving QAT" with a CAM-consistency loss on rice-leaf images | Images; no calibration |
| De Simone et al., arXiv 2511.00966 (Nov 2025) | **CirCor, INT8 on an STM32 board**, MC-Dropout selective prediction (+3% sensitivity) | No explanations, ECE/Brier, subgroups or external validation; *its future work lists all of these* |
| Rohr et al., PLOS Digit Health 2024 | Multiple-instance learning on CirCor; scores activations inside the annotated murmur phase | FP32 only; scores model outputs, not post-hoc attributions |
| Althaph & Challa, Sci Rep 2025 | Grad-CAM/SHAP overlap with systolic and diastolic annotations | FP32 only; HeartWave dataset |
| Sun et al., Front AI 2026 | KL attribution regularizer toward frequency-band priors on CirCor | No quantization; explicitly does **not** use S1/S2/systole phases |
| JCM benchmark, 17 Oct 2026 | Cross-dataset PCG benchmark with 11 methods | FP32 only, as far as could be seen [S]; **read the full text before submitting** |

### 3.2 Novelty claims: final verdicts
| Claim | Verdict | How to phrase it |
|---|---|---|
| A. First study of how quantization affects explanations, calibration and subgroups on **1D physiological signals** | **SAFE** | "To our knowledge, the first systematic study on physiological acoustic signals." Frame it as answering the stated future work of Ozdemir & Li and De Simone et al. |
| B. **Joint** audit of explanation + calibration + subgroups (+ shift) under quantization | **SAFE** (no paper combines these on any data type) | This is the paper's headline |
| C. Phase-grounded explanation metric | **AT RISK as a new metric** (Rohr 2024, Althaph 2025) | Present it as **ΔPRR, the change in phase-relevance across precisions**, using CirCor's semi-manual segmentation. Cite both precedents |
| D. A new "explanation-preserving QAT loss" | **TAKEN** (Park 2020, Quant-XAI 2026, qZACH-ViT 2026) | Use it only as a **baseline**. Any method novelty must be specific to heart sounds (see RQ5) |
| E. **Mechanism:** *why* quantization hurts soft murmurs (dynamic range) and shifted inputs (activation-range saturation) | **SAFE** (no PCG/ECG study found) | This is the scientific centre of the paper (Section 4) |
| F. Quantized PCG under device and noise shift | **SAFE** (no study found; the closest is an object-detection paper) | — |

---

## 4. The scientific core: two mechanistic hypotheses

The study should do more than report that quantization changes things. It should test *why* and *for whom*. Two hypotheses come from how quantization works and how heart sounds behave.

### H1: "Quiet murmurs fall below the quantization step"
INT8 has 256 levels. In an S1/S2-dominated recording, the per-tensor activation scale is set by the **loud heart sounds**. A grade-I murmur is low-amplitude, so its features may land within a few quantization levels of zero and be rounded away.

**Predictions:**
- (i) Accuracy loss under PTQ is concentrated in grade I, not grade II–III (Hooker-style "compression-identified exemplars").
- (ii) The effect is **stronger for raw-waveform 1D models** than for log-mel models, because the log compresses dynamic range.
- (iii) Per-channel quantization and QAT reduce it.

### H2: "Shifted inputs saturate ranges calibrated in-domain"
Static PTQ fixes activation ranges using CirCor calibration data. Noisy recordings and other stethoscopes produce activations **outside those ranges**, which get clipped.

**Predictions:**
- (i) The FP32 → INT8 performance gap **grows** under shift (external dataset, added noise).
- (ii) The layer-wise clipping rate predicts the errors.
- (iii) A cheap fix, PTQ calibration on noise-augmented data, recovers much of the gap at no runtime cost.

**Both outcomes are publishable.** If H1 and H2 hold, the paper shows a clinically dangerous failure plus a cheap fix. If they don't, it is reassuring negative evidence that INT8 heart-sound models are safe to deploy, with confidence intervals. Each finding still reaches every metric regulators ask for.

---

## 5. Research questions

| RQ | Question | Primary metrics |
|---|---|---|
| **RQ1** (subgroups, H1) | Does INT8/INT4 PTQ/QAT disproportionately degrade detection of **grade-I murmurs**, infants, particular auscultation sites and low-quality recordings? Does this differ between raw-waveform and log-mel models? | Weighted accuracy (WA), murmur recall, AUROC per subgroup; worst-group gap; patient-bootstrap 95% CI |
| **RQ2** (calibration and abstention) | Does quantization degrade calibration and selective prediction? Do temperature scaling after PTQ and an "Unknown/low-quality → refer" rule restore it? | ECE (15 bins), Brier, AURC (risk–coverage), Challenge outcome cost |
| **RQ3** (explanations) | Does quantization change *why* the model decides, even when predictions agree? | FP32 vs. quantized attribution Spearman ρ and top-20% IoU; deletion/insertion AUC; **ΔPRR** (systolic murmurs only) |
| **RQ4** (shift, H2) | Does the FP32 → INT gap grow under internal shift (campaign 2014 → 2015), external shift (ZCHSound, PhysioNet 2016) and added noise (SNR 20 → 0 dB)? Does the clipping rate explain it? | Δ(FP32 − INT) in-domain vs. shifted; per-layer clipping rate; correlation with errors |
| **RQ5** (mitigation) | Which cheap fix best preserves trustworthiness? (a) per-channel PTQ, (b) shift-aware PTQ calibration data, (c) QAT, (d) QAT + KD (calibration), (e) Park-style attribution-preserving QAT (baseline for claim D), (f) *optional heart-sound-specific addition:* **phase-masked attribution alignment**, which matches attribution mass only inside the systole/diastole segments | Everything above + size, latency, RAM on Raspberry Pi 4 / x86 CPU (+ ESP32-S3 optional) |

---

## 6. Data (verified)

### 6.1 Training and in-domain: CirCor DigiScope v1.0.3 (PhysioNet, ODC-By, open access)
- **Size:** 942 patients and 3,163 recordings in the public set. 4 kHz, 16-bit; 4.8–80.4 s per recording [V/C].
- **Recording sites:** MV 861, AV 800, PV 766, TV 732, Phc 4. 586 patients have exactly one recording per main site; 12 more have all four sites plus a repeat recording, so 598 cover all four sites [C, re-verified locally 2026-10-20 by `scripts/p0_verify_data.py`].
- **Segmentation coverage:** all 3,163 training recordings have a `.tsv`; on the first 200 files a mean 57% of each recording's duration is annotated (min 4%), so phase-based metrics must be restricted to annotated spans [C, local].
- **Campaign split:** CC2014 385 patients, CC2015 557 [C, local].
- **Murmur labels:** Absent 695, Present 179, Unknown 68 (patients) [C].
- **Systolic grade:** I 104, II 28, III 46. Timing: holosystolic 101, early 59, mid 17 [C].
- **Diastolic murmurs:** only 5, so exclude them from phase analysis [C].
- **Age:** Child 664, Infant 126, Adolescent 72, Neonate 6, missing 74 (70 of those are pregnant) [C]. **Merge Neonate+Infant** into one subgroup.
- **Other fields:** Sex F 486 / M 456; campaign CC2014/CC2015 (use for internal shift).
- **Segmentation `.tsv` files** (S1/systole/S2/diastole):
  - **semi-automatic**: three algorithms vote, then a cardiac physiologist corrects them;
  - high-quality segments only (label 0 = unannotated);
  - training set only [V].
- **The hidden test set is still not released** [V]. Use **patient-level stratified 5-fold CV**. **Do not compare** directly with the Challenge's 0.780.
- **Official scoring code:** github.com/physionetchallenges/evaluation-2022 (weighted accuracy uses weights 5/3/1; outcome cost) [V]. Vendored unmodified in `third_party/`, wrapped in `pcg/scoring.py`; checked against known answers (oracle WA 1.000, all-Absent WA 0.387, all-Present WA 0.499) [C, local].
- **Folds:** `splits/folds_v1.json`, patient-level, 5 folds, seed 0, stratified by murmur x campaign; 188–189 patients and 35–36 Present patients per fold [C, local].
- **Download note:** physionet.org served ~4 KB/s from the dev network; the AWS Open Data mirror (`s3://physionet-open/circor-heart-sound/1.0.3/`) gave ~4 MB/s. Use `scripts/download_circor_s3.py`.

### 6.2 External validation (zero-shot)
| Dataset | Why use it | Caution |
|---|---|---|
| **ZCHSound** (TBME 2024) | 1,259 **children**, echo-confirmed CHD vs. normal, 3 Chinese centers, a **different stethoscope**, clean and noisy subsets | Labels are CHD/normal, not murmur. Map Present → abnormal and report AUROC, sensitivity and specificity. Check how to access it |
| **PhysioNet/CinC 2016** | Multi-device, multi-site | Mostly adults; abnormal/normal labels |
| **HLS-CMDS** (Zenodo) | Controlled noise and lung-sound mixing | Manikin recordings: a stress test, not clinical evidence |
| Synthetic | AudioSet/ICU noise at graded SNR; stethoscope impulse responses | Document the generation protocol |

---

## 7. Models and compression conditions

| Role | Model | Why |
|---|---|---|
| Raw waveform (tests H1) | M5-style 1D CNN (~0.5M params) | Most exposed to dynamic-range loss; ESP-DL supports Conv1D |
| Log-mel (tests H1) | Small 2D CNN, e.g. MobileNetV3-Small or a ResNet in the style of the top Challenge entry, on log-mel | Represents the Challenge winners; log compression is the contrast with the raw-waveform model |
| Tiny edge model | ~20–30k-param CNN in the style of De Simone (Light CNN, ~23 KB INT8) | Direct comparison with the closest heart-sound edge paper |

**Precision conditions:**
- FP32
- INT8 dynamic
- INT8 static PTQ (per-tensor and per-channel; calibration data in-domain or noise-augmented)
- INT8 QAT
- INT4 weight-only (**simulated with fake-quant**; no CPU INT4 conv kernels exist in PyTorch or ONNX Runtime [V])
- the RQ5 mitigations

**Training recipe:** patient-level folds; 3 seeds; SpecAugment/mixup for log-mel; time-shift/gain augmentation for raw waveform; class-balanced loss; cosine LR + warmup; EMA. Aggregate recordings to a patient-level decision with the official rule (Present > Unknown > Absent).

---

## 8. Tooling (verified 2026-10)

- **PyTorch quantization:**
  - Eager `torch.ao.quantization` QAT/PTQ **still works** on Conv1d (tested on torch 2.14, with a deprecation warning) [C]. It is the fastest route for Conv1d.
  - The recommended path is **torchao PT2E** (`torchao.quantization.pt2e`, with `X86InductorQuantizer` or the XNNPACK quantizer for ARM) [V].
  - `torch.ao.quantization.quantize_pt2e` no longer exists [C]. Pin your versions.
- **Gradient-based explanations on quantized models:**
  - Converted INT8 models have **no autograd**. Compute IG/Grad-CAM on the **fake-quant model** in eval mode with observers frozen. Gradients pass through the straight-through estimator [C].
  - Report agreement between fake-quant and real INT8 logits (max |Δ| and top-1 agreement).
  - Cross-check with **Occlusion/KernelSHAP on the real INT8 model** (ONNX Runtime).
- **Deployment:**
  - ONNX Runtime static QDQ S8S8 (x86 and Raspberry Pi ARM) [V].
  - Optional: ESP32-S3 via **ESP-DL** (w8a8, supports Conv1D) [V].
  - FunnelNet reports 9.3 ms on a Pi 4B (TFLite) as a reference point [V].
- **Starting code:**
  - Benjamin-Walker/heart-murmur-detection (PyTorch, WA 0.780 reported; ownership not confirmed)
  - am2234/parallel-hsmm-murmur (CUED_Acoustics)
  - DeepPSP/cinc2022
  - all team sources are on PhysioNet `challenge-2022/1.0.0/sources` [V]

---

## 9. Evaluation protocol (what reviewers will check)
1. **Patient-level** splits everywhere, stratified by murmur label and campaign. No recording-level leakage.
2. **Patient-level bootstrap 95% CIs** for every metric (an FDA requirement). Paired tests (Wilcoxon, or McNemar on predictions) for FP32 vs. INT.
3. Report **WA, AUROC, murmur recall, specificity, ECE, Brier, AURC and outcome cost** together. Never accuracy alone.
4. **Subgroup tables:** grade (I vs. II–III), age (Neonate+Infant / Child / Adolescent), sex, site (AV/PV/TV/MV), campaign, signal quality.
5. **Explanations:** at least one gradient method (IG), one CAM method and one perturbation method. Deletion restricted to a phase plus a pseudo-phase control (Rohr's design) before trusting ΔPRR.
6. **Sanity check** (Adebayo-style randomization) so attributions are not just edge detectors.
7. State clearly what is **out of scope**: innocent vs. pathological murmurs (CirCor has no such labels), CHD without a murmur, and diastolic murmurs (n = 5).

---

## 10. Implementation roadmap

| Phase | Weeks | Deliverable |
|---|---|---|
| **P0 Setup** | 1 | Download CirCor; recompute the counts above; patient-level folds; official scorer wired in; request ZCHSound access; read the JCM 2026 benchmark and qZACH-ViT in full |
| **P1 FP32 baselines** | 2–3 | Raw-waveform 1D CNN, log-mel 2D CNN and tiny CNN × 5 folds × 3 seeds. Target CV WA ~0.70–0.78 |
| **P2 Compression sweep** | 4 | All PTQ/QAT/INT4-sim conditions; fake-quant vs. INT8 agreement; ONNX export; Pi latency |
| **P3 Trust audit** | 5–6 | RQ1–RQ3: subgroup, calibration and explanation tables; H1 analysis (grade × representation × precision) |
| **P4 Shift** | 7–8 | RQ4: campaign shift, ZCHSound, PhysioNet 2016, SNR sweep; per-layer clipping analysis (H2) |
| **P5 Mitigation** | 9–10 | RQ5 fixes plus the baseline from claim D; optional phase-masked alignment; ESP32-S3 demo |
| **P6 Writing** | 11–12 | Paper, figures, code release (GitHub + model card) |

**Compute:** about 30 FP32 runs (~15 GPU-h) plus ~20 QAT runs (~15 GPU-h) plus evaluation (~10 h). That is roughly 40 GPU-h, or about 1.5 weeks of Kaggle quota spread over the project.

**Figures to plan from day 1:**
1. PCG waveform with phase shading and FP32 vs. INT8 attribution overlays.
2. Accuracy loss by grade × representation × precision (H1).
3. Reliability diagrams for FP32/PTQ/QAT.
4. FP32 → INT gap, in-domain vs. shifted, with clipping rate (H2).
5. Pareto plot of trustworthiness vs. latency for the mitigations.

---

## 11. Publication plan

| Stage | Content | Venue (verify dates) |
|---|---|---|
| Short paper | P1–P3 (RQ1–3, H1) | **CAIDS-2026** (Riphah, Dec 24–25; deadline ~**Nov 10**, aggregator listing; MMU Press). Only if the deadline is confirmed |
| Full conference paper | P1–P5 | **ICRAI 2027** (NUST; deadline likely ~Feb 2027), **iCoMET 2027**, **ICIT 2027** (FAST), **24th IBCAST** (~Apr–May 2027), **ICET 2027** (GIKI, ~Jun 2027). Also strongly fitting: **Computing in Cardiology 2027** (the Challenge's own community; abstract deadline usually ~April) |
| Journal (strongest for PhD applications) | Full study + code | *Computers in Biology and Medicine*, *Biomedical Signal Processing and Control*, *IEEE Access*; stretch: *IEEE JBHI* (published the CirCor paper) |

Avoid the "Conference Alerts.co.in" events (ICAIEB etc.). The same title is listed in Macau, Frankfurt and Lahore, which matches the pay-to-present mill pattern.

---

## 12. Risks and mitigations
| Risk | Mitigation |
|---|---|
| Someone publishes signal-domain quantization × explanations first (the image-domain versions came out at about 4 papers in 4 months) | Move fast; post to arXiv after P3; re-search the week before submitting. The mechanism (H1/H2) and the shift/regulatory framing are hard to scoop together |
| Small subgroups (grade II n = 28, neonates 6) | Merge groups; bootstrap CIs; report effect sizes, not just p-values |
| Label mismatch on external sets | Report AUROC/sensitivity/specificity for a binary abnormal/murmur mapping; state it as a limitation |
| No hidden test set | CV only; never claim to beat the Challenge leaderboard |
| Single annotator; noisy CirCor labels | Acknowledge; use leave-campaign-out as a robustness check |
| Explanations on fake-quant ≠ real INT8 | Report agreement; perturbation explanations on the real INT8 model |
| H1/H2 turn out negative | Still publishable as evidence that INT8 is safe, with CIs and regulator-aligned reporting |

---

## 13. Collaboration lead
The **Karachi digital-stethoscope deep-learning protocol** (Ali et al., BMJ Open 2021) means a Pakistani group already collects auscultation data for RHD. Contact them early:
- local data would turn F3 (shift) from a proxy into a real Pakistan validation;
- a clinical co-author strengthens every venue above.

---

## 14. Key references (verified citations)
- Oliveira J, et al. *The CirCor DigiScope Dataset: From Murmur Detection to Murmur Classification.* IEEE JBHI 2022;26(6):2524–2535. doi:10.1109/JBHI.2021.3137048 [V]
- Reyna MA, et al. *Heart murmur detection from phonocardiogram recordings: The George B. Moody PhysioNet Challenge 2022.* PLOS Digit Health 2023;2(9):e0000324 [V]
- Prince J, et al. (Eko) JAHA 2023;12(20):e030377 [V]
- Papunen et al. Eur J Pediatr 2024;183:4951–58 [V]
- Ozdemir & Li. CHIL 2026, PMLR 333:952–965 [V]
- Rabbi, Farabe, Rahman. arXiv:2607.22872 [V]
- Angelakis. qZACH-ViT, arXiv:2607.15421 [V]
- Khalid et al. arXiv:2609.07803 [V]
- Woergaard & Selvan. FairQuant, arXiv:2602.23192 [V]
- Park et al. NeurIPS 2020, arXiv:2010.15054 [V]
- Hooker et al. arXiv:1911.05248; arXiv:2010.03058 [S]
- De Simone et al. arXiv:2511.00966 [V]
- Rohr et al. PLOS Digit Health 2024, doi:10.1371/journal.pdig.0000461 [V]
- Althaph & Challa. Sci Rep 2025 (PMC12575608) [V]
- Sun et al. Front AI 2026, doi:10.3389/frai.2026.1780496 [V]
- Al Tawil et al. Front Physiol 2026, doi:10.3389/fphys.2026.1927399 [V]
- *Cross-Dataset Generalization in PCG Murmur Classification…* J Clin Med 2026;15(20):8058 [S]
- ZCHSound. IEEE TBME 2024, doi:10.1109/tbme.2023.3348800 [V]
- FDA. *Artificial Intelligence-Enabled Device Software Functions* (draft), Jan 2025 [V]
- MDCG 2025-6 / AIB 2025-1 [V]
- Malik et al. 2024 (PMC12102029); Haq et al. 2026 (PMC13386109); Ali et al. BMJ Open 2021 (PMC8344289) [V]
