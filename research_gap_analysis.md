# Research Gap Analysis: Biomedical × Edge × Explainable AI

> **SUPERSEDED by `FINAL_research_problem.md` (2026-10-20).** The EC-QAT loss proposed below turned out to be already published (Park 2020, Quant-XAI 2026, qZACH-ViT 2026). Keep this file only for the venue table.

*Prepared 2026-10-20. Literature swept with live web search up to this date. Any paper marked ⚠ was found only in search snippets. Read it before you cite it.*

---

## 0. Bottom line

1. **Recommended paper:** *"Same Murmur, Different Reasons? How INT8/INT4 Quantization Changes Explanations, Calibration and Subgroup Performance in Pediatric Heart-Sound Classification."*
   - This combines Edge AI, XAI and biomedical AI on CirCor DigiScope.
   - It is open for 1D cardiac signals. The closest work (2026) does each piece separately and only on images.
2. **Main change from your earlier plan:** make *quantization → explanation/calibration/fairness* the central contribution, and make LoRA secondary.
   - LoRA on small models is a weak hook. A 2026 ECG study (Section 3, thread 3) found LoRA did worse than partial fine-tuning.
   - The explanation-under-quantization angle has fresh 2026 image-domain papers to build on, and nobody has done it on signals yet.
3. **Conference reality check:** the two you circled are closed.
   - ICET 2026 (GIKI): deadline was 15 Jun 2026.
   - ICRAI 2026 (NUST): deadline was 8 Feb 2026, and the conference was held in May 2026.
   - Realistic venues are in Section 1. Plan for **two papers**: a short empirical paper now and the full method paper in 2027.
4. **Speed matters.** The image-domain versions of this idea appeared Jul–Sep 2026, so someone may publish the signal version soon. Re-check arXiv the week before you submit.

---

## 1. Venue analysis (target conferences)

| Venue | Dates | Deadline | Indexing | Verdict |
|---|---|---|---|---|
| ICET 2026, GIKI (IEEE) | 18–19 Nov 2026 | 15 Jun 2026 | IEEE Xplore | ❌ Closed |
| ICRAI 2026, NUST (IEEE) | 4–5 May 2026 | 8 Feb 2026 | IEEE Xplore | ❌ Already held. Watch for ICRAI 2027 (CFP likely ~Dec 2026, deadline ~Feb 2027) |
| **CAIDS-2026, Riphah** | 24–25 Dec 2026 | **~10 Nov 2026** (aggregator only; confirm on caids.org.pk) | MMU Press (Malaysia) journals, not IEEE | ⚠ Possible for a **short paper**. Weak indexing |
| **ICIT 2027, FAST-NUCES CFD** | 28–31 Mar 2027 | Not yet found (check "Important Dates" on icit.nu.edu.pk) | Site links IEEE Xplore (not confirmed) | ✅ **Best fit.** Has "Healthcare Systems" and "Edge/Fog Computing" tracks |
| iCoMET 2027, Sukkur IBA | ~May 2027 (2026 ran 22–23 May) | No CFP yet | IEEE Xplore (past editions) | ✅ Good backup. The 2026 edition had a **Medical AI** session |
| ICACS'27, UoL | 2–3 Feb 2027 | 25 Sep 2026 | — | ❌ Closed |
| ICAIEB / ICAIDSGM / ICIADAS / ICAPD / ICIADSA ("Conference Alerts.co.in") | various | — | none credible | 🚫 **Avoid.** Same ICAIEB title is also listed for Macau (Apr 2026) and Frankfurt (Nov 2026), organiser "Research Fora". This matches the pay-to-present mill pattern and would weaken a PhD profile |
| ICES&T 2026, IUB Bahawalpur (IEEE) | 10–12 Nov 2026 | Not published, almost certainly passed | IEEE | ❌ Treat as closed |
| 23rd IBCAST, CESAT | 17–20 Aug 2026 | May 2026 | IEEE | ❌ Held. Watch for 24th IBCAST (deadline ~Apr–May 2027) |

**Recommended strategy**

- **Paper A, short empirical (CAIDS-2026, or hold for ICIT 2027):** PTQ only, 2 models. Measure explanation stability and calibration under INT8/INT4. This is doable in 3 weeks.
- **Paper B, full method:** add QAT, subgroups and the proposed EC-QAT loss, with Raspberry Pi/MCU latency.
  - Target: ICIT 2027 / iCoMET 2027 / ICRAI 2027.
  - Stretch target: a journal such as *Biomedical Signal Processing and Control*, *Computers in Biology and Medicine*, or *IEEE Access*. A journal counts for more in PhD applications than a local conference.
  - B must extend A substantially (new method plus new experiments). Also check whether CAIDS's MMU Press journal publication counts as prior publication for B's venue.

---

## 2. How the gaps were judged

I swept six threads that meet at Biomedical × Edge × XAI. Each was rated **CLOSED** (saturated, no publishable novelty), **PARTIAL** (done in some modality or partly), or **OPEN** (no work found).

| # | Thread | Images | ECG | PCG (heart sound) |
|---|---|---|---|---|
| 1 | Quantization/pruning → explanation faithfulness | PARTIAL (2026) | OPEN | **OPEN** |
| 2 | Compression of cardiac classifiers for MCU | n/a | **CLOSED** (saturated) | PARTIAL |
| 3 | LoRA/QLoRA on small audio/biosignal encoders → INT8 deployable | — | OPEN (but LoRA underperforms) | OPEN |
| 4 | On-device XAI | PARTIAL | PARTIAL | **OPEN** |
| 5 | Calibration/selective prediction under quantization | PARTIAL (CHIL 2026) | OPEN | **OPEN** |
| 6 | Subgroup fairness under compression | CLOSED (derm), PARTIAL (CXR) | OPEN | **OPEN** |

**The key point:** column 3 is open on five of six rows. Each 2026 image paper covers *one* row, and none covers signals. A paper that measures rows 1, 5 and 6 together on PCG (with row 2 as the deployment context) is new on three counts, with recent baselines to cite.

---

## 3. Evidence per thread

### Thread 1: Compression vs. explanation faithfulness
- **Rabbi, Farabe, Rahman (arXiv 2607.22872, Jul 2026), "Same Predictions, Different Reasons: The Effect of Quantization on Model Explanations"** ✓ abstract read.
  - Static PTQ INT8/INT4 on 5 CNNs.
  - Grad-CAM and LIME; PCC, SSIM, IoU, deletion/insertion.
  - Finding: accuracy does not predict explanation stability, and stability depends on the architecture.
  - *Missing:* signals, IG/SHAP, QAT, calibration.
- **Khalid, Saleem, Saqib, Wunsch, Yaqub (arXiv 2609.07803, Sep 2026)** ✓. Pruning (not quantization) on long-tailed medical images. Rare classes degrade first, and SNIP preserves explanations best.
- **Attribution Preservation in Network Compression (NeurIPS 2020, arXiv 2010.15054).** Older foundation work: attribution-preserving fine-tuning. Natural images only. **This is the methodological ancestor of the proposed EC-QAT loss.**
- ⚠ qZACH-ViT (2607.15421): QAT with an attribution-retention regularizer, from a secondary listing. **Read it first.** If it is a general method, the EC-QAT loss must be positioned against it.
- ⚠ Faithful Faithfulness Evaluations, Breast MRI (arXiv 2609.25978): perturbation protocol choice changes conclusions. Use this to justify reporting several faithfulness metrics.

### Thread 2: Cardiac model compression for edge
- **De Simone et al. (arXiv 2511.00966, Nov 2025)** ✓. Uncertainty-aware murmur detection on CirCor for low-resource edge use, with MC-Dropout selective classification. **No quantization.** This is your closest PCG baseline.
- ⚠ FunnelNet (arXiv 2405.09570): ~5.4k-parameter CNN, TinyML, Raspberry Pi 4B.
- ⚠ Knowledge distillation for heart sounds (EMBC 2023). Uses PhysioNet 2016 data, not CirCor.
- ECG on MCUs is **saturated**: Arduino UNO INT8 (Bioengineering 2026), ESP32-S3 (ArrythML 2606.02256), 1-bit ECG-LDC (2607.09680), qHuBERT. **Do not pitch "ECG on a microcontroller" as the contribution.**

### Thread 3: LoRA/PEFT + quantization
- ⚠ Florea (arXiv 2501.13884): LoRA on Qwen2-Audio (large, not edge) for CirCor murmur features.
- ⚠ CinC 2025 audio foundation models for heart sounds: notes **no PCG foundation model exists**.
- ⚠ arXiv 2604.23385: **LoRA on ECG-FM did worse than partial fine-tuning.** ← A reviewer may raise this against a LoRA-centred paper.
- ⚠ ECG-at-the-edge review (2604.02501) proposes QA-LoRA but does not test it.
- *Verdict:* the gap is open, but LoRA on a 2–5M-parameter model has a weak payoff. Keep it as an **optional ablation in Paper B**: QLoRA from a small audio encoder (e.g., BEATs / EfficientAT) → merged → INT8.

### Thread 4: On-device XAI
- ⚠ Grad-CAM vs. cardiologist reasoning on heart sounds (PMC12189486). Not on edge.
- ⚠ EXGnet (2506.12404), CLARITY-AI 2.0. ECG XAI aimed at edge devices.
- ⚠ Human-centred XAI for TinyML (2608.07091). Skin lesions.
- *Open:* on-device explanation for murmur detection. A cheap INT8 CAM computed on the device vs. float IG/SHAP has not been benchmarked.

### Thread 5: Calibration under quantization
- **Ozdemir & Li, CHIL 2026 (PMLR 333)** ✓. INT8 PTQ shifts ECE/Brier depending on architecture, and QAT recovers it. Brain MRI, CXR, skin. **No signals, subgroups or explanations.** This is a direct template for your calibration section.

### Thread 6: Fairness under compression
- Hooker et al. 2019/2020 (1911.05248, 2010.03058): compression hurts underrepresented groups disproportionately.
- ⚠ FairQuant (2602.23192, 2026, with code): group-aware mixed-precision QAT on Fitzpatrick17k.
- ⚠ Pruning on long-tailed CXR (MICCAI 2023).
- *Open for PCG:* CirCor has **age group (neonate → adolescent), sex, pregnancy status, and auscultation location**, all usable as subgroups, and nobody has used them for compression analysis.

---

## 4. The recommended project

### Working title
**"Same Murmur, Different Reasons? Quantization-Induced Explanation Drift, Miscalibration and Subgroup Disparity in Edge Heart-Sound Classifiers"**

### Why CirCor is especially strong for XAI (your unique hook)
CirCor DigiScope provides **segmentation annotations (S1, systole, S2, diastole)** and **murmur timing labels (systolic/diastolic)**. That allows a **physiologically grounded explanation metric** that image papers cannot have:

> **Phase-Relevance Ratio (PRR):** the share of attribution mass that falls in the clinically correct cardiac phase. For example, for a systolic murmur, attribution inside systole ÷ total attribution.

So instead of only asking "did the saliency map change after INT8?" (Rabbi 2026), you can ask **"did the INT8 model stop looking at systole?"** That is a clinically meaningful explanation-faithfulness question and a clear novelty claim. Verify the annotation files (`.tsv` per recording) in the data in Week 1.

### Research questions
- **RQ1:** How do INT8/INT4 PTQ and QAT affect murmur detection (CinC 2022 weighted accuracy) across a 1D raw-waveform CNN and a 2D mel-spectrogram CNN?
- **RQ2:** Does quantization change *why* the model decides? Measure with attribution similarity (Spearman, top-k IoU), deletion/insertion AUC and **PRR**, even when accuracy is unchanged.
- **RQ3:** Does quantization degrade calibration (ECE, Brier) and selective-prediction risk-coverage, and does post-PTQ temperature scaling fix it?
- **RQ4:** Are these effects uneven across subgroups (age group, sex, auscultation location)?
- **RQ5 (Paper B, the method):** Can an **Explanation- and Calibration-preserving QAT loss (EC-QAT)** keep faithfulness and calibration at INT4/INT8 without costing accuracy or latency?

### Method sketch for EC-QAT (Paper B contribution)
```
L = CE(y, f_q(x))                                  # task loss on fake-quant model
  + λ1 · (1 − cos(A_fp32(x), A_q(x)))              # attribution alignment (Grad-CAM / grad×input)
  + λ2 · KL(softmax(z_fp32/T) || softmax(z_q/T))   # distillation from FP32 teacher → keeps calibration
  [+ λ3 · PRR-hinge (optional, needs phase labels)]
```
- It builds on attribution-preservation (NeurIPS 2020) and FairQuant / qZACH-ViT. It is new because it is **signal-domain** and **phase-grounded**, and it **targets calibration jointly**.
- Ablations: λ1 = 0, λ2 = 0, both off (= plain QAT).

### Models (deliberately reduced from the earlier 3-model plan)
| Role | Model | Params | Why |
|---|---|---|---|
| 1D raw waveform | M5-style 1D-CNN | ~0.5M | Fast, quantization-friendly, 1D attributions map directly onto time-phase |
| 2D spectrogram | MobileNetV3-Small (ImageNet-pretrained, 1-channel mel) | ~1.5–2M | Standard efficient-audio baseline, depthwise convs are known to be quantization-sensitive (interesting!) |
| *(Paper B optional)* | EfficientAT-MN small / BEATs + QLoRA | <10M | Only if time allows; tests the "foundation-model → INT8" path |

Drop the transformer and distillation Path B from Paper A. They multiply compute without serving the core question.

### Compression conditions
FP32 (reference) · INT8 dynamic · INT8 static PTQ · INT4 PTQ (weight-only, or simulated fake-quant) · INT8 QAT · INT4 QAT · *(Paper B)* EC-QAT INT8/INT4.

### Evaluation metrics
- **Task:** CinC 2022 murmur weighted accuracy (weights 5/3/1 for present/unknown/absent), outcome cost metric, macro-F1, recall on the murmur-present class.
- **Explanation:** Grad-CAM (2D) / Grad-CAM-1D, Integrated Gradients, grad×input (cheap, on-device candidate), Occlusion/KernelSHAP (perturbation-based, so it works on true INT8 kernels).
  - Metrics: Spearman ρ and top-20% IoU vs. FP32, deletion/insertion AUC, **PRR**.
- **Calibration:** ECE (15 bins), Brier, reliability diagrams, AURC (risk-coverage), with and without temperature scaling.
- **Fairness:** per-subgroup weighted accuracy and ECE; worst-group gap; Δ(gap) FP32 → quantized.
- **Edge:** model size (KB), latency on CPU / Raspberry Pi 4 (or ESP32-S3 via TFLite-Micro for the 1D-CNN), peak RAM.

### ⚠ Technical pitfall to handle explicitly (reviewers will ask)
Truly integer PyTorch/TFLite kernels **don't support backprop**, so gradient-based XAI (Grad-CAM, IG) cannot run on the real INT8 model.
- Compute gradient methods on the **fake-quant (simulated) model**.
- Show that fake-quant and true INT8 logits agree (report max |Δlogit| and top-1 agreement %).
- Cross-check with **perturbation methods (Occlusion/SHAP) run on the true INT8 model**.
- State this in the Methods section. Getting it right on your own will already set the paper above many in this area.

### Protocol essentials
- **Patient-level** stratified 5-fold CV. A patient has several recordings, so splitting at recording level leaks.
- 3 seeds per condition.
- Report mean ± std and paired tests (Wilcoxon) for FP32 vs. quantized.
- Use the modern training recipe from your earlier plan: SpecAugment, mixup, label smoothing, cosine + warmup, EMA, class-balanced sampling/focal loss.

### Compute budget (Kaggle P100/T4, 30 h/week)
- Paper A: 2 models × 5 folds × 3 seeds × FP32 training ≈ 30 runs × ~20–40 min ≈ **12–18 GPU-h**.
  - PTQ is minutes per run, and XAI and calibration evaluation take ~3–5 h.
  - **Fits in ~1 week of quota.**
- Paper B: QAT/EC-QAT add ~2× training. Total ≈ 40–50 GPU-h over 3–4 weeks. Feasible.

---

## 5. Three-week sprint for Paper A (if targeting CAIDS ~10 Nov)

Confirm the deadline on caids.org.pk **today**. If the deadline is not confirmed or the scope is too tight, skip CAIDS and fold this into Paper B for ICIT 2027.

| Days | Work |
|---|---|
| 1–2 | Download CirCor v1.0.3 (PhysioNet). EDA. Patient-level folds. Verify segmentation/timing `.tsv` files. Read qZACH-ViT and Rabbi 2026 in full |
| 3–6 | Train M5 and MobileNetV3 FP32 (5 folds × 3 seeds). Log CinC weighted accuracy |
| 7–8 | PTQ: INT8 dynamic, INT8 static, INT4 weight-only. Fake-quant vs. true-INT8 agreement check |
| 9–12 | XAI pipeline (Captum: IG, Grad-CAM, grad×input, Occlusion). Similarity metrics plus **PRR**. Calibration (ECE, Brier, AURC, temperature scaling). Subgroup tables |
| 13–14 | CPU / Raspberry Pi latency and size |
| 15–19 | Write a 6–8 page paper (IEEE template). Figures: attribution overlays on PCG with phase shading, reliability diagrams, subgroup-gap bars |
| 20–21 | arXiv re-check, internal review, submit |

Paper B (Dec 2026 – Feb 2027): QAT, EC-QAT loss and ablations, MCU deployment, optional QLoRA path. Then submit to ICIT 2027 / iCoMET 2027 / a journal.

---

## 6. Backup topics (if the main gap gets scooped)

1. **On-device explanation benchmark for PCG.** Which cheap explanation (CAM, grad×input, attention) computed in INT8 on a Raspberry Pi/ESP32 best matches float IG/SHAP, measured by PRR? (Thread 4, OPEN.)
2. **QLoRA from audio foundation models to INT8 edge for heart sounds.** (Thread 3, OPEN, but riskier given negative LoRA results in ECG.)
3. **Quantization × fairness on PTB-XL** (age/sex subgroups). Easier data, but ECG is a crowded field.

---

## 7. Corrections to the earlier plan (pasted proposal)

- **"LoRA as central contribution":** downgrade to optional. Evidence (2604.23385) suggests LoRA can underperform on small biosignal encoders, and the novelty is weaker than the XAI angle.
- **"First systematic study of compression on CinC weighted score":** partly undercut by De Simone 2025 (edge CirCor) and FunnelNet. Reframe around explanations, calibration and fairness, where the "first" claim holds as of today.
- **3 architectures plus distillation:** cut to 2 architectures for Paper A. Distillation now appears only as the KL term inside EC-QAT.
- **References from memory:** the earlier proposal's bibliography still needs checking against the actual papers. The ✓ items in this doc were confirmed from the abstract page; the ⚠ items need a read.

---

## 8. Key references (with verification status)

| Status | Reference |
|---|---|
| ✓ | Rabbi, Farabe, Rahman. *Same Predictions, Different Reasons: The Effect of Quantization on Model Explanations.* arXiv:2607.22872, 2026 |
| ✓ | Khalid, Saleem, Saqib, Wunsch, Yaqub. *Model Pruning, Long-Tail Forgetting and Explanation Reliability in Medical Imaging.* arXiv:2609.07803, 2026 |
| ✓ | Ozdemir & Li. *Bridging the Reliability Gap: INT8 Quantization Effects on Discrimination and Calibration in Medical Imaging.* CHIL 2026, PMLR 333 |
| ✓ | De Simone et al. *Optimizing Uncertainty-Aware DL for On-the-Edge Murmur Detection in Low-Resource Settings.* arXiv:2511.00966, 2025 |
| ⚠ | Attribution Preservation in Network Compression. NeurIPS 2020, arXiv:2010.15054 |
| ⚠ | qZACH-ViT, arXiv:2607.15421 |
| ⚠ | FairQuant (Woergaard & Selvan), arXiv:2602.23192 |
| ⚠ | Hooker et al. arXiv:1911.05248; arXiv:2010.03058 |
| ⚠ | FunnelNet, arXiv:2405.09570 |
| ⚠ | Florea, arXiv:2501.13884; ECG-FM LoRA, arXiv:2604.23385; ECG edge review, arXiv:2604.02501 |
| ⚠ | Faithfulness evaluation pitfalls (Breast MRI), arXiv:2609.25978 |
| ⚠ | Oliveira et al. / Reyna et al. CirCor DigiScope dataset & George B. Moody PhysioNet Challenge 2022 papers (confirm exact citations on physionet.org) |

---

## 9. Immediate next actions
1. Confirm the CAIDS-2026 deadline and publication terms (caids.org.pk). Find ICIT 2027 "Important Dates".
2. Read qZACH-ViT (2607.15421) and Rabbi 2026 in full. They are the main novelty risks.
3. Download CirCor and confirm the phase/timing annotations exist for murmur-present recordings.
4. Approach a co-supervisor with biomedical signal experience. A clinical co-author can also check PRR's validity.
