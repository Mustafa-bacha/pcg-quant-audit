# System Specification (checked 2026-10-20)

Machine used for local development of the "Does Quantization Forget the Quiet Murmurs?" project.

## Hardware

| Item | Value |
|---|---|
| CPU | Intel Core Ultra 7 265U, 12 cores / 14 threads (2 P + 8 E + 2 LP-E, Arrow Lake-U laptop part) |
| RAM | 15.5 GB (16 GB, no upgrade path assumed) |
| GPU | Intel integrated Graphics only, ~2 GB shared. **No NVIDIA GPU, no CUDA.** |
| NPU | Intel AI Boost NPU is present on this SoC, but no PyTorch backend uses it; ignore |
| Disk C: | 106 GB free of 199 GB (OS) |
| Disk D: | 277 GB free of 277 GB (project drive, use for data) |
| OS | Windows 11 Pro 10.0.26200 |

## Software

| Tool | Status |
|---|---|
| Python | 3.14.7 (Windows Store/PythonCore install), pip 26.2 |
| Project venv | `D:\Own\home\research_paper\.venv` (created 2026-10-20, CPU PyTorch) |
| PyTorch | 2.14.1 CPU wheel available for 3.14 (no CUDA build applies) |
| torchao / captum / onnxruntime | 0.18.0 / 0.9.0 / 1.31.0 available on PyPI for 3.14 |
| git | **not installed** (code was fetched as zips; install Git for Windows before the code-release phase) |
| WSL | not installed |
| conda / uv / docker / ffmpeg / kaggle CLI | not installed |
| Network | PhysioNet reachable; CirCor zip is 449 MB |

## What this means for the project

**This laptop is the analysis and quantization machine, not the training machine.**

- **Training (P1, P5):** CPU-only. The tiny CNN (~25k params) and the M5-style 1D CNN (~0.5M params) train on CPU in tens of minutes per fold on 3,163 recordings. The log-mel 2D CNN (MobileNetV3-Small) is slower; 5 folds x 3 seeds should go to **Kaggle (P100/T4)**. Rule: develop and smoke-test every training script locally on 1 fold with a short schedule, then run the full grid on Kaggle and bring the checkpoints back.
- **Quantization (P2):** CPU is the *correct* target. INT8 static PTQ/QAT via torchao PT2E with `X86InductorQuantizer`, and ONNX Runtime QDQ S8S8, both run natively here. Latency numbers from this CPU are reportable as "x86 laptop" but the edge numbers need a Raspberry Pi (not yet owned, see below).
- **Explanations (P3):** Captum IG / Grad-CAM on fake-quant models is CPU-friendly. Memory is the limit: keep batch sizes small (16-32) and 16 GB is enough.
- **Data:** CirCor extracted is ~1 GB on D:. PhysioNet 2016 (~1 GB) and ZCHSound fit easily.
- **RAM caution:** load spectrograms lazily or cache them as float16 `.npy` per recording. Do not hold all raw waveforms in memory at once with Jupyter open (~3,163 x ~23 s x 4 kHz x 4 bytes = ~1.2 GB float32, fine alone, but avoid duplicating).
- **Threads:** set `torch.set_num_threads(12)` for training; leave 2 threads for the OS.

## Hardware still needed (buy or borrow)

| Item | Phase | Purpose |
|---|---|---|
| Raspberry Pi 4B/5 | P2 | Reportable ARM INT8 latency (FunnelNet reference: 9.3 ms on Pi 4B) |
| ESP32-S3 dev board (optional) | P5 | ESP-DL w8a8 demo; makes the "edge" claim concrete |
| Digital stethoscope (optional, Littmann 3200 or any) | P4 | A handful of local recordings for a device-shift sanity check; not required for the paper |

## Quick start

```powershell
D:\Own\home\research_paper\.venv\Scripts\Activate.ps1
python -c "import torch; print(torch.__version__, torch.get_num_threads())"
```
