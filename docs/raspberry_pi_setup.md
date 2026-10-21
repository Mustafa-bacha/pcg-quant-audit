# Raspberry Pi for reportable edge latency

## What to buy (one of)
| Option | Why | Approx. price |
|---|---|---|
| **Raspberry Pi 5, 4 GB** (recommended) | Cortex-A76, current mainstream edge board; ONNX Runtime has ARM64 wheels; fastest INT8 (dot-product instructions) | ~US$60 board; ~US$90 with PSU + microSD + case |
| Raspberry Pi 4 Model B, 4 GB | Same board as FunnelNet's reported 9.3 ms (arXiv 2405.09570), so numbers are directly comparable to prior PCG work | ~US$55 board |
| Orange Pi Zero 2 W (Cortex-A53) | The exact board used by Ludera & Kersten-Oertel (MICCAI 2026); cheapest, slowest | ~US$36 |

Also needed: official USB-C power supply (5 V / 5 A for Pi 5), 32 GB microSD, and a way to connect (Ethernet
cable to the laptop, or Wi-Fi). An ESP32-S3 dev board (~US$10) is optional for the P5 microcontroller demo.

## Setup (about 30 minutes)
1. Flash **Raspberry Pi OS Lite 64-bit** with Raspberry Pi Imager; enable SSH and Wi-Fi in the imager's settings.
2. On the Pi:
   ```bash
   sudo apt update && sudo apt install -y python3-venv
   python3 -m venv ~/pcgenv && ~/pcgenv/bin/pip install onnxruntime numpy
   ```
3. From the laptop, copy the benchmark script and one run's ONNX files:
   ```powershell
   scp scripts\pi_benchmark.py runs\tiny_cnn\fold0\seed0\quant\backbone_*.onnx pi@raspberrypi.local:~/
   ```
4. On the Pi:
   ```bash
   ~/pcgenv/bin/python pi_benchmark.py --models backbone_fp32.onnx backbone_int8.onnx backbone_ort_int8_pct.onnx \
       --input-shape 1 20000 --threads 1 4 --out pi_latency_tiny_cnn.json
   ```
   For log-mel models use `--input-shape 1 1 64 201` (the float front-end runs outside the quantized graph).
5. Copy `pi_latency_*.json` back into `runs/<model>/fold0/seed0/quant/` and cite it in table T7.

## What to report
Median and p90 batch-1 latency at 1 and 4 threads, model size on disk, board, OS, onnxruntime version. Compare
FP32 vs INT8 on the *same* board; the laptop x86 numbers in T7 are not edge numbers.
