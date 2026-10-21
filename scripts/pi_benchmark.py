"""Edge latency benchmark for the exported ONNX backbones (Raspberry Pi / any ARM or x86 CPU).

Copy a run's quant/ folder (or just the .onnx files) to the device, then:

    python3 -m venv ~/pcgenv && ~/pcgenv/bin/pip install onnxruntime numpy
    ~/pcgenv/bin/python pi_benchmark.py --models backbone_fp32.onnx backbone_int8.onnx backbone_ort_int8_pct.onnx \
        --input-shape 1 20000 --threads 1 4 --out pi_latency.json

Reports median / p90 latency per model x thread count, model size, and (if psutil is installed)
process CPU time. Input shape: tiny_cnn/m5 take (1, 20000) raw waveform windows; logmel_cnn/effnet_b0 backbones take
the log-mel (1, 1, 64, 201) because the float front-end is computed outside the quantized graph.
No project imports, so it runs on a bare Pi.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def bench(path: Path, shape: tuple[int, ...], threads: int, n_warm: int = 20, n_iter: int = 200) -> dict:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    x = np.random.randn(*shape).astype(np.float32)
    for _ in range(n_warm):
        sess.run(None, {name: x})
    t = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        t.append((time.perf_counter() - t0) * 1e3)
    t = np.array(t)
    return {"model": path.name, "threads": threads, "median_ms": float(np.median(t)), "p90_ms": float(np.percentile(t, 90)),
            "mean_ms": float(t.mean()), "min_ms": float(t.min()), "size_kb": path.stat().st_size / 1024, "n_iter": n_iter}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--input-shape", nargs="+", type=int, default=[1, 20000])
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 4])
    ap.add_argument("--out", default="pi_latency.json")
    args = ap.parse_args()
    info = {"machine": platform.machine(), "platform": platform.platform(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "onnxruntime": ort.__version__, "input_shape": args.input_shape}
    try:
        info["cpu_model"] = next(l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.lower().startswith(("model name", "hardware", "model")))
    except Exception:  # noqa: BLE001
        pass
    rows = []
    for m in args.models:
        for th in args.threads:
            r = bench(Path(m), tuple(args.input_shape), th)
            rows.append(r)
            print(f"{r['model']:32s} threads={th} median={r['median_ms']:.2f} ms p90={r['p90_ms']:.2f} ms size={r['size_kb']:.0f} KB", flush=True)
    Path(args.out).write_text(json.dumps({"device": info, "results": rows}, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
