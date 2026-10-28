#!/usr/bin/env python3
"""
Benchmark a RIFE ONNX model on this machine with OpenVINO (CPU, f32 and bf16)
and, when installed, ONNX Runtime (CPU). Prints ms/frame and fps for the raw
inference only (no VapourSynth involved), which is the upper bound of what the
plugin can reach.

    pip install openvino numpy           # required
    pip install onnxruntime               # optional comparison
    python scripts/benchmark.py models/rife_v4.26.onnx --width 960 --height 576
    python scripts/benchmark.py models/rife_v4.26.onnx --width 1920 --height 1080 --threads 16 --iters 10

Width/height are rounded up to the padding multiple of the model (64 for
v4.26, 128 for v4.25_lite, else 32) exactly like the plugin does.
"""

import argparse
import os
import re
import time

import numpy as np


def padding_for(name):
    name = os.path.basename(name).lower()
    m = re.search(r"v(\d+)[._](\d+)", name)
    if not m:
        return 32
    major, minor = int(m.group(1)), int(m.group(2))
    lite = "lite" in name
    heavy = "heavy" in name or "large" in name
    if (major, minor) >= (4, 26):
        return 64
    if (major, minor) == (4, 25) and lite:
        return 128
    if (major, minor) == (4, 25) and heavy:
        return 64
    return 32


def make_input(channels, h, w):
    rng = np.random.default_rng(0)
    x = np.zeros((1, channels, h, w), dtype=np.float32)
    x[0, 0:6] = rng.random((6, h, w), dtype=np.float32)
    x[0, 6] = 0.5
    if channels == 11:
        xs = 2.0 * np.arange(w, dtype=np.float32) / (w - 1) - 1.0
        ys = 2.0 * np.arange(h, dtype=np.float32) / (h - 1) - 1.0
        x[0, 7] = xs[None, :]
        x[0, 8] = ys[:, None]
        x[0, 9] = 2.0 / (w - 1)
        x[0, 10] = 2.0 / (h - 1)
    return x


def timeit(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    ms = (time.perf_counter() - t) / iters * 1000.0
    return ms


def bench_openvino(model_path, x, precision, threads, hint, iters, warmup):
    import openvino as ov

    core = ov.Core()
    model = core.read_model(model_path)
    model.reshape({model.input().get_any_name(): list(x.shape)})
    cfg = {"INFERENCE_PRECISION_HINT": precision, "PERFORMANCE_HINT": hint}
    if threads:
        cfg["INFERENCE_NUM_THREADS"] = threads
        cfg["ENABLE_HYPER_THREADING"] = True  # otherwise the CPU plugin caps threads at physical cores
    t = time.perf_counter()
    compiled = core.compile_model(model, "CPU", cfg)
    compile_s = time.perf_counter() - t
    req = compiled.create_infer_request()
    inp = compiled.input(0)
    actual = compiled.get_property("INFERENCE_PRECISION_HINT")
    nthreads = compiled.get_property("INFERENCE_NUM_THREADS")

    def run():
        req.infer({inp: x})

    ms = timeit(run, warmup, iters)
    return ms, compile_s, str(actual), nthreads


def bench_onnxruntime(model_path, x, threads, iters, warmup):
    import onnxruntime as ort

    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name

    def run():
        sess.run(None, {name: x})

    return timeit(run, warmup, iters)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=576)
    ap.add_argument("--threads", type=int, default=0, help="inference threads (0 = runtime default: one per physical core)")
    ap.add_argument("--hint", default="LATENCY", choices=["LATENCY", "THROUGHPUT"])
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--precisions", default="f32,bf16")
    ap.add_argument("--no-ort", action="store_true", help="skip ONNX Runtime even if installed")
    args = ap.parse_args()

    import openvino as ov
    model = ov.Core().read_model(args.model)
    channels = model.input().get_partial_shape()[1].get_length()
    pad = padding_for(args.model)
    if channels == 11:
        w = (args.width + pad - 1) // pad * pad
        h = (args.height + pad - 1) // pad * pad
    else:
        w, h = args.width, args.height
    x = make_input(channels, h, w)
    print(f"model: {args.model}  input: {list(x.shape)} (requested {args.width}x{args.height}, pad {pad})")
    print(f"cpu: {ov.Core().get_property('CPU', 'FULL_DEVICE_NAME')}  "
          f"caps: {ov.Core().get_property('CPU', 'OPTIMIZATION_CAPABILITIES')}")

    for prec in args.precisions.split(","):
        try:
            ms, cs, actual, nt = bench_openvino(args.model, x, prec, args.threads, args.hint, args.iters, args.warmup)
            print(f"OpenVINO CPU {prec:>4} (hint {args.hint}, threads {nt}): {ms:8.1f} ms/frame  {1000 / ms:6.2f} fps"
                  f"   [compile {cs:.1f}s, effective precision {actual}]")
        except Exception as e:  # noqa: BLE001
            print(f"OpenVINO CPU {prec:>4}: failed: {e}")

    if not args.no_ort:
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            print("ONNX Runtime not installed (pip install onnxruntime) - skipping comparison")
            return
        try:
            ms = bench_onnxruntime(args.model, x, args.threads, args.iters, args.warmup)
            print(f"ONNX Runtime CPU  f32: {ms:8.1f} ms/frame  {1000 / ms:6.2f} fps")
        except Exception as e:  # noqa: BLE001
            print(f"ONNX Runtime CPU f32: failed: {e}")


if __name__ == "__main__":
    main()
