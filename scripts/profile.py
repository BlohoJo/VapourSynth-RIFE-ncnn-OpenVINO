#!/usr/bin/env python3
"""
Per-layer profile of a RIFE ONNX/IR model on the OpenVINO CPU device.

    python scripts/profile.py models/rife_v4.25.onnx --width 1920 --height 1080
    python scripts/profile.py models/rife_v4.25_heavy.onnx --precision bf16 --threads 16 --top 15

Prints the frame time, then the time spent per operation type (Convolution,
GridSample = the optical-flow warps, Interpolate = resizes, Reorder = layout
conversions, ...) and the slowest individual layers, using OpenVINO's built-in
performance counters. Needs `pip install openvino numpy`.
"""
import argparse, time, sys
from collections import defaultdict
import numpy as np
import openvino as ov

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("model")
ap.add_argument("--width", type=int, default=1920)
ap.add_argument("--height", type=int, default=1080)
ap.add_argument("--pad", type=int, default=64, help="pad dimensions to a multiple of this (plugin uses 32 or 64)")
ap.add_argument("--precision", default="f32", choices=["f32", "bf16", "f16"])
ap.add_argument("--hint", default="latency", choices=["latency", "throughput"])
ap.add_argument("--threads", type=int, default=0, help="inference threads (0 = OpenVINO default)")
ap.add_argument("--iters", type=int, default=8)
ap.add_argument("--top", type=int, default=10, help="number of slowest layers to list")
a = ap.parse_args()

H = (a.height + a.pad - 1) // a.pad * a.pad
W = (a.width + a.pad - 1) // a.pad * a.pad
core = ov.Core()
m = core.read_model(a.model)
m.reshape({m.inputs[0].get_any_name(): [1, m.inputs[0].get_partial_shape()[1].get_length(), H, W]})
props = {"PERF_COUNT": True, "PERFORMANCE_HINT": a.hint.upper(), "INFERENCE_PRECISION_HINT": a.precision}
if a.threads:
    props["INFERENCE_NUM_THREADS"] = a.threads
cm = core.compile_model(m, "CPU", props)
req = cm.create_infer_request()
x = np.random.rand(*m.inputs[0].shape).astype(np.float32)
req.infer({0: x})  # warm-up
ts = []
for _ in range(a.iters):
    t = time.perf_counter()
    req.infer({0: x})
    ts.append(time.perf_counter() - t)

bytype = defaultdict(float)
rows = []
total = 0.0
for pc in req.profiling_info:
    us = pc.real_time.total_seconds() * 1e6
    if us <= 0:
        continue
    total += us
    bytype[pc.node_type] += us
    rows.append((us, pc.node_type, pc.exec_type, pc.node_name))

print(f"{a.model}  {W}x{H}  {a.precision}  hint={a.hint}  threads={cm.get_property('INFERENCE_NUM_THREADS')}")
print(f"cpu: {core.get_property('CPU', 'FULL_DEVICE_NAME').strip()}")
print(f"frame time: {np.median(ts) * 1000:.1f} ms median / {min(ts) * 1000:.1f} ms best of {a.iters}  "
      f"(sum of layer counters {total / 1000:.1f} ms)\n")
print(f"{'operation type':26s} {'ms':>9s} {'share':>7s}")
for t, us in sorted(bytype.items(), key=lambda kv: -kv[1]):
    if us / total < 0.002:
        break
    print(f"{t:26s} {us / 1000:9.1f} {100 * us / total:6.1f}%")
print(f"\nslowest {a.top} layers:")
for us, t, et, name in sorted(rows, reverse=True)[: a.top]:
    print(f"  {us / 1000:8.1f} ms  {t:22s} {et:26s} {name[:60]}")
