#!/usr/bin/env python3
"""
Convert a RIFE ONNX model to OpenVINO IR (.xml + .bin), optionally with the
input shape fixed to a padded resolution. The plugin can load either the
.onnx or the .xml directly, so this is purely optional; a pre-converted IR
avoids the ONNX frontend import at every filter construction (compile time),
and a static-shape IR is slightly faster to compile. With cache_dir set in the
plugin this matters little.

    pip install openvino
    python scripts/convert_to_ir.py models/rife_v4.26.onnx
    python scripts/convert_to_ir.py models/rife_v4.26.onnx --width 1920 --height 1088 --out models/ir/
    python scripts/convert_to_ir.py models/rife_v4.26.onnx --fp16   # compress weights to f16 (smaller .bin)
"""

import argparse
import os

import openvino as ov


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx")
    ap.add_argument("--out", help="output directory (default: next to the input)")
    ap.add_argument("--width", type=int, help="fix the input width (must already be padded to the model multiple)")
    ap.add_argument("--height", type=int, help="fix the input height (must already be padded to the model multiple)")
    ap.add_argument("--fp16", action="store_true", help="store weights as f16 (compress_to_fp16)")
    args = ap.parse_args()

    core = ov.Core()
    model = core.read_model(args.onnx)
    shape = model.input().get_partial_shape()
    print(f"input: {model.input().get_any_name()} {shape}")

    if args.width or args.height:
        if not (args.width and args.height):
            ap.error("--width and --height must be given together")
        channels = shape[1].get_length()
        model.reshape({model.input().get_any_name(): [1, channels, args.height, args.width]})
        print(f"reshaped to [1, {channels}, {args.height}, {args.width}]")

    out_dir = args.out or os.path.dirname(os.path.abspath(args.onnx))
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.onnx))[0]
    if args.width:
        base += f"_{args.width}x{args.height}"
    xml = os.path.join(out_dir, base + ".xml")
    ov.save_model(model, xml, compress_to_fp16=args.fp16)
    print(f"saved {xml} (+ .bin)")


if __name__ == "__main__":
    main()
