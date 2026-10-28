# VapourSynth-RIFE-OpenVINO

Real-Time Intermediate Flow Estimation (RIFE) frame interpolation for VapourSynth, running
**entirely on the CPU** through the [OpenVINO](https://github.com/openvinotoolkit/openvino) runtime.

This is a fork of [styler00dollar/VapourSynth-RIFE-ncnn-Vulkan](https://github.com/styler00dollar/VapourSynth-RIFE-ncnn-Vulkan)
(itself based on [HolyWu/VapourSynth-RIFE-ncnn-Vulkan](https://github.com/HolyWu/VapourSynth-RIFE-ncnn-Vulkan) and
[nihui/rife-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan)). The VapourSynth side of the plugin -
parameter semantics, frame mapping, scene-change handling, PSNR based skipping, parallel request pool - is kept
as close to the original as practical; the ncnn + Vulkan inference backend has been replaced by OpenVINO's CPU
plugin loading the RIFE **ONNX** models exported by the [vs-mlrt](https://github.com/AmusementClub/vs-mlrt) project.

It is meant for machines without a usable GPU: VPS boxes, servers, or desktops such as an AMD Ryzen 9 9950X where
you would rather keep the GPU free.

Plugin namespace: `rifeov` - plugin id: `com.blohojo.rifeov` - function: `rifeov.RIFE`.

## Why OpenVINO on an AMD CPU?

OpenVINO is an Intel project, but its **CPU plugin is vendor-agnostic**: it is built on oneDNN and dispatches
kernels at runtime on CPUID feature flags, not on the vendor string. On AMD Zen 4 / Zen 5 (Ryzen 7000/9000,
EPYC 9004/9005) it picks the AVX-512, AVX-512 VNNI and AVX-512 BF16 kernels, so `bf16` inference works on a
Ryzen 9950X exactly like on an Intel Sapphire Rapids. Verified on this project's test VM (AMD EPYC 9254, Zen 4):
OpenVINO reports `OPTIMIZATION_CAPABILITIES = ['BF16', 'WINOGRAD', 'FP32', 'INT8', 'BIN', 'EXPORT_IMPORT']` and bf16
inference is measurably faster than f32.

What does **not** work on AMD: the OpenVINO `GPU` (Intel iGPU/Arc) and `NPU` plugins are Intel-only. Leave
`device="CPU"` (the default).

**Prior art:** vs-mlrt's `vsov` backend (`Backend.OV_CPU()`) also runs these very same RIFE ONNX models on OpenVINO
from VapourSynth. vs-mlrt is a generic inference plugin with a Python wrapper that assembles the RIFE inputs; this
plugin is a single self-contained C++ filter that keeps the familiar `rife.RIFE(...)` interface, frame props and
scene-change / skip logic of the ncnn plugin. Use whichever fits your pipeline.

## Usage

```
rifeov.RIFE(vnode clip[, int model=72, int factor_num=2, int factor_den=1, int fps_num=None, int fps_den=None,
            string model_path=None, string device="CPU", int threads=1, int ov_threads=0, string precision="f32",
            string hint=None, string cache_dir=None, int pad=0, bint sc=False, bint skip=False,
            float skip_threshold=60.0, bint list_devices=False])
```

```python
import vapoursynth as vs
core = vs.core
# core.std.LoadPlugin("/path/to/librife_openvino.so")   # if not in the autoload directory

clip = core.resize.Bicubic(src, format=vs.RGBS, matrix_in_s="709")
clip = core.rifeov.RIFE(clip, model=72, factor_num=2, precision="bf16", ov_threads=16)
clip = core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")
```

- **clip**: Clip to process. Only RGB format with float sample type of 32 bit depth is supported (RGBS).

### Models

Models are **not** bundled. Download the ONNX files with `scripts/download_models.py` (see below) into a `models`
directory next to the plugin binary, or point `model_path` anywhere you like. The file names follow vs-mlrt:
`rife_v4.26.onnx`, `rife_v4.25_lite.onnx`, `rife_v4.18_ensemble.onnx`, ...

- **model**: Model to use. The numbering is the one of the ncnn-Vulkan plugin so existing scripts keep working.
  Indices without an ONNX export raise an error. Default: 72 (rife v4.26).

  | model | file | | model | file |
  |---|---|---|---|---|
  | 0-10 | *(rife v1/v2/v3, HD, UHD, anime - not available)* | | 43 / 44 | rife_v4.14 / rife_v4.14_ensemble |
  | 11 / 12 | rife_v4.0 / rife_v4.0_ensemble | | 45 / 46 | rife_v4.14_lite / rife_v4.14_lite_ensemble |
  | 13 / 14 | *(rife v4.1 - not available)* | | 47 / 48 | rife_v4.15 / rife_v4.15_ensemble |
  | 15 / 16 | rife_v4.2 / rife_v4.2_ensemble | | 49 / 50 | rife_v4.15_lite / rife_v4.15_lite_ensemble |
  | 17 / 18 | rife_v4.3 / rife_v4.3_ensemble | | 51 / 52 | rife_v4.16_lite / rife_v4.16_lite_ensemble |
  | 19 / 20 | rife_v4.4 / rife_v4.4_ensemble | | 53 / 54 | rife_v4.17 / rife_v4.17_ensemble |
  | 21 / 22 | rife_v4.5 / rife_v4.5_ensemble | | 55 / 56 | rife_v4.17_lite / rife_v4.17_lite_ensemble |
  | 23 / 24 | rife_v4.6 / rife_v4.6_ensemble | | 57 / 58 | rife_v4.18 / rife_v4.18_ensemble |
  | 25 / 26 | rife_v4.7 / rife_v4.7_ensemble | | 59 / 60 | rife_v4.19 / rife_v4.19_ensemble |
  | 27 / 28 | rife_v4.8 / rife_v4.8_ensemble | | 61 / 62 | rife_v4.20 / rife_v4.20_ensemble |
  | 29 / 30 | rife_v4.9 / rife_v4.9_ensemble | | 63 | rife_v4.21 |
  | 31 / 32 | rife_v4.10 / rife_v4.10_ensemble | | 64 | rife_v4.22 |
  | 33 / 34 | rife_v4.11 / rife_v4.11_ensemble | | 65 | rife_v4.22_lite |
  | 35 / 36 | rife_v4.12 / rife_v4.12_ensemble | | 66 | rife_v4.23 |
  | 37 / 38 | rife_v4.12_lite / rife_v4.12_lite_ensemble | | 67 / 68 | rife_v4.24 / rife_v4.24_ensemble |
  | 39 / 40 | rife_v4.13 / rife_v4.13_ensemble | | 69 | rife_v4.25 |
  | 41 / 42 | rife_v4.13_lite / rife_v4.13_lite_ensemble | | 70 | rife_v4.25_lite |
  | | | | 71 | rife_v4.25_heavy |
  | | | | **72** | **rife_v4.26** (default) |
  | | | | 73 | rife_v4.26_heavy (ncnn "v4.26-large") |

  Note that vs-mlrt does not ship ensemble variants for every version (none for v4.21-v4.23 and v4.25+); a missing
  file produces a clear "model file ... not found" error.

- **factor_num, factor_den**: Factor of target frame rate. For example `factor_num=5, factor_den=2` will multiply
  input clip FPS by 2.5.

- **fps_num, fps_den**: Target frame rate. Supersedes `factor_num`/`factor_den` parameter if specified.

- **model_path**: Either a directory containing the `.onnx` / `.xml` model files (the file is then chosen by
  `model`), or the path of a single `.onnx` or OpenVINO IR `.xml` file (then `model` is ignored). Default:
  `models` directory next to the plugin binary.

- **device**: OpenVINO device name passed straight to `ov::Core::compile_model`. `"CPU"` (default) is the target
  of this plugin; `"GPU"` may work on Intel iGPU/Arc machines, `"AUTO"`, `"HETERO:..."` etc. are passed through
  untested. Use `list_devices=True` to see what your OpenVINO build reports.

- **threads**: Number of parallel OpenVINO inference requests (equivalent of `gpu_thread` in the ncnn plugin).
  Each request is a full copy of the input/output tensors; VapourSynth frame threads block when all requests are
  busy. Default 1, which is sensible because one OpenVINO inference already uses all cores. With `threads>1`
  the compiled model defaults to the `THROUGHPUT` hint (several independent streams), see tuning below.

- **ov_threads**: `INFERENCE_NUM_THREADS` - the number of CPU threads OpenVINO uses for one inference. 0 (default)
  = automatic: OpenVINO uses one thread per **physical** core (SMT siblings are ignored). When set explicitly,
  hyper-threading is enabled so the value is honoured even above the physical core count.

- **precision**: `"f32"` (default) or `"bf16"` -> `INFERENCE_PRECISION_HINT`. bf16 needs AVX-512 BF16 (AMD Zen 4/5,
  Intel Cooper Lake / Sapphire Rapids and newer) and is roughly 10-25 % faster at a small, usually invisible
  accuracy cost. `"f16"` is accepted for non-CPU devices.

- **hint**: `"latency"`, `"throughput"` or `"none"` -> `PERFORMANCE_HINT`. Default: `latency` when `threads=1`,
  `throughput` otherwise.

- **cache_dir**: OpenVINO compiled-model cache directory (`ov::cache_dir`). The first run for a given model /
  resolution / settings writes a blob there, later filter constructions load it instead of recompiling. Optional.

- **pad**: Padding multiple. Frames are padded (edge replication) to a multiple of this value before inference
  and cropped afterwards, like the ncnn plugin. 0 (default) = auto-detect from the model file name using the
  vs-mlrt table: 64 for v4.26 and newer, 128 for v4.25_lite, 64 for v4.25_heavy, 32 otherwise.

- **sc**: Avoid interpolating frames over scene changes. You must invoke `misc.SCDetect` on YUV or Gray format
  of the input beforehand so as to set the `_SceneChangeNext` frame property.

- **skip**: Skip interpolating static frames. Requires the
  [VMAF](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-VMAF) plugin.

- **skip_threshold**: PSNR threshold to determine whether the current frame and the next one are static.

- **list_devices**: Simply print a list of available OpenVINO devices on the frame and do no interpolation
  (analog of `list_gpu`).

### Removed compared to the ncnn-Vulkan plugin

- **tta** and **uhd**: the vs-mlrt ONNX graphs do not contain the TTA / UHD (downscaled flow) paths, so these
  options were removed rather than silently ignored.
- **gpu_id** -> replaced by `device`; **gpu_thread** -> `threads`; **list_gpu** -> `list_devices`.
- rife v1 / v2.x / v3.x / HD / UHD / anime models: no ONNX exports exist, the indices raise an error.

### Output frame properties

`_DurationNum`/`_DurationDen` are rescaled like in the original plugin; the output clip's `fps` and `num_frames`
are `fps * factor_num / factor_den` and `num_frames * factor_num / factor_den`.

## Installation

### Linux

1. Get the plugin: build it (below) or take `librife_openvino.so` from a release/CI artifact.
2. Get the OpenVINO runtime (2026.4.1 was used for development):
   [`openvino_toolkit_ubuntu24_2026.4.1.22982.07f9c262b05_x86_64.tgz`](https://storage.openvinotoolkit.org/repositories/openvino/packages/2026.4.1/linux/openvino_toolkit_ubuntu24_2026.4.1.22982.07f9c262b05_x86_64.tgz).
   The plugin needs `libopenvino.so.*`, `libopenvino_intel_cpu_plugin.so`, `libopenvino_onnx_frontend.so.*`,
   `libopenvino_ir_frontend.so.*` (all in `runtime/lib/intel64/`) and `libtbb.so.12`
   (`runtime/3rdparty/tbb/lib/` or your distro's `libtbb12`). OpenVINO 2024+ registers its device plugins
   statically, so there is no `plugins.xml` to copy (`cache.json` from the same directory is optional).
   Either put them **next to the plugin** (the plugin has `$ORIGIN` and `$ORIGIN/openvino` in its rpath, and
   links `libtbb` directly so it is found there too), keep the extracted archive and add
   `runtime/lib/intel64` to `LD_LIBRARY_PATH` (`source <archive>/setupvars.sh` does that), or install the
   runtime system-wide.
3. Copy `librife_openvino.so` into your VapourSynth plugin directory (e.g. `/usr/local/lib/vapoursynth`) or load
   it with `core.std.LoadPlugin`.
4. Download models: `python3 scripts/download_models.py v4.26 --out /usr/local/lib/vapoursynth/models`
   (`--list` shows every available name, `--all` fetches everything; needs `7z` or `pip install py7zr`).

### Windows

1. Take `RIFE-OpenVINO.dll` from a release/CI artifact or build it with Visual Studio 2022 (below).
2. Download the OpenVINO Windows archive
   [`openvino_toolkit_windows_2026.4.1.22982.07f9c262b05_x86_64.zip`](https://storage.openvinotoolkit.org/repositories/openvino/packages/2026.4.1/windows/openvino_toolkit_windows_2026.4.1.22982.07f9c262b05_x86_64.zip)
   and copy from `runtime\bin\intel64\Release\`: `openvino.dll`, `openvino_intel_cpu_plugin.dll`,
   `openvino_onnx_frontend.dll`, `openvino_ir_frontend.dll` (there is no `plugins.xml` in OpenVINO 2024+;
   `cache.json` is optional), plus `tbb12.dll` (and `tbbbind_2_5.dll`, `tbbmalloc.dll`) from
   `runtime\3rdparty\tbb\bin\`. Put them **in the same folder as `RIFE-OpenVINO.dll`** (VapourSynth loads
   plugins with `LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR`, so dependencies next to the plugin are found) or anywhere
   on `PATH`. The CI artifact `VapourSynth-RIFE-OpenVINO-win64.zip` already contains this exact set.
3. Put `RIFE-OpenVINO.dll` into your VapourSynth plugins folder (or `LoadPlugin` it).
4. `python scripts\download_models.py v4.26 --out <plugins folder>\models` (needs 7-Zip's `7z.exe` on PATH or
   `pip install py7zr`).

## Compilation

Requires CMake >= 3.16, a C++20 compiler and an extracted OpenVINO runtime archive (or an installed OpenVINO
package providing `OpenVINOConfig.cmake`). VapourSynth headers are vendored, no VapourSynth SDK is needed.

```sh
# Linux
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DOpenVINO_DIR=/path/to/openvino/runtime/cmake
cmake --build build -j$(nproc)
sudo cmake --install build          # -> <vapoursynth libdir>/vapoursynth/librife_openvino.so (or -DVS_INSTALL_DIR=...)

# Windows (Visual Studio 2022)
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DOpenVINO_DIR=C:\path\to\openvino\runtime\cmake
cmake --build build --config Release   # -> build\Release\RIFE-OpenVINO.dll
```

`.github/workflows/build.yml` is a GitHub Actions workflow that builds both platforms the same way, packages
the plugin together with the needed OpenVINO runtime files, smoke-tests that the packaged plugin loads with
only those files next to it, uploads the result as an artifact and creates a release on `v*` tags.

`ci/github-workflow-build.yml` is the maintained copy of that workflow: the credentials used to push this fork
are not allowed to touch `.github/workflows/`, so after pulling a change to it, sync it with

```sh
cp ci/github-workflow-build.yml .github/workflows/build.yml && git add .github && git commit -m "Sync CI workflow"
```

## Scripts

- `scripts/download_models.py` - downloads `rife_*.7z` packs from the vs-mlrt
  [external-models](https://github.com/AmusementClub/vs-mlrt/releases/tag/external-models) release and extracts
  the ONNX files with the expected names (`--v2` also keeps the 7-channel `rife_v2` variants).
- `scripts/benchmark.py` - raw inference benchmark: OpenVINO CPU f32 / bf16 vs ONNX Runtime CPU on a given
  ONNX file and resolution, printing ms/frame and fps (`pip install openvino numpy`, optionally `onnxruntime`).
- `scripts/convert_to_ir.py` - optional ONNX -> OpenVINO IR (`.xml` + `.bin`) conversion, optionally with a fixed
  input shape. The plugin loads `.onnx` directly, so this is only a convenience.
- `scripts/rawclip.py` - helper used by the tests to load `ffmpeg -pix_fmt gbrpf32le -f rawvideo` output as an
  RGBS clip without any source plugin.

## Performance tuning (e.g. Ryzen 9 9950X, 16 cores / 32 threads)

The CPU plugin is already multi-threaded, so the first thing to decide is whether you want one fast inference at
a time (`threads=1`, LATENCY) or several independent ones (`threads>1`, THROUGHPUT):

- **`threads=1` + `ov_threads=16`** (one per physical core; this is also what the automatic setting does) and
  `hint="latency"` gives the lowest per-frame latency. Start here.
- **`threads=2..4`** + `hint="throughput"` (the default when `threads>1`) splits the cores into that many
  independent streams, each processing a different frame. This usually gives the best total fps on large core
  counts because RIFE is a sequence of fairly small convolutions that do not scale perfectly across 16 cores; it
  costs `threads` times the tensor memory. Try `threads=2` and `threads=4` with `ov_threads=0`; also raise
  VapourSynth's `core.num_threads` so enough frames are requested in parallel.
- **`precision="bf16"`**: ~10-25 % faster on Zen 4/5 thanks to AVX-512 BF16. Compare a few frames against f32
  if you are picky about accuracy; the difference is normally far below what is visible after encoding.
- **`ov_threads=32`** (all SMT threads) is rarely faster than 16 for convolution workloads; measure.
- **`cache_dir`** removes the ONNX import + graph compilation (a few hundred ms to a couple of seconds for 4K)
  on each script start.
- Keep the input at the real content resolution; padding is only to the next multiple of 64, but 4K at
  `rife_v4.26` is ~16x the work of 960x540.
- On a VPS that exposes vCPUs as SMT siblings (e.g. 2 vCPUs = 1 core + HT), set `ov_threads` to the vCPU count
  explicitly (see the numbers below), otherwise OpenVINO uses only one thread per physical core.

## Measured on the development VM

All numbers below come from a small cloud VM: **AMD EPYC 9254 (Zen 4), 2 vCPUs = 1 physical core + SMT, 7 GB
RAM, no GPU**, OpenVINO 2026.4.1, model `rife_v4.26`, 2x interpolation through VapourSynth (so the times include
the frame copies, padding and VapourSynth overhead). They are useful for relative comparisons only - a 16-core
desktop will be many times faster.

| resolution | settings | ms per interpolated frame | output fps (2x) |
|---|---|---|---|
| 640x360 | f32, defaults | 224 | 8.9 |
| 640x360 | bf16 | 190 | 10.5 |
| 640x360 | bf16, ov_threads=2 | 123 | 16.2 |
| 640x360 | bf16, threads=2 | 107 | 18.6 |
| 960x576 | f32, defaults | 495 | 4.0 |
| 960x576 | bf16 | 462 | 4.3 |
| 960x576 | f32, threads=2 | 282 | 7.1 |
| 960x576 | f32, ov_threads=2 | 356 | 5.6 |
| 960x576 | bf16, ov_threads=2 | 328 | 6.1 |
| 960x576 | bf16, threads=2 | 244 | 8.2 |

Raw inference (`scripts/benchmark.py`, 960x576, no VapourSynth): OpenVINO f32 536 ms/frame, OpenVINO bf16 455
ms/frame, ONNX Runtime CPU f32 1021 ms/frame.

Quality check: a 24 fps 640x360 `testsrc2` clip had its odd frames dropped and was re-interpolated 2x with
`rife_v4.26`; the interpolated frames reach **27.6-27.9 dB PSNR** against the real frames, versus 21.9-22.6 dB for
a plain 50/50 blend of the neighbours.

## Limitations

- RIFE **v4.x** models only (ONNX exports from vs-mlrt); no v1/v2/v3/HD/UHD/anime models.
- No `tta` / `uhd` modes.
- Batch size is fixed at 1 (one frame pair per inference); use `threads` for parallelism.
- The `rife_v2/` (7-channel, internal padding) ONNX variants are loaded on a best-effort basis - they work with
  OpenVINO 2026.4.1 in testing, but the `rife/` (11-channel) variants are the supported layout.
- Only the CPU device is tested. Other OpenVINO devices are passed through unmodified.

## License

MIT - see [LICENSE](LICENSE). Original plugin (c) HolyWu and styler00dollar; OpenVINO port (c) BlohoJo.
RIFE models (c) hzwer et al. (MIT), ONNX exports courtesy of the vs-mlrt project (AmusementClub).
