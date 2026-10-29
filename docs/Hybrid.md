# Using RIFE-OpenVINO with Hybrid

[Hybrid](https://www.selur.de/) (Windows build 2026.10.10.1 is the reference for this page) ships RIFE through the
ncnn-Vulkan plugin (`vsfilters\FrameFilter\RIFE\librife.dll`, namespace `rife`). This page covers two things:

1. **For users today:** how to drop `RIFE-OpenVINO.dll` into a Hybrid installation and call it from Hybrid's
   *Filtering → Vapoursynth → Custom* tab.
2. **For Hybrid's developer:** a proposal for native support that reuses the existing RIFE GUI almost unchanged.

Everything was written against Hybrid's installed file tree and a script Hybrid generated; the plugin itself was
tested through `vspipe` on Linux and on a Ryzen 9 9950X on Windows, but not inside the Hybrid GUI, so treat the
Custom-tab details as "verified by reading Hybrid's output", not as something the author clicked through.

## 1. Installing the plugin into Hybrid (user side)

### 1.1 Where to put it

Hybrid keeps every frame filter in its own folder together with that filter's private dependencies
(`FrameFilter\Bore\` carries its own `libgsl-28.dll`, `FrameFilter\RIFE\` carries `librife*.dll` plus a `models\`
tree). Follow the same pattern and create:

```
C:\Program Files\Hybrid\64bit\vsfilters\FrameFilter\RIFE-OpenVINO\
```

Writing below `C:\Program Files` needs an elevated Explorer/PowerShell. Do **not** put the files into the
existing `FrameFilter\RIFE\` folder: it is Hybrid's, gets overwritten on every Hybrid update, and the OpenVINO
`.onnx` models and the ncnn `models\rife-v4.x\` folders would end up side by side with the same names.

### 1.2 What to copy

Take `VapourSynth-RIFE-OpenVINO-win64.zip` from the GitHub release (or the CI artifact of the same name) and copy
its contents into that folder. The archive holds exactly the set the plugin needs:

| file | role |
|---|---|
| `RIFE-OpenVINO.dll` | the VapourSynth plugin (namespace `rifeov`) |
| `openvino.dll` | OpenVINO runtime core |
| `openvino_intel_cpu_plugin.dll` | CPU device plugin (the one actually doing the inference) |
| `openvino_onnx_frontend.dll` | reads the `.onnx` model files |
| `openvino_ir_frontend.dll` | reads OpenVINO IR `.xml` models (optional format, see `scripts\convert_to_ir.py`) |
| `tbb12.dll`, `tbbmalloc.dll`, `tbbmalloc_proxy.dll`, `tbbbind_2_5.dll` | oneTBB threading runtime used by OpenVINO |
| `cache.json` | optional OpenVINO plugin registry (only present when the OpenVINO archive had one) |
| `scripts\` | `download_models.py`, `benchmark.py`, `profile.py`, `convert_to_ir.py` - helpers, not needed at run time |

VapourSynth loads plugins with `LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR`, so the OpenVINO and TBB DLLs **must sit in the
same folder as `RIFE-OpenVINO.dll`** (or be on `PATH`). Nothing goes into `plugins64`/`plugins2`: Hybrid does not
autoload from `vsfilters`, it `LoadPlugin`s every filter explicitly, and so will your Custom-tab snippet.

Hybrid's `64bit\` root already contains an OpenVINO runtime (`openvino.dll`, `openvino_intel_cpu_plugin.dll`,
`tbb12.dll`, ... dated 2026-05-25, not the same build as the 2026.4.1 the plugin is compiled against - the file
sizes differ). Leave it alone and do not try to share it: that folder is not on `vspipe.exe`'s DLL search path
(`vspipe.exe` lives in `64bit\Vapoursynth\Lib\site-packages\vapoursynth\`), and OpenVINO's CPU plugin and core
library have to come from the same build. Keeping the plugin's own copy next to it is the only combination that
is known to work.

### 1.3 Models

The ONNX models are not in the archive. Create `...\RIFE-OpenVINO\models\` and download into it, e.g. with
Hybrid's own Python (it is a normal CPython 3.14 in `C:\Program Files\Hybrid\64bit\Vapoursynth\`):

```bat
cd "C:\Program Files\Hybrid\64bit\vsfilters\FrameFilter\RIFE-OpenVINO"
"C:\Program Files\Hybrid\64bit\Vapoursynth\python.exe" scripts\download_models.py v4.25_heavy v4.26 --out models
```

`--list` prints every available name, `--all` fetches them all (~22-87 MB each). The script needs 7-Zip's
`7z.exe` on `PATH` or `pip install py7zr`; see the README for the `CERTIFICATE_VERIFY_FAILED` workaround. If the
models already exist somewhere else (e.g. `E:\USER\PythonVE\RIFE\models\`), skip the download and pass that
directory as `model_path=` in the script instead.

The `model` numbers are the ncnn plugin's, so whatever Hybrid shows in its RIFE *Model* dropdown maps 1:1:
`rife-v4.25-heavy (ensemble=False)` is `model=71` in Hybrid's generated script and is `model=71` here too
(`rife_v4.25_heavy.onnx`). 69 = v4.25, 70 = v4.25_lite, 72 = v4.26 (default), 73 = v4.26_heavy (ncnn
"v4.26-large"). Ensemble variants exist as ONNX only up to v4.24.

### 1.4 What to enter on the Custom tab

Hybrid's generated script (see `Hybrid_VapourSynth_Script_Example.txt`-style output) does, for its own RIFE:

```python
core.std.LoadPlugin(path="C:/Program Files/Hybrid/64bit/vsfilters/FrameFilter/RIFE/librife.dll")
...
clip = core.resize.Spline36(clip=clip, format=vs.RGBS, matrix_in_s="709", range_in_s="limited", range_s="full")
clip = core.rife.RIFE(clip, model=71, fps_num=60, fps_den=1, sc=False)
clip = core.resize.Spline36(clip=clip, format=vs.YUV420P8, matrix_s="709", range_in_s="full", range_s="limited", dither_type="error_diffusion")
clip = core.std.AssumeFPS(clip=clip, fpsnum=60, fpsden=1)
```

The OpenVINO version is the same four lines with a different plugin and two extra tuning arguments. Turn **off**
Hybrid's *Frame Interpolation* checkbox (otherwise both run) and put this into the Custom section, at a position
where `clip` is still the filtered YUV clip (after denoising/resizing, before the encoder's colour conversion):

```python
# RIFE-OpenVINO (CPU) frame interpolation
core.std.LoadPlugin(path="C:/Program Files/Hybrid/64bit/vsfilters/FrameFilter/RIFE-OpenVINO/RIFE-OpenVINO.dll")
clip = core.resize.Spline36(clip=clip, format=vs.RGBS, matrix_in_s="709", range_in_s="limited", range_s="full")
clip = core.rifeov.RIFE(clip, model=71, fps_num=60, fps_den=1, precision="bf16", ov_threads=32, sc=False)
clip = core.resize.Spline36(clip=clip, format=vs.YUV420P8, matrix_s="709", range_in_s="full", range_s="limited", dither_type="error_diffusion")
clip = core.std.AssumeFPS(clip=clip, fpsnum=60, fpsden=1)
```

Adjust `matrix_in_s`/`range_in_s` and the output `format` to what the rest of your script uses (the lines above
are for a BT.709, limited-range YUV420P8 source, i.e. exactly what Hybrid emitted in the example). Notes:

- `precision="bf16"` needs AVX-512 BF16 (Zen 4/5, Sapphire Rapids+). On other CPUs use `"f32"` (the default).
- `ov_threads=32` = all SMT threads of a 9950X. `ov_threads=0` (default) uses one thread per physical core.
  Alternative that usually gives the highest total fps on 16+ cores: `threads=2` (or 3-4) with `ov_threads=0`;
  the plugin then switches to OpenVINO's THROUGHPUT hint and runs that many frames in parallel. Keep
  Hybrid's VapourSynth thread count at its default or higher so enough frames are requested at once.
- `cache_dir="C:/Users/<you>/AppData/Local/rifeov-cache"` (any writable folder) skips the ONNX import and graph
  compilation on every job start - worth it for batch jobs, a few hundred ms to a couple of seconds each.
- `sc=True` requires `_SceneChangeNext` frame props, i.e. `clip = core.misc.SCDetect(clip, threshold=0.1)` on
  the YUV clip **before** the RGBS conversion - the same thing Hybrid's own *Scene change* checkbox adds. `skip`
  needs the VMAF plugin, which Hybrid does not ship.
- `factor_num=2, factor_den=1` instead of `fps_num/fps_den` gives a plain 2x, like Hybrid's *Factor* checkbox.
- Whatever changes the frame rate inside a Custom section is invisible to Hybrid's own bookkeeping (the
  *Output: FPS / Frame count* line at the bottom of the window and the values it hands to the muxer). If the
  Custom tab offers a way to declare the new frame rate / frame count, set it; otherwise check the muxed file's
  frame rate and duration once. This is one of the reasons native support is worth asking for.

Sanity check before encoding: Hybrid's *Preview* / *vsViewer* should show the new frame count, and a `vspipe
--info` of the generated `.vpy` must report `Frames: <about 1.0225 x the input>` and `FPS: 60/1` for a 58.678 →
60 conversion.

### 1.5 If it does not load

`core.std.LoadPlugin` failing with "The specified module could not be found" means a dependency is missing next
to the DLL - re-check the file list above (`tbb12.dll` is the usual culprit). "model file ... not found" means
the `models\` folder or the `.onnx` for that `model` index is missing. A crash only when *another* OpenVINO- or
TBB-using plugin is loaded in the same script would be a `tbb12.dll` version collision (Windows loads one module
per name per process); none is known, but if it happens, load `RIFE-OpenVINO.dll` first so its TBB wins.

## 2. Proposal for native support in Hybrid (developer side)

This section is written so it can be posted to the [Hybrid forum](https://forum.selur.net/) mostly as-is.

### 2.1 What the plugin is

[VapourSynth-RIFE-ncnn-OpenVINO](https://github.com/BlohoJo/VapourSynth-RIFE-ncnn-OpenVINO) is a port of the
RIFE ncnn-Vulkan plugin that Hybrid already uses to Intel's OpenVINO runtime, CPU device. Same filter structure,
same frame-property handling, same `sc`/`skip`/`skip_threshold` logic, same `model` numbering; the inference
backend is the only thing that changed. It targets machines **without a usable GPU** (servers, VPSes, laptops with
a broken Vulkan stack, AMD boxes where the ncnn-Vulkan plugin misbehaves) and uses the ONNX exports of the vs-mlrt
project, so it runs every RIFE 4.x model from v4.0 to v4.26_heavy. OpenVINO's CPU plugin is plain x86-64
(AVX2 / AVX-512 / AVX-512 BF16 code paths); "Intel" in the name does not mean Intel-only - it has been verified on
AMD Zen 4 (EPYC 9254) and Zen 5 (Ryzen 9 9950X). MIT licence, Windows x64 and Linux builds via GitHub Actions,
tagged releases.

### 2.2 Why it cannot be a `CustomSynthScripts` XML filter

Hybrid's XML custom-filter mechanism (forum thread 3231) is explicitly restricted to filters that do not change
resolution, frame count or frame rate. RIFE changes the last two, so the only places it can live are the Custom
tab (today) or the existing *Frame → Interpolation* page (the proposal).

### 2.3 Smallest possible GUI change

The call is deliberately a drop-in for `core.rife.RIFE`:

```
rifeov.RIFE(clip, model, factor_num, factor_den, fps_num, fps_den, model_path, device="CPU", threads=1,
            ov_threads=0, precision="f32", hint, cache_dir, pad=0, sc, skip, skip_threshold, list_devices)
```

| Hybrid's RIFE (ncnn) GUI element | ncnn argument | OpenVINO argument | note |
|---|---|---|---|
| Model dropdown | `model=N` | `model=N` | **identical numbering**; v1/v2/v3/HD/UHD/anime entries (0-10, 13-14) have no ONNX and must be hidden or greyed |
| Target FPS / Factor | `fps_num/fps_den`, `factor_num/factor_den` | same | unchanged |
| Scene change + SCD tools | `sc=True` + `misc.SCDetect` | same | unchanged |
| Skip threshold | `skip`, `skip_threshold` | same | unchanged (needs VMAF plugin, as today) |
| GPU-threads | `gpu_thread=N` | `threads=N` | same meaning: parallel inferences; default 1 |
| Force gpu / gpu id | `gpu_id` | *(drop)* | `device="CPU"` is the default; no GUI needed |
| TTA, UHD | `tta`, `uhd` | *(drop / disable)* | not available in the ONNX graphs |
| *(new)* Precision | - | `precision="f32"` / `"bf16"` | bf16 ≈ 10-25 % faster on AVX-512 BF16 CPUs (Zen 4/5, Sapphire Rapids+) |
| *(new, optional)* OV threads | - | `ov_threads=N` | 0 = one per physical core (auto); users with SMT may want "all logical cores" |
| *(new, optional)* Model cache | - | `cache_dir=<Hybrid temp dir>` | removes 0.3-2 s model compile per job start |

Concretely, the least work is a backend selector on the existing *Frame Interpolation → RIFE* page (or a second
dropdown entry "RIFE (OpenVINO/CPU)" next to "RIFE"), which:

1. writes `core.std.LoadPlugin(path=".../vsfilters/FrameFilter/RIFE-OpenVINO/RIFE-OpenVINO.dll")` instead of
   `librife.dll`;
2. emits `core.rifeov.RIFE(` instead of `core.rife.RIFE(` with the same `model`, `fps_num/fps_den` or
   `factor_*`, `sc`, `skip`, `skip_threshold`;
3. passes *GPU-threads* as `threads=`, drops `gpu_id`, disables TTA/UHD, hides models without ONNX;
4. adds a *Precision* combo (f32/bf16) and, optionally, an *OV threads* spin box.

The surrounding RGBS conversion, `AssumeFPS`, frame-count bookkeeping and the SCDetect handling are exactly what
Hybrid already generates for the ncnn plugin (see the script above) and need no change. The output clip's `fps`
and `num_frames` are set by the plugin the same way `rife.RIFE` sets them.

### 2.4 Packaging

- Plugin folder: `64bit\vsfilters\FrameFilter\RIFE-OpenVINO\` with the 9 DLLs listed in section 1.2 (about 69 MB
  unpacked with OpenVINO 2026.4.1: `openvino_intel_cpu_plugin.dll` alone is 46 MB). The GitHub release zip
  `VapourSynth-RIFE-OpenVINO-win64.zip` is exactly that folder plus `README`/`LICENSE`/`scripts`, so the addon
  can be built by unzipping it.
- Hybrid already ships an OpenVINO runtime in `64bit\` (dated 2026-05-25). If it is convenient to keep a single
  copy, the plugin can be rebuilt against that exact OpenVINO version (CMake `-DOpenVINO_DIR=...`; any OpenVINO
  ≥ 2024.0 should work, the code uses only the stable `ov::Core` API) and the DLLs dropped from the plugin folder,
  provided `vspipe.exe` can find them (PATH or a copy beside the plugin). Mixing `openvino.dll` from one build
  with `openvino_intel_cpu_plugin.dll` from another does not work, and two different `tbb12.dll` in one process
  is a risk, so "one OpenVINO per process" is the rule to design around.
- Models: the `.onnx` files are roughly 20-90 MB each (the `heavy` models are the large ones) and are published by vs-mlrt
  as `models.v*.7z` / `ext-models.v*.7z` archives on its GitHub releases. They fit the existing
  "optional addon download" model Hybrid uses for other large components; the file names are
  `rife_v4.26.onnx`, `rife_v4.25_heavy.onnx`, `rife_v4.18_ensemble.onnx`, ... and the plugin picks them by the
  `model` index from a `models\` folder next to the DLL (or from `model_path=`). A minimal addon with v4.25,
  v4.25_lite, v4.25_heavy, v4.26 and v4.26_heavy stays in the low hundreds of MB.
- Linux: the Linux build (`librife_openvino.so`) and the OpenVINO `.so` set are covered in the README and would
  fit the `hybrid-vapoursynth-addon` build the same way as the ncnn plugin.

### 2.5 Defaults worth baking in

`threads=1`, `ov_threads=0`, `precision="f32"` are safe everywhere. For speed on modern desktops: `bf16` when
the CPU reports AVX-512 BF16 (OpenVINO exposes this through `ov::device::capabilities`; or simply let the user
tick it), and either `ov_threads = logical core count` or `threads=2` with the THROUGHPUT hint - both measured
faster than the defaults on a 16-core Zen 5; see the README's performance section. A `cache_dir` inside
Hybrid's temp folder is free speed for every job after the first. Realistic expectations: a 16-core desktop CPU
does single-digit fps at 1080p with the heavy models - usable for offline conversions, not a replacement for a
GPU; the README's "Where a 1080p frame goes" paragraph explains why the heavy models cost little extra on CPU.

### 2.6 Alternative with zero new plugin: vs-mlrt

vs-mlrt's `vsov` backend (`Backend.OV_CPU()`) runs the same ONNX models on OpenVINO from its Python wrapper
(`vsmlrt.RIFE(...)`). Hybrid already carries a `vsmlrt-cuda` folder, so part of vs-mlrt may already be present. The
trade-off: vs-mlrt's RIFE wrapper has a different signature (`multi`, `model=vsmlrt.RIFEModel.v4_26`, no `sc`/`skip`
path, `ensemble` flag) and would need its own code path in Hybrid; `rifeov.RIFE` keeps the GUI change to a
namespace swap plus a precision combo. Both are valid; the plugin exists to make the second option cheap.
