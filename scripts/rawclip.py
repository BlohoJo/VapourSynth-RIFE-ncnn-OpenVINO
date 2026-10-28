"""
Tiny helper to turn a raw planar float RGB file (ffmpeg -pix_fmt gbrpf32le
-f rawvideo) into a VapourSynth RGBS clip without any source plugin.

    import rawclip
    clip = rawclip.load(core, "clip.raw", 640, 360, fps=(24, 1))

Used by the test scripts; also handy on machines without ffms2/bestsource.
"""

import numpy as np
import vapoursynth as vs


def load(core, path, width, height, fps=(24, 1)):
    frame_floats = width * height * 3
    data = np.memmap(path, dtype=np.float32, mode="r")
    num_frames = data.size // frame_floats
    if num_frames < 1:
        raise ValueError(f"{path}: file too small for a {width}x{height} frame")

    # ffmpeg gbrpf32le plane order is G, B, R -> VapourSynth RGB is R, G, B
    plane_map = {0: 2, 1: 0, 2: 1}

    blank = core.std.BlankClip(width=width, height=height, format=vs.RGBS,
                               length=num_frames, fpsnum=fps[0], fpsden=fps[1], keep=True)

    def fill(n, f):
        fout = f.copy()
        base = n * frame_floats
        for plane in range(3):
            src = data[base + plane_map[plane] * width * height:
                       base + (plane_map[plane] + 1) * width * height].reshape(height, width)
            np.asarray(fout[plane])[:] = src
        return fout

    return core.std.ModifyFrame(blank, blank, fill)
