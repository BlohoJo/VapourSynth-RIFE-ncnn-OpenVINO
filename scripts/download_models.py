#!/usr/bin/env python3
"""
Download RIFE ONNX models from the vs-mlrt "external-models" release and
extract them into a models/ directory with the file names this plugin expects
(rife_v4.26.onnx, rife_v4.25_lite.onnx, rife_v4.6_ensemble.onnx, ...).

    python scripts/download_models.py                 # default: rife_v4.26
    python scripts/download_models.py v4.26 v4.25_lite v4.22_lite
    python scripts/download_models.py --all
    python scripts/download_models.py --list
    python scripts/download_models.py --out /usr/local/lib/vapoursynth/models v4.18

Each pack is a .7z archive containing rife/<name>.onnx (v1 layout, 11 input
channels - the one this plugin uses by default) and rife_v2/<name>.onnx
(v2 layout, 7 channels). By default only the v1 files are extracted; pass
--v2 to also keep the v2 variants (as <name>.v2.onnx).

Extraction needs either a `7z` executable (p7zip / 7-Zip) on PATH or the
`py7zr` Python package (pip install py7zr).

If the download fails with CERTIFICATE_VERIFY_FAILED (typical for python.org
builds on Windows whose certificate store lacks GitHub's root CA), the script
automatically retries with the `certifi` bundle if installed (pip install
certifi) and then with the system `curl`; `--insecure` skips verification as a
last resort.

Models are (c) hzwer et al. (RIFE, MIT) and were exported by the vs-mlrt
project: https://github.com/AmusementClub/vs-mlrt/releases/tag/external-models
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request

RELEASE = "https://github.com/AmusementClub/vs-mlrt/releases/download/external-models/"

# model name -> pack file in the external-models release
PACKS = {
    "v4.0_ensemble": "rife_ensemble_v1.7z",
    "v4.2_ensemble": "rife_ensemble_v1.7z",
    "v4.3_ensemble": "rife_ensemble_v1.7z",
    "v4.4_ensemble": "rife_ensemble_v1.7z",
    "v4.5_ensemble": "rife_ensemble_v1.7z",
    "v4.6_ensemble": "rife_ensemble_v1.7z",
}
for _v in ["4.7", "4.8", "4.9", "4.10", "4.11", "4.12", "4.13", "4.14", "4.15", "4.17", "4.18",
           "4.19", "4.20", "4.21", "4.22", "4.23", "4.24", "4.25", "4.26"]:
    PACKS[f"v{_v}"] = f"rife_v{_v}.7z"
for _v in ["4.7", "4.8", "4.9", "4.10", "4.11", "4.12", "4.13", "4.14", "4.15", "4.17", "4.18",
           "4.19", "4.20", "4.24"]:
    PACKS[f"v{_v}_ensemble"] = f"rife_v{_v}.7z"
for _v in ["4.12", "4.13", "4.14", "4.15", "4.16", "4.17", "4.22", "4.25"]:
    PACKS[f"v{_v}_lite"] = f"rife_v{_v}_lite.7z"
for _v in ["4.12", "4.13", "4.14", "4.15", "4.16", "4.17"]:
    PACKS[f"v{_v}_lite_ensemble"] = f"rife_v{_v}_lite.7z"
PACKS["v4.25_heavy"] = "rife_v4.25_heavy.7z"
PACKS["v4.26_heavy"] = "rife_v4.26_heavy.7z"

# v4.0 - v4.6 non-ensemble v1 exports are not in external-models; they ship in the
# big models.<version>.7z of the regular vs-mlrt releases.
BASE_MODELS = ["v4.0", "v4.2", "v4.3", "v4.4", "v4.5", "v4.6"]
BASE_PACK_URL = "https://github.com/AmusementClub/vs-mlrt/releases/download/v15.16/models.v15.16.7z"


def have_7z():
    return shutil.which("7z") or shutil.which("7za") or shutil.which("7zz")


def extract(archive, dest, patterns):
    exe = have_7z()
    if exe:
        cmd = [exe, "x", "-y", f"-o{dest}", archive] + patterns
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        return
    try:
        import py7zr  # type: ignore
    except ImportError:
        sys.exit("error: neither a 7z executable nor the py7zr package is available "
                 "(apt install p7zip-full / pip install py7zr)")
    with py7zr.SevenZipFile(archive, "r") as z:
        names = [n for n in z.getnames() if any(n.replace("\\", "/") == p for p in patterns)]
        z.extract(path=dest, targets=names)


def _ssl_contexts(insecure):
    """Yield (label, ssl.SSLContext) candidates in the order they should be tried."""
    import ssl
    yield "system certificates", ssl.create_default_context()
    try:
        import certifi  # type: ignore
        yield "certifi bundle", ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    if insecure:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        yield "NO certificate verification (--insecure)", ctx


def _download_python(url, path, ctx):
    req = urllib.request.Request(url, headers={"User-Agent": "download_models.py"})
    last = [-1]
    with urllib.request.urlopen(req, context=ctx) as resp, open(path, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total > 0 and sys.stdout.isatty() and done // 1048576 != last[0]:
                last[0] = done // 1048576
                sys.stdout.write(f"\r  {done / 1048576:.1f} / {total / 1048576:.1f} MB")
                sys.stdout.flush()
    print()


def _download_curl(url, path):
    exe = shutil.which("curl")
    if not exe:
        return False
    # curl on Windows uses Schannel (the OS certificate machinery, including its
    # automatic root-certificate update), on Linux/macOS the system bundle.
    r = subprocess.run([exe, "-L", "--fail", "--retry", "3", "-o", path, url])
    return r.returncode == 0 and os.path.exists(path) and os.path.getsize(path) > 0


SSL_HELP = """
The HTTPS certificate of {host} could not be verified by this Python
installation ("{err}").
This is almost always a missing root/intermediate certificate in the store that
Python uses - it is common with python.org builds on Windows - and not a problem
with the download itself. Any ONE of the following fixes it:

  1. pip install certifi           (then simply re-run this script)
  2. install curl                   (Windows 10 1803+ / 11 already ship curl.exe
                                     in C:\\Windows\\System32 - re-run the script)
  3. open {url}
     once in Microsoft Edge/Chrome; Windows then fetches the missing root
     certificate into its store and Python can verify it afterwards.
  4. re-run with --insecure to skip certificate verification for this download
     (the archives come from GitHub releases; only do this if you accept the
     risk of an unverified connection).
"""


def download(url, path, insecure=False):
    import ssl
    import urllib.error

    print(f"downloading {url}", flush=True)
    ssl_error = None
    for label, ctx in _ssl_contexts(insecure):
        try:
            if ssl_error is not None:
                print(f"  retrying with {label}")
            _download_python(url, path, ctx)
            return
        except urllib.error.URLError as e:
            if isinstance(getattr(e, "reason", None), ssl.SSLCertVerificationError):
                ssl_error = e.reason
                continue
            raise
        except ssl.SSLCertVerificationError as e:
            ssl_error = e
            continue

    # Python could not verify the certificate: try the platform's curl, which
    # uses the OS certificate store directly.
    if _download_curl(url, path):
        return
    if os.path.exists(path):
        os.remove(path)
    host = urllib.parse.urlsplit(url).netloc
    sys.exit(SSL_HELP.format(host=host, err=ssl_error, url=url))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", default=["v4.26"], help="model names, e.g. v4.26 v4.25_lite v4.18_ensemble")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))
    ap.add_argument("--all", action="store_true", help="download every available model")
    ap.add_argument("--list", action="store_true", help="list available model names and exit")
    ap.add_argument("--v2", action="store_true", help="also extract the rife_v2/ (7-channel) variants as <name>.v2.onnx")
    ap.add_argument("--keep", action="store_true", help="keep the downloaded .7z packs next to the models")
    ap.add_argument("--insecure", action="store_true",
                    help="skip HTTPS certificate verification if Python's certificate store cannot verify GitHub "
                         "(see the message printed when that happens for safer alternatives)")
    args = ap.parse_args()

    names = sorted(list(PACKS) + BASE_MODELS, key=lambda s: ([int(x) if x.isdigit() else x for x in s.replace("v", "").replace("_", ".").split(".")]))
    if args.list:
        for n in names:
            print(n)
        return

    wanted = names if args.all else args.models
    bad = [m for m in wanted if m not in PACKS and m not in BASE_MODELS]
    if bad:
        sys.exit(f"error: unknown model(s): {', '.join(bad)} (use --list)")

    os.makedirs(args.out, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="rife_models_")
    try:
        # group by pack so each archive is downloaded once
        packs = {}
        for m in wanted:
            pack = BASE_PACK_URL if m in BASE_MODELS else RELEASE + PACKS[m]
            packs.setdefault(pack, []).append(m)

        for url, models in packs.items():
            archive = os.path.join(args.out if args.keep else tmp, url.rsplit("/", 1)[1])
            if not os.path.exists(archive):
                download(url, archive, insecure=args.insecure)
            patterns = [f"rife/rife_{m}.onnx" for m in models]
            if args.v2:
                patterns += [f"rife_v2/rife_{m}.onnx" for m in models]
            extract(archive, tmp, patterns)
            for m in models:
                src = os.path.join(tmp, "rife", f"rife_{m}.onnx")
                if not os.path.exists(src):
                    print(f"  warning: rife/rife_{m}.onnx not found in {os.path.basename(archive)}")
                else:
                    dst = os.path.join(args.out, f"rife_{m}.onnx")
                    shutil.move(src, dst)
                    print(f"  -> {dst}")
                if args.v2:
                    src2 = os.path.join(tmp, "rife_v2", f"rife_{m}.onnx")
                    if os.path.exists(src2):
                        dst2 = os.path.join(args.out, f"rife_{m}.v2.onnx")
                        shutil.move(src2, dst2)
                        print(f"  -> {dst2}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
