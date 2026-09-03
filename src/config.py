"""Shared paths / constants for the 2D->3D millirobot tracking pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIDEO = ROOT / "Video.mp4"
CSV   = ROOT / "Path 2.csv"
OUT   = ROOT / "out"
DATA  = ROOT / "data"
for d in (OUT, DATA, OUT / "frames", OUT / "figs"):
    d.mkdir(parents=True, exist_ok=True)

FPS = 30.0
W, H = 960, 720


# --- Windows unicode-path-safe OpenCV IO -------------------------------------
# cv2.imread/imwrite go through the ANSI CRT on Windows and SILENTLY FAIL (no
# exception, just a None or a no-op) when any component of the path is
# non-ASCII. Always use these two instead.
import numpy as _np, cv2 as _cv2


def imread_u(path, flags=_cv2.IMREAD_COLOR):
    buf = _np.fromfile(str(path), dtype=_np.uint8)
    img = _cv2.imdecode(buf, flags)
    if img is None:
        raise IOError(f"could not decode image: {path}")
    return img


def imwrite_u(path, img):
    path = str(path)
    ext = "." + path.rsplit(".", 1)[-1]
    ok, buf = _cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"could not encode image: {path}")
    buf.tofile(path)
    return True
