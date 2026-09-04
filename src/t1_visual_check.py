"""T1 - is the tracked dark blob actually a robot INSIDE the lumen?

Challenge raised: looking at the video, the object does not appear to be
running inside the tube. If the detector locked onto something else - a shadow,
an external driving magnet, a marker on the outside of the phantom - then the
whole pipeline is tracking the wrong object and every downstream number is
about that wrong object.

This does not argue from the fitted geometry (which would be circular). It
crops the raw frames around the detection and lets the pixels answer.
"""
import cv2, numpy as np
from config import VIDEO, DATA, OUT, imwrite_u

uv = np.load(DATA / "track2d.npz")["uv"]
cap = cv2.VideoCapture(str(VIDEO))
PICKS = [120, 300, 450, 620, 780, 900]
R = 90
tiles = []
for k in PICKS:
    cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, f = cap.read()
    if not ok or np.isnan(uv[k, 0]): continue
    x, y = int(uv[k, 0]), int(uv[k, 1])
    x0, y0 = max(0, x - R), max(0, y - R)
    x1, y1 = min(f.shape[1], x + R), min(f.shape[0], y + R)
    crop = f[y0:y1, x0:x1].copy()
    crop = cv2.resize(crop, (360, 360), interpolation=cv2.INTER_NEAREST)
    cx = int((x - x0) / (x1 - x0) * 360); cy = int((y - y0) / (y1 - y0) * 360)
    cv2.circle(crop, (cx, cy), 26, (0, 255, 255), 2)
    cv2.putText(crop, f"f{k}", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2)
    tiles.append(crop)
cap.release()
grid = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:6])])
imwrite_u(OUT / "figs" / "t1_zoom.png", grid)
print(f"[T1] saved out/figs/t1_zoom.png  ({len(tiles)} crops, 180x180 px each, 2x zoom)")
