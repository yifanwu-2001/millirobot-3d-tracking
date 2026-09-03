"""Stage D (exploration): confirm the robot is detectable via static-camera background subtraction.

Camera is static -> temporal median is a clean background plate. The robot is the
only large moving structure, and it is DARK against the orange vessel lumen.
"""
import cv2, numpy as np
from config import VIDEO, OUT, imwrite_u

cap = cv2.VideoCapture(str(VIDEO))
N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# --- 1. temporal median background from a strided subsample -------------------
idx = np.linspace(0, N - 1, 60).astype(int)
buf = []
for i in idx:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, f = cap.read()
    if ok: buf.append(f)
bg = np.median(np.stack(buf), axis=0).astype(np.uint8)
imwrite_u(OUT / "background_median.png", bg)
print(f"background from {len(buf)} frames")

# --- 2. per-frame difference, report the strongest moving blob ----------------
bg_g = cv2.GaussianBlur(cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
rows = []
for k in range(N):
    ok, f = cap.read()
    if not ok: break
    g = cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
    d = bg_g - g                      # positive where frame is DARKER than background
    d[d < 0] = 0
    m = d.max()
    yx = np.unravel_index(int(np.argmax(d)), d.shape)
    rows.append((k, m, yx[1], yx[0], float((d > 25).sum())))
cap.release()

rows = np.array(rows)
np.save(OUT / "explore_diff.npy", rows)
print("frame  maxdiff   x     y    area(px>25)")
for r in rows[::64]:
    print(f"{int(r[0]):5d}  {r[1]:6.1f}  {int(r[2]):4d} {int(r[3]):4d}  {r[4]:8.0f}")
print(f"\nmaxdiff over video: min {rows[:,1].min():.1f}  median {np.median(rows[:,1]):.1f}  max {rows[:,1].max():.1f}")
print(f"area  over video: min {rows[:,4].min():.0f}  median {np.median(rows[:,4]):.0f}  max {rows[:,4].max():.0f}")
