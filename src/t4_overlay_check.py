"""T4 - does the PROJECTED CENTERLINE drawn in the animation actually lie on the
tubes? If the grey curve in the left panel wanders off the vessels, the
animation looks like a robot running outside the pipes even when the detection
itself is fine - and that is a defect in the deliverable, not in the tracking.
"""
import cv2, numpy as np
from config import DATA, OUT, W, H, imread_u, imwrite_u

bg = imread_u(OUT / "background_median.png")
hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
tight = (((h < 25) | (h > 170)) & (s > 40) & (v > 40)).astype(np.uint8)
tight = cv2.morphologyEx(tight, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

PG = np.load(DATA / "q1_affine_downstream.npz")["PG"]     # what the animation draws
uv = np.load(DATA / "track2d.npz")["uv"]

def frac_on_vessel(P, name):
    xi = np.clip(np.rint(P[:, 0]).astype(int), 0, W - 1)
    yi = np.clip(np.rint(P[:, 1]).astype(int), 0, H - 1)
    inb = (P[:, 0] >= 0) & (P[:, 0] < W) & (P[:, 1] >= 0) & (P[:, 1] < H)
    on = tight[yi, xi].astype(bool) & inb
    print(f"     {name:<38} in frame {100*inb.mean():5.1f}%   on a vessel {100*on.mean():5.1f}%")
    return on

print("[T4] how much of each curve lies on an orange tube?")
frac_on_vessel(PG, "projected centerline (animation)")
ok = ~np.isnan(uv[:, 0])
frac_on_vessel(uv[ok], "detected robot track")

vis = bg.copy()
vis[tight > 0] = (0.55 * vis[tight > 0] + 0.45 * np.array([255, 60, 60])).astype(np.uint8)
for i in range(len(PG) - 1):
    p0, p1 = PG[i].astype(int), PG[i + 1].astype(int)
    if np.all(np.abs(p0) < 5000) and np.all(np.abs(p1) < 5000):
        cv2.line(vis, tuple(p0), tuple(p1), (0, 255, 255), 2)
for p in uv[ok][::6].astype(int):
    cv2.circle(vis, tuple(p), 2, (0, 0, 255), -1)
imwrite_u(OUT / "figs" / "t4_overlay.png", vis)
print("[T4] saved out/figs/t4_overlay.png  (blue tint = orange lumen,"
      " yellow = projected Path 2, red dots = detections)")
