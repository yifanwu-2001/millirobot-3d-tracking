"""T6 - WHERE is the projected centerline off-vessel, and is that region empty
background or an unfilled (clear) tube the orange mask cannot see?"""
import cv2, numpy as np
from config import DATA, OUT, W, H, imread_u, imwrite_u

bg = imread_u(OUT / "background_median.png")
z = np.load(DATA / "t5_anatomical.npz")
on = z["on|affine (no depth)"]; d = z["d|affine (no depth)"]
from scipy.spatial.transform import Rotation as Rot
from a_centerline import load as load_cl
s3, C3, _, _ = load_cl()
SG = np.arange(0, s3[-1] + 1e-9, 0.25)
CG = np.stack([np.interp(SG, s3, C3[:, c]) for c in range(3)], 1)
pa = np.load(DATA / "l2_affine.npy")
P = (CG @ Rot.from_rotvec(pa[:3]).as_matrix().T)[:, :2] * np.exp(pa[3]) + pa[4:6]

# contiguous off-vessel runs
bad = ~on
dd = np.diff(np.r_[0, bad.astype(int), 0])
st, en = np.where(dd == 1)[0], np.where(dd == -1)[0]
runs = sorted(zip(en - st, st, en), reverse=True)[:4]
print("[T6] longest off-vessel stretches of the projected centerline:")
for ln, a, b in runs:
    print(f"     s = {SG[a]:6.1f} - {SG[b-1]:6.1f} mm  ({ln} samples, {SG[b-1]-SG[a]:5.1f} mm)"
          f"  max {d[a:b].max():5.1f} px off  centroid ({P[a:b,0].mean():.0f},{P[a:b,1].mean():.0f})")

tiles = []
for ln, a, b in runs:
    cx, cy = int(P[a:b, 0].mean()), int(P[a:b, 1].mean())
    R = 130
    x0, y0 = max(0, cx - R), max(0, cy - R)
    crop = bg[y0:min(H, cy + R), x0:min(W, cx + R)].copy()
    if crop.shape[0] < 60 or crop.shape[1] < 60: continue
    for i in range(a, b - 1):
        p0 = (int(P[i, 0]) - x0, int(P[i, 1]) - y0)
        p1 = (int(P[i + 1, 0]) - x0, int(P[i + 1, 1]) - y0)
        cv2.line(crop, p0, p1, (0, 255, 255), 2)
    crop = cv2.resize(crop, (330, 330), interpolation=cv2.INTER_NEAREST)
    cv2.putText(crop, f"s={SG[a]:.0f}-{SG[b-1]:.0f}mm", (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 255), 2)
    tiles.append(crop)
if tiles:
    imwrite_u(OUT / "figs" / "t6_offvessel.png", np.hstack(tiles))
    print(f"[T6] saved out/figs/t6_offvessel.png ({len(tiles)} crops, yellow = the off-vessel curve)")
