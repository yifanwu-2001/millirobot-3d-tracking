"""T7 - the orange mask cannot see unfilled tubes, so 'off-vessel' was overstated.

T6 showed the longest off-vessel stretch runs along a pale, fluid-free tube
segment. The phantom's background is flat, bright and unsaturated; tubes -
filled or not - are darker, tinted, or textured. This builds a mask of ANY
tube-like structure and re-scores the cameras, so the anatomical-plausibility
metric is not just measuring where the dye went.
"""
import cv2, numpy as np
from scipy.spatial.transform import Rotation as Rot
from config import DATA, OUT, W, H, imread_u, imwrite_u
from a_centerline import load as load_cl

bg = imread_u(OUT / "background_median.png")
hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)

orange = ((h < 25) | (h > 170)) & (s > 40) & (v > 40)
# background: bright, nearly unsaturated, and locally flat
flat = cv2.absdiff(gray, cv2.GaussianBlur(gray, (0, 0), 6)) < 4
background = (v > 215) & (s < 28) & flat
anytube = (~background)
anytube = cv2.morphologyEx(anytube.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
anytube = cv2.morphologyEx(anytube, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
imwrite_u(OUT / "figs" / "t7_anytube_mask.png", anytube * 255)
print(f"[T7] mask coverage: orange-only {100*orange.mean():.1f}%,"
      f" any-tube {100*anytube.mean():.1f}%")

dist_out = cv2.distanceTransform(1 - anytube, cv2.DIST_L2, 5)
s3, C3, _, _ = load_cl()
SG = np.arange(0, s3[-1] + 1e-9, 0.25)
CG = np.stack([np.interp(SG, s3, C3[:, c]) for c in range(3)], 1)


def pin(rv, t, f, cx, cy):
    Xc = CG @ Rot.from_rotvec(rv).as_matrix().T + t
    return Xc[:, :2] / Xc[:, 2:3] * f + np.array([cx, cy])


p11 = np.load(DATA / "c8_11.npy"); z10 = np.load(DATA / "c10_ba.npz")
pa = np.load(DATA / "l2_affine.npy")
cams = {
    "pinhole f=1298": pin(p11[:3], p11[3:6], np.exp(p11[6]), p11[9], p11[10]),
    "pinhole f=516": pin(z10["rv"], z10["t"], float(z10["f"]), float(z10["cx"]), float(z10["cy"])),
    "affine (no depth)": (CG @ Rot.from_rotvec(pa[:3]).as_matrix().T)[:, :2] * np.exp(pa[3]) + pa[4:6],
}
oldz = np.load(DATA / "t5_anatomical.npz")
print(f"\n[T7] projected centerline on ANY tube (vs orange-only in brackets)")
print(f"{'camera':<22}{'on tube':>10}{'was':>9}{'off >10px':>12}{'worst px':>10}")
for nm, P in cams.items():
    xi = np.clip(np.rint(P[:, 0]).astype(int), 0, W - 1)
    yi = np.clip(np.rint(P[:, 1]).astype(int), 0, H - 1)
    d = dist_out[yi, xi]
    old = 100 * oldz[f"on|{nm}"].mean()
    print(f"{nm:<22}{100*np.mean(d==0):9.1f}%{old:8.1f}%{100*np.mean(d>10):11.1f}%{d.max():10.1f}")

uvz = np.load(DATA / "track2d.npz")["uv"]; ok = ~np.isnan(uvz[:, 0])
xi = np.clip(np.rint(uvz[ok, 0]).astype(int), 0, W - 1)
yi = np.clip(np.rint(uvz[ok, 1]).astype(int), 0, H - 1)
print(f"\n[T7] detected robot track on ANY tube: {100*np.mean(dist_out[yi, xi] == 0):.1f}%")
print(f"[T7] random image point: {100*anytube.mean():.1f}%  <- the metric is weaker now,")
print(f"     because the any-tube mask covers much more of the frame. Use both:")
print(f"     orange-only is specific but blind to unfilled tubes; any-tube is complete")
print(f"     but permissive. The truth for each stretch is between them.")
