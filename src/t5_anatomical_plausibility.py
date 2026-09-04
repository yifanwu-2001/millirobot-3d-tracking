"""T5 - a metric nothing in this project had: is the projected model ON the
anatomy?

Every measure used so far - reprojection, off-curve distance, out-and-back
agreement, held-out residual - compares the ESTIMATE to the OBSERVATION. None
asks whether the projected centerline lies inside a vessel at all. A camera can
place its curve close to every detected robot position and still route long
stretches of that curve through open space between tubes, because the robot
only ever visits part of the curve.

That is exactly what the animation revealed by eye. Here it is as a number, and
as a fourth discriminator for the camera family, computed against the image and
not against any camera's own fit.
"""
import cv2, numpy as np
from scipy.spatial.transform import Rotation as Rot
from config import DATA, OUT, W, H, imread_u
from a_centerline import load as load_cl

bg = imread_u(OUT / "background_median.png")
hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
lum = (((h < 25) | (h > 170)) & (s > 40) & (v > 40)).astype(np.uint8)
lum = cv2.morphologyEx(lum, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
dist_out = cv2.distanceTransform(1 - lum, cv2.DIST_L2, 5)   # px outside the lumen

s3, C3, _, _ = load_cl(); L = s3[-1]
SG = np.arange(0, L + 1e-9, 0.25)
CG = np.stack([np.interp(SG, s3, C3[:, c]) for c in range(3)], 1)


def pin(rv, t, f, cx, cy):
    Xc = CG @ Rot.from_rotvec(rv).as_matrix().T + t
    return Xc[:, :2] / Xc[:, 2:3] * f + np.array([cx, cy])


cams = {}
p11 = np.load(DATA / "c8_11.npy")
cams["pinhole f=1298"] = pin(p11[:3], p11[3:6], np.exp(p11[6]), p11[9], p11[10])
z10 = np.load(DATA / "c10_ba.npz")
cams["pinhole f=516"] = pin(z10["rv"], z10["t"], float(z10["f"]),
                            float(z10["cx"]), float(z10["cy"]))
pa = np.load(DATA / "l2_affine.npy")
Xa = CG @ Rot.from_rotvec(pa[:3]).as_matrix().T
cams["affine (no depth)"] = Xa[:, :2] * np.exp(pa[3]) + pa[4:6]

# the traversed interval matters: only judge the stretch the robot actually used
uvz = np.load(DATA / "track2d.npz")["uv"]
ok = ~np.isnan(uvz[:, 0])
print("[T5] fraction of the PROJECTED CENTERLINE lying inside an orange lumen")
print(f"{'camera':<22}{'on vessel':>11}{'off by >10px':>14}{'worst off (px)':>16}")
res = {}
for nm, P in cams.items():
    xi = np.clip(np.rint(P[:, 0]).astype(int), 0, W - 1)
    yi = np.clip(np.rint(P[:, 1]).astype(int), 0, H - 1)
    d = dist_out[yi, xi]
    on = d == 0
    res[nm] = (on, d)
    print(f"{nm:<22}{100*on.mean():10.1f}%{100*np.mean(d>10):13.1f}%{d.max():16.1f}")

print(f"\n[T5] for reference, the DETECTED robot track: "
      f"{100*np.mean(dist_out[np.clip(np.rint(uvz[ok,1]).astype(int),0,H-1), np.clip(np.rint(uvz[ok,0]).astype(int),0,W-1)] == 0):.1f}% on vessel")
print(f"[T5] a uniformly random image point: {100*np.mean(lum>0):.1f}%")

best = max(res, key=lambda k: res[k][0].mean())
print(f"\n[T5] best on this axis: {best} ({100*res[best][0].mean():.1f}%)")
print(f"[T5] NOTE this is not a camera-selection verdict on its own: the lumen mask")
print(f"     is a 2D projection of a 3D tree, so a curve can be 'on a vessel' while")
print(f"     lying in the WRONG vessel. It is a necessary condition, not sufficient -")
print(f"     but a curve that is off-vessel is definitely wrong there.")
np.savez(DATA / "t5_anatomical.npz", **{f"on|{k}": res[k][0] for k in res},
         **{f"d|{k}": res[k][1] for k in res})
