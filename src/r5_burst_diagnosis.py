"""Stage R5 - where does V4's f=516-over-affine evidence preference come from?

R3 found the 56.6-nat return-leg preference is not diffuse: 49.5 nats sit in
twelve frames, in two bursts (t ~ 27.7 s and ~ 28.7 s), and none of them is
in the junction episode. This stage looks at those frames - detection quality,
raw-vs-trend disagreement, distance to each camera's projected curve, the
nearest-point arc length under each camera - and saves native-resolution crops
with both curves drawn, so the cause can be read off the pixels.
"""
import numpy as np, cv2
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from config import DATA, OUT, VIDEO, imwrite_u
from a_centerline import load as load_cl

s3, C3, _, _ = load_cl(); L = s3[-1]; SG = np.arange(0, L + 1e-9, .25)
CG = np.stack([np.interp(SG, s3, C3[:, c]) for c in range(3)], 1)
raw = np.load(DATA / "track2d.npz"); uv_raw, conf, area = raw["uv"], raw["conf"], raw["area"]
tr = np.load(DATA / "track2d_trend.npz")["trend"]
z10 = np.load(DATA / "c10_ba.npz")
Xc = CG @ Rot.from_rotvec(z10["rv"]).as_matrix().T + np.asarray(z10["t"]).ravel()
P516 = Xc[:, :2] / Xc[:, 2:3] * float(z10["f"]) + np.array([float(z10["cx"]), float(z10["cy"])])
pa = np.load(DATA / "l2_affine.npy")
Paff = (CG @ Rot.from_rotvec(pa[:3]).as_matrix().T)[:, :2] * np.exp(pa[3]) + pa[4:6]
r3 = np.load(DATA / "r3_evidence_family.npz")
inc516 = r3["inc|pinhole f=516 (c10_ba)"]; incaff = r3["inc|affine, no depth (l2)"]
t516, taff = cKDTree(P516), cKDTree(Paff)

print("[R5] frames 824-869: detection quality, distances to each camera's curve, evidence increment gap")
print(f"{'frame':>5}{'t':>6}{'conf':>6}{'area':>6}{'|raw-tr|':>9}{'d516':>7}{'daff':>7}{'s516':>7}{'saff':>7}{'inc516-aff':>12}")
for k in range(824, 870):
    d5, i5 = t516.query(tr[k]); da, ia = taff.query(tr[k])
    flag = "  <--" if abs(inc516[k] - incaff[k]) > 2 else ""
    print(f"{k:5d}{k/30:6.2f}{conf[k]:6.2f}{area[k]:6.0f}{np.linalg.norm(uv_raw[k]-tr[k]):9.1f}"
          f"{d5:7.1f}{da:7.1f}{SG[i5]:7.1f}{SG[ia]:7.1f}{inc516[k]-incaff[k]:+12.2f}{flag}")

med_area = np.nanmedian(area[area > 0])
b1, b2 = np.arange(824, 844), np.arange(855, 870)
print(f"\n[R5] burst 1 (824-843): median blob area {np.median(area[b1]):.0f} px vs {med_area:.0f} px overall; "
      f"min confidence {conf[b1].min():.2f}; raw-trend gap max {np.max(np.linalg.norm(uv_raw[b1]-tr[b1],axis=1)):.0f} px "
      f"(two off-screen outliers at 836-837 bridged by d2_clean)")
print(f"[R5] burst 2 (855-869): median blob area {np.median(area[b2]):.0f} px, min confidence {conf[b2].min():.2f}; "
      f"nearest-point s jumps {SG[taff.query(tr[860])[1]]:.0f} -> {SG[taff.query(tr[861])[1]]:.0f} mm (affine) and "
      f"{SG[t516.query(tr[865])[1]]:.0f} -> {SG[t516.query(tr[866])[1]]:.0f} mm (f=516): a self-approach of the curve")
print(f"[R5] evidence gap in burst 1: {np.sum(inc516[b1]-incaff[b1]):+.1f} nats; burst 2: {np.sum(inc516[b2]-incaff[b2]):+.1f} nats; "
      f"whole return leg: {np.sum(inc516[671:]-incaff[671:]):+.1f} nats")

cap = cv2.VideoCapture(str(VIDEO)); tiles = []
def draw_curve(img, P, x0, y0, sc, col):
    pts = ((P - [x0, y0]) * sc).astype(int)
    for i in range(len(pts) - 1):
        if all(0 <= pts[j, 0] < img.shape[1] and 0 <= pts[j, 1] < img.shape[0] for j in (i, i + 1)):
            cv2.line(img, tuple(pts[i]), tuple(pts[i + 1]), col, 1)
for k in (826, 831, 836, 855, 860, 865):
    cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, fr = cap.read()
    x, y = int(tr[k, 0]), int(tr[k, 1]); r = 50; x0, y0 = max(0, x - r), max(0, y - r)
    crop = cv2.resize(fr[y0:y + r, x0:x + r].copy(), None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    draw_curve(crop, P516, x0, y0, 4, (255, 0, 255)); draw_curve(crop, Paff, x0, y0, 4, (255, 255, 0))
    cv2.circle(crop, (int((tr[k, 0] - x0) * 4), int((tr[k, 1] - y0) * 4)), 6, (0, 255, 255), 2)
    cv2.drawMarker(crop, (int((uv_raw[k, 0] - x0) * 4), int((uv_raw[k, 1] - y0) * 4)), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
    cv2.putText(crop, f"{k} t={k/30:.2f}s", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
    tiles.append(crop)
imwrite_u(OUT / "figs" / "r5_burst_crops.png", np.hstack(tiles))
print("[R5] saved out/figs/r5_burst_crops.png (magenta = f=516 curve, cyan = affine curve, yellow o = trend, red x = raw)")
