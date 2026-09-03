"""Stage K5 - does tracking the principal-axis midpoint instead of the blob
centroid remove the roll-driven appearance wobble that K4 identified?

K4 showed blob AREA and ORIENTATION carry more 0.8-2.0 Hz band power than the
centroid does (12.6x / 10.1x vs 4.3x / 4.2x), consistent with the elongated
robot's rolling silhouette moving the centroid without the body actually
translating. The centroid is not the only point we could track: the midpoint
of the blob's minAreaRect (center of the rectangle spanning its long and short
axis) is less sensitive to *which side* bulges during a roll, because it is
defined by the extremal extent along the principal axis, not by the mass
distribution within it.

Robustness: if the robot bends, minAreaRect's angle/elongation can jump
frame-to-frame in a way that is not a roll artefact (a genuine shape change),
so a frame whose elongation ratio jumps more than RATIO_JUMP from its neighbour
falls back to the centroid and is flagged low-confidence.

Two checks:
  1. does the new track's band power ratio (centroid vs axis-midpoint) drop,
     i.e. is it less modulated at the roll frequency?
  2. fed into the SAME centerline-only HMM fit as K2's baseline, does
     reprojection error / tail rate / log Z improve?
"""
import cv2, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch, savgol_filter
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from config import VIDEO, DATA, OUT, FPS, W, H, imread_u
from d_detect import vessel_mask, DIFF_TH
from e_hmm import MultiViewArcHMM
from k_offaxis import SG, CG, cam_terms

RATIO_JUMP = 0.40   # elongation-ratio frame-to-frame jump that triggers fallback

bg = imread_u(OUT / "background_median.png")
vm = vessel_mask(bg)
bg_g = cv2.GaussianBlur(cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.int16)
kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

uv = np.load(DATA / "track2d.npz")["uv"]
cap = cv2.VideoCapture(str(VIDEO))
K = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

uv_axis = uv.copy()
ratio = np.full(K, np.nan)
low_conf = np.zeros(K, bool)
prev_ratio = None
for k in range(K):
    ok, fr = cap.read()
    if not ok or np.isnan(uv[k, 0]):
        prev_ratio = None
        continue
    g = cv2.GaussianBlur(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.int16)
    d = np.clip(bg_g - g, 0, 255).astype(np.uint8); d[~vm] = 0
    binm = cv2.morphologyEx((d > DIFF_TH).astype(np.uint8), cv2.MORPH_OPEN, kern)
    n, lab, st, ce = cv2.connectedComponentsWithStats(binm, 8)
    x, y = int(round(uv[k, 0])), int(round(uv[k, 1]))
    if not (0 <= x < lab.shape[1] and 0 <= y < lab.shape[0]) or lab[y, x] == 0:
        prev_ratio = None
        continue
    li = lab[y, x]
    ys, xs = np.where(lab == li)
    if len(xs) < 30:
        prev_ratio = None
        continue
    pts = np.stack([xs, ys], 1).astype(np.float32)
    (rcx, rcy), (rw, rh), ang = cv2.minAreaRect(pts)
    r = max(rw, rh) / max(min(rw, rh), 1e-6)
    ratio[k] = r
    jump = prev_ratio is not None and abs(r - prev_ratio) > RATIO_JUMP
    if jump:
        low_conf[k] = True   # keep uv_axis[k] = original centroid (already copied)
    else:
        uv_axis[k] = (rcx, rcy)
    prev_ratio = r
cap.release()

good = ~np.isnan(ratio)
print(f"[K5] axis-midpoint computed on {good.sum()}/{K} frames, "
      f"{low_conf.sum()} ({100*low_conf.sum()/good.sum():.1f}%) fell back to centroid "
      f"on an elongation jump > {RATIO_JUMP}")
moved = np.linalg.norm(uv_axis - uv, axis=1)
print(f"[K5] axis-midpoint vs centroid displacement: median {np.nanmedian(moved[good]):.2f} px, "
      f"p90 {np.nanpercentile(moved[good],90):.2f} px")

# ---------------------------------------------------------------- band power
k_idx = np.arange(K)
def detrend(v, gd):
    x = np.interp(k_idx, k_idx[gd], v[gd])
    return x - savgol_filter(x, 41, 2)

sig = {"centroid u": (uv[:, 0], good), "centroid v": (uv[:, 1], good),
       "axis-mid u": (uv_axis[:, 0], good), "axis-mid v": (uv_axis[:, 1], good)}
print(f"\n[K5] band power (0.8-2.0 Hz vs out-of-band), centroid vs axis-midpoint")
print(f"{'quantity':<16}{'band power ratio':>18}")
ratios = {}
for nm, (v, gd) in sig.items():
    d = detrend(v, gd)
    fr, P = welch(d, fs=FPS, nperseg=256)
    inband = (fr > 0.8) & (fr < 2.0)
    rr = P[inband].mean() / P[~inband & (fr > 0)].mean()
    ratios[nm] = rr
    print(f"{nm:<16}{rr:18.1f}")
print(f"\n[K5] READING: if axis-mid u/v ratios are lower than centroid u/v, the roll "
      f"modulation shrank; if similar, the axis-midpoint is picking up the same "
      f"appearance signal K4 found (rolling changes the rect extent too).")

# --------------------------------------------------------- does the FIT improve?
p = np.load(DATA / "c8_11.npy")
rv, t, f, sa, sb, cx, cy = p[:3], p[3:6], np.exp(p[6]), p[7], p[8], p[9], p[10]
R = Rot.from_rotvec(rv).as_matrix()
A, B1, B2 = cam_terms(R, t)
c = np.array([cx, cy])

zc = np.load(DATA / "track2d_clean.npz")
k_turn = int(zc["k_turn"])
Kn = len(zc["uv"])
UV_c = np.where(np.isnan(uv[:Kn]), zc["uv"], uv[:Kn])
UV_a = np.where(np.isnan(uv_axis[:Kn]), zc["uv"], uv_axis[:Kn])

hmm = MultiViewArcHMM(SG, sigma_px=6.0, v_max=30., n_v=41, accel_sigma=20., p_lost=0.05)


def fit_centerline_only(UV, tag):
    fn_cache = {}

    def fn(k):
        if k not in fn_cache:
            Xc = A
            fn_cache[k] = [Xc[:, :2] / Xc[:, 2:3] * f + c]
        return fn_cache[k]

    obs = UV[:, None, :]
    s_hat, s_std, lz = hmm.forward(obs, fn, 1 / FPS)
    idx = np.clip(np.rint(s_hat / (SG[1] - SG[0])).astype(int), 0, len(SG) - 1)
    Q = np.stack([fn(k)[0][idx[k]] for k in range(Kn)])
    rp = np.linalg.norm(Q - UV, axis=1)
    Aa, Bb = np.arange(k_turn + 1), np.arange(k_turn + 1, Kn)
    d_img, ia = cKDTree(UV[Aa]).query(UV[Bb]); sel = d_img < 8.0
    dsr = np.abs(s_hat[Bb][sel] - s_hat[Aa][ia[sel]])
    print(f"{tag:<26}{np.median(rp):9.1f}{np.percentile(rp,90):9.1f}"
          f"{100*np.mean(rp>30):9.1f}{np.median(dsr):11.2f}{lz:12.0f}")
    return rp, dsr, lz


print(f"\n[K5] centerline-only fit: centroid track (K2 baseline) vs axis-midpoint track")
print(f"{'track':<26}{'reproj':>9}{'p90':>9}{'>30px%':>9}{'out-back':>11}{'log Z':>12}")
fit_centerline_only(UV_c, "centroid (K2 baseline)")
fit_centerline_only(UV_a, "axis-midpoint")

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
sl = slice(200, 460)
ax[0].plot(k_idx[sl], detrend(uv[:, 0], good)[sl], label="centroid u", lw=1.1)
ax[0].plot(k_idx[sl], detrend(uv_axis[:, 0], good)[sl], label="axis-mid u", lw=1.1)
ax[0].set_xlabel("frame"); ax[0].legend(fontsize=8); ax[0].set_title("K5: centroid vs axis-midpoint, detrended u")
names = list(ratios.keys()); vals = [ratios[n] for n in names]
ax[1].bar(names, vals); ax[1].set_ylabel("band power ratio"); ax[1].tick_params(axis='x', rotation=30)
ax[1].set_title("0.8-2.0 Hz band power")
plt.tight_layout(); plt.savefig(OUT / "figs" / "k5_axis_track.png", dpi=110)
np.savez(DATA / "k5_axis_track.npz", uv_axis=uv_axis, ratio=ratio, low_conf=low_conf)
print("\n[K5] saved out/figs/k5_axis_track.png, data/k5_axis_track.npz")
