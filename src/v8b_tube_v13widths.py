"""V8b - V8's tube-aware decode with the width provenance V13 exposed.

V8 bounds the transverse offset by R_tube(s) taken from L1's width profile. That
profile was (a) sampled along the OLD c8_11 projection - the camera Q2/R3 reject -
and (b) NaN on 142/761 samples, filled with the global median and clipped to
3-6 mm. V13 re-measured along the current f=516 projection requiring both edges
and found NO valid two-sided width anywhere in the s=109-119 mm junction - the one
stretch where V8's headline gain (failure 3.8% -> 0.6%, junction max 45.8 -> 33.4
px) is claimed. V8's "5.3 mm at the junction" is an imputed number.

This stage asks whether the gain survives when the junction radius is not
invented. Same decoder, same camera, four radius profiles:

    A  V8 as published: L1 widths, median-imputed, clipped 3-6 mm     (reference)
    B  re-measured on the f=516 projection, both edges required; gaps filled by
       linear interpolation from the nearest measured neighbours, clipped 3-6 mm
    C  as B, but gaps get the conservative floor, 3.0 mm
    D  as B, but gaps get NO transverse freedom (axis-only there)

If A and B/C agree, V8's gain is real and the imputation was harmless. If the
junction gain only appears under A, it was manufactured by the imputed radius.
"""
import numpy as np, cv2
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from config import DATA, OUT, W, H, FPS, imread_u
from a_centerline import load as load_cl
from e_hmm import ArcHMM

V_MAX, SIGMA_PX = 60.0, 9.0
SIGMA_D0, SIGMA_DD = 1.2, 1.5
s3c, C3, _, _ = load_cl(); L = s3c[-1]
DS = 0.25
SG = np.arange(0, L + 1e-9, DS)
CG = np.stack([np.interp(SG, s3c, C3[:, c]) for c in range(3)], 1)
fin = np.load(DATA / "i_final.npz"); UV, k_turn = fin["UV"], int(fin["k_turn"])
K = len(UV); dt = 1 / FPS
OUT_LEG = np.zeros(K, bool); OUT_LEG[:k_turn + 1] = True; RET = ~OUT_LEG

z10 = np.load(DATA / "c10_ba.npz")
R516 = Rot.from_rotvec(z10["rv"]).as_matrix(); t516 = np.asarray(z10["t"]).ravel(); f516 = float(z10["f"])
c516 = np.array([float(z10["cx"]), float(z10["cy"])])
Xc = CG @ R516.T + t516; Zc = Xc[:, 2]
P_ax = Xc[:, :2] / Zc[:, None] * f516 + c516
sc = f516 / Zc
T2 = np.gradient(P_ax, SG, axis=0); T2 /= np.clip(np.linalg.norm(T2, axis=1, keepdims=True), 1e-9, None)
N2 = np.stack([-T2[:, 1], T2[:, 0]], 1)
JUNC = (SG > 109) & (SG < 119)


def proj516(X):
    Xq = X @ R516.T + t516
    return Xq[:, :2] / Xq[:, 2:3] * f516 + c516


# ------------------------------------------ width re-measured on THIS projection
bg = imread_u(OUT / "background_median.png")
hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV).astype(float)
hue = np.minimum(hsv[:, :, 0], 180 - hsv[:, :, 0])
orange = np.clip(1 - hue / 25, 0, 1) * np.clip(hsv[:, :, 1] / 60, 0, 1) * np.clip(hsv[:, :, 2] / 60, 0, 1)
orange = cv2.GaussianBlur(orange, (3, 3), 0)
offs = np.arange(-100, 100.01, .25)
w_new = np.full(len(SG), np.nan)
for i in range(len(SG)):
    p = P_ax[i] + offs[:, None] * N2[i]
    prof = map_coordinates(orange, [p[:, 1], p[:, 0]], order=1, mode='constant')
    c = len(offs) // 2
    if prof[c] < .5: continue
    a = b = c
    while a > 0 and prof[a - 1] >= .5: a -= 1
    while b < len(offs) - 1 and prof[b + 1] >= .5: b += 1
    if a == 0 or b == len(offs) - 1: continue          # edge not found inside the window
    lo = offs[a - 1] + .25 * (.5 - prof[a - 1]) / (prof[a] - prof[a - 1])
    hi = offs[b] + .25 * (.5 - prof[b]) / (prof[b + 1] - prof[b])
    if hi - lo < 5 or hi - lo > 100: continue
    w_new[i] = hi - lo
meas = np.isfinite(w_new)
print(f"[V8b] width re-measured on the f=516 projection, both edges required: "
      f"{meas.sum()}/{len(SG)} samples; junction s=109-119: {int((meas & JUNC).sum())} valid")

w_old = np.load(DATA / "l1_size_cue.npz")["width_px"]
old_meas = np.isfinite(w_old)
print(f"      L1 (old camera) widths: {old_meas.sum()}/{len(SG)} valid; junction: {int((old_meas & JUNC).sum())} valid")


def interp_gaps(w):
    out = w.copy(); ok = np.isfinite(w)
    out[~ok] = np.interp(SG[~ok], SG[ok], w[ok])
    return out


R_A = np.clip(np.where(old_meas, w_old, np.nanmedian(w_old)) / 2 / sc, 3.0, 6.0)
R_B = np.clip(interp_gaps(w_new) / 2 / sc, 3.0, 6.0)
R_C = np.where(meas, np.clip(w_new / 2 / sc, 3.0, 6.0), 3.0)
R_D = np.where(meas, np.clip(w_new / 2 / sc, 3.0, 6.0), 0.05)
VARIANTS = {"A  V8 as published (L1, imputed)": R_A,
            "B  f=516 widths, gaps interpolated": R_B,
            "C  f=516 widths, gaps floored 3 mm": R_C,
            "D  f=516 widths, gaps axis-only": R_D}
print(f"\n{'radius profile':<38}{'median':>8}{'p10':>7}{'p90':>7}{'junction med':>14}{'junction measured?':>20}")
for nm, R in VARIANTS.items():
    jm = "yes" if (meas & JUNC).any() and "L1" not in nm else ("L1: " + ("yes" if (old_meas & JUNC).any() else "NO") if "L1" in nm else "NO")
    print(f"{nm:<38}{np.median(R):8.2f}{np.percentile(R,10):7.2f}{np.percentile(R,90):7.2f}{np.median(R[JUNC]):14.2f}{jm:>20}")

FR = np.array([-1.0, -2 / 3, -1 / 3, 0.0, 1 / 3, 2 / 3, 1.0])


def run_variant(R_tube):
    D_mm = FR[None, :] * R_tube[:, None]; D_px = D_mm * sc[:, None]
    W_D = np.exp(-0.5 * (D_mm / SIGMA_D0) ** 2); W_D /= W_D.sum(1, keepdims=True)

    class TubeArcHMM(ArcHMM):
        def loglik(self, uv):
            if uv is None or not np.isfinite(uv).all(): return np.zeros(self.S)
            pos = P_ax[:, None, :] + D_px[:, :, None] * N2[:, None, :]
            d2 = ((pos - uv[None, None, :]) ** 2).sum(-1)
            g = np.exp(-0.5 * d2 / self.sigma ** 2) / (2 * np.pi * self.sigma ** 2)
            return np.log((1 - self.p_lost) * (W_D * g).sum(1) + self.p_lost / (960.0 * 720.0))

    hmm = TubeArcHMM(s_grid=SG, proj_xy=P_ax, sigma_px=SIGMA_PX, v_max=V_MAX, n_v=41, accel_sigma=20., p_lost=0.05)
    s_tb = hmm.viterbi(UV, dt)
    ix = np.clip(np.rint(s_tb / DS).astype(int), 0, hmm.S - 1)
    # exact 1-D DP for d given the s-path
    D = len(FR); em = np.zeros((K, D))
    for k in range(K):
        if not np.isfinite(UV[k, 0]): continue
        pos = P_ax[ix[k]][None, :] + D_px[ix[k]][:, None] * N2[ix[k]][None, :]
        em[k] = -0.5 * ((pos - UV[k][None, :]) ** 2).sum(1) / SIGMA_PX ** 2 + np.log(np.maximum(W_D[ix[k]], 1e-300))
    diffs = D_mm[ix][1:][:, None, :] - D_mm[ix][:-1][:, :, None]
    Td = -0.5 * diffs ** 2 / SIGMA_DD ** 2
    delta = em[0].copy(); back = np.zeros((K, D), int)
    for k in range(1, K):
        cand = delta[:, None] + Td[k - 1]
        back[k] = np.argmax(cand, axis=0); delta = cand[back[k], np.arange(D)] + em[k]
    di = np.zeros(K, int); di[-1] = int(np.argmax(delta))
    for k in range(K - 1, 0, -1): di[k - 1] = back[k, di[k]]
    d_mm, d_px = D_mm[ix, di], D_px[ix, di]
    X = CG[ix].copy()
    for k in range(K):
        if abs(d_px[k]) < 1e-6: continue
        i = ix[k]; z0 = Zc[i]; x0, y0 = Xc[i, 0], Xc[i, 1]
        J = f516 * np.array([[1 / z0, 0, -x0 / z0 ** 2], [0, 1 / z0, -y0 / z0 ** 2]]) @ R516.T
        dl = np.linalg.pinv(J) @ (d_px[k] * N2[i]); n = np.linalg.norm(dl)
        if n > R_tube[i]: dl *= R_tube[i] / n
        X[k] = CG[i] + dl
    rp = np.linalg.norm(proj516(X) - UV, axis=1)
    A = np.arange(k_turn + 1); B = np.arange(k_turn + 1, K)
    d_img, ia = cKDTree(UV[A]).query(UV[B]); sel = d_img < 8.0
    ob = np.median(np.abs(s_tb[B][sel] - s_tb[A][ia[sel]]))
    ep = np.arange(885, 920)
    return dict(rp=rp, s=s_tb, d=d_mm, X=X, ob=ob,
                med=np.median(rp), p90=np.percentile(rp, 90), fail=100 * np.mean(rp > 30),
                out=np.median(rp[OUT_LEG]), ret=np.median(rp[RET]), retf=100 * np.mean(rp[RET] > 30),
                ep_med=np.median(rp[ep]), ep_max=rp[ep].max(), d_ep=d_mm[ep][np.argmax(np.abs(d_mm[ep]))],
                at_bound=100 * np.mean(np.abs(d_mm) >= 0.99 * R_tube[ix]))


# axis-only reference (same camera)
hmm0 = ArcHMM(SG, P_ax, sigma_px=SIGMA_PX, v_max=V_MAX, n_v=41, accel_sigma=20., p_lost=0.05)
s0 = hmm0.viterbi(UV, dt); ix0 = np.clip(np.rint(s0 / DS).astype(int), 0, hmm0.S - 1)
rp0 = np.linalg.norm(P_ax[ix0] - UV, axis=1); ep = np.arange(885, 920)
print(f"\n{'model':<38}{'med':>6}{'p90':>7}{'>30px':>7}{'out':>6}{'RET':>6}{'RET>30':>8}"
      f"{'ep885-920 med/max':>20}{'d@ep':>7}{'|d|=R':>7}{'out-back':>9}")
print(f"{'axis-only (f=516)':<38}{np.median(rp0):6.2f}{np.percentile(rp0,90):7.2f}{100*np.mean(rp0>30):6.1f}%"
      f"{np.median(rp0[OUT_LEG]):6.2f}{np.median(rp0[RET]):6.2f}{100*np.mean(rp0[RET]>30):7.1f}%"
      f"{np.median(rp0[ep]):11.1f}/{rp0[ep].max():5.1f}{'-':>7}{'-':>7}{'-':>9}")
res = {}
for nm, R in VARIANTS.items():
    r = run_variant(R); res[nm] = r
    print(f"{nm:<38}{r['med']:6.2f}{r['p90']:7.2f}{r['fail']:6.1f}%{r['out']:6.2f}{r['ret']:6.2f}{r['retf']:7.1f}%"
          f"{r['ep_med']:11.1f}/{r['ep_max']:5.1f}{r['d_ep']:+7.1f}{r['at_bound']:6.1f}%{r['ob']:9.2f}")

print(f"\n[V8b] reading: compare the junction column and the failure rate between A and B/C/D.")
print(f"      If B and C keep A's numbers, V8's gain did not depend on the imputed junction radius.")
print(f"      If the junction improvement only exists under A, it was manufactured by a radius")
print(f"      the image does not support - and D shows what the decoder does with honest geometry.")
np.savez(DATA / "v8b_tube_v13widths.npz", w_new=w_new, R_A=R_A, R_B=R_B, R_C=R_C, R_D=R_D,
         **{f"rp|{k[0]}": v["rp"] for k, v in res.items()}, **{f"d|{k[0]}": v["d"] for k, v in res.items()},
         **{f"X|{k[0]}": v["X"] for k, v in res.items()}, rp_axis=rp0)
print("[V8b] saved data/v8b_tube_v13widths.npz")
