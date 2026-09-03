"""P0 - head-to-head of every candidate camera under ONE composite metric.

Three things this settles that the project had left open:

1. L2 found a depth-free affine camera fits the 2D track BETTER (3.81 px) than
   any pinhole (4.92 px) with fewer parameters - but never fed it through the
   tracker. Affine has no `f` at all, so C11's 4.2 mm focal-length ambiguity
   simply does not exist for it. Whether that is an advantage or a bias is an
   empirical question, answered here.

2. M3 found v_max=30 mm/s is too conservative and 60 is a free win, but it was
   never applied. Note it interacts with M2: relaxing the speed cap also relaxes
   the very constraint that makes Viterbi physically honest, so the violation
   rate has to be re-checked, not assumed.

3. M2 showed reprojection error and out-and-back repeatability each have a blind
   spot (a memoryless lookup beats the HMM on both by teleporting). Every row
   below therefore reports FIVE numbers, and no single one is allowed to decide:
     - reprojection median / p90            (2D fit)
     - speed-violation rate                 (physical plausibility)
     - out-and-back repeatability           (self-consistency)
     - HELD-OUT return-leg reprojection     (generalisation, per M1)
     - and across rows, the 3D spread       (camera-model systematic)
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from config import DATA, W, H, FPS
from a_centerline import load as load_cl
from e_hmm import ArcHMM

s3, C3, _, _ = load_cl(); L = s3[-1]
DS = 0.25
SG = np.arange(0, L + 1e-9, DS)
CG = np.stack([np.interp(SG, s3, C3[:, c]) for c in range(3)], 1)

fin = np.load(DATA / "i_final.npz")
UV, k_turn = fin["UV"], int(fin["k_turn"])
K = len(UV)
A_LEG, B_LEG = np.arange(k_turn + 1), np.arange(k_turn + 1, K)

# ------------------------------------------------------------------- cameras
def pinhole(p):
    rv, t, f, cx, cy = p[:3], p[3:6], np.exp(p[6]), p[9], p[10]
    Xc = CG @ Rot.from_rotvec(rv).as_matrix().T + t
    P = Xc[:, :2] / Xc[:, 2:3] * f + np.array([cx, cy])
    return P, f / np.median(Xc[:, 2])


def affine(p):
    rv, sc, cxy = p[:3], np.exp(p[3]), p[4:6]
    Xc = CG @ Rot.from_rotvec(rv).as_matrix().T
    return Xc[:, :2] * sc + cxy, sc


cams = {}
p11 = np.load(DATA / "c8_11.npy")
cams["pinhole f=1298 (c8_11)"] = pinhole(p11)
cams["affine, no depth term (L2)"] = affine(np.load(DATA / "l2_affine.npy"))
try:
    z10 = np.load(DATA / "c10_ba.npz")
    key = [k for k in z10.files if z10[k].shape == (11,) or z10[k].shape == (13,)]
    if key: cams["pinhole f=516 (c10_ba)"] = pinhole(z10[key[0]])
except Exception:
    pass

# ------------------------------------------------------------------- metrics
def evaluate(P, scale, v_max, decode):
    hmm = ArcHMM(SG, P, sigma_px=9.0, v_max=v_max, n_v=41, accel_sigma=20., p_lost=0.05)
    if decode == "causal":
        s_hat = hmm.forward(UV, 1 / FPS)[0]
    else:
        s_hat = hmm.viterbi(UV, 1 / FPS)
    idx = np.clip(np.rint(s_hat / DS).astype(int), 0, len(SG) - 1)
    rp = np.linalg.norm(P[idx] - UV, axis=1)
    spd = np.abs(np.diff(s_hat)) * FPS
    viol = 100 * np.mean(spd > v_max + 1e-6)
    d_img, ia = cKDTree(UV[A_LEG]).query(UV[B_LEG])
    sel = d_img < 8.0
    ob = np.abs(s_hat[B_LEG][sel] - s_hat[A_LEG][ia[sel]])
    X = np.stack([np.interp(s_hat, SG, CG[:, c]) for c in range(3)], 1)
    return dict(rp_med=np.median(rp), rp_p90=np.percentile(rp, 90),
                fail=100 * np.mean(rp > 30), viol=viol, spd_max=spd.max(),
                ob=np.median(ob), held=np.median(rp[B_LEG]),
                held_fail=100 * np.mean(rp[B_LEG] > 30), X=X, s=s_hat, scale=scale)


rows = {}
print(f"{'camera / decode / v_max':<44}{'reproj':>8}{'p90':>7}{'fail%':>7}"
      f"{'v-viol%':>9}{'max mm/s':>10}{'out-back':>10}{'HELD-OUT':>10}{'held f%':>9}")
for cname, (P, scale) in cams.items():
    for decode in ("causal", "viterbi"):
        for v_max in (30., 60.):
            r = evaluate(P, scale, v_max, decode)
            rows[(cname, decode, v_max)] = r
            tag = f"{cname[:26]:<27}{decode:<8}{v_max:.0f}"
            print(f"{tag:<44}{r['rp_med']:8.1f}{r['rp_p90']:7.1f}{r['fail']:7.1f}"
                  f"{r['viol']:9.1f}{r['spd_max']:10.0f}{r['ob']:10.2f}"
                  f"{r['held']:10.1f}{r['held_fail']:9.1f}")

# ------------------------------------------- camera-model systematic (C11 style)
print(f"\n[P0] 3D disagreement between camera models (viterbi, v_max=60):")
keys = [k for k in rows if k[1] == "viterbi" and k[2] == 60.]
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        d = np.linalg.norm(rows[keys[i]]["X"] - rows[keys[j]]["X"], axis=1)
        print(f"     {keys[i][0][:24]:<26} vs {keys[j][0][:24]:<26}"
              f" median {np.median(d):5.2f} mm  p90 {np.percentile(d,90):6.2f} mm")

np.savez(DATA / "p0_headtohead.npz",
         **{f"{c}|{d}|{int(v)}": rows[(c, d, v)]["s"] for (c, d, v) in rows})
print(f"\n[P0] saved data/p0_headtohead.npz")
