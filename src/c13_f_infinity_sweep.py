"""Stage C13 - extend C11's two-model f-sensitivity test with Stage I2's
offline smoothing (not just the causal filter), plus a third, extreme point:
f -> infinity, i.e. orthographic / weak-perspective projection, using the same
rotation and image scale as model A. If 3D disagreement keeps growing from
f=1298 to f=516 to f=infinity, the ambiguity is unbounded within the fitted
family; if it saturates, there's an effective floor on how wrong "not knowing
f" can make the 3D output.
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial import cKDTree
from config import DATA, OUT, FPS
from a_centerline import load as load_cl
from e_hmm import ArcHMM

s3, C3, _, _ = load_cl(); L = s3[-1]
ds = 0.25
sg = np.arange(0, L + 1e-9, ds)
Cg = np.stack([np.interp(sg, s3, C3[:, c]) for c in range(3)], 1)

zt = np.load(DATA / "track2d_trend.npz"); UV = zt["trend"]; k_turn = int(zt["k_turn"])
K = len(UV)


def perspective(rv, t, f, cx, cy):
    R = Rot.from_rotvec(rv).as_matrix()
    Xc = Cg @ R.T + t
    return Xc[:, :2] / Xc[:, 2:3] * f + np.array([cx, cy])


pA = np.load(DATA / "c8_11.npy")
rvA, tA, fA, cxA, cyA = pA[:3], pA[3:6], np.exp(pA[6]), pA[9], pA[10]
PgA = perspective(rvA, tA, fA, cxA, cyA)

zB = np.load(DATA / "c10_ba.npz")
fB, cxB, cyB = float(zB["f"]), float(zB["cx"]), float(zB["cy"])
rvB, tB = zB["rv"], zB["t"]
PgB = perspective(rvB, tB, fB, cxB, cyB)

# orthographic limit: same rotation as model A, image scale fixed at A's
# fitted px/mm, translation replaced by a 2D offset matched to the track
RA = Rot.from_rotvec(rvA).as_matrix()
XcA = Cg @ RA.T + tA
scaleA = fA / np.median(XcA[:, 2])
xy = XcA[:, :2] * scaleA
offset = UV.mean(0) - xy[np.searchsorted(sg, (sg[0] + sg[-1]) / 2)]
PgC = xy + offset
print(f"[C13] model C (orthographic): scale={scaleA:.2f} px/mm, same rotation as model A")

hmm_kw = dict(sigma_px=9.0, v_max=30., n_v=41, accel_sigma=20., p_lost=0.05)
models = {"A (f=1298)": PgA, "B (f=516)": PgB, "C (f=inf, orthographic)": PgC}
S = {}
for name, Pg in models.items():
    hmm = ArcHMM(sg, Pg, **hmm_kw)
    s_hat, s_std, v_hat = hmm.forward(UV, 1 / FPS)
    idx = np.clip(np.rint(s_hat / ds).astype(int), 0, len(sg) - 1)
    rp = np.linalg.norm(Pg[idx] - UV, axis=1)
    s_vit = hmm.viterbi(UV, 1 / FPS)
    idx_v = np.clip(np.rint(s_vit / ds).astype(int), 0, len(sg) - 1)
    rp_vit = np.linalg.norm(Pg[idx_v] - UV, axis=1)
    S[name] = dict(s_hat=s_hat, rp=rp, s_vit=s_vit, rp_vit=rp_vit)
    print(f"[C13] {name}: causal reproj med/p90 {np.median(rp):.1f}/{np.percentile(rp,90):.1f} px  "
          f"Viterbi reproj med/p90 {np.median(rp_vit):.1f}/{np.percentile(rp_vit,90):.1f} px")

X = {name: np.stack([np.interp(v["s_hat"], sg, Cg[:, c]) for c in range(3)], 1) for name, v in S.items()}
Xv = {name: np.stack([np.interp(v["s_vit"], sg, Cg[:, c]) for c in range(3)], 1) for name, v in S.items()}

names = list(models.keys())
print(f"\n[C13] pairwise 3D disagreement, CAUSAL filter (median / p90 mm):")
for i in range(3):
    for j in range(i + 1, 3):
        d = np.linalg.norm(X[names[i]] - X[names[j]], axis=1)
        print(f"      {names[i]} vs {names[j]}: {np.median(d):.2f} / {np.percentile(d,90):.2f} mm")

print(f"\n[C13] pairwise 3D disagreement, VITERBI (offline, median / p90 mm):")
for i in range(3):
    for j in range(i + 1, 3):
        d = np.linalg.norm(Xv[names[i]] - Xv[names[j]], axis=1)
        print(f"      {names[i]} vs {names[j]}: {np.median(d):.2f} / {np.percentile(d,90):.2f} mm")

np.savez(DATA / "c13_f_infinity_sweep.npz",
         **{f"s_hat_{i}": S[n]["s_hat"] for i, n in enumerate(names)},
         **{f"s_vit_{i}": S[n]["s_vit"] for i, n in enumerate(names)})
print("\n[C13] saved data/c13_f_infinity_sweep.npz")
