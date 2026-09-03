"""P0b - does the affine camera fix P1's "no good match anywhere" class?

P1 split the remaining failures into branch-selection errors (a better arc
length existed) and no-match errors (nothing on the curve fit that pixel), and
found the no-match class is location-bound and NOT explained by excursion,
centerline sampling, radial distortion, or small rigid deformation.

P0 then found the affine camera more than halves the held-out failure rate.
If that gain lands on the no-match class, the cause was camera-model
mis-specification after all - just not focal length specifically, since affine
has no focal length. If the no-match class survives, it is something else and
P1's "unidentified" verdict stands for it.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from config import DATA, FPS
from a_centerline import load as load_cl
from e_hmm import ArcHMM

s3, C3, _, _ = load_cl(); L = s3[-1]
DS = 0.25
SG = np.arange(0, L + 1e-9, DS)
CG = np.stack([np.interp(SG, s3, C3[:, c]) for c in range(3)], 1)
fin = np.load(DATA / "i_final.npz"); UV = fin["UV"]; K = len(UV)

def pinhole(p):
    Xc = CG @ Rot.from_rotvec(p[:3]).as_matrix().T + p[3:6]
    return Xc[:, :2] / Xc[:, 2:3] * np.exp(p[6]) + np.array([p[9], p[10]])

def affine(p):
    Xc = CG @ Rot.from_rotvec(p[:3]).as_matrix().T
    return Xc[:, :2] * np.exp(p[3]) + p[4:6]

cams = {"pinhole (c8_11)": pinhole(np.load(DATA / "c8_11.npy")),
        "affine  (L2)":    affine(np.load(DATA / "l2_affine.npy"))}

print(f"{'camera':<20}{'fails':>7}{'branch':>9}{'no-match':>10}"
      f"{'off-curve med px':>19}{'in ep. s=110-170':>18}")
out = {}
for name, P in cams.items():
    hmm = ArcHMM(SG, P, sigma_px=9.0, v_max=60., n_v=41, accel_sigma=20., p_lost=0.05)
    s_hat = hmm.viterbi(UV, 1 / FPS)
    idx = np.clip(np.rint(s_hat / DS).astype(int), 0, len(SG) - 1)
    rp = np.linalg.norm(P[idx] - UV, axis=1)
    d_min, jn = cKDTree(P).query(UV)          # distance to the WHOLE projected curve
    fail = rp > 30
    nomatch = fail & (d_min > 30)
    branch = fail & ~nomatch
    s_at = SG[jn]
    inzone = nomatch & (s_at > 100) & (s_at < 175)
    out[name] = dict(rp=rp, d_min=d_min, fail=fail, nomatch=nomatch, s_hat=s_hat)
    print(f"{name:<20}{fail.sum():7d}{branch.sum():9d}{nomatch.sum():10d}"
          f"{np.median(d_min):19.1f}{inzone.sum():18d}")

a, b = out["pinhole (c8_11)"], out["affine  (L2)"]
print(f"\n[P0b] the pinhole camera's {a['nomatch'].sum()} no-match frames, re-scored under affine:")
still = b["nomatch"][a["nomatch"]]
print(f"      still no-match under affine : {still.sum()}/{len(still)} ({100*still.mean():.0f}%)")
print(f"      their off-curve distance    : pinhole median {np.median(a['d_min'][a['nomatch']]):.1f} px"
      f"  ->  affine median {np.median(b['d_min'][a['nomatch']]):.1f} px")
print(f"\n[P0b] and the whole-track off-curve distance (how well the CURVE itself sits):")
for name in cams:
    d = out[name]["d_min"]
    print(f"      {name:<20} median {np.median(d):5.1f} px   p90 {np.percentile(d,90):6.1f} px"
          f"   >30px {100*np.mean(d>30):5.1f}%")
print(f"\n[P0b] READING: the off-curve distance is a property of the CAMERA alone (no")
print(f"      estimator involved). If it drops under affine, the projected geometry")
print(f"      genuinely sits closer to the observed vessel, and P1's localised residual")
print(f"      was model mis-specification - not focal length, since affine has none.")
np.savez(DATA / "p0b_failure_classes.npz",
         **{f"{k}|{f}": out[k][f] for k in out for f in ("rp", "d_min", "s_hat")})
