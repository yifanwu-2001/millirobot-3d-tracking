"""Q2 - the complete camera family, and the 3D spread that is the number to quote.

P0's table was missing C10's f=516 model because `c10_ba.npz` stores rv/t/f/cx/cy
as separate arrays rather than one parameter vector. With it included, all four
mutually-incompatible cameras that fit this 2D track are scored side by side, and
the pairwise 3D disagreement across the family is the honest systematic error bar
on absolute position - the single number to put in front of anyone deciding
whether a calibration shot is worth taking.

The family is not equiprobable: R3's bug-immune held-out evidence rejects
f=1298 by 266 nats and it is last on every column here, so the spread
excluding f=1298 is the one to quote. (An earlier version excluded f=516 on
L1's 9.8-sigma claim, which was retracted in Stage R1.)
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from config import DATA, FPS
from a_centerline import load as load_cl
from e_hmm import ArcHMM

V_MAX = 60.0
s3, C3, _, _ = load_cl(); L = s3[-1]
DS = 0.25
SG = np.arange(0, L + 1e-9, DS)
CG = np.stack([np.interp(SG, s3, C3[:, c]) for c in range(3)], 1)
fin = np.load(DATA / "i_final.npz"); UV, k_turn = fin["UV"], int(fin["k_turn"])
K = len(UV)
A_LEG, B_LEG = np.arange(k_turn + 1), np.arange(k_turn + 1, K)


def pin(rv, t, f, cx, cy):
    Xc = CG @ Rot.from_rotvec(rv).as_matrix().T + t
    return Xc[:, :2] / Xc[:, 2:3] * f + np.array([cx, cy]), f / np.median(Xc[:, 2])


cams = {}
p11 = np.load(DATA / "c8_11.npy")
cams["pinhole f=1298 (c8_11)"] = pin(p11[:3], p11[3:6], np.exp(p11[6]), p11[9], p11[10])
z10 = np.load(DATA / "c10_ba.npz")
cams["pinhole f=516 (c10_ba)"] = pin(z10["rv"], z10["t"], float(z10["f"]),
                                     float(z10["cx"]), float(z10["cy"]))
pa = np.load(DATA / "l2_affine.npy")
Xa = CG @ Rot.from_rotvec(pa[:3]).as_matrix().T
cams["affine, no depth (l2)"] = (Xa[:, :2] * np.exp(pa[3]) + pa[4:6], np.exp(pa[3]))

print(f"[Q2] {len(cams)} camera models, all fitting the same 2D track, Viterbi v_max={V_MAX:.0f}\n")
print(f"{'camera':<26}{'px/mm':>7}{'off-curve':>11}{'reproj':>8}{'p90':>7}"
      f"{'fail%':>7}{'v-viol%':>9}{'held-out':>10}{'held f%':>9}")
res = {}
for nm, (P, sc) in cams.items():
    hmm = ArcHMM(SG, P, sigma_px=9.0, v_max=V_MAX, n_v=41, accel_sigma=20., p_lost=0.05)
    s_v = hmm.viterbi(UV, 1 / FPS)
    idx = np.clip(np.rint(s_v / DS).astype(int), 0, len(SG) - 1)
    rp = np.linalg.norm(P[idx] - UV, axis=1)
    d_min, _ = cKDTree(P).query(UV)
    spd = np.abs(np.diff(s_v)) * FPS
    X = np.stack([np.interp(s_v, SG, CG[:, c]) for c in range(3)], 1)
    res[nm] = dict(s=s_v, X=X, rp=rp, d_min=d_min)
    print(f"{nm:<26}{sc:7.2f}{np.median(d_min):11.1f}{np.median(rp):8.1f}"
          f"{np.percentile(rp,90):7.1f}{100*np.mean(rp>30):7.1f}"
          f"{100*np.mean(spd>V_MAX+1e-6):9.1f}{np.median(rp[B_LEG]):10.1f}"
          f"{100*np.mean(rp[B_LEG]>30):9.1f}")

print(f"\n[Q2] pairwise 3D trajectory disagreement (mm)")
print(f"{'pair':<52}{'median':>9}{'p90':>9}{'max':>9}")
ks = list(res); pair_med = {}
for i in range(len(ks)):
    for j in range(i + 1, len(ks)):
        d = np.linalg.norm(res[ks[i]]["X"] - res[ks[j]]["X"], axis=1)
        pair_med[(ks[i], ks[j])] = np.median(d)
        print(f"{ks[i][:24]+' vs '+ks[j][:24]:<52}{np.median(d):9.2f}"
              f"{np.percentile(d,90):9.2f}{d.max():9.2f}")

# per-frame spread across the whole family
Xs = np.stack([res[k]["X"] for k in ks])
ctr = Xs.mean(0)
spread = np.linalg.norm(Xs - ctr, axis=2).max(0)
print(f"\n[Q2] per-frame spread across ALL {len(ks)} models (max deviation from their mean):")
print(f"     median {np.median(spread):.2f} mm   p90 {np.percentile(spread,90):.2f} mm"
      f"   max {spread.max():.2f} mm")

# NOTE: an earlier version of this block excluded f=516 on L1's "9.8 sigma"
# claim. L1's verdict was an artefact of scoring every camera against c8_11's
# depth profile (retracted, Stage R1). The exclusion supported by the
# bug-immune evidence (R3, f=1298 worse by 266 nats held out) and by every Q2
# column is f=1298. The old key `spread_nof516` is still written for any
# downstream reader, but labelled as the superseded exclusion.
keep = [k for k in ks if "1298" not in k]
Xs2 = np.stack([res[k]["X"] for k in keep]); c2 = Xs2.mean(0)
sp2 = np.linalg.norm(Xs2 - c2, axis=2).max(0)
old = [k for k in ks if "516" not in k]
Xo = np.stack([res[k]["X"] for k in old]); spo = np.linalg.norm(Xo - Xo.mean(0), axis=2).max(0)
print(f"\n[Q2] excluding f=1298 (rejected by R3's held-out evidence, 266 nats, and last on every column above):")
print(f"     median {np.median(sp2):.2f} mm   p90 {np.percentile(sp2,90):.2f} mm"
      f"   max {sp2.max():.2f} mm")
print(f"     (superseded exclusion of f=516 on L1's retracted claim, for the record: "
      f"median {np.median(spo):.2f} / p90 {np.percentile(spo,90):.2f} / max {spo.max():.2f} mm)")
print(f"\n[Q2] THE NUMBER TO QUOTE: absolute 3D position from this video is uncertain")
print(f"     by about +-{np.median(sp2):.1f} mm (median) / {np.percentile(sp2,90):.1f} mm (p90)")
print(f"     purely because the camera was never calibrated. Tracking failure rate is")
print(f"     now {min(100*np.mean(res[k]['rp']>30) for k in keep):.1f}% - the estimator is not the limit.")
np.savez(DATA / "q2_camera_family.npz", spread=spread, spread_excl_1298=sp2, spread_nof516=spo,
         **{f"X|{k}": res[k]["X"] for k in res})
print(f"\n[Q2] saved data/q2_camera_family.npz")
