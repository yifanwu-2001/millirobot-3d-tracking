"""Stage R3 - sequential predictive evidence for the FULL camera family.

Why this exists. R2a's elimination of pinhole f=1298 rested on its out-and-back
tail (4.9% of tight pairs disagreeing by >5 mm). That tail was produced by the
Viterbi backtrack bug V11 later found in `ArcHMM.viterbi` (backtracking with
the new velocity's shift instead of the previous velocity's). Re-run with the
fix, the tail is gone: 0.0% >5 mm for every camera. R2a no longer separates
the family, so the "two surviving models" premise under S1/V4/V5/V8/V11-V13
needs a replacement leg to stand on - or it falls.

V4 introduced the strongest bug-immune discriminator available - the
sequential predictive log-evidence  sum_k log p(u_k | u_1..k-1)  under each
camera, forward-only, same sigma and motion model - but scored only f=516 vs
affine, taking f=1298's rejection from R2a. This stage scores all three, split
by leg (outbound = calibration set, return = never entered any fit).

What the evidence is and is not. It IS bug-immune (forward pass only) and it
integrates over the whole posterior, so it rewards a camera whose arc-length
parameterisation lets the motion prior explain the observed sequence - the
speed-consistency signal R2 was going to look for. It is NOT free of the
camera's own projection: the likelihood is still Gaussian in the distance to
that camera's curve. Q2's off-curve column (affine 2.7 px < f=516 3.5 px) and
V4's evidence (f=516 > affine by 56 nats) already disagree, which shows the
evidence is carrying information beyond nearest-curve distance - but "less
circular" is the honest description, not "non-circular".
"""
import numpy as np
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
K, dt = len(UV), 1 / FPS


def pin(rv, t, f, cx, cy):
    Xc = CG @ Rot.from_rotvec(rv).as_matrix().T + t
    return Xc[:, :2] / Xc[:, 2:3] * f + np.array([cx, cy])


cams = {}
p11 = np.load(DATA / "c8_11.npy")
cams["pinhole f=1298 (c8_11)"] = pin(p11[:3], p11[3:6], np.exp(p11[6]), p11[9], p11[10])
z10 = np.load(DATA / "c10_ba.npz")
cams["pinhole f=516 (c10_ba)"] = pin(z10["rv"], z10["t"], float(z10["f"]),
                                     float(z10["cx"]), float(z10["cy"]))
pa = np.load(DATA / "l2_affine.npy")
cams["affine, no depth (l2)"] = (CG @ Rot.from_rotvec(pa[:3]).as_matrix().T)[:, :2] * np.exp(pa[3]) + pa[4:6]


def evidence(P):
    """Forward-only. Returns (logZ_all, logZ_outbound, logZ_return, per-frame incr)."""
    hmm = ArcHMM(SG, P, sigma_px=9.0, v_max=V_MAX, n_v=41, accel_sigma=20., p_lost=0.05)
    logp = np.full((hmm.S, hmm.V), -np.log(hmm.S * hmm.V))
    inc = np.zeros(K)
    for k in range(K):
        if k > 0:
            logp = hmm._advect(logp, dt)
        logp = logp + hmm.loglik(UV[k])[:, None]
        m = logp.max()
        inc[k] = m + np.log(np.exp(logp - m).sum())
        w = np.exp(logp - m); w /= w.sum()
        logp = np.log(np.maximum(w, 1e-300))
    return inc.sum(), inc[:k_turn + 1].sum(), inc[k_turn + 1:].sum(), inc


print(f"[R3] sequential predictive log-evidence, forward-only (bug-immune), "
      f"sigma=9, v_max={V_MAX:.0f}\n")
print(f"{'camera':<26}{'logZ all':>11}{'outbound (fit)':>16}{'RETURN (held out)':>19}")
res = {}
for nm, P in cams.items():
    lz, lzo, lzr, inc = evidence(P)
    res[nm] = dict(lz=lz, lzo=lzo, lzr=lzr, inc=inc)
    print(f"{nm:<26}{lz:11.1f}{lzo:16.1f}{lzr:19.1f}")

names = list(res)
best_ret = max(names, key=lambda n: res[n]["lzr"])
print(f"\n[R3] preference on the HELD-OUT return leg, relative to the best ({best_ret}):")
for n in names:
    d = res[best_ret]["lzr"] - res[n]["lzr"]
    print(f"     {n:<26} {-d:+8.1f} nats" + ("   <- best" if n == best_ret else ""))
print(f"\n     (rule of thumb: >5 nats is strong, >10 decisive - on the same sigma and prior,")
print(f"      a nats gap is a log likelihood ratio for the observed sequence)")

# where does the return-leg preference come from - a few frames or everywhere?
print(f"\n[R3] is the return-leg gap concentrated or diffuse? per-frame increment difference,")
print(f"     best camera minus each other, return leg only:")
ret = np.arange(k_turn + 1, K)
for n in names:
    if n == best_ret: continue
    d = res[best_ret]["inc"][ret] - res[n]["inc"][ret]
    top = np.sort(d)[::-1]
    print(f"     vs {n:<24} total {d.sum():+7.1f}   median/frame {np.median(d):+.3f}   "
          f"top-10 frames carry {100*top[:10].sum()/max(d.sum(),1e-9):.0f}%   frames favouring other: {100*np.mean(d<0):.0f}%")

np.savez(DATA / "r3_evidence_family.npz",
         **{f"lz|{n}": res[n]["lz"] for n in names},
         **{f"lzo|{n}": res[n]["lzo"] for n in names},
         **{f"lzr|{n}": res[n]["lzr"] for n in names},
         **{f"inc|{n}": res[n]["inc"] for n in names})
print(f"\n[R3] saved data/r3_evidence_family.npz")
