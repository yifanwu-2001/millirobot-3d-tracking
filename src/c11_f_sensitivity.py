"""Stage C11 - decisive test: C8/C10 found TWO camera models (f=1298 and
f=521) that fit the same 2D correspondences almost equally well (4.92 vs
4.62 px trimmed residual). Does that unidentifiability actually reach the
quantity anyone cares about - the recovered 3D trajectory - or is it a
nuisance parameter that cancels out?

Because the estimator's state IS the arc length s and the 3D output is the
camera-independent map X(s) = C(s) (Stage A's fixed physical centerline), this
reduces to one question: fed the SAME real pixel track, do the two camera
models' HMM fits recover the same s_hat(k)? If yes, the two "different"
cameras are the same estimator in disguise. If no, calibration degeneracy is
not just theoretical.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
from config import DATA, OUT, FPS
from a_centerline import load as load_cl
from e_hmm import ArcHMM

s3, C3, _, _ = load_cl(); L = s3[-1]
ds = 0.25
sg = np.arange(0, L + 1e-9, ds)
Cg = np.stack([np.interp(sg, s3, C3[:, c]) for c in range(3)], 1)


def build(rv, t, f, cx, cy):
    R = Rot.from_rotvec(rv).as_matrix()
    Xc = Cg @ R.T + t
    Pg = Xc[:, :2] / Xc[:, 2:3] * f + np.array([cx, cy])
    return Pg


pA = np.load(DATA / "c8_11.npy")
rvA, tA, fA, cxA, cyA = pA[:3], pA[3:6], np.exp(pA[6]), pA[9], pA[10]
PgA = build(rvA, tA, fA, cxA, cyA)

zB = np.load(DATA / "c10_ba.npz")
fB, cxB, cyB = float(zB["f"]), float(zB["cx"]), float(zB["cy"])
rvB, tB = zB["rv"], zB["t"]
PgB = build(rvB, tB, fB, cxB, cyB)

print(f"[C11] model A (c8_11, in production use): f={fA:.0f} px, principal ({cxA:.0f},{cyA:.0f})")
print(f"[C11] model B (c10 damped BA):             f={fB:.0f} px, principal ({cxB:.0f},{cyB:.0f})")

zt = np.load(DATA / "track2d_trend.npz"); UV = zt["trend"]; k_turn = int(zt["k_turn"])
K = len(UV)

hmmA = ArcHMM(sg, PgA, sigma_px=9.0, v_max=30., n_v=41, accel_sigma=20., p_lost=0.05)
hmmB = ArcHMM(sg, PgB, sigma_px=9.0, v_max=30., n_v=41, accel_sigma=20., p_lost=0.05)

s_hatA, s_stdA, _ = hmmA.forward(UV, 1 / FPS)
s_hatB, s_stdB, _ = hmmB.forward(UV, 1 / FPS)

XA = np.stack([np.interp(s_hatA, sg, Cg[:, c]) for c in range(3)], 1)
XB = np.stack([np.interp(s_hatB, sg, Cg[:, c]) for c in range(3)], 1)
d3 = np.linalg.norm(XA - XB, axis=1)
ds_arc = np.abs(s_hatA - s_hatB)

QA = PgA[np.clip(np.rint(s_hatA / ds).astype(int), 0, len(sg) - 1)]
QB = PgB[np.clip(np.rint(s_hatB / ds).astype(int), 0, len(sg) - 1)]
rpA = np.linalg.norm(QA - UV, axis=1)
rpB = np.linalg.norm(QB - UV, axis=1)

print(f"\n[C11] each model's own fit quality (should both be reasonable - neither is 'the bad one')")
print(f"      model A reproj: median {np.median(rpA):.1f} px, p90 {np.percentile(rpA,90):.1f} px")
print(f"      model B reproj: median {np.median(rpB):.1f} px, p90 {np.percentile(rpB,90):.1f} px")

print(f"\n[C11] SAME pixel track, two camera models -> arc-length disagreement:")
print(f"      |s_A - s_B|: median {np.median(ds_arc):.2f} mm, p90 {np.percentile(ds_arc,90):.2f} mm, "
      f"max {ds_arc.max():.2f} mm")
print(f"\n[C11] -> resulting 3D TRAJECTORY disagreement (camera-independent, X=C(s)):")
print(f"      |X_A - X_B|: median {np.median(d3):.2f} mm, p90 {np.percentile(d3,90):.2f} mm, "
      f"max {d3.max():.2f} mm")

near_turn = np.abs(np.arange(K) - k_turn) < 40
print(f"\n[C11] concentration check: near the turnaround / self-crossing region (n={near_turn.sum()}) "
      f"vs elsewhere (n={K-near_turn.sum()})")
print(f"      3D disagreement near crossing: median {np.median(d3[near_turn]):.2f} mm, "
      f"p90 {np.percentile(d3[near_turn],90):.2f} mm")
print(f"      3D disagreement elsewhere:     median {np.median(d3[~near_turn]):.2f} mm, "
      f"p90 {np.percentile(d3[~near_turn],90):.2f} mm")

verdict = ("NUISANCE: f is unidentifiable but the 3D trajectory barely moves"
           if np.median(d3) < 0.5 and np.percentile(d3, 90) < 2.0 else
           "LOCALIZED: disagreement concentrates at the ambiguous region"
           if np.median(d3[near_turn]) > 2 * max(np.median(d3[~near_turn]), 1e-6) else
           "SYSTEMATIC: the two camera models disagree on the 3D trajectory globally")
print(f"\n[C11] VERDICT: {verdict}")

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
k = np.arange(K)
ax[0].plot(k, s_hatA, lw=1.1, label=f"model A (f={fA:.0f})")
ax[0].plot(k, s_hatB, lw=1.1, label=f"model B (f={fB:.0f})", alpha=.8)
ax[0].axvline(k_turn, color='r', ls='--', lw=.8)
ax[0].set_xlabel("frame"); ax[0].set_ylabel("s (mm)"); ax[0].legend(fontsize=8)
ax[0].set_title("recovered arc length: two camera models, same pixels")
ax[1].semilogy(k, np.maximum(d3, 1e-3), lw=.8)
ax[1].axvline(k_turn, color='r', ls='--', lw=.8)
ax[1].set_xlabel("frame"); ax[1].set_ylabel("3D disagreement (mm)")
ax[1].set_title("|X_A(k) - X_B(k)|")
ax[2].hist(d3, bins=50)
ax[2].set_xlabel("3D disagreement (mm)"); ax[2].set_title(f"distribution (median {np.median(d3):.2f} mm)")
plt.tight_layout(); plt.savefig(OUT / "figs" / "c11_f_sensitivity.png", dpi=110)
np.savez(DATA / "c11_f_sensitivity.npz", s_hatA=s_hatA, s_hatB=s_hatB, d3=d3, ds_arc=ds_arc,
         rpA=rpA, rpB=rpB)
print("\n[C11] saved out/figs/c11_f_sensitivity.png, data/c11_f_sensitivity.npz")
