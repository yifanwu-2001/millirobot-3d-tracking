"""Stage C12 - 10-minute diagnostic: is the raw-track centerline-only
residual (p90 86 px, K2's baseline) structured by ARC LENGTH (a fixed part of
the vessel/camera model is wrong) or by TIME (an episodic, dynamic cause -
fast motion, detection dropout, wobble bursts)?

This matters because the two point in opposite directions: arc-length
structure says "go fix the geometry/camera at that location"; time structure
says "look at what the robot/detector was doing at that moment" (independent
of where on the vessel it was).
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
from config import DATA, OUT, FPS
from e_hmm import MultiViewArcHMM
from k_offaxis import SG, CG, cam_terms

p = np.load(DATA / "c8_11.npy")
rv, t, f, cx, cy = p[:3], p[3:6], np.exp(p[6]), p[9], p[10]
R = Rot.from_rotvec(rv).as_matrix()
A, B1, B2 = cam_terms(R, t)
c = np.array([cx, cy])

uv_raw = np.load(DATA / "track2d.npz")["uv"]
zc = np.load(DATA / "track2d_clean.npz")
uv_cln = zc["uv"]; k_turn = int(zc["k_turn"])
K = len(uv_cln)
UV = np.where(np.isnan(uv_raw), uv_cln, uv_raw)
obs = UV[:, None, :]

hmm = MultiViewArcHMM(SG, sigma_px=6.0, v_max=30., n_v=41, accel_sigma=20., p_lost=0.05)


def fn(k):
    return [A[:, :2] / A[:, 2:3] * f + c]


s_hat, s_std, lz = hmm.forward(obs, fn, 1 / FPS)
idx = np.clip(np.rint(s_hat / (SG[1] - SG[0])).astype(int), 0, len(SG) - 1)
Q = fn(0)[0][idx]
rp = np.linalg.norm(Q - UV, axis=1)
print(f"[C12] baseline reproduced: median {np.median(rp):.1f} px, p90 {np.percentile(rp,90):.1f} px "
      f"(K2 reference: 7.7 / 86.0)")

# --- by arc length ---
nbin = 20
bins_s = np.linspace(SG.min(), SG.max(), nbin + 1)
which_s = np.clip(np.digitize(s_hat, bins_s) - 1, 0, nbin - 1)
med_by_s = np.array([np.median(rp[which_s == i]) if (which_s == i).any() else np.nan for i in range(nbin)])
n_by_s = np.array([(which_s == i).sum() for i in range(nbin)])
worst_s = np.nanargmax(med_by_s)
print(f"\n[C12] BY ARC LENGTH ({nbin} bins across s=0..{SG.max():.0f} mm):")
print(f"      median-of-bin-medians {np.nanmedian(med_by_s):.1f} px, "
      f"spread (bin p90 - bin p10 of the medians) {np.nanpercentile(med_by_s,90)-np.nanpercentile(med_by_s,10):.1f} px")
print(f"      worst bin: s={bins_s[worst_s]:.0f}-{bins_s[worst_s+1]:.0f} mm, "
      f"median {med_by_s[worst_s]:.1f} px, n={n_by_s[worst_s]}")

# --- by time ---
nbin_t = 20
bins_t = np.linspace(0, K, nbin_t + 1).astype(int)
which_t = np.clip(np.digitize(np.arange(K), bins_t) - 1, 0, nbin_t - 1)
med_by_t = np.array([np.median(rp[which_t == i]) for i in range(nbin_t)])
worst_t = np.argmax(med_by_t)
print(f"\n[C12] BY TIME ({nbin_t} bins across {K} frames):")
print(f"      median-of-bin-medians {np.median(med_by_t):.1f} px, "
      f"spread (bin p90 - bin p10 of the medians) {np.percentile(med_by_t,90)-np.percentile(med_by_t,10):.1f} px")
print(f"      worst bin: frames {bins_t[worst_t]}-{bins_t[worst_t+1]}, median {med_by_t[worst_t]:.1f} px")

# --- correlate with instantaneous speed (a "dynamic" cause) ---
v_hat = np.gradient(s_hat) * FPS
corr_speed = np.corrcoef(np.abs(v_hat), np.log10(np.maximum(rp, .1)))[0, 1]
print(f"\n[C12] corr(|speed|, log rp) = {corr_speed:.2f}  "
      f"(time-clustering signature: high error during fast motion)")

verdict = ("TIME-CLUSTERED: error concentrates in specific time windows, not specific vessel locations"
           if (np.percentile(med_by_t, 90) - np.percentile(med_by_t, 10)) >
              1.5 * (np.nanpercentile(med_by_s, 90) - np.nanpercentile(med_by_s, 10))
           else "ARC-LENGTH-CLUSTERED: error concentrates at specific vessel locations regardless of when visited"
           if (np.nanpercentile(med_by_s, 90) - np.nanpercentile(med_by_s, 10)) >
              1.5 * (np.percentile(med_by_t, 90) - np.percentile(med_by_t, 10))
           else "MIXED / NEITHER DOMINATES")
print(f"\n[C12] VERDICT: {verdict}")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
ax[0].bar(0.5 * (bins_s[:-1] + bins_s[1:]), med_by_s, width=(bins_s[1] - bins_s[0]) * .9)
ax[0].axvline(SG[SG.searchsorted(0)], color='0.6', lw=.5)
ax[0].set_xlabel("arc length s (mm)"); ax[0].set_ylabel("median reproj (px)")
ax[0].set_title("residual by WHERE on the vessel")
ax[1].bar(0.5 * (bins_t[:-1] + bins_t[1:]), med_by_t, width=(bins_t[1] - bins_t[0]) * .9)
ax[1].axvline(k_turn, color='r', ls='--', lw=.8)
ax[1].set_xlabel("frame"); ax[1].set_title("residual by WHEN in the video")
ax[2].scatter(np.abs(v_hat), rp, s=3, alpha=.3)
ax[2].set_xlabel("|speed| (mm/s)"); ax[2].set_ylabel("reproj (px)"); ax[2].set_yscale("log")
ax[2].set_title(f"speed vs error, corr(log)={corr_speed:.2f}")
plt.tight_layout(); plt.savefig(OUT / "figs" / "c12_residual_diagnosis.png", dpi=110)
print("\n[C12] saved out/figs/c12_residual_diagnosis.png")
