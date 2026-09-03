"""Stage I4 - is the causal filter's point estimate the right summary of the
posterior? `ArcHMM.forward()` returns the POSTERIOR MEAN of s at each step
(`s_hat = (ps * s).sum()`). At a self-crossing the posterior can be genuinely
bimodal - two separated arc-length regions both plausible. The mean of a
bimodal distribution can land in the valley BETWEEN the modes, a location
with near-zero actual probability and no physical meaning, while the MODE
(argmax) always picks a real candidate.

This does not touch e_hmm.py (every other stage's s_hat stays exactly as
reported); it duplicates the recursion locally and computes both estimators
side by side on the same real track, to see whether the mode is a better
real-time (still causal, still O(1) per frame) estimator than the mean.
"""
import numpy as np
from scipy.spatial import cKDTree
from config import DATA, OUT, FPS
from e_hmm import ArcHMM

z = np.load(DATA / "i_final.npz")
sg, Pg, UV, k_turn = z["sg"], z["Pg"], z["UV"], int(z["k_turn"])
rp_mean_ref, scale = z["rp"], float(z["scale"])
K, dt = len(UV), 1 / FPS

hmm = ArcHMM(sg, Pg, sigma_px=9.0, v_max=30., n_v=41, accel_sigma=20., p_lost=0.05)
S, V = hmm.S, hmm.V

logp = np.full((S, V), -np.log(S * V))
s_mean = np.zeros(K); s_map = np.zeros(K)
for k in range(K):
    if k > 0:
        logp = hmm._advect(logp, dt)
    logp = logp + hmm.loglik(UV[k])[:, None]
    logp -= logp.max()
    w = np.exp(logp); w /= w.sum()
    ps = w.sum(1)                       # marginal over v -> p(s_k | obs_1:k)
    s_mean[k] = (ps * hmm.s).sum()
    s_map[k] = hmm.s[np.argmax(ps)]
    logp = np.log(np.maximum(w, 1e-300))

ds = sg[1] - sg[0]
idx_mean = np.clip(np.rint(s_mean / ds).astype(int), 0, S - 1)
idx_map = np.clip(np.rint(s_map / ds).astype(int), 0, S - 1)
rp_mean = np.linalg.norm(Pg[idx_mean] - UV, axis=1)
rp_map = np.linalg.norm(Pg[idx_map] - UV, axis=1)

print(f"[I4] sanity: recomputed mean-estimator matches Stage I's stored rp "
      f"(median {np.median(rp_mean):.1f} vs {np.median(rp_mean_ref):.1f} px)")

print(f"\n{'estimator':<22}{'reproj px (med)':>16}{'p90 px':>9}{'>30 px %':>10}")
print(f"{'posterior mean (I)':<22}{np.median(rp_mean):16.1f}{np.percentile(rp_mean,90):9.1f}"
      f"{100*np.mean(rp_mean>30):10.1f}")
print(f"{'posterior mode (I4)':<22}{np.median(rp_map):16.1f}{np.percentile(rp_map,90):9.1f}"
      f"{100*np.mean(rp_map>30):10.1f}")

diverge = np.abs(s_mean - s_map) > 3.0
print(f"\n[I4] mean and mode disagree by >3 mm on {diverge.sum()} frames "
      f"({100*diverge.mean():.1f}%)")
print(f"     on those frames: mean-estimator reproj median {np.median(rp_mean[diverge]):.1f} px, "
      f"mode-estimator reproj median {np.median(rp_map[diverge]):.1f} px")

A, B = np.arange(k_turn + 1), np.arange(k_turn + 1, K)
d_img, ia = cKDTree(UV[A]).query(UV[B]); sel = d_img < 8.0
dsr_mean = np.abs(s_mean[B][sel] - s_mean[A][ia[sel]])
dsr_map = np.abs(s_map[B][sel] - s_map[A][ia[sel]])
print(f"\n[I4] out-and-back repeatability: mean {np.median(dsr_mean):.2f} mm, "
      f"mode {np.median(dsr_map):.2f} mm")

np.savez(DATA / "i4_map_filter.npz", s_mean=s_mean, s_map=s_map, rp_mean=rp_mean, rp_map=rp_map)
print("\n[I4] saved data/i4_map_filter.npz")
