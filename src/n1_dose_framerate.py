"""Stage N1 - frame-rate / dose ablation.

Directly answers the README's own "What changes for real fluoroscopy" section:
clinical fluoroscopy trades acquisition rate (and detector gain/exposure, i.e.
dose) for patient radiation exposure. This subsamples the ALREADY-COLLECTED
30 fps track in time (simulating a lower pulse rate) and injects additional
detection noise (simulating lower-dose, noisier detection) to trace out an
accuracy-vs-acquisition-rate curve and an accuracy-vs-noise curve, at zero new
data cost. Re-uses the existing camera fit and centerline; only the HMM's
observation sequence changes.
"""
import numpy as np
from scipy.spatial import cKDTree
from config import DATA, FPS
from e_hmm import ArcHMM

z = np.load(DATA / "i_final.npz")
sg, Pg, UV_full, k_turn = z["sg"], z["Pg"], z["UV"], int(z["k_turn"])
K_full = len(UV_full)
rng = np.random.default_rng(0)


def run_hmm(UV, dt, sigma_px=9.0):
    hmm = ArcHMM(sg, Pg, sigma_px=sigma_px, v_max=30., n_v=41, accel_sigma=20., p_lost=0.05)
    s_hat, s_std, v_hat = hmm.forward(UV, dt)
    Q_hat = np.stack([np.interp(s_hat, sg, Pg[:, c]) for c in range(2)], 1)
    rp = np.linalg.norm(Q_hat - UV, axis=1)
    return s_hat, rp


def outback_repeat(s_hat, idx_full, k_turn_full):
    """Map subsampled-frame arc-length estimates back onto the full-track
    out/back split to compute repeatability on the same physical criterion
    used throughout the project."""
    A = idx_full[idx_full <= k_turn_full]; B = idx_full[idx_full > k_turn_full]
    if len(A) < 5 or len(B) < 5: return np.nan
    UVA, UVB = UV_full[A], UV_full[B]
    sA, sB = s_hat[idx_full <= k_turn_full], s_hat[idx_full > k_turn_full]
    d_img, ia = cKDTree(UVA).query(UVB); sel = d_img < 8.0
    if sel.sum() < 5: return np.nan
    return float(np.median(np.abs(sB[sel] - sA[ia[sel]])))


print("[N1] (a) FRAME-RATE ablation - subsample the 30 fps track in time\n")
print(f"{'fps':>6}{'keep 1/N':>9}{'n frames':>10}{'reproj med px':>15}{'p90':>8}{'>30px%':>9}{'out-back mm':>13}")
rows_fps = []
for target_fps in (30, 15, 10, 7.5, 6, 5, 3, 2):
    step = max(1, round(FPS / target_fps))
    idx = np.arange(0, K_full, step)
    UV = UV_full[idx]
    s_hat, rp = run_hmm(UV, dt=step / FPS)
    ob = outback_repeat(s_hat, idx, k_turn)
    print(f"{FPS/step:6.1f}{step:9d}{len(idx):10d}{np.median(rp):15.1f}{np.percentile(rp,90):8.1f}"
          f"{100*np.mean(rp>30):9.1f}{ob:13.2f}")
    rows_fps.append((FPS/step, np.median(rp), np.percentile(rp,90), 100*np.mean(rp>30), ob))
rows_fps = np.array(rows_fps)

print("\n[N1] (b) DOSE ablation - inject extra detection noise on TOP of the real 9 px floor")
print("    (Gaussian, simulating a noisier low-dose detector; the real video keeps its own")
print("    detection noise on top of whatever is injected, so 0 px extra = the real baseline)\n")
print(f"{'extra sigma px':>16}{'eff. sigma':>12}{'reproj med px':>15}{'p90':>8}{'>30px%':>9}{'out-back mm':>13}")
rows_noise = []
for extra in (0, 5, 10, 15, 20, 30, 45):
    UV = UV_full + rng.normal(0, extra, UV_full.shape)
    eff_sigma = np.hypot(9.0, extra)
    s_hat, rp = run_hmm(UV, dt=1/FPS, sigma_px=eff_sigma)
    ob = outback_repeat(s_hat, np.arange(K_full), k_turn)
    print(f"{extra:16.0f}{eff_sigma:12.1f}{np.median(rp):15.1f}{np.percentile(rp,90):8.1f}"
          f"{100*np.mean(rp>30):9.1f}{ob:13.2f}")
    rows_noise.append((extra, eff_sigma, np.median(rp), np.percentile(rp,90), 100*np.mean(rp>30), ob))
rows_noise = np.array(rows_noise)

print("\n[N1] (c) COMBINED: low frame rate AND extra noise together (worst realistic case)")
print(f"{'fps':>6}{'extra sigma':>13}{'reproj med px':>15}{'>30px%':>9}{'out-back mm':>13}")
for target_fps, extra in [(15, 10), (10, 15), (7.5, 20), (5, 20)]:
    step = max(1, round(FPS / target_fps))
    idx = np.arange(0, K_full, step)
    UV = UV_full[idx] + rng.normal(0, extra, UV_full[idx].shape)
    eff_sigma = np.hypot(9.0, extra)
    s_hat, rp = run_hmm(UV, dt=step/FPS, sigma_px=eff_sigma)
    ob = outback_repeat(s_hat, idx, k_turn)
    print(f"{FPS/step:6.1f}{extra:13.0f}{np.median(rp):15.1f}{100*np.mean(rp>30):9.1f}{ob:13.2f}")

# where does it break?
med_break = rows_fps[rows_fps[:, 1] > 2*np.median(rp)][0] if (rows_fps[:,1] > 2*rows_fps[0,1]).any() else None
print(f"\n[N1] baseline (30 fps, real noise): reproj med {rows_fps[0,1]:.1f} px, >30px {rows_fps[0,3]:.1f}%")
thresh = rows_fps[0, 1] * 1.5
bad = rows_fps[rows_fps[:, 1] > thresh]
if len(bad):
    print(f"     median reprojection exceeds 1.5x baseline at or below {bad[0,0]:.1f} fps")
else:
    print(f"     median reprojection stays within 1.5x baseline across all tested rates -")
    print(f"     the estimator is remarkably insensitive to acquisition rate down to "
          f"{rows_fps[-1,0]:.1f} fps (the motion model absorbs the larger inter-frame gaps).")

np.savez(DATA / "n1_dose_framerate.npz", rows_fps=rows_fps, rows_noise=rows_noise)
print("\n[N1] saved data/n1_dose_framerate.npz")
