"""Stage K3 - why did the off-centerline model work in simulation and fail on
the real video?

K2 fitted r = 0.3 mm where D3's image-space measurement implies about 1.3 mm,
and the evidence gain was 16 nats against 283 nats in simulation. The
hypothesis: the real residual is dominated by REGISTRATION error (p90 = 86 px,
about 13 mm) which is ten times the wobble signal (8.6 px, about 1.3 mm), so
there is nothing left for r to explain.

Test: refit using only the frames where the baseline registration is good. If r
climbs towards 1.3 mm there, the hypothesis holds and the off-axis model is
sound but blocked by registration - which is the same bottleneck as everywhere
else. If r stays near zero even on clean frames, the model itself is wrong.
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from config import DATA, OUT, W, H, FPS
from e_hmm import MultiViewArcHMM
from k_offaxis import SG, CG, cam_terms, make_proj_fn, fit_offaxis

p = np.load(DATA / "c8_11.npy")
rv, t, f, cx, cy = p[:3], p[3:6], np.exp(p[6]), p[9], p[10]
R = Rot.from_rotvec(rv).as_matrix(); c = np.array([cx, cy])
A, B1, B2 = cam_terms(R, t)
scale = f / np.median((CG @ R.T + t)[:, 2])

zr = np.load(DATA / "track2d.npz"); zc = np.load(DATA / "track2d_clean.npz")
UV = np.where(np.isnan(zr["uv"]), zc["uv"], zr["uv"])
K = len(UV)
OM = 2 * np.pi * 1.17
hmm = MultiViewArcHMM(SG, sigma_px=6.0, v_max=30., n_v=41, accel_sigma=20., p_lost=0.05)

# baseline pass to get the per-frame registration residual
fn0 = make_proj_fn(A, B1, B2, f, c, 0.0, 0.0, OM, K)
s0, _, _ = hmm.forward(UV[:, None, :], fn0, 1 / FPS)
idx = np.clip(np.rint(s0 / (SG[1] - SG[0])).astype(int), 0, len(SG) - 1)
rp0 = np.linalg.norm(np.stack([fn0(k)[0][idx[k]] for k in range(K)]) - UV, axis=1)
print(f"[K3] baseline reprojection px: median {np.median(rp0):.1f}  p90 {np.percentile(rp0,90):.1f}")
print(f"[K3] wobble signal to fit (D3): 8.6 px = {8.6/scale:.2f} mm at {scale:.2f} px/mm\n")

print(f"{'gate (keep frames with resid <)':<34}{'n kept':>8}{'best r mm':>11}"
      f"{'log Z gain':>12}{'signal/noise':>14}")
for gate in (np.inf, 40., 25., 15., 10.):
    keep = rp0 < gate
    if keep.sum() < 150:
        print(f"{f'< {gate:.0f} px':<34}{keep.sum():8d}   too few frames"); continue
    obs = np.where(keep[:, None, None], UV[:, None, :], np.nan)
    lz0 = hmm.forward(obs, fn0, 1 / FPS)[2]
    lz, r_h, p_h, _, _ = fit_offaxis(obs, A, B1, B2, f, c, OM, hmm, K,
                                     r_grid=(0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8),
                                     n_phi=16, verbose=False)
    med_resid_mm = np.median(rp0[keep]) / scale
    tag = "no gate (all frames)" if np.isinf(gate) else f"< {gate:.0f} px"
    print(f"{tag:<34}{keep.sum():8d}{r_h:11.1f}{lz-lz0:12.1f}"
          f"{8.6/scale/max(med_resid_mm,1e-6):14.2f}")

print(f"\n[K3] for reference, the synthetic case gained 283 nats and recovered r exactly.")
print(f"     Interpretation is in the README under 'K2/K3 - the real video'.")
