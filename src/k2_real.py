"""Stage K2 - apply the off-centerline model to the REAL video.

Two things are being tested at once:

1. Does modelling (r, phi) improve real tracking - reprojection error, the
   fraction of failed frames, and the out-and-back repeatability?

2. Does the fitted radial offset r agree with Stage D3? D3 measured the wobble
   in IMAGE space (8.6 px median at 7.35 px/mm -> about 1.2 mm). K2 estimates r
   from the 3D model and the camera, a completely different route. Agreement
   would be strong evidence that the wobble really is the robot riding off-axis
   rather than a detector artefact.

Unlike the synthetic case, the raw (not de-wobbled) track is the right input
here: the whole point is that the wobble is now signal, not noise.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from config import DATA, OUT, W, H, FPS
from e_hmm import MultiViewArcHMM
from k_offaxis import SG, CG, N1G, N2G, cam_terms, make_proj_fn, fit_offaxis

# camera from Stage C8 (free principal point, best available)
p = np.load(DATA / "c8_11.npy")
rv, t, f, sa, sb, cx, cy = p[:3], p[3:6], np.exp(p[6]), p[7], p[8], p[9], p[10]
R = Rot.from_rotvec(rv).as_matrix()
c = np.array([cx, cy])
A, B1, B2 = cam_terms(R, t)
scale = f / np.median((CG @ R.T + t)[:, 2])
print(f"[K2] camera f={f:.0f} px, principal ({cx:.0f},{cy:.0f}), scale {scale:.2f} px/mm")

# RAW track: the wobble is signal now, so do not use the de-wobbled trend
zr = np.load(DATA / "track2d.npz")
uv_raw = zr["uv"]
zc = np.load(DATA / "track2d_clean.npz")
uv_cln = zc["uv"]; k_turn = int(zc["k_turn"])
K = len(uv_cln)
# fill the few lost frames from the cleaned track, keep the wobble everywhere else
UV = np.where(np.isnan(uv_raw), uv_cln, uv_raw)
obs = UV[:, None, :]
print(f"[K2] {K} frames, using the RAW (wobbling) track")

hmm = MultiViewArcHMM(SG, sigma_px=6.0, v_max=30., n_v=41, accel_sigma=20., p_lost=0.05)


def evaluate(r, phi0, omega, tag):
    fn = make_proj_fn(A, B1, B2, f, c, r, phi0, omega, K)
    s_hat, s_std, lz = hmm.forward(obs, fn, 1 / FPS)
    idx = np.clip(np.rint(s_hat / (SG[1] - SG[0])).astype(int), 0, len(SG) - 1)
    Q = np.stack([fn(k)[0][idx[k]] for k in range(K)])
    rp = np.linalg.norm(Q - UV, axis=1)
    Aa, Bb = np.arange(k_turn + 1), np.arange(k_turn + 1, K)
    d_img, ia = cKDTree(UV[Aa]).query(UV[Bb]); sel = d_img < 8.0
    dsr = np.abs(s_hat[Bb][sel] - s_hat[Aa][ia[sel]])
    print(f"{tag:<30}{np.median(rp):9.1f}{np.percentile(rp,90):9.1f}"
          f"{100*np.mean(rp>30):9.1f}{np.median(dsr):11.2f}{np.percentile(dsr,90):9.2f}{lz:12.0f}")
    return s_hat, s_std, rp, dsr, lz


print(f"\n{'model':<30}{'reproj':>9}{'p90':>9}{'>30px%':>9}"
      f"{'out-back':>11}{'p90':>9}{'log Z':>12}")
OM = 2 * np.pi * 1.17
base = evaluate(0.0, 0.0, OM, "centerline only")

print(f"\n[K2] fitting (r, phi0) at the measured 1.17 Hz roll rate")
lz, r_h, p_h, s_h, sd_h = fit_offaxis(obs, A, B1, B2, f, c, OM, hmm, K, verbose=True)
print(f"\n{'model':<30}{'reproj':>9}{'p90':>9}{'>30px%':>9}"
      f"{'out-back':>11}{'p90':>9}{'log Z':>12}")
evaluate(0.0, 0.0, OM, "centerline only")
fit = evaluate(r_h, p_h, OM, f"off-axis r={r_h:.1f} mm")

print(f"\n[K2] CROSS-CHECK of the radial offset")
print(f"     Stage D3, from image-space wobble : 8.6 px / {scale:.2f} px per mm = "
      f"{8.6/scale:.2f} mm")
print(f"     Stage K2, from the 3D model fit    : {r_h:.2f} mm")
print(f"     -> {'AGREE' if abs(r_h - 8.6/scale) < 0.5 else 'DISAGREE'} "
      f"(difference {abs(r_h - 8.6/scale):.2f} mm)")

# is the roll rate right? scan it
print(f"\n[K2] scanning the roll rate around the 1.17 Hz measured in D3")
print(f"{'roll Hz':>10}{'best r mm':>12}{'log Z':>14}")
best_hz = None
for hz in (0.9, 1.05, 1.17, 1.3, 1.45, 2.34):
    lz2, r2, p2, _, _ = fit_offaxis(obs, A, B1, B2, f, c, 2 * np.pi * hz, hmm, K,
                                    r_grid=(0.6, 0.9, 1.2, 1.5), n_phi=8, verbose=False)
    print(f"{hz:10.2f}{r2:12.1f}{lz2:14.0f}")
    if best_hz is None or lz2 > best_hz[0]:
        best_hz = (lz2, hz, r2)
print(f"     best roll rate by evidence: {best_hz[1]:.2f} Hz "
      f"(D3 spectral peak was 1.17 Hz)")

np.savez(DATA / "k2_real.npz", r=r_h, phi0=p_h, s_hat=s_h, s_std=sd_h,
         rp_base=base[2], rp_fit=fit[2], dsr_base=base[3], dsr_fit=fit[3])
print(f"\n[K2] saved data/k2_real.npz")
