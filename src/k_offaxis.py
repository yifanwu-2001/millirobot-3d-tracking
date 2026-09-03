"""Stage K - model the robot OFF the centerline: state (s, sdot, r, phi).

Stage D3 measured that the robot rides about 1.2 mm off the centerline and
precesses at 1.17 Hz. Up to now that motion was treated as measurement noise,
which inflates the effective sigma and caps the median error. Here it becomes
part of the model:

    X(s, r, phi) = C(s) + r * ( cos(phi) N1(s) + sin(phi) N2(s) )

with (N1, N2) a parallel-transport frame (k_frame.py; the Frenet normal would
flip at curvature zeros).

The roll phase advances deterministically at the measured rate, phi_k = phi0 +
omega * k / fps, so the state does NOT need two extra dimensions - only two
extra GLOBAL parameters (r, phi0), which are chosen by maximising the HMM's own
model evidence log Z. The per-frame offset is folded into the projection, which
MultiViewArcHMM already supports through proj_fn(k).

This is the only remaining lever on the MEDIAN error: no viewing geometry
removes the off-centerline term (Stage J showed biplane only fixes the tail).
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from config import DATA, OUT, W, H, FPS
from a_centerline import load as load_cl
from k_frame import transport_frame
from e_hmm import MultiViewArcHMM

s3, C3, _, _ = load_cl()
L = s3[-1]
DS = 0.25
SG = np.arange(0, L + 1e-9, DS)
CG = np.stack([np.interp(SG, s3, C3[:, c]) for c in range(3)], 1)
_, N1G, N2G = transport_frame(CG)


def cam_terms(R, t):
    """Precompute the camera-frame constant and the two offset directions, so a
    change of (r, phi) costs only a couple of vector operations per frame."""
    return CG @ R.T + t, N1G @ R.T, N2G @ R.T


def make_proj_fn(A, B1, B2, f, c, r, phi0, omega, K):
    """proj_fn(k) -> [ (S,2) ] for the offset centerline at frame k."""
    ph = phi0 + omega * np.arange(K) / FPS
    cs, sn = np.cos(ph), np.sin(ph)
    cache = {}

    def fn(k):
        if k not in cache:
            Xc = A + r * (cs[k] * B1 + sn[k] * B2)
            cache[k] = [Xc[:, :2] / Xc[:, 2:3] * f + c]
        return cache[k]
    return fn


def fit_offaxis(obs, A, B1, B2, f, c, omega, hmm, K,
                r_grid=(0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8),
                n_phi=12, verbose=True):
    """Grid search (r, phi0) by model evidence."""
    phis = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    best = None
    table = np.zeros((len(r_grid), n_phi))
    for i, r in enumerate(r_grid):
        for j, p0 in enumerate(phis):
            fn = make_proj_fn(A, B1, B2, f, c, r, p0, omega, K)
            sh, sd, lz = hmm.forward(obs, fn, 1 / FPS)
            table[i, j] = lz
            if best is None or lz > best[0]:
                best = (lz, r, p0, sh, sd)
            if r == 0.0:
                break                      # phi is meaningless at r = 0
    if verbose:
        print(f"[K] evidence log Z by radial offset (max over phase):")
        for i, r in enumerate(r_grid):
            row = table[i, :1] if r == 0 else table[i]
            mark = "  <== best" if abs(r - best[1]) < 1e-9 else ""
            print(f"     r = {r:4.1f} mm   log Z = {row.max():12.1f}{mark}")
    return best


# --------------------------------------------------------------- synthetic test
def camera_for(view_dir, f=1015.):
    v = np.asarray(view_dir, float); v /= np.linalg.norm(v)
    a = np.array([0, 0, 1.]) if abs(v[2]) < .9 else np.array([1., 0, 0])
    e1 = np.cross(v, a); e1 /= np.linalg.norm(e1); e2 = np.cross(v, e1)
    R = np.stack([e1, e2, v]); ctr = CG.mean(0)
    ext = np.ptp((CG - ctr) @ R.T, axis=0)
    dist = f * max(ext[0] / (.7 * W), ext[1] / (.7 * H))
    return R, -R @ ctr + np.array([0, 0, dist]), f


def truth(n_frames, v_mm_s=9.0, turn=0.62):
    k = np.arange(n_frames); kt = int(turn * n_frames)
    v = np.where(k < kt, 1.0, -1.0) * v_mm_s
    v *= .6 + .4 * np.sin(np.pi * np.clip(np.where(k < kt, k / kt, (n_frames - k) / (n_frames - kt)), 0, 1))
    s = np.cumsum(v) / FPS
    return np.clip(5 + (s - s.min()) / (s.max() - s.min()) * (L - 10), 0, L)


def synth(seed=0, r_true=1.2, omega=2 * np.pi * 1.17, sigma_px=6.0, K=900,
          view=(-0.086, 0.382, -0.920)):
    rng = np.random.default_rng(seed)
    R, t, f = camera_for(view); c = np.array([W / 2., H / 2.])
    s_true = truth(K)
    phi0 = rng.uniform(0, 2 * np.pi)
    ph = phi0 + omega * np.arange(K) / FPS
    Ct = np.stack([np.interp(s_true, SG, CG[:, i]) for i in range(3)], 1)
    n1 = np.stack([np.interp(s_true, SG, N1G[:, i]) for i in range(3)], 1)
    n2 = np.stack([np.interp(s_true, SG, N2G[:, i]) for i in range(3)], 1)
    X_true = Ct + r_true * (np.cos(ph)[:, None] * n1 + np.sin(ph)[:, None] * n2)
    Xc = X_true @ R.T + t
    U = Xc[:, :2] / Xc[:, 2:3] * f + c + rng.normal(0, sigma_px, (K, 2))
    return U[:, None, :], (R, t, f, c), s_true, X_true, phi0, omega


def err3d(s_hat, r, phi0, omega, X_true, K):
    ph = phi0 + omega * np.arange(K) / FPS
    Ch = np.stack([np.interp(s_hat, SG, CG[:, i]) for i in range(3)], 1)
    n1 = np.stack([np.interp(s_hat, SG, N1G[:, i]) for i in range(3)], 1)
    n2 = np.stack([np.interp(s_hat, SG, N2G[:, i]) for i in range(3)], 1)
    Xh = Ch + r * (np.cos(ph)[:, None] * n1 + np.sin(ph)[:, None] * n2)
    return np.linalg.norm(Xh - X_true, axis=1)


if __name__ == "__main__":
    K = 900
    print("[K] SYNTHETIC: does modelling (r, phi) beat treating the wobble as noise?")
    print(f"    truth r = 1.2 mm, roll 1.17 Hz, detection noise 6 px, {K} frames\n")
    rows = {"centerline only (baseline)": [], "oracle (true r, phi0)": [],
            "estimated (grid search)": []}
    r_est, phi_err = [], []
    for seed in range(4):
        obs, (R, t, f, c), s_true, X_true, phi0_t, om = synth(seed)
        A, B1, B2 = cam_terms(R, t)
        hmm = MultiViewArcHMM(SG, sigma_px=6.0, v_max=25., n_v=41, accel_sigma=18., p_lost=0.02)

        sh, _, _ = hmm.forward(obs, make_proj_fn(A, B1, B2, f, c, 0.0, 0.0, om, K), 1 / FPS)
        rows["centerline only (baseline)"].append(err3d(sh, 0.0, 0.0, om, X_true, K))

        sh, _, _ = hmm.forward(obs, make_proj_fn(A, B1, B2, f, c, 1.2, phi0_t, om, K), 1 / FPS)
        rows["oracle (true r, phi0)"].append(err3d(sh, 1.2, phi0_t, om, X_true, K))

        lz, r_h, p_h, sh, sd = fit_offaxis(obs, A, B1, B2, f, c, om, hmm, K,
                                           verbose=(seed == 0))
        rows["estimated (grid search)"].append(err3d(sh, r_h, p_h, om, X_true, K))
        r_est.append(r_h)
        d = (p_h - phi0_t) % (2 * np.pi)
        phi_err.append(min(d, 2 * np.pi - d))

    print(f"\n{'model':<32}{'med mm':>9}{'p90 mm':>9}{'max mm':>9}{'>5mm %':>9}")
    for k, v in rows.items():
        e = np.concatenate(v)
        print(f"{k:<32}{np.median(e):9.2f}{np.percentile(e,90):9.2f}{e.max():9.2f}"
              f"{100*np.mean(e>5):9.1f}")
    print(f"\n[K] recovered r over 4 seeds: {np.round(r_est,2)}  (truth 1.2 mm)")
    print(f"[K] roll-phase error: {np.degrees(phi_err).round(0)} deg  "
          f"(grid step {360/12:.0f} deg)")
    np.savez(DATA / "k_synth.npz", **{k: np.concatenate(v) for k, v in rows.items()})
