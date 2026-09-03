"""Stage J - biplane and rotating-C-arm extension.

Single-view tracking failed on 16% of real frames by latching onto the wrong
arc length where the projected centerline self-intersects. A second view fixes
exactly that failure mode: two arc lengths that collide in one projection
almost never collide in both. This stage measures how much a second view is
actually worth, and whether a rotating single detector can buy the same thing.

  J1  biplane, sweeping the angle between the two views
  J2  rotating C-arm, sweeping the rotation rate
  J3  which view PAIR is best (pairwise observability)
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from config import DATA, OUT, W, H, FPS
from a_centerline import load as load_cl
from e_hmm import MultiViewArcHMM
from f_synthetic import camera, project, truth_trajectory, offaxis

s3, C3, T3, _ = load_cl(); L = s3[-1]
DS = 0.25
SG = np.arange(0, L + 1e-9, DS)
CG = np.stack([np.interp(SG, s3, C3[:, c]) for c in range(3)], 1)

V0 = np.array([-0.086, 0.382, -0.920])          # the view actually fitted in Stage C8
V0 /= np.linalg.norm(V0)
AXIS = np.array([-0.176, -0.983, 0.052])        # path long axis ~ "patient axis" a C-arm turns about
AXIS /= np.linalg.norm(AXIS)


def rot_about(v, axis, deg):
    return Rot.from_rotvec(np.radians(deg) * axis).apply(v)


def simulate(views, s_true, sigma_px, amp_mm, seed):
    """Project the true (off-centerline) robot into every view, add pixel noise."""
    rng = np.random.default_rng(seed + 991)
    obs = []
    for (R, t, f) in views:
        X = np.stack([np.interp(s_true, s3, C3[:, c]) for c in range(3)], 1)
        X = X + offaxis(s_true, R, amp_mm, seed=seed)
        U, _ = project(X, R, t, f)
        obs.append(U + rng.normal(0, sigma_px, U.shape))
    return np.stack(obs, 1)                      # (K, n_views, 2)


def score(s_hat, s_true):
    Xh = np.stack([np.interp(s_hat, SG, CG[:, c]) for c in range(3)], 1)
    Xt = np.stack([np.interp(s_true, s3, C3[:, c]) for c in range(3)], 1)
    e = np.linalg.norm(Xh - Xt, axis=1)
    return (np.median(e), np.percentile(e, 90), e.max(),
            100 * np.mean(e > 5.0), np.median(np.abs(s_hat - s_true)))


def run(view_dirs_fn, n_frames=900, sigma_px=6.0, amp_mm=1.2, f=1015., seed=0,
        static=True):
    """view_dirs_fn(k) -> list of view directions active at frame k.
    static=True when those directions do not depend on k (fixed detectors)."""
    s_true = truth_trajectory(n_frames, seed=seed)
    v_at0 = view_dirs_fn(0)
    n_views = len(v_at0)
    cams0 = [camera(v, f) for v in v_at0]
    obs = simulate(cams0, s_true, sigma_px, amp_mm, seed) if static else None

    if not static:   # rotating detector: rebuild the camera every frame
        obs = np.full((n_frames, n_views, 2), np.nan)
        rng = np.random.default_rng(seed + 991)
        Xt = np.stack([np.interp(s_true, s3, C3[:, c]) for c in range(3)], 1)
        cams_k = []
        for k in range(n_frames):
            cams = [camera(v, f) for v in view_dirs_fn(k)]
            cams_k.append(cams)
            for i, (R, t, ff) in enumerate(cams):
                Xk = Xt[k] + offaxis(s_true[k:k+1], R, amp_mm, seed=seed)[0]
                u, _ = project(Xk[None], R, t, ff)
                obs[k, i] = u[0] + rng.normal(0, sigma_px, 2)
        proj_cache = [[project(CG, R, t, ff)[0] for (R, t, ff) in cams] for cams in cams_k]
        proj_fn = lambda k: proj_cache[k]
    else:
        P0 = [project(CG, R, t, ff)[0] for (R, t, ff) in cams0]
        proj_fn = lambda k: P0

    hmm = MultiViewArcHMM(SG, sigma_px=max(sigma_px, 4.0), v_max=25., n_v=41,
                          accel_sigma=18., p_lost=0.02)
    s_hat, s_std, _ = hmm.forward(obs, proj_fn, 1 / FPS)
    return score(s_hat, s_true) + (np.median(s_std),)


if __name__ == "__main__":
    HDR = (f"{'case':<36}{'med mm':>8}{'p90 mm':>8}{'max mm':>8}"
           f"{'>5mm %':>8}{'s err':>8}{'sigma':>8}")
    SEEDS = (0, 1, 2)

    def show(tag, r):
        print(f"{tag:<36}{r[0]:8.2f}{r[1]:8.2f}{r[2]:8.2f}{r[3]:8.1f}{r[4]:8.2f}{r[5]:8.2f}")

    print("[J1] BIPLANE: second detector at an angle to the fitted view")
    print(HDR)
    rows = {}
    for th in (0, 10, 20, 30, 45, 60, 75, 90):
        vd = [V0] if th == 0 else [V0, rot_about(V0, AXIS, th)]
        r = np.mean([run(lambda k, vd=vd: vd, seed=sd, static=True) for sd in SEEDS], axis=0)
        rows[th] = r
        show("single view (baseline)" if th == 0 else f"biplane, {th} deg separation", r)

    b = rows[0]
    print(f"\n     biplane at 90 deg vs single view:"
          f"  median {b[0]:.2f} -> {rows[90][0]:.2f} mm ({b[0]/rows[90][0]:.1f}x)"
          f"   p90 {b[1]:.2f} -> {rows[90][1]:.2f} mm ({b[1]/rows[90][1]:.1f}x)"
          f"   catastrophic {b[3]:.1f}% -> {rows[90][3]:.1f}%")

    print("\n[J2] ROTATING C-ARM: one detector, view direction sweeping over time")
    print(HDR)
    n_frames = 900
    for w in (0, 2, 5, 10, 20, 40):
        fn = (lambda k, w=w: [rot_about(V0, AXIS, w * k / FPS)])
        r = np.mean([run(fn, n_frames=n_frames, seed=sd, static=(w == 0)) for sd in SEEDS], axis=0)
        total = w * n_frames / FPS
        show(f"sweep {w} deg/s  (total {total:.0f} deg)", r)

    print("\n[J2b] SAME, but the sweep is a limited-angle rock back and forth (+-20 deg)")
    print(HDR)
    for period_s in (2.0, 4.0, 8.0):
        fn = (lambda k, T=period_s: [rot_about(V0, AXIS, 20 * np.sin(2 * np.pi * k / FPS / T))])
        r = np.mean([run(fn, n_frames=n_frames, seed=sd, static=False) for sd in SEEDS], axis=0)
        show(f"rock +-20 deg, period {period_s:.0f} s", r)
