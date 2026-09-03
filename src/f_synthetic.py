"""Stage F - synthetic validation: known truth in, measured error out.

Everything upstream (detector, camera fit) is bypassed so the estimator itself
can be scored against ground truth, and so the error can be attributed:
  - in-plane error  (what the image directly constrains)
  - depth error     (what only the vessel constraint + motion model recover)

Noise sources modelled from the real data measured in Stages D/D3:
  detection noise   sigma_px
  off-centerline    the robot rides the wall, not the axis: 1.2 mm at 1.17 Hz
                    (measured: 8.6 px median wobble / 7.35 px per mm)
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from config import DATA, OUT, W, H, FPS
from a_centerline import load as load_cl
from e_hmm import ArcHMM

C0 = np.array([W / 2., H / 2.])
s3, C3, T3, _ = load_cl(); L = s3[-1]


def camera(view_dir, f=1015., dist=None, roll=0.0):
    """Build R, t for a camera looking at the path centroid along -view_dir."""
    v = np.asarray(view_dir, float); v /= np.linalg.norm(v)
    a = np.array([0, 0, 1.]) if abs(v[2]) < .9 else np.array([1., 0, 0])
    e1 = np.cross(v, a); e1 /= np.linalg.norm(e1); e2 = np.cross(v, e1)
    Rz = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
    R = Rz @ np.stack([e1, e2, v])
    ctr = C3.mean(0)
    if dist is None:                        # frame the path to ~70% of image width
        ext = np.ptp((C3 - ctr) @ R.T, axis=0)
        dist = f * max(ext[0] / (0.7 * W), ext[1] / (0.7 * H))
    t = -R @ ctr + np.array([0, 0, dist])
    return R, t, f


def project(X, R, t, f):
    Xc = X @ R.T + t
    return Xc[:, :2] / Xc[:, 2:3] * f + C0, Xc[:, 2]


def truth_trajectory(n_frames=900, v_mm_s=9.0, turn_frac=0.62, seed=0):
    """Out-and-back traversal like the real video, with a smooth speed profile."""
    rng = np.random.default_rng(seed)
    k = np.arange(n_frames); kt = int(turn_frac * n_frames)
    v = np.where(k < kt, 1.0, -1.0) * v_mm_s
    v *= 0.6 + 0.4 * np.sin(np.pi * np.clip(np.where(k < kt, k / kt, (n_frames - k) / (n_frames - kt)), 0, 1))
    s = np.cumsum(v) / FPS
    s = 5 + (s - s.min()) / (s.max() - s.min()) * (L - 10)
    return np.clip(s, 0, L)


def offaxis(s_true, R, amp_mm=1.2, f_hz=1.17, seed=0):
    """Displace the robot off the centerline, perpendicular to the local tangent."""
    rng = np.random.default_rng(seed)
    T = np.stack([np.interp(s_true, s3, T3[:, c]) for c in range(3)], 1)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    a = np.tile([0, 0, 1.], (len(T), 1))
    n1 = np.cross(T, a); n1 /= np.clip(np.linalg.norm(n1, axis=1, keepdims=True), 1e-9, None)
    n2 = np.cross(T, n1)
    ph = 2 * np.pi * f_hz * np.arange(len(T)) / FPS + rng.uniform(0, 2 * np.pi)
    return amp_mm * (np.cos(ph)[:, None] * n1 + np.sin(ph)[:, None] * n2)


def run_case(view_dir, sigma_px=6.0, amp_mm=1.2, f=1015., n_frames=900, seed=0,
             hmm_sigma=None, ds=0.5, causal=True):
    rng = np.random.default_rng(seed)
    R, t, f = camera(view_dir, f)
    s_true = truth_trajectory(n_frames, seed=seed)
    X_true = np.stack([np.interp(s_true, s3, C3[:, c]) for c in range(3)], 1)
    X_true = X_true + offaxis(s_true, R, amp_mm, seed=seed)
    U, _ = project(X_true, R, t, f)
    U = U + rng.normal(0, sigma_px, U.shape)

    sg = np.arange(0, L + 1e-9, ds)
    Cg = np.stack([np.interp(sg, s3, C3[:, c]) for c in range(3)], 1)
    Pg, zg = project(Cg, R, t, f)
    hmm = ArcHMM(sg, Pg, sigma_px=hmm_sigma or max(sigma_px, 4.0), v_max=25., n_v=41,
                 accel_sigma=18., p_lost=0.02)
    s_hat = hmm.forward(U, 1 / FPS)[0] if causal else hmm.viterbi(U, 1 / FPS)

    X_hat = np.stack([np.interp(s_hat, sg, Cg[:, c]) for c in range(3)], 1)
    Xc_t = np.stack([np.interp(s_true, s3, C3[:, c]) for c in range(3)], 1)   # on-centerline truth
    err = np.linalg.norm(X_hat - Xc_t, axis=1)
    e_vec = (X_hat - Xc_t) @ R.T
    return dict(err=err, s_true=s_true, s_hat=s_hat,
                e_depth=np.abs(e_vec[:, 2]), e_inplane=np.linalg.norm(e_vec[:, :2], axis=1),
                s_err=np.abs(s_hat - s_true), R=R, scale=f / np.median(zg))


if __name__ == "__main__":
    print("[F] synthetic validation, causal (real-time) filter\n")
    print(f"{'view direction':>22}{'px/mm':>7}{'3D err mm (med/p90/max)':>26}"
          f"{'depth mm':>16}{'in-plane mm':>14}{'s err mm':>10}")
    views = {
        "C6 fit  [.51 .19 -.84]": [0.514, 0.192, -0.836],
        "best obs [.58 .09 -.81]": [0.581, 0.092, -0.809],
        "worst obs[.58 .12  .81]": [0.575, 0.122, 0.809],
        "along +X": [1., 0, 0], "along +Y": [0, 1., 0], "along +Z": [0, 0, 1.],
    }
    res = {}
    for name, v in views.items():
        r = run_case(v)
        res[name] = r
        e, d, ip, se = r["err"], r["e_depth"], r["e_inplane"], r["s_err"]
        print(f"{name:>22}{r['scale']:7.2f}"
              f"{f'{np.median(e):.2f} / {np.percentile(e,90):.2f} / {e.max():.2f}':>26}"
              f"{f'{np.median(d):.2f}/{np.percentile(d,90):.2f}':>16}"
              f"{f'{np.median(ip):.2f}/{np.percentile(ip,90):.2f}':>14}"
              f"{np.median(se):10.2f}")
    np.savez(DATA / "f_synth.npz", **{k: v["err"] for k, v in res.items()})
