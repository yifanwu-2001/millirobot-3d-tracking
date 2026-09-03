"""Stage C9 - intrinsic calibration from the trajectory correspondences.

Follows the suggestion of using cv2.calibrateCamera on the correspondences.
This is now the right move: C8 showed the camera MODEL (specifically the fixed
principal point) was the limiting factor, and we finally have (a) a de-wobbled
2D track and (b) a pose good enough to produce trustworthy correspondences.

Procedure
  1. DTW correspondences under the best C8 camera (3D point <-> 2D point)
  2. cv2.calibrateCamera, single view, iterated with re-established correspondences
  3. BOOTSTRAP over subsets of the correspondences, to measure empirically whether
     "more points" actually pins down the focal length, or whether the single-view
     focal/depth coupling keeps it loose no matter how many points we add.
"""
import numpy as np, cv2
from scipy.spatial.transform import Rotation as Rot
from config import DATA, OUT, W, H
from a_centerline import load as load_cl

s3, C3, _, _ = load_cl(); L = s3[-1]
z3 = np.load(DATA / "c3_global_trend.npz"); U = z3["U"].astype(np.float64)
p11 = np.load(DATA / "c8_11.npy")          # best free-principal-point solution
rv0, t0, f0, sa, sb, cx0, cy0 = p11[:3], p11[3:6], np.exp(p11[6]), p11[7], p11[8], p11[9], p11[10]
print(f"[C9] seed from C8: f={f0:.0f} px, principal ({cx0:.0f},{cy0:.0f}), s={sa:.1f}..{sb:.1f} mm")


def curve(sa, sb, n=250):
    ss = np.linspace(sa, sb, n)
    return np.stack([np.interp(ss, s3, C3[:, c]) for c in range(3)], 1), ss


def proj(X, rv, t, K):
    q, _ = cv2.projectPoints(X.astype(np.float64), rv.astype(np.float64),
                             t.astype(np.float64), K, None)
    return q.reshape(-1, 2)


def dtw_pairs(A, B):
    n, m = len(A), len(B)
    D = np.linalg.norm(A[:, None] - B[None], axis=-1)
    acc = np.full((n+1, m+1), np.inf); acc[0, 0] = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            acc[i, j] = D[i-1, j-1] + min(acc[i-1, j], acc[i, j-1], acc[i-1, j-1])
    i, j, pr = n, m, []
    while i > 0 and j > 0:
        pr.append((i-1, j-1))
        k = int(np.argmin([acc[i-1, j], acc[i, j-1], acc[i-1, j-1]]))
        if k == 0: i -= 1
        elif k == 1: j -= 1
        else: i -= 1; j -= 1
    return np.array(pr[::-1])


K = np.array([[f0, 0, cx0], [0, f0, cy0], [0, 0, 1.]])
rv, t = rv0.copy(), t0.copy()
X3, ss = curve(sa, sb)

BASE = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO |
        cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
FIXPP = BASE | cv2.CALIB_FIX_PRINCIPAL_POINT


def calib(obj, img, Kin, flags):
    """cv2.calibrateCamera, returning None instead of raising when it diverges."""
    try:
        return cv2.calibrateCamera(obj.reshape(1, -1, 3).astype(np.float32),
                                   img.reshape(1, -1, 2).astype(np.float32),
                                   (W, H), Kin.copy(), None, flags=flags)
    except cv2.error:
        return None


# ---------------------------------------------------------------- experiment A
print("\n[C9-A] FREE principal point (what cv2.calibrateCamera does by default here)")
K = np.array([[f0, 0, W/2.], [0, f0, H/2.], [0, 0, 1.]])
rv, t = rv0.copy(), t0.copy()
X3, ss = curve(sa, sb)
for it in range(4):
    Q = proj(X3, rv, t, K); pr = dtw_pairs(U, Q)
    out = calib(X3[pr[:, 1]], U[pr[:, 0]], K, BASE)
    if out is None:
        print(f"   it {it}: DIVERGED - principal point left the image, calibration aborted")
        break
    rms, K, _, rvs, tvs = out; rv, t = rvs[0].ravel(), tvs[0].ravel()
    print(f"   it {it}: rms {rms:6.2f} px   f {K[0,0]:8.0f}   principal ({K[0,2]:7.0f},{K[1,2]:7.0f})")

# ---------------------------------------------------------------- experiment B
print("\n[C9-B] principal point FIXED at the image centre, solve focal length only")
K = np.array([[f0, 0, W/2.], [0, f0, H/2.], [0, 0, 1.]])
rv, t = rv0.copy(), t0.copy()
print(f"{'it':>4}{'n pairs':>9}{'rms px':>9}{'f':>10}{'dist mm':>18}{'px/mm':>8}")
for it in range(8):
    Q = proj(X3, rv, t, K); pr = dtw_pairs(U, Q)
    out = calib(X3[pr[:, 1]], U[pr[:, 0]], K, FIXPP)
    if out is None: print(f"   it {it}: diverged"); break
    rms, K, _, rvs, tvs = out; rv, t = rvs[0].ravel(), tvs[0].ravel()
    Xc = X3 @ Rot.from_rotvec(rv).as_matrix().T + t
    print(f"{it:4d}{len(pr):9d}{rms:9.2f}{K[0,0]:10.0f}"
          f"{f'{Xc[:,2].min():.0f}..{Xc[:,2].max():.0f}':>18}{K[0,0]/np.median(Xc[:,2]):8.2f}")

# ------------------------------------------------- experiment C: does data help?
print("\n[C9-C] bootstrap with the principal point FIXED: does more data pin down f?")
print("        (image noise set to the measured 9 px floor = the off-centerline wobble)")
print(f"{'n pts':>8}{'f median':>11}{'f 5-95%':>24}{'spread':>9}{'n ok':>6}")
Q = proj(X3, rv, t, K); pr = dtw_pairs(U, Q)
objA = X3[pr[:, 1]]; imgA = U[pr[:, 0]]
rng = np.random.default_rng(0)
for npts in (20, 40, 80, 160, len(pr)):
    fs = []
    for b in range(60):
        idx = rng.choice(len(pr), min(npts, len(pr)), replace=(npts > len(pr)))
        jit = imgA[idx] + rng.normal(0, 9.0, (len(idx), 2))
        out = calib(objA[idx], jit, K, FIXPP)
        if out is not None and 100 < out[1][0, 0] < 1e6:
            fs.append(out[1][0, 0])
    fs = np.array(fs)
    if len(fs) < 5:
        print(f"{npts:8d}   too many divergences ({len(fs)}/60 usable)"); continue
    lo, hi = np.percentile(fs, [5, 95])
    print(f"{npts:8d}{np.median(fs):11.0f}{f'{lo:.0f} .. {hi:.0f}':>24}{hi/lo:8.1f}x{len(fs):6d}")

np.savez(DATA / "c9_calib.npz", K=K, rv=rv, t=t, sa=sa, sb=sb)
print("\n[C9] saved data/c9_calib.npz")
