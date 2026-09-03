"""Stage J4 - does a second view fix the calibration degeneracy found in C7/C9?

C7 showed the focal length was free over a 50x range from one view; C9 showed
cv2.calibrateCamera runs the principal point out of the image. Here we answer
the question analytically rather than by re-running an optimiser: build the
Fisher information of the projection with respect to the camera parameters and
read off the Cramer-Rao standard errors.

This is the BEST case for the single view - it assumes exact correspondences
(each pixel is known to belong to a known arc length), which the real pipeline
never has. If f is still poorly determined here, it is hopeless in practice.
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from config import DATA, W, H
from a_centerline import load as load_cl

s3, C3, _, _ = load_cl(); L = s3[-1]
N = 200
ss = np.linspace(0, L, N)
X = np.stack([np.interp(ss, s3, C3[:, c]) for c in range(3)], 1)
SIGMA = 6.0                     # px, detection noise
V0 = np.array([-0.086, 0.382, -0.920]); V0 /= np.linalg.norm(V0)
AX = np.array([-0.176, -0.983, 0.052]); AX /= np.linalg.norm(AX)


def pose_for(v, f):
    v = v / np.linalg.norm(v)
    a = np.array([0, 0, 1.]) if abs(v[2]) < .9 else np.array([1., 0, 0])
    e1 = np.cross(v, a); e1 /= np.linalg.norm(e1); e2 = np.cross(v, e1)
    R = np.stack([e1, e2, v]); ctr = X.mean(0)
    ext = np.ptp((X - ctr) @ R.T, axis=0)
    dist = f * max(ext[0] / (.7 * W), ext[1] / (.7 * H))
    return Rot.from_matrix(R).as_rotvec(), -R @ ctr + np.array([0, 0, dist])


def project(rv, t, f, cx, cy):
    Xc = X @ Rot.from_rotvec(rv).as_matrix().T + t
    return Xc[:, :2] / Xc[:, 2:3] * f + np.array([cx, cy])


def jacobian(views, shared):
    """views: list of (rv, t). shared: [f, cx, cy] common to all views.
    Parameter vector = per-view pose (6 each) + shared intrinsics (3)."""
    nv = len(views); npar = 6 * nv + 3
    rows = []
    base = [project(rv, t, *shared) for rv, t in views]
    for p in range(npar):
        step = np.zeros(npar)
        h = 1e-5 if p < 6 * nv else (1e-3 if p == 6 * nv else 1e-3)
        step[p] = h
        pert = []
        for i, (rv, t) in enumerate(views):
            rv2 = rv + step[6*i:6*i+3]; t2 = t + step[6*i+3:6*i+6]
            sh = np.array(shared) + step[6*nv:]
            pert.append(project(rv2, t2, *sh))
        d = np.concatenate([(a - b).ravel() for a, b in zip(pert, base)]) / h
        rows.append(d)
    return np.stack(rows, 1)            # (2*N*nv, npar)


def crb(views, shared):
    J = jacobian(views, shared)
    F = J.T @ J / SIGMA ** 2
    # the parameters have wildly different units; report via pseudo-inverse with
    # a rank check so a genuine degeneracy shows up as a huge / infinite error
    U, S, Vt = np.linalg.svd(F)
    rank = int((S > S.max() * 1e-12).sum())
    Finv = Vt.T @ np.diag(1.0 / np.maximum(S, S.max() * 1e-12)) @ U.T
    npar = F.shape[0]
    return np.sqrt(np.diag(Finv))[-3:], rank, npar, S.min() / S.max()


F_TRUE = 1015.0
print(f"[J4] Cramer-Rao standard errors on the shared intrinsics")
print(f"     true f = {F_TRUE:.0f} px, principal at image centre, {N} exact correspondences,")
print(f"     detection noise sigma = {SIGMA:.0f} px\n")
print(f"{'configuration':<34}{'sigma_f px':>13}{'sigma_cx px':>13}{'sigma_cy px':>13}"
      f"{'rank':>7}{'cond':>10}")

rv0, t0 = pose_for(V0, F_TRUE)
sh = [F_TRUE, W / 2., H / 2.]
e, rank, npar, cnd = crb([(rv0, t0)], sh)
print(f"{'single view':<34}{e[0]:13.1f}{e[1]:13.1f}{e[2]:13.1f}{rank:4d}/{npar:<2d}{cnd:10.1e}")
single = e.copy()

for th in (10, 20, 30, 45, 60, 90):
    v2 = Rot.from_rotvec(np.radians(th) * AX).apply(V0)
    rv2, t2 = pose_for(v2, F_TRUE)
    e, rank, npar, cnd = crb([(rv0, t0), (rv2, t2)], sh)
    print(f"{f'biplane, {th} deg separation':<34}{e[0]:13.1f}{e[1]:13.1f}{e[2]:13.1f}"
          f"{rank:4d}/{npar:<2d}{cnd:10.1e}")

print(f"\n[J4] a second view at 90 deg improves sigma_f by "
      f"{single[0]/crb([(rv0,t0), pose_for(Rot.from_rotvec(np.radians(90)*AX).apply(V0), F_TRUE)], sh)[0][0]:.0f}x")
print(f"[J4] single-view sigma_f = {single[0]:.0f} px on a true f of {F_TRUE:.0f} px"
      f"  = {100*single[0]/F_TRUE:.0f}% relative")
print("     NOTE: this is far BETTER than the ~50x flat valley measured in C7, because the")
print("     bound above assumes EXACT correspondences. The real pipeline matches curve to")
print("     curve, which adds a sliding degree of freedom along the path, so C7 is the")
print("     achievable number and this is only a lower bound that practice does not reach.")
print("     The useful content here is the RATIO: a second view buys ~3x on f and ~3x on cx,")
print("     and ~10x on Fisher conditioning -- but NEITHER configuration is well conditioned")
print("     (cond ~1e-11 to 1e-10). Biplane reduces the calibration problem, it does not")
print("     remove it. A calibration shot still beats both.")
np.save(DATA / "j4_crb.npy", single)
