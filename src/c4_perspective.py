"""Stage C4 - full perspective refinement of the viewing geometry.

7 free parameters: R (axis-angle, 3), t (3), focal length f (1).
Principal point fixed at the image centre (freed in a second pass).
Cost: trimmed one-sided Chamfer, pixels, from the detected 2D track to the
projected 3D centerline. Multi-start from the top orthographic hypotheses of C3
crossed with a range of focal lengths (large f -> orthographic limit).
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot
from config import DATA, OUT, W, H

TRIM = 0.90
C0 = np.array([W / 2, H / 2])


def project(X, rv, t, f, c=C0):
    Xc = X @ Rot.from_rotvec(rv).as_matrix().T + t
    z = Xc[:, 2]
    return Xc[:, :2] / z[:, None] * f + c, z


def cost(p, U, X, tree_pts=None):
    rv, t, f = p[:3], p[3:6], p[6]
    if f <= 0 or t[2] <= 0: return 1e6
    Q, z = project(X, rv, t, f)
    if (z <= 1e-3).any(): return 1e6
    if not np.isfinite(Q).all(): return 1e6
    d, _ = cKDTree(Q).query(U)
    n = int(TRIM * len(U))
    return float(np.sort(d)[:n].mean())


def refine(U, X, rv0, t0, f0, maxiter=4000):
    p0 = np.r_[rv0, t0, f0]
    scale = np.r_[[0.05]*3, [np.abs(t0[2])*0.02]*3, [f0*0.05]]
    res = minimize(lambda q: cost(p0 + q*scale, U, X), np.zeros(7),
                   method="Nelder-Mead",
                   options=dict(maxiter=maxiter, xatol=1e-4, fatol=1e-4, adaptive=True))
    return p0 + res.x * scale, res.fun


def main():
    z3 = np.load(DATA / "c3_global.npz")
    dirs, angs, top, U, X, Ur = z3["dirs"], z3["angs"], z3["top"], z3["U"], z3["X"], float(z3["Ur"])
    Uc = U.mean(0)

    best = None
    print(f"[C4] multi-start perspective refinement")
    print(f"{'seed':>5}{'f0 px':>9}{'ortho px':>10}{'refined px':>12}   view dir after refine")
    for r in range(8):
        sc, di, ri = top[r]; di, ri = int(di), int(ri)
        v = dirs[di]; ang = angs[ri]
        a = np.array([0, 0, 1.0]) if abs(v[2]) < 0.9 else np.array([1.0, 0, 0])
        e1 = np.cross(v, a); e1 /= np.linalg.norm(e1); e2 = np.cross(v, e1)
        Rz = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
        R0 = Rz @ np.stack([e1, e2, v])          # rows: image x, image y, view axis
        Q = X @ R0.T
        Qr = np.sqrt(((Q[:, :2] - Q[:, :2].mean(0)) ** 2).sum(1).mean())
        k_px_per_mm = Ur / Qr                     # orthographic scale
        for f0 in (1200., 2500., 5000., 12000., 40000.):
            tz = f0 / k_px_per_mm
            Xc0 = X @ R0.T
            t0 = np.r_[-Xc0[:, :2].mean(0), tz - Xc0[:, 2].mean()]
            rv0 = Rot.from_matrix(R0).as_rotvec()
            p, c = refine(U, X, rv0, t0, f0)
            if best is None or c < best[1]:
                best = (p, c, r, f0)
                vd = Rot.from_rotvec(p[:3]).as_matrix()[2]
                print(f"{r:5d}{f0:9.0f}{sc*Ur:10.1f}{c:12.2f}   {vd.round(3)}  f={p[6]:.0f}")
    p, c, r, f0 = best
    rv, t, f = p[:3], p[3:6], p[6]
    Rm = Rot.from_rotvec(rv).as_matrix()
    Q, zc = project(X, rv, t, f)
    print(f"\n[C4] BEST  trimmed Chamfer {c:.2f} px   (orthographic best was {top[0][0]*Ur:.1f} px)")
    print(f"     focal length      {f:.0f} px")
    print(f"     camera distance   {t[2]:.0f} mm  (depth range of path along view axis: {np.ptp(zc):.1f} mm)")
    print(f"     perspective ratio z_max/z_min = {zc.max()/zc.min():.3f}  "
          f"-> {'perspective matters' if zc.max()/zc.min()>1.05 else 'near-orthographic'}")
    print(f"     view axis         {Rm[2].round(4)}")
    print(f"     FOV horizontal    {2*np.degrees(np.arctan(W/2/f)):.1f} deg")
    np.savez(DATA / "c4_camera.npz", rv=rv, t=t, f=f, c=C0, cost=c, R=Rm, U=U, X=X, Q=Q, z=zc)

    d, _ = cKDTree(Q).query(U)
    print(f"\n[C4] per-point residual px: median {np.median(d):.1f}  p75 {np.percentile(d,75):.1f}"
          f"  p90 {np.percentile(d,90):.1f}  max {d.max():.1f}")
    print(f"     fraction of track points within 10 px of the projected path: {100*np.mean(d<10):.0f}%")


if __name__ == "__main__":
    main()
