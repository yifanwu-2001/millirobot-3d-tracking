"""Stage C6 - joint fit of camera AND the traversed arc-length interval.

C4/C5 used a ONE-SIDED cost (every track point must lie near the projected
path). That is satisfied by a projected path far longer than the track, which
is exactly the failure we saw: the first ~28 mm of Path 2 had a 300 px residual
because nothing in the track corresponds to it.

Here we estimate, jointly:
    rv (3)  t (3)  log f (1)   -- the camera
    s_a, s_b (2)                -- the stretch of Path 2 the robot actually covered
under a SYMMETRIC trimmed cost, so the fitted interval cannot be longer than
what the track supports.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot
from config import DATA, OUT, W, H
from a_centerline import load as load_cl

TRIM = 0.90
C0 = np.array([W / 2., H / 2.])
s3, C3, T3, _ = load_cl()
L = s3[-1]


def curve(sa, sb, n=250):
    ss = np.linspace(sa, sb, n)
    return np.stack([np.interp(ss, s3, C3[:, c]) for c in range(3)], 1), ss


def project(X, rv, t, f):
    Xc = X @ Rot.from_rotvec(rv).as_matrix().T + t
    z = Xc[:, 2]
    if (z <= 1e-3).any(): return None, z
    return Xc[:, :2] / z[:, None] * f + C0, z


def trimmed(d, trim=TRIM):
    n = max(3, int(trim * len(d)))
    return float(np.sort(d)[:n].mean())


TREE_U = None   # KD-tree of the (fixed) 2D track, built once


def cost(p, U):
    rv, t, lf, sa, sb = p[:3], p[3:6], p[6], p[7], p[8]
    f = np.exp(lf)
    if not (0 <= sa < sb <= L) or (sb - sa) < 20 or t[2] <= 0 or f <= 50:
        return 1e6
    X, _ = curve(sa, sb)
    Q, z = project(X, rv, t, f)
    if Q is None or not np.isfinite(Q).all(): return 1e6
    d1, _ = cKDTree(Q).query(U)     # track -> curve
    d2, _ = TREE_U.query(Q)         # curve -> track   (this is what C4 was missing)
    return 0.5 * (trimmed(d1) + trimmed(d2))


def main():
    z3 = np.load(DATA / "c3_global_trend.npz")
    dirs, angs, top, U, Ur = z3["dirs"], z3["angs"], z3["top"], z3["U"], float(z3["Ur"])
    global TREE_U; TREE_U = cKDTree(U)

    print("[C6] joint camera + traversed-interval fit, symmetric trimmed cost\n")
    print(f"{'seed':>5}{'f0':>8}{'init sa,sb':>14}{'cost px':>10}{'f':>8}{'t_z mm':>9}"
          f"{'sa..sb mm':>14}{'FOV w mm':>10}")
    best = None
    inits = [(0, L), (0.25 * L, L), (0.15 * L, 0.85 * L)]
    for r in range(4):
        sc, di, ri = top[r]; di, ri = int(di), int(ri)
        v = dirs[di]; ang = angs[ri]
        a = np.array([0, 0, 1.]) if abs(v[2]) < .9 else np.array([1., 0, 0])
        e1 = np.cross(v, a); e1 /= np.linalg.norm(e1); e2 = np.cross(v, e1)
        Rz = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
        R0 = Rz @ np.stack([e1, e2, v])
        rv0 = Rot.from_matrix(R0).as_rotvec()
        Xf, _ = curve(0, L)
        Qo = Xf @ R0.T
        Qr = np.sqrt(((Qo[:, :2] - Qo[:, :2].mean(0)) ** 2).sum(1).mean())
        kpm = Ur / Qr
        for f0 in (1500., 5000., 25000.):
            tz = f0 / kpm
            t0 = np.r_[-Qo[:, :2].mean(0), tz - Qo[:, 2].mean()]
            for (sa0, sb0) in inits:
                p0 = np.r_[rv0, t0, np.log(f0), sa0, sb0]
                scale = np.r_[[.05]*3, [abs(tz)*.02]*3, [.05], [6.], [6.]]
                res = minimize(lambda q: cost(p0 + q*scale, U), np.zeros(9),
                               method="Nelder-Mead",
                               options=dict(maxiter=2500, xatol=1e-3, fatol=1e-3, adaptive=True))
                p = p0 + res.x * scale
                if best is None or res.fun < best[1]:
                    best = (p, res.fun)
                    f = np.exp(p[6]); X, _ = curve(p[7], p[8]); Q, zz = project(X, p[:3], p[3:6], f)
                    fovw = 2*zz.min()*np.tan(np.arctan(W/2/f))
                    print(f"{r:5d}{f0:8.0f}{f'{sa0:.0f},{sb0:.0f}':>14}{res.fun:10.2f}"
                          f"{f:8.0f}{p[5]:9.0f}{f'{p[7]:.0f}..{p[8]:.0f}':>14}{fovw:10.0f}")

    p, c = best
    rv, t, f, sa, sb = p[:3], p[3:6], np.exp(p[6]), p[7], p[8]
    X, ss = curve(sa, sb); Q, zz = project(X, rv, t, f)
    Rm = Rot.from_rotvec(rv).as_matrix()
    print(f"\n[C6] BEST symmetric trimmed cost {c:.2f} px")
    print(f"     traversed interval  s = {sa:.1f} .. {sb:.1f} mm   "
          f"({sb-sa:.1f} of {L:.1f} mm = {100*(sb-sa)/L:.0f}% of Path 2)")
    print(f"     focal length {f:.0f} px, FOV {2*np.degrees(np.arctan(W/2/f)):.1f} deg")
    print(f"     camera distance {zz.min():.0f}..{zz.max():.0f} mm, depth range {np.ptp(zz):.1f} mm")
    print(f"     visible width at nearest depth {2*zz.min()*np.tan(np.arctan(W/2/f)):.0f} mm")
    print(f"     view axis {Rm[2].round(4)}")
    d1, _ = cKDTree(Q).query(U); d2, _ = cKDTree(U).query(Q)
    print(f"\n[C6] track->curve px: median {np.median(d1):.1f}  p90 {np.percentile(d1,90):.1f}  max {d1.max():.1f}")
    print(f"     curve->track px: median {np.median(d2):.1f}  p90 {np.percentile(d2,90):.1f}  max {d2.max():.1f}")
    print(f"     scale at mid-depth: {f/np.median(zz):.2f} px/mm")
    np.savez(DATA / "c6_camera_trend.npz", rv=rv, t=t, f=f, c=C0, sa=sa, sb=sb,
             cost=c, R=Rm, U=U, X=X, Q=Q, z=zz, ss=ss)


if __name__ == "__main__":
    main()
