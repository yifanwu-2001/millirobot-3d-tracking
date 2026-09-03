"""Stage C8 - decisive test: is the residual the CAMERA MODEL or the PATH?

Adds radial distortion (k1,k2) and a free principal point on top of the C7 model
and re-optimises. Interpretation:
  residual collapses  -> the camera model was the limitation, Path 2 is correct
  residual unchanged  -> Path 2 genuinely deviates from the route travelled
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot
from config import DATA, OUT, W, H
from a_centerline import load as load_cl

TRIM = .90
s3, C3, _, _ = load_cl(); L = s3[-1]
z3 = np.load(DATA / "c3_global_trend.npz")
dirs, angs, top, U, Ur = z3["dirs"], z3["angs"], z3["top"], z3["U"], float(z3["Ur"])
TREE_U = cKDTree(U)


def curve(sa, sb, n=250):
    ss = np.linspace(sa, sb, n)
    return np.stack([np.interp(ss, s3, C3[:, c]) for c in range(3)], 1)


def proj(X, rv, t, f, cx, cy, k1, k2):
    Xc = X @ Rot.from_rotvec(rv).as_matrix().T + t
    z = Xc[:, 2]
    if (z <= 1e-3).any(): return None
    xn = Xc[:, :2] / z[:, None]
    r2 = (xn ** 2).sum(1)
    xn = xn * (1 + k1 * r2 + k2 * r2 ** 2)[:, None]
    return xn * f + np.array([cx, cy])


def trimmed(d): return float(np.sort(d)[:max(3, int(TRIM*len(d)))].mean())


def make_cost(nparam):
    def cost(p):
        rv, t, lf, sa, sb = p[:3], p[3:6], p[6], p[7], p[8]
        cx, cy = (p[9], p[10]) if nparam > 9 else (W/2., H/2.)
        k1, k2 = (p[11], p[12]) if nparam > 11 else (0., 0.)
        f = np.exp(lf)
        if not (0 <= sa < sb <= L) or (sb-sa) < 20 or t[2] <= 0 or f <= 50: return 1e6
        if abs(k1) > 1.5 or abs(k2) > 3.0: return 1e6
        if not (0.2*W < cx < 0.8*W and 0.2*H < cy < 0.8*H): return 1e6
        Q = proj(curve(sa, sb), rv, t, f, cx, cy, k1, k2)
        if Q is None or not np.isfinite(Q).all() or np.abs(Q).max() > 1e5: return 1e6
        d1, _ = cKDTree(Q).query(U); d2, _ = TREE_U.query(Q)
        return .5*(trimmed(d1) + trimmed(d2))
    return cost


def seeds():
    out = []
    for r in range(4):
        sc, di, ri = top[r]; di, ri = int(di), int(ri)
        v = dirs[di]; ang = angs[ri]
        a = np.array([0,0,1.]) if abs(v[2]) < .9 else np.array([1.,0,0])
        e1 = np.cross(v, a); e1 /= np.linalg.norm(e1); e2 = np.cross(v, e1)
        Rz = np.array([[np.cos(ang), -np.sin(ang), 0],[np.sin(ang), np.cos(ang), 0],[0,0,1]])
        R0 = Rz @ np.stack([e1, e2, v])
        Qo = curve(0, L) @ R0.T
        kpm = Ur/np.sqrt(((Qo[:,:2]-Qo[:,:2].mean(0))**2).sum(1).mean())
        for f0 in (1200., 2500., 8000.):
            tz = f0/kpm
            out.append((Rot.from_matrix(R0).as_rotvec(), np.r_[-Qo[:,:2].mean(0), tz-Qo[:,2].mean()], f0, tz))
    return out


print("[C8] does a richer camera model explain the residual?\n")
print(f"{'model':<44}{'params':>8}{'cost px':>10}{'improvement':>13}")
base = None
for name, npar in [("pinhole, fixed principal point", 9),
                   ("+ free principal point", 11),
                   ("+ radial distortion k1,k2", 13)]:
    best = None
    for rv0, t0, f0, tz in seeds():
        for (sa0, sb0) in [(0, L), (.15*L, .95*L)]:
            p0 = np.r_[rv0, t0, np.log(f0), sa0, sb0, W/2., H/2., 0., 0.][:npar]
            scl = np.r_[[.05]*3, [abs(tz)*.02]*3, [.05], [6.], [6.], [20.], [20.], [.05], [.05]][:npar]
            res = minimize(lambda q: make_cost(npar)(np.r_[p0+q*scl, np.zeros(13-npar)]),
                           np.zeros(npar), method="Nelder-Mead",
                           options=dict(maxiter=3500, xatol=1e-3, fatol=1e-3, adaptive=True))
            if best is None or res.fun < best[0]: best = (res.fun, p0+res.x*scl)
    if base is None: base = best[0]
    print(f"{name:<44}{npar:>8}{best[0]:10.2f}{100*(1-best[0]/base):12.0f}%")
    np.save(DATA / f"c8_{npar}.npy", best[1])
    if npar == 13:
        p = best[1]
        print(f"\n     fitted principal point ({p[9]:.0f}, {p[10]:.0f})  vs image centre ({W/2:.0f}, {H/2:.0f})")
        print(f"     fitted distortion k1={p[11]:+.4f}  k2={p[12]:+.4f}")
        print(f"     focal {np.exp(p[6]):.0f} px, traversed s = {p[7]:.1f}..{p[8]:.1f} mm")
