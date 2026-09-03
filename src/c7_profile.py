"""Stage C7 - is the camera identifiable from ONE centerline? Profile the cost.

C6 returned two very different cameras with nearly identical cost (f=4735/FOV12
and f=1015/FOV51). Rather than pick one, fix the focal length on a grid and
re-optimise everything else, to map out how flat the valley really is, and to
see which quantities ARE pinned down.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot
from config import DATA, OUT, W, H
from a_centerline import load as load_cl

TRIM = .90; C0 = np.array([W/2., H/2.])
s3, C3, _, _ = load_cl(); L = s3[-1]
z3 = np.load(DATA / "c3_global_trend.npz")
dirs, angs, top, U, Ur = z3["dirs"], z3["angs"], z3["top"], z3["U"], float(z3["Ur"])
TREE_U = cKDTree(U)


def curve(sa, sb, n=250):
    ss = np.linspace(sa, sb, n)
    return np.stack([np.interp(ss, s3, C3[:, c]) for c in range(3)], 1)


def proj(X, rv, t, f):
    Xc = X @ Rot.from_rotvec(rv).as_matrix().T + t
    z = Xc[:, 2]
    if (z <= 1e-3).any(): return None, z
    return Xc[:, :2] / z[:, None] * f + C0, z


def trimmed(d): return float(np.sort(d)[:max(3, int(TRIM*len(d)))].mean())


def cost_fixed_f(p, f):
    rv, t, sa, sb = p[:3], p[3:6], p[6], p[7]
    if not (0 <= sa < sb <= L) or (sb-sa) < 20 or t[2] <= 0: return 1e6
    X = curve(sa, sb); Q, z = proj(X, rv, t, f)
    if Q is None or not np.isfinite(Q).all(): return 1e6
    d1, _ = cKDTree(Q).query(U); d2, _ = TREE_U.query(Q)
    return .5*(trimmed(d1) + trimmed(d2))


print("[C7] focal-length profile: fix f, re-optimise pose + traversed interval\n")
print(f"{'f px':>8}{'FOV deg':>9}{'cost px':>9}{'t_z mm':>9}{'depth mm':>10}"
      f"{'sa..sb':>13}{'px/mm':>8}   view axis")
rows = []
for f in (600, 800, 1000, 1400, 2000, 3000, 5000, 8000, 15000, 40000):
    best = None
    for r in range(4):
        sc, di, ri = top[r]; di, ri = int(di), int(ri)
        v = dirs[di]; ang = angs[ri]
        a = np.array([0,0,1.]) if abs(v[2]) < .9 else np.array([1.,0,0])
        e1 = np.cross(v, a); e1 /= np.linalg.norm(e1); e2 = np.cross(v, e1)
        Rz = np.array([[np.cos(ang), -np.sin(ang), 0],[np.sin(ang), np.cos(ang), 0],[0,0,1]])
        R0 = Rz @ np.stack([e1, e2, v]); rv0 = Rot.from_matrix(R0).as_rotvec()
        Qo = curve(0, L) @ R0.T
        Qr = np.sqrt(((Qo[:,:2]-Qo[:,:2].mean(0))**2).sum(1).mean()); kpm = Ur/Qr
        tz = f/kpm
        for (sa0, sb0) in [(0, L), (0.15*L, 0.95*L)]:
            p0 = np.r_[rv0, -Qo[:,:2].mean(0), tz-Qo[:,2].mean(), sa0, sb0]
            sc_ = np.r_[[.05]*3, [abs(tz)*.02]*3, [6.], [6.]]
            res = minimize(lambda q: cost_fixed_f(p0+q*sc_, f), np.zeros(8),
                           method="Nelder-Mead",
                           options=dict(maxiter=2000, xatol=1e-3, fatol=1e-3, adaptive=True))
            if best is None or res.fun < best[0]: best = (res.fun, p0+res.x*sc_)
    c, p = best
    X = curve(p[6], p[7]); Q, zz = proj(X, p[:3], p[3:6], f)
    va = Rot.from_rotvec(p[:3]).as_matrix()[2]
    print(f"{f:8d}{2*np.degrees(np.arctan(W/2/f)):9.1f}{c:9.2f}{p[5]:9.0f}{np.ptp(zz):10.1f}"
          f"{f'{p[6]:.0f}..{p[7]:.0f}':>13}{f/np.median(zz):8.2f}   {va.round(3)}")
    rows.append((f, c, p, va, np.ptp(zz), f/np.median(zz)))

rows_a = np.array([(r[0], r[1], r[4], r[5]) for r in rows])
cmin = rows_a[:,1].min()
ok = rows_a[rows_a[:,1] < cmin*1.15]
print(f"\n[C7] cost minimum {cmin:.2f} px; focal lengths within 15% of it: "
      f"{ok[:,0].min():.0f} .. {ok[:,0].max():.0f} px  (ratio {ok[:,0].max()/ok[:,0].min():.0f}x)")
print(f"     => FOCAL LENGTH IS NOT IDENTIFIABLE from a single centerline.")
print(f"     image scale px/mm across those solutions: {ok[:,3].min():.2f} .. {ok[:,3].max():.2f}"
      f"  (spread {100*(ok[:,3].max()/ok[:,3].min()-1):.0f}%)  <- this IS well determined")
vas = np.array([r[3] for r in rows])[rows_a[:,1] < cmin*1.15]
ang_spread = np.degrees(np.arccos(np.clip(np.abs(vas @ vas[0]), -1, 1)))
print(f"     view-axis spread across those solutions: {ang_spread.max():.1f} deg")
np.save(DATA / "c7_profile.npy", rows_a)
