"""Stage L2 - if f cannot be pinned down, does the data even NEED it?

A weak-perspective (scaled orthographic) camera drops the 1/Z term entirely:
image = s * (R @ X)[:2] + c, with a single global scale s replacing f/Z. It has
NO depth-dependent term at all, so it cannot even express the size-modulation
L1 went looking for - if it fits the observed 2D track about as well as the
best perspective model (c8_11, 4.92 px, C's table), that is direct evidence
the 2D data does not require - or usefully constrain - a perspective camera at
all, independent of the size-cue argument in L1.
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
dirs, angs, top, U = z3["dirs"], z3["angs"], z3["top"], z3["U"]
TREE_U = cKDTree(U)


def curve(sa, sb, n=250):
    ss = np.linspace(sa, sb, n)
    return np.stack([np.interp(ss, s3, C3[:, c]) for c in range(3)], 1)


def trimmed(d): return float(np.sort(d)[:max(3, int(TRIM*len(d)))].mean())


def proj_affine(X, rv, sc, cxy):
    Xc = X @ Rot.from_rotvec(rv).as_matrix().T
    return Xc[:, :2] * sc + cxy


def cost_affine(p):
    rv, lsc, cx, cy, sa, sb = p[:3], p[3], p[4], p[5], p[6], p[7]
    if not (0 <= sa < sb <= L) or (sb - sa) < 20: return 1e6
    Q = proj_affine(curve(sa, sb), rv, np.exp(lsc), np.array([cx, cy]))
    if not np.isfinite(Q).all(): return 1e6
    d1, _ = cKDTree(Q).query(U); d2, _ = TREE_U.query(Q)
    return .5*(trimmed(d1) + trimmed(d2))


print("[L2] weak-perspective (affine) camera: 8 free params (R 3, log-scale 1,")
print("     principal point 2, traversed interval sa/sb 2) - NO depth dependence at all\n")

best = None
for r in range(6):
    sc0, di, ri = top[r]; di, ri = int(di), int(ri)
    v = dirs[di]; ang = angs[ri]
    a = np.array([0, 0, 1.]) if abs(v[2]) < .9 else np.array([1., 0, 0])
    e1 = np.cross(v, a); e1 /= np.linalg.norm(e1); e2 = np.cross(v, e1)
    Rz = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    R0 = Rz @ np.stack([e1, e2, v])
    Qo = curve(0, L) @ R0.T
    Ur = np.sqrt(((U - U.mean(0))**2).sum(1).mean())
    kpm = Ur / np.sqrt(((Qo[:,:2]-Qo[:,:2].mean(0))**2).sum(1).mean())
    for sa0, sb0 in [(0, L), (.15*L, .95*L)]:
        p0 = np.r_[Rot.from_matrix(R0).as_rotvec(), np.log(kpm), U.mean(0) - kpm*Qo[:,:2].mean(0), sa0, sb0]
        scl = np.r_[[.05]*3, [.05], [15.], [15.], [6.], [6.]]
        res = minimize(lambda q: cost_affine(p0+q*scl), np.zeros(8), method="Nelder-Mead",
                       options=dict(maxiter=3000, xatol=1e-3, fatol=1e-3, adaptive=True))
        if best is None or res.fun < best[0]: best = (res.fun, p0+res.x*scl)

cbest, pbest = best
print(f"weak-perspective (affine), 8 params           {cbest:8.2f} px")
print(f"\n comparison table (from README / C8):")
print(f"  pinhole, fixed principal point,  9 params    9.37 px")
print(f"  pinhole, + free principal point, 11 params   4.92 px")
print(f"  pinhole, + radial distortion,    13 params   5.71 px")
print(f"  weak-perspective (affine),        8 params  {cbest:6.2f} px   <- this stage")

if cbest < 6.0:
    print(f"\n[L2] READING: an affine camera with FEWER parameters than the free-PP pinhole")
    print(f"     matches or beats it. Perspective depth (f) is not just unidentifiable here -")
    print(f"     it is not needed to explain the observed 2D track at all. This is the cleanest")
    print(f"     resolution of the C7-C10 degeneracy: stop trying to recover f, and report the")
    print(f"     3D trajectory's invariance to it directly (see also L1, m1_holdout f-invariance).")
else:
    print(f"\n[L2] READING: the affine model fits meaningfully worse than the free-PP pinhole -")
    print(f"     perspective (some real 1/Z falloff) IS needed to explain the 2D shape, even")
    print(f"     though f itself stays unidentified. The degeneracy is in WHICH f, not IN f vs no-f.")

rv, lsc, cx, cy, sa, sb = pbest[:3], pbest[3], pbest[4], pbest[5], pbest[6], pbest[7]
print(f"\n     fitted: scale {np.exp(lsc):.3f} px/mm, principal ({cx:.0f},{cy:.0f}), s={sa:.1f}..{sb:.1f} mm")
np.save(DATA / "l2_affine.npy", pbest)
print("\n[L2] saved data/l2_affine.npy")
