"""P1d - the residual is location-bound but neither an excursion (P1b) nor a
centerline sampling artefact (P1c). Is it RADIALLY structured, i.e. lens
distortion the C8 ablation failed to isolate?

Lead: all three bad episodes sit in the lower-left of the frame, 284-324 px
from the image centre, while the well-matched track runs up the diagonal to the
upper right. And C8's 11-parameter fit pushed the principal point to (348, 574)
- also lower-left, i.e. the optimiser dragged the principal point TOWARD the
region it could not otherwise fit. That is what a distortion-shaped error does
to a model that has no distortion term.

C8 did test distortion (13 params, 5.71 px, worse than 11 params' 4.92 px) but
it freed the principal point and k1/k2 SIMULTANEOUSLY. Those two are strongly
coupled - a shifted principal point mimics low-order radial distortion - so
that ablation cannot separate them. Here the principal point is held at the
image centre so k1/k2 have to do the work alone.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot
from config import DATA, W, H
from a_centerline import load as load_cl

fin = np.load(DATA / "i_final.npz")
UV, Pg, scale = fin["UV"], fin["Pg"], float(fin["scale"])
p1 = np.load(DATA / "p1_excursion.npz")
d_min, nomatch, fail = p1["d_min"], p1["nomatch"], p1["fail"]
ctr = np.array([W / 2., H / 2.])
rad = np.linalg.norm(UV - ctr, axis=1)

print("[P1d] does the off-curve residual grow with radius from the image centre?")
qs = np.percentile(rad, [0, 20, 40, 60, 80, 100])
print(f"{'radius band px':>18}{'n':>7}{'median off-curve px':>22}{'no-match rate':>16}")
for i in range(5):
    m = (rad >= qs[i]) & (rad <= qs[i + 1])
    print(f"{f'{qs[i]:.0f}-{qs[i+1]:.0f}':>18}{m.sum():7d}{np.median(d_min[m]):22.1f}"
          f"{100*nomatch[m].mean():15.1f}%")
r_bad, r_good = rad[nomatch], rad[~fail]
print(f"\n      radius of no-match frames : median {np.median(r_bad):.0f} px")
print(f"      radius of matched frames  : median {np.median(r_good):.0f} px")
c = np.corrcoef(rad, np.log10(np.maximum(d_min, .1)))[0, 1]
print(f"      corr(radius, log off-curve) = {c:+.2f}")

# ---------------------------------------------------------------- refit test
s3, C3, _, _ = load_cl(); L = s3[-1]
z3 = np.load(DATA / "c3_global_trend.npz")
U = z3["U"]; TREE_U = cKDTree(U)
TRIM = .90


def curve(sa, sb, n=250):
    ss = np.linspace(sa, sb, n)
    return np.stack([np.interp(ss, s3, C3[:, c]) for c in range(3)], 1)


def proj(X, rv, t, f, cx, cy, k1, k2):
    Xc = X @ Rot.from_rotvec(rv).as_matrix().T + t
    z = Xc[:, 2]
    if (z <= 1e-3).any(): return None
    xn = Xc[:, :2] / z[:, None]
    r2 = (xn ** 2).sum(1)
    return xn * (1 + k1 * r2 + k2 * r2 ** 2)[:, None] * f + np.array([cx, cy])


def trimmed(d): return float(np.sort(d)[:max(3, int(TRIM * len(d)))].mean())


def cost(p, mode):
    rv, t, lf, sa, sb = p[:3], p[3:6], p[6], p[7], p[8]
    if mode == "pp":       cx, cy, k1, k2 = p[9], p[10], 0., 0.
    elif mode == "dist":   cx, cy, k1, k2 = W/2., H/2., p[9], p[10]
    else:                  cx, cy, k1, k2 = p[9], p[10], p[11], p[12]
    f = np.exp(lf)
    if not (0 <= sa < sb <= L) or (sb-sa) < 20 or t[2] <= 0 or f <= 50: return 1e6
    if abs(k1) > 1.5 or abs(k2) > 3.0: return 1e6
    if not (0.2*W < cx < 0.8*W and 0.2*H < cy < 0.8*H): return 1e6
    Q = proj(curve(sa, sb), rv, t, f, cx, cy, k1, k2)
    if Q is None or not np.isfinite(Q).all() or np.abs(Q).max() > 1e5: return 1e6
    d1, _ = cKDTree(Q).query(U); d2, _ = TREE_U.query(Q)
    return .5 * (trimmed(d1) + trimmed(d2))


p11 = np.load(DATA / "c8_11.npy")
print(f"\n[P1d] refit with the principal point PINNED to the image centre, so k1/k2")
print(f"      cannot hide inside a shifted principal point")
print(f"{'model':<44}{'free params':>13}{'trimmed residual':>19}")
seeds = [p11[:9]]
for mode, npar, label in [("pp", 11, "free principal point (C8's 11-param)"),
                          ("dist", 11, "distortion k1,k2, principal point PINNED"),
                          ("both", 13, "free principal point + distortion")]:
    best = None
    for s0 in seeds:
        if mode == "pp":   p0 = np.r_[s0, p11[9], p11[10]]
        elif mode == "dist": p0 = np.r_[s0, 0., 0.]
        else:              p0 = np.r_[s0, p11[9], p11[10], 0., 0.]
        scl = np.r_[[.05]*3, [abs(s0[5])*.02]*3, [.05], [6.], [6.]]
        scl = np.r_[scl, ([20., 20.] if mode == "pp" else [.05, .05] if mode == "dist"
                          else [20., 20., .05, .05])]
        for trial in range(3):
            r = minimize(lambda q: cost(p0 + q*scl, mode), np.zeros(len(p0)),
                         method="Nelder-Mead",
                         options=dict(maxiter=4000, xatol=1e-3, fatol=1e-3, adaptive=True))
            p0 = p0 + r.x * scl
            if best is None or r.fun < best[0]: best = (r.fun, p0.copy())
    print(f"{label:<44}{npar:>13}{best[0]:19.2f}")
    if mode == "dist":
        print(f"{'':<44}{'':>13}   k1={best[1][9]:+.4f} k2={best[1][10]:+.4f}")
    np.save(DATA / f"p1d_{mode}.npy", best[1])
print(f"\n[P1d] if 'distortion, PP pinned' beats 'free PP', the 251 px principal-point")
print(f"      offset was standing in for distortion all along.")
