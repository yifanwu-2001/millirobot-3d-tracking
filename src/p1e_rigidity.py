"""P1e - is the localised residual a violation of the RIGID-BODY assumption?

Refuted so far: branch excursion (P1b), centerline undersampling (P1c), radial
lens distortion (P1d). What is left has this signature: location-bound, ~6 mm,
confined to s ~ 110-170 mm, not radial, while 90% of the track sits 4.3 px
(0.66 mm) from the projected curve.

An intrinsic-calibration error cannot produce that. A wrong focal length warps
the projection SMOOTHLY and GLOBALLY - it cannot leave most of the path at
sub-millimetre agreement and two specific stretches 6 mm out.

A violated rigid-body assumption can. Stage C fits ONE rigid transform for the
whole model. If the silicone phantom sat differently while filming than when
Path 2.csv was measured (mounted in a flow rig vs. scanned), a global rigid fit
lands on the bulk of the path and misses wherever the model deformed.

Test: fit the camera to the BAD stretch alone. If a rigid transform explains it
perfectly on its own but is incompatible with the transform that explains the
rest, the geometry moved between measurement and filming.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot
from config import DATA, W, H
from a_centerline import load as load_cl

s3, C3, _, _ = load_cl(); L = s3[-1]
fin = np.load(DATA / "i_final.npz")
UV, Pg, sg, scale = fin["UV"], fin["Pg"], fin["sg"], float(fin["scale"])
p1 = np.load(DATA / "p1_excursion.npz"); nomatch = p1["nomatch"]; d_min = p1["d_min"]
_, jn = cKDTree(Pg).query(UV); s_at = sg[jn]

BAD = (100.0, 190.0)      # the stretch carrying the no-match frames
GOOD = (0.0, 100.0)


def curve(sa, sb, n=250):
    ss = np.linspace(sa, sb, n)
    return np.stack([np.interp(ss, s3, C3[:, c]) for c in range(3)], 1)


def proj(X, p):
    rv, t, f, cx, cy = p[:3], p[3:6], np.exp(p[6]), p[7], p[8]
    Xc = X @ Rot.from_rotvec(rv).as_matrix().T + t
    z = Xc[:, 2]
    if (z <= 1e-3).any(): return None
    return Xc[:, :2] / z[:, None] * f + np.array([cx, cy])


def fit(obs, sa, sb, seed, iters=3):
    T = cKDTree(obs)
    def cost(p):
        if np.exp(p[6]) <= 50 or p[5] <= 0: return 1e6
        if not (0.2*W < p[7] < 0.8*W and 0.2*H < p[8] < 0.8*H): return 1e6
        Q = proj(curve(sa, sb), p)
        if Q is None or not np.isfinite(Q).all() or np.abs(Q).max() > 1e5: return 1e6
        d1, _ = cKDTree(Q).query(obs); d2, _ = T.query(Q)
        n1 = max(3, int(.9*len(d1))); n2 = max(3, int(.9*len(d2)))
        return .5*(np.sort(d1)[:n1].mean() + np.sort(d2)[:n2].mean())
    p0 = seed.copy()
    scl = np.r_[[.03]*3, [abs(seed[5])*.015]*3, [.03], [15.], [15.]]
    best = (cost(p0), p0.copy())
    # Explicit initial simplex. Starting Nelder-Mead at x0=zeros with the default
    # simplex gives every coordinate a step of 2.5e-4, which with fatol=1e-3
    # terminates before the search moves at all - that silently returned the seed.
    for it in range(iters):
        simp = np.vstack([np.zeros(9), np.eye(9)])
        r = minimize(lambda q: cost(p0 + q*scl), np.zeros(9), method="Nelder-Mead",
                     options=dict(maxiter=6000, xatol=1e-4, fatol=1e-4,
                                  adaptive=True, initial_simplex=simp))
        p0 = p0 + r.x*scl
        if r.fun < best[0] - 1e-9: best = (r.fun, p0.copy())
    return best


p11 = np.load(DATA / "c8_11.npy")
seed = np.r_[p11[:7], p11[9], p11[10]]

in_bad = (s_at >= BAD[0]) & (s_at <= BAD[1])
in_good = (s_at >= GOOD[0]) & (s_at < GOOD[1])
print(f"[P1e] observations nearest s in {GOOD}: {in_good.sum()} frames")
print(f"      observations nearest s in {BAD}: {in_bad.sum()} frames "
      f"({nomatch[in_bad].sum()} of them no-match)")

print(f"\n[P1e] fit ONE rigid camera to each stretch separately")
print(f"{'fitted on':<26}{'own residual px':>18}{'f':>10}{'principal point':>20}")
res = {}
for name, sel, (sa, sb) in [("global (Stage C8)", None, (0., L)),
                            ("good stretch only", in_good, GOOD),
                            ("bad stretch only", in_bad, BAD)]:
    if sel is None:
        p = seed; c = None
    else:
        c, p = fit(UV[sel], sa, sb, seed)
    res[name] = p
    own = c if c is not None else np.nan
    print(f"{name:<26}{own:18.2f}{np.exp(p[6]):10.0f}"
          f"{f'({p[7]:.0f}, {p[8]:.0f})':>20}")

print(f"\n[P1e] does the BAD stretch admit a good rigid fit of its own?")
cb, pb = fit(UV[in_bad], *BAD, seed)
Qb = proj(curve(*BAD), pb)
db, _ = cKDTree(Qb).query(UV[in_bad])
Qg = proj(curve(*BAD), seed)
dg, _ = cKDTree(Qg).query(UV[in_bad])
print(f"      with the GLOBAL camera : median {np.median(dg):6.1f} px = {np.median(dg)/scale:5.2f} mm")
print(f"      with its OWN camera    : median {np.median(db):6.1f} px = {np.median(db)/scale:5.2f} mm")
print(f"      -> improvement {np.median(dg)/max(np.median(db),1e-6):.1f}x")

print(f"\n[P1e] how far apart are the two cameras?")
Rg = Rot.from_rotvec(seed[:3]).as_matrix(); Rb = Rot.from_rotvec(pb[:3]).as_matrix()
ang = np.degrees(np.arccos(np.clip((np.trace(Rg.T @ Rb) - 1) / 2, -1, 1)))
print(f"      rotation differs by {ang:.1f} deg, translation by "
      f"{np.linalg.norm(seed[3:6]-pb[3:6]):.1f} mm, f by "
      f"{abs(np.exp(seed[6])-np.exp(pb[6])):.0f} px")
print(f"      -> 50 deg is NOT a deformed phantom, it is a different camera. Fitting 9")
print(f"         parameters to 90 mm of curve is under-constrained (M1 saw the same")
print(f"         thing on a half-leg refit), so the free fit above proves nothing.")

# ------------------------------------------------------------- constrained test
print(f"\n[P1e-b] CONSTRAINED test: fix f and the principal point at the global values,")
print(f"        allow only a SMALL rigid perturbation of the model (<=8 deg, <=8 mm).")
print(f"        A phantom that sagged in its mount is a small perturbation. Overfitting")
print(f"        needs a large one.")

Xbad = curve(*BAD)
TU = cKDTree(UV[in_bad])
f_g, cx_g, cy_g = np.exp(seed[6]), seed[7], seed[8]
Rg0 = Rot.from_rotvec(seed[:3]).as_matrix(); tg0 = seed[3:6]


def perturbed_cost(q, lim_deg, lim_mm):
    dr = q[:3] * (np.radians(lim_deg) / 1.0)
    dt = q[3:6] * (lim_mm / 1.0)
    if np.linalg.norm(q[:3]) > 1 or np.linalg.norm(q[3:6]) > 1: return 1e6
    Xw = (Xbad - Xbad.mean(0)) @ Rot.from_rotvec(dr).as_matrix().T + Xbad.mean(0) + dt
    Xc = Xw @ Rg0.T + tg0
    if (Xc[:, 2] <= 1e-3).any(): return 1e6
    Q = Xc[:, :2] / Xc[:, 2:3] * f_g + np.array([cx_g, cy_g])
    if not np.isfinite(Q).all(): return 1e6
    d1, _ = cKDTree(Q).query(UV[in_bad]); d2, _ = TU.query(Q)
    n1 = max(3, int(.9*len(d1))); n2 = max(3, int(.9*len(d2)))
    return .5*(np.sort(d1)[:n1].mean() + np.sort(d2)[:n2].mean())


print(f"{'allowed perturbation':<28}{'residual px':>13}{'mm':>8}{'vs global':>12}")
base = perturbed_cost(np.zeros(6), 8, 8)
print(f"{'none (global fit)':<28}{base:13.2f}{base/scale:8.2f}{'1.0x':>12}")
for lim_deg, lim_mm in [(2, 2), (5, 5), (8, 8), (15, 15)]:
    best = (base, None)
    for trial in range(4):
        x0 = np.zeros(6) if trial == 0 else np.random.default_rng(trial).normal(0, .3, 6)
        simp = np.vstack([x0, x0 + np.eye(6) * .35])
        r = minimize(lambda q: perturbed_cost(q, lim_deg, lim_mm), x0,
                     method="Nelder-Mead",
                     options=dict(maxiter=4000, xatol=1e-4, fatol=1e-4,
                                  adaptive=True, initial_simplex=simp))
        if r.fun < best[0]: best = (r.fun, r.x)
    print(f"{f'<= {lim_deg} deg, {lim_mm} mm':<28}{best[0]:13.2f}{best[0]/scale:8.2f}"
          f"{base/max(best[0],1e-9):11.1f}x")

print(f"\n[P1e] READING: if a <=5 deg / <=5 mm perturbation already recovers most of the")
print(f"      6.2 px gap, local deformation of the phantom is a live explanation and a")
print(f"      checkerboard would NOT fix it. If it takes 15 deg, the residual is not")
print(f"      deformation either, and the cause remains unidentified.")
np.savez(DATA / "p1e_rigidity.npz", p_bad=pb, p_seed=seed, d_bad_global=dg, d_bad_own=db)
