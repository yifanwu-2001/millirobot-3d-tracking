"""Stage M1 - out-of-sample validation: was Stage C's camera fit checked on data
it never saw?

Stage C (c1..c10) registers the camera ONLY against the outbound leg's 2D shape
(track2d_trend.py's `trend[:k_turn+1]`, ICP against Path 2). The return leg
(k_turn+1..N) never enters calibration - the HMM only sees it downstream, at
Stage I/I2. So splitting i_final's per-frame reprojection error by leg is a
genuine held-out check, already implicit in the pipeline but never reported.

We also ask whether the gap shrinks with more calibration data (overfitting to
a short outbound leg) or stays fixed (a leg-specific effect - e.g. the robot's
appearance/orientation differs outbound vs return, or the arc-length coverage
differs) by refitting calibration on HALF the outbound leg and comparing the
gap on the other half vs the full return leg.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot
from config import DATA, OUT, W, H
from a_centerline import load as load_cl
from e_hmm import ArcHMM

s3, C3, _, _ = load_cl(); L = s3[-1]
zt = np.load(DATA / "track2d_trend.npz"); trend = zt["trend"]; k_turn = int(zt["k_turn"])
z3 = np.load(DATA / "c3_global_trend.npz")
dirs, angs, top = z3["dirs"], z3["angs"], z3["top"]

print(f"[M1] outbound frames 0..{k_turn} ({k_turn+1}), return frames {k_turn+1}..{len(trend)-1} "
      f"({len(trend)-1-k_turn})\n")

# ---------------------------------------------------------------------------
# Part 1: the fit already in hand (c8_11) - split its OWN reprojection by leg.
# This is the headline number: zero new fitting, just re-slicing i_final.npz.
# ---------------------------------------------------------------------------
fin = np.load(DATA / "i_final.npz"); rp = fin["rp"]
vit = np.load(DATA / "i2_viterbi.npz"); rp_vit = vit["rp_vit"]
A = slice(0, k_turn + 1); B = slice(k_turn + 1, len(rp))
print("[M1] Part 1 - c8_11 camera, calibrated on outbound ONLY, scored on both legs")
print(f"{'':<24}{'reproj px med':>15}{'p90':>8}{'>30px%':>9}")
for nm, r in [("causal, outbound (fit)", rp[A]), ("causal, return (held out)", rp[B]),
              ("Viterbi, outbound (fit)", rp_vit[A]), ("Viterbi, return (held out)", rp_vit[B])]:
    print(f"{nm:<24}{np.median(r):15.1f}{np.percentile(r,90):8.1f}{100*np.mean(r>30):9.1f}")
gap = np.median(rp[B]) / np.median(rp[A])
print(f"\n[M1] return-leg median error is {gap:.1f}x the outbound (calibration-set) median.")
print("     This is the honest generalisation number for the C8_11 camera + Path 2 fit.")

# ---------------------------------------------------------------------------
# Part 2: does the gap shrink with more calibration data, or is it structural?
# Refit calibration (C8's free-PP model, 11 params) on the FIRST HALF of the
# outbound leg only, and score on (a) the second half of outbound (b) all of
# return. If more/less calibration data barely moves the held-out number, the
# gap is not an overfitting artefact of a short calibration arc.
# ---------------------------------------------------------------------------
TRIM = .90
C0 = np.array([W / 2., H / 2.])


def curve(sa, sb, n=250):
    ss = np.linspace(sa, sb, n)
    return np.stack([np.interp(ss, s3, C3[:, c]) for c in range(3)], 1)


def proj11(X, rv, t, f, cx, cy):
    Xc = X @ Rot.from_rotvec(rv).as_matrix().T + t
    z = Xc[:, 2]
    if (z <= 1e-3).any(): return None
    return Xc[:, :2] / z[:, None] * f + np.array([cx, cy])


def trimmed(d): return float(np.sort(d)[:max(3, int(TRIM * len(d)))].mean())


def fit_calibration(U, extra_seed=None):
    """Same free-principal-point model as C8_11, ICP against curve(sa,sb) vs U.
    extra_seed: an (11,) param vector from a known-good fit on a different U
    (e.g. the full-outbound solution), added as an additional starting point -
    a short leg may not have enough shape diversity for blind global search to
    find the basin at all, which is a search-difficulty artefact, not a
    generalisation signal.
    """
    Ur = np.sqrt(((U - U.mean(0)) ** 2).sum(1).mean())
    tree_u = cKDTree(U)

    def cost(p):
        rv, t, lf, sa, sb, cx, cy = p[:3], p[3:6], p[6], p[7], p[8], p[9], p[10]
        f = np.exp(lf)
        if not (0 <= sa < sb <= L) or (sb - sa) < 20 or t[2] <= 0 or f <= 50: return 1e6
        if not (0.1 * W < cx < 0.9 * W and 0.1 * H < cy < 0.9 * H): return 1e6
        Q = proj11(curve(sa, sb), rv, t, f, cx, cy)
        if Q is None or not np.isfinite(Q).all() or np.abs(Q).max() > 1e5: return 1e6
        d1, _ = cKDTree(Q).query(U); d2, _ = tree_u.query(Q)
        return .5 * (trimmed(d1) + trimmed(d2))

    best = None
    seeds = []
    for r in range(6):
        sc, di, ri = top[r]; di, ri = int(di), int(ri)
        v = dirs[di]; ang = angs[ri]
        a = np.array([0, 0, 1.]) if abs(v[2]) < .9 else np.array([1., 0, 0])
        e1 = np.cross(v, a); e1 /= np.linalg.norm(e1); e2 = np.cross(v, e1)
        Rz = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
        R0 = Rz @ np.stack([e1, e2, v])
        Qo = curve(0, L) @ R0.T
        kpm = Ur / np.sqrt(((Qo[:, :2] - Qo[:, :2].mean(0)) ** 2).sum(1).mean())
        for f0 in (1200., 2500.):
            tz = f0 / kpm
            for sa0, sb0 in [(0, L), (.15 * L, .95 * L)]:
                p0 = np.r_[Rot.from_matrix(R0).as_rotvec(), -Qo[:, :2].mean(0), tz - Qo[:, 2].mean(),
                          np.log(f0), sa0, sb0, W / 2., H / 2.]
                scl = np.r_[[.05]*3, [abs(tz)*.02]*3, [.05], [6.], [6.], [20.], [20.]]
                seeds.append((p0, scl))
    if extra_seed is not None:
        p0 = extra_seed.copy()
        scl = np.r_[[.05]*3, [abs(p0[3:6]).max()*.02]*3, [.1], [4.], [4.], [15.], [15.]]
        seeds.append((p0, scl))          # exact restart
        seeds.append((p0, scl * 3))      # wider basin search around the good fit
    for p0, scl in seeds:
        res = minimize(lambda q: cost(p0 + q * scl), np.zeros(11), method="Nelder-Mead",
                       options=dict(maxiter=3000, xatol=1e-3, fatol=1e-3, adaptive=True))
        if best is None or res.fun < best[0]: best = (res.fun, p0 + res.x * scl)
    return best


half = (k_turn + 1) // 2
U_full_out = trend[:k_turn + 1]
U_half_out = trend[:half]
U_2nd_half = trend[half:k_turn + 1]
U_return = trend[k_turn + 1:]

def score(p, U_eval, tag):
    rv, t, f, sa, sb, cx, cy = p[:3], p[3:6], np.exp(p[6]), p[7], p[8], p[9], p[10]
    Xg = curve(0, L, 761)
    Qg = proj11(Xg, rv, t, f, cx, cy)
    tree = cKDTree(Qg)
    d, idx = tree.query(U_eval)
    print(f"     {tag:<38} n={len(U_eval):4d}  chamfer median {np.median(d):6.2f} px  p90 {np.percentile(d,90):6.2f} px")
    return d


print("\n[M1] Part 2 - refit on HALF the outbound leg, score on the rest")
print("     (fitting full-outbound first, so the half fit can be seeded from a known-good")
print("     basin - without that, blind search on a short/low-diversity arc gets stuck)")
cost_full, p_full = fit_calibration(U_full_out)
print(f"     full-outbound fit: ICP cost {cost_full:.3f} (norm)")
score(p_full, U_return, "-> full RETURN leg (calibrated on ALL outbound)")

cost_half, p_half = fit_calibration(U_half_out, extra_seed=p_full)
print(f"\n     half-outbound fit: ICP cost {cost_half:.3f} (norm)")
score(p_half, U_2nd_half, "-> 2nd half of OUTBOUND (same leg)")
score(p_half, U_return, "-> full RETURN leg")

print("\n[M1] READING: if the half-calibration gap on RETURN is close to the full-calibration")
print("     gap, the held-out error is not primarily an overfitting/data-starvation artefact -")
print("     it is consistent with a leg-specific effect (systematic 2D-shape difference between")
print("     outbound and return, e.g. from the appearance/roll effect K4 already found).")

np.savez(DATA / "m1_holdout.npz",
         rp_outbound=rp[A], rp_return=rp[B], rp_vit_outbound=rp_vit[A], rp_vit_return=rp_vit[B],
         p_half=p_half, p_full=p_full, k_turn=k_turn)
print("\n[M1] saved data/m1_holdout.npz")
