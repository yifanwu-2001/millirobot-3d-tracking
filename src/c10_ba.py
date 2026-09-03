"""Stage C10 - redo the free-principal-point refit with a damped least-squares
solver instead of cv2.calibrateCamera, to test whether the (1923, -19)
divergence C9 hit is a SOLVER stability problem (fixable) or the structural
ill-conditioning C7/J4 already quantified (not fixable by a better solver).

Procedure, per the plan: seed from the STABLE fixed-principal-point fit
(c8_9.npy), then free (cx, cy) with a Tikhonov prior pulling them toward the
image centre - a soft_l1-robust, bounded scipy.optimize.least_squares in place
of cv2.calibrateCamera's unconstrained Levenberg-Marquardt. This is not
expected to beat C8's own free-PP number (4.92 px, already a stable fit found
by bounded Nelder-Mead) - the question is whether a properly damped BA
converges to something similar and stable, or whether even damping cannot
stop it from running toward the boundary, which would show the divergence is
about the cost surface, not the optimizer.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from config import DATA, OUT, W, H
from a_centerline import load as load_cl

TRIM = .90
s3, C3, _, _ = load_cl(); L = s3[-1]
z3 = np.load(DATA / "c3_global_trend.npz"); U = z3["U"].astype(np.float64)
TREE_U = cKDTree(U)

p9 = np.load(DATA / "c8_9.npy")
rv0, t0, f0, sa0, sb0 = p9[:3], p9[3:6], np.exp(p9[6]), p9[7], p9[8]
print(f"[C10] seed = C8 fixed-principal-point fit: f={f0:.0f} px, s={sa0:.1f}..{sb0:.1f} mm")


def curve(sa, sb, n=250):
    ss = np.linspace(sa, sb, n)
    return np.stack([np.interp(ss, s3, C3[:, c]) for c in range(3)], 1)


X3 = curve(sa0, sb0)


def proj(X, rv, t, f, cx, cy):
    Xc = X @ Rot.from_rotvec(rv).as_matrix().T + t
    return Xc[:, :2] / Xc[:, 2:3] * f + np.array([cx, cy])


def dtw_pairs(A, B):
    n, m = len(A), len(B)
    D = np.linalg.norm(A[:, None] - B[None], axis=-1)
    acc = np.full((n + 1, m + 1), np.inf); acc[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc[i, j] = D[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    i, j, pr = n, m, []
    while i > 0 and j > 0:
        pr.append((i - 1, j - 1))
        k = int(np.argmin([acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1]]))
        if k == 0: i -= 1
        elif k == 1: j -= 1
        else: i -= 1; j -= 1
    return np.array(pr[::-1])


def trimmed(d):
    return float(np.sort(d)[:max(3, int(TRIM * len(d)))].mean())


def symmetric_residual(rv, t, f, cx, cy):
    Q = proj(X3, rv, t, f, cx, cy)
    d1, _ = cKDTree(Q).query(U); d2, _ = TREE_U.query(Q)
    return .5 * (trimmed(d1) + trimmed(d2))


LO = np.r_[[-np.inf] * 6, np.log(100.), 0., 0.]
HI = np.r_[[np.inf] * 6, np.log(50000.), float(W), float(H)]


def run(lam, n_outer=8):
    rv, t, lf, cx, cy = rv0.copy(), t0.copy(), np.log(f0), W / 2., H / 2.
    traj = []
    for it in range(n_outer):
        f = np.exp(lf)
        Q = proj(X3, rv, t, f, cx, cy)
        pr = dtw_pairs(U, Q)
        Xm, Um = X3[pr[:, 1]], U[pr[:, 0]]

        def res(q):
            rv_, t_, lf_, cx_, cy_ = q[:3], q[3:6], q[6], q[7], q[8]
            f_ = np.exp(lf_)
            Qm = proj(Xm, rv_, t_, f_, cx_, cy_)
            rdat = (Qm - Um).ravel()
            rpri = np.sqrt(lam) * np.array([cx_ - W / 2., cy_ - H / 2.])
            return np.concatenate([rdat, rpri])

        p0 = np.r_[rv, t, lf, cx, cy]
        sol = least_squares(res, p0, bounds=(LO, HI), loss="soft_l1", f_scale=5.0, max_nfev=4000)
        rv, t, lf, cx, cy = sol.x[:3], sol.x[3:6], sol.x[6], sol.x[7], sol.x[8]
        f = np.exp(lf)
        rms = symmetric_residual(rv, t, f, cx, cy)
        traj.append((f, cx, cy, rms))
    return rv, t, np.exp(lf), cx, cy, sol, traj


print(f"\n[C10] lambda=0 (undamped free PP, same cost surface C9's calibrateCamera hit)")
*_, sol0, traj0 = run(0.0)
for it, (f, cx, cy, rms) in enumerate(traj0):
    print(f"   it {it}: f {f:8.0f}   principal ({cx:7.0f},{cy:7.0f})   trimmed {rms:6.2f} px")

print(f"\n[C10] damped free PP: Tikhonov prior toward image centre, sweeping lambda")
print(f"{'lambda':>10}{'final f':>10}{'final cx,cy':>18}{'trimmed px':>12}{'hit bound?':>12}")
results = {}
for lam in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
    rv, t, f, cx, cy, sol, traj = run(lam)
    rms = traj[-1][3]
    hit = "yes" if (cx <= 1 or cx >= W - 1 or cy <= 1 or cy >= H - 1 or f <= 101 or f >= 49999) else "no"
    results[lam] = (f, cx, cy, rms, traj, rv, t)
    print(f"{lam:10.0e}{f:10.0f}{f'({cx:.0f},{cy:.0f})':>18}{rms:12.2f}{hit:>12}")

best_lam = min(results, key=lambda L: results[L][3])
f_b, cx_b, cy_b, rms_b, traj_b, rv_b, t_b = results[best_lam]
J = results[best_lam][4]
print(f"\n[C10] best-residual damped fit: lambda={best_lam:.0e}, {rms_b:.2f} px "
      f"(reference: C7 fixed-PP 9.37 px, C8 undamped free-PP 4.92 px)")

# parameter uncertainty from the final least_squares Jacobian, for comparison to J4's CRB
rv, t, lf, cx, cy, sol, _ = run(best_lam)
try:
    J_ = sol.jac
    resid_var = float((sol.fun ** 2).sum() / max(len(sol.fun) - len(sol.x), 1))
    cov = np.linalg.pinv(J_.T @ J_) * resid_var
    sig_f = f_b * np.sqrt(max(cov[6, 6], 0))   # d f/d(logf) = f
    sig_cx, sig_cy = np.sqrt(max(cov[7, 7], 0)), np.sqrt(max(cov[8, 8], 0))
    print(f"[C10] parameter std from this fit's Jacobian: sigma_f={sig_f:.0f} px, "
          f"sigma_cx={sig_cx:.0f} px, sigma_cy={sig_cy:.0f} px")
    print(f"      (J4's single-view Cramer-Rao bound, best case with EXACT correspondences: "
          f"sigma_f=175 px, sigma_cx=98 px)")
except np.linalg.LinAlgError:
    print("[C10] Jacobian singular at the solution - covariance not estimable")

np.savez(DATA / "c10_ba.npz", f=f_b, cx=cx_b, cy=cy_b, rms=rms_b, best_lam=best_lam,
         rv=rv_b, t=t_b)
print("\n[C10] saved data/c10_ba.npz")
