"""Stage H - end-to-end run on the REAL video.

There is no 3D ground truth for the real sequence, so validation uses the
out-and-back structure: the robot retraces the same vessel, so the arc length
recovered on the return leg must match the outbound leg at the same image
position. That is an independent check the estimator never sees.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot
from config import DATA, OUT, W, H, FPS
from a_centerline import load as load_cl
from e_hmm import ArcHMM

C0 = np.array([W/2., H/2.]); TRIM = .90
s3, C3, T3, _ = load_cl(); L = s3[-1]
F_FIX = 2000.0                     # from C7: cost-minimising focal in the flat valley

z3 = np.load(DATA / "c3_global_trend.npz")
dirs, angs, top, U160, Ur = z3["dirs"], z3["angs"], z3["top"], z3["U"], float(z3["Ur"])
TREE_U = cKDTree(U160)


def curve(sa, sb, n=250):
    ss = np.linspace(sa, sb, n)
    return np.stack([np.interp(ss, s3, C3[:, c]) for c in range(3)], 1)


def proj(X, R, t, f=F_FIX):
    Xc = X @ R.T + t; z = Xc[:, 2]
    if (z <= 1e-3).any(): return None, z
    return Xc[:, :2] / z[:, None] * f + C0, z


def trimmed(d): return float(np.sort(d)[:max(3, int(TRIM*len(d)))].mean())


def cost(p):
    R = Rot.from_rotvec(p[:3]).as_matrix(); t = p[3:6]
    if not (0 <= p[6] < p[7] <= L) or (p[7]-p[6]) < 20 or t[2] <= 0: return 1e6
    Q, z = proj(curve(p[6], p[7]), R, t)
    if Q is None or not np.isfinite(Q).all(): return 1e6
    d1, _ = cKDTree(Q).query(U160); d2, _ = TREE_U.query(Q)
    return .5*(trimmed(d1) + trimmed(d2))


print(f"[H] re-fitting pose at the fixed focal length f={F_FIX:.0f} px (C7 valley minimum)")
best = None
for r in range(5):
    sc, di, ri = top[r]; di, ri = int(di), int(ri)
    v = dirs[di]; ang = angs[ri]
    a = np.array([0,0,1.]) if abs(v[2]) < .9 else np.array([1.,0,0])
    e1 = np.cross(v, a); e1 /= np.linalg.norm(e1); e2 = np.cross(v, e1)
    Rz = np.array([[np.cos(ang), -np.sin(ang), 0],[np.sin(ang), np.cos(ang), 0],[0,0,1]])
    R0 = Rz @ np.stack([e1, e2, v]); rv0 = Rot.from_matrix(R0).as_rotvec()
    Qo = curve(0, L) @ R0.T
    kpm = Ur/np.sqrt(((Qo[:,:2]-Qo[:,:2].mean(0))**2).sum(1).mean()); tz = F_FIX/kpm
    for (sa0, sb0) in [(0, L), (.15*L, .95*L)]:
        p0 = np.r_[rv0, -Qo[:,:2].mean(0), tz-Qo[:,2].mean(), sa0, sb0]
        scl = np.r_[[.05]*3, [abs(tz)*.02]*3, [6.], [6.]]
        res = minimize(lambda q: cost(p0+q*scl), np.zeros(8), method="Nelder-Mead",
                       options=dict(maxiter=3000, xatol=1e-3, fatol=1e-3, adaptive=True))
        if best is None or res.fun < best[0]: best = (res.fun, p0+res.x*scl)
c, p = best
R = Rot.from_rotvec(p[:3]).as_matrix(); t = p[3:6]; sa, sb = p[6], p[7]
print(f"[H] pose cost {c:.2f} px   view axis {R[2].round(3)}   traversed s = {sa:.1f}..{sb:.1f} mm")

# --- project the centerline grid ---------------------------------------------
ds = 0.25
sg = np.arange(0, L + 1e-9, ds)
Cg = np.stack([np.interp(sg, s3, C3[:, c]) for c in range(3)], 1)
Pg, zg = proj(Cg, R, t)
print(f"[H] image scale {F_FIX/np.median(zg):.2f} px/mm   depth range {np.ptp(zg):.1f} mm")

# --- run the causal estimator on the FULL real track (out AND back) -----------
zt = np.load(DATA / "track2d_trend.npz")
UV = zt["trend"]; k_turn = int(zt["k_turn"])
hmm = ArcHMM(sg, Pg, sigma_px=9.0, v_max=30., n_v=41, accel_sigma=20., p_lost=0.05)
import time; t0 = time.time()
s_hat, s_std, v_hat = hmm.forward(UV, 1/FPS)
dt_ms = 1000*(time.time()-t0)/len(UV)
print(f"[H] causal filter: {dt_ms:.2f} ms/frame  ({1000/dt_ms:.0f} fps)  <- real time at 30 fps: "
      f"{'YES' if dt_ms < 33 else 'NO'}")

X_hat = np.stack([np.interp(s_hat, sg, Cg[:, c]) for c in range(3)], 1)
Q_hat, _ = proj(X_hat, R, t)
reproj = np.linalg.norm(Q_hat - UV, axis=1)
print(f"[H] reprojection error px: median {np.median(reproj):.1f}  p90 {np.percentile(reproj,90):.1f}"
      f"  -> {np.median(reproj)/(F_FIX/np.median(zg)):.2f} mm")

# --- out-and-back self-consistency -------------------------------------------
A, B = np.arange(k_turn+1), np.arange(k_turn+1, len(UV))
treeA = cKDTree(UV[A])
d_img, ia = treeA.query(UV[B])
sel = d_img < 8.0                      # only where the two legs really overlap in the image
ds_recovered = np.abs(s_hat[B][sel] - s_hat[A][ia[sel]])
print(f"\n[H] OUT-AND-BACK SELF-CONSISTENCY  (n={sel.sum()} frames overlapping within 8 px)")
print(f"    |s_return - s_outbound| mm: median {np.median(ds_recovered):.2f}"
      f"  p90 {np.percentile(ds_recovered,90):.2f}  max {ds_recovered.max():.2f}")
print(f"    -> independent estimate of 3D localisation repeatability")

print(f"\n[H] arc length travelled: {s_hat.min():.1f} .. {s_hat.max():.1f} mm"
      f"  (span {s_hat.max()-s_hat.min():.1f} mm of the {L:.1f} mm path)")
print(f"[H] posterior std: median {np.median(s_std):.2f} mm  p90 {np.percentile(s_std,90):.2f} mm")
np.savez(DATA / "h_result.npz", R=R, t=t, f=F_FIX, sa=sa, sb=sb, sg=sg, Cg=Cg, Pg=Pg,
         s_hat=s_hat, s_std=s_std, v_hat=v_hat, X_hat=X_hat, Q_hat=Q_hat, UV=UV,
         k_turn=k_turn, reproj=reproj, cost=c)
