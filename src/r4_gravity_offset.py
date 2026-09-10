"""Stage R4 - a WORLD-FIXED direction offset (gravity / field gradient), tested.

Every off-centerline model tried so far defines its direction in a frame that
travels with the curve:
    K1-K5  rotating at 1.17 Hz in the parallel-transport frame
    U1     constant angle in the parallel-transport frame, per leg
    V1     locked to the Frenet normal (toward the centre of curvature)
None is a fixed direction in the phantom's own coordinate system. Along Path 2
the transport frame turns 132-664 deg relative to the CSV axes (checked), so
"constant theta in the transport frame" and "always the same wall of the
world" are different model families - and the second is the physically obvious
one. A NdFeB-cored millirobot (density ~7.5 g/cc) in a water-filled tube rests
on the lowest wall; a gradient-driven one is pulled toward the magnet. Either
way the offset direction is a fixed world vector g:

    X(s) = C(s) + r * g_perp(s),    g_perp = (g - (g.T)T) / |g - (g.T)T|

Three parameters: the direction of g (2) and the offset r (1). The offset
becomes a known function of PLACE (the local tube orientation), identical on
both passes, with no leg parameter. If this holds, the tube model's transverse
position (V8) turns from a free latent into a prediction, and V13's along-ray
depth ambiguity collapses with it.

Variant B adds one parameter: g_perp rotated by +-alpha about the tube axis,
sign set by travel direction. A body rolling on a wall climbs the side its
rotation sense pushes it toward, so reversing travel can flip the side - the
mechanism T10's "opposite walls" reading needs if gravity alone does not fit.

Protocol is V1's, unchanged: fit on the OUTBOUND leg by HMM evidence; predict
the RETURN leg with nothing refit; report both legs' absolute residuals, never
a ratio alone; reverse transfer (fit on return, predict outbound); Viterbi
under the best model; the paired view of the centerline decode's failure
frames. A cheap screen (residual at the baseline decode's own s-path, outbound
frames only) ranks directions before evidence is spent on the top few. The
affine camera is used, as in U1/V1, so the rows are comparable to theirs.
"""
import numpy as np, matplotlib, io, cv2
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
from config import DATA, OUT, FPS, imread_u, imwrite_u
from a_centerline import load as load_cl
from k_frame import transport_frame
from e_hmm import ArcHMM

V_MAX = 60.0
s3, C3, T3, _ = load_cl(); L = s3[-1]
DS = 0.25
SG = np.arange(0, L + 1e-9, DS)
CG = np.stack([np.interp(SG, s3, C3[:, c]) for c in range(3)], 1)
Tg = np.stack([np.interp(SG, s3, T3[:, c]) for c in range(3)], 1)
Tg /= np.clip(np.linalg.norm(Tg, axis=1, keepdims=True), 1e-9, None)

fin = np.load(DATA / "i_final.npz"); UV, k_turn = fin["UV"], int(fin["k_turn"])
K = len(UV); dt = 1 / FPS
OUT_LEG = np.zeros(K, bool); OUT_LEG[:k_turn + 1] = True; RET = ~OUT_LEG
pa = np.load(DATA / "l2_affine.npy")
R_aff = Rot.from_rotvec(pa[:3]).as_matrix(); SC = np.exp(pa[3]); C0 = pa[4:6]
A0 = (CG @ R_aff.T)[:, :2] * SC + C0
print(f"[R4] affine camera, {SC:.2f} px/mm; {OUT_LEG.sum()} outbound / {RET.sum()} return frames")


# ------------------------------------------------------------- the model
def gperp(g):
    g = np.asarray(g, float); g = g / np.linalg.norm(g)
    gp = g[None, :] - (Tg @ g)[:, None] * Tg
    n = np.linalg.norm(gp, axis=1, keepdims=True)
    return gp / np.clip(n, 1e-9, None)


def rot_about_T(v, a):
    """Rotate v (S,3), v perpendicular to Tg, by angle a about Tg."""
    return v * np.cos(a) + np.cross(Tg, v) * np.sin(a)


def proj_world(g, r, alpha=0.0):
    d = gperp(g)
    if alpha != 0.0: d = rot_about_T(d, alpha)
    return A0 + r * (d @ R_aff.T)[:, :2] * SC


def fib_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n); th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def score(P, k0=0, k1=None):
    """Causal pass over frames k0..k1-1 from a uniform prior (V1's score, truncated
    to the frames that count - frames with zeroed likelihood contribute ~0 to
    logZ under advection, so this is equivalent and 2x cheaper)."""
    k1 = K if k1 is None else k1
    hmm = ArcHMM(SG, P, sigma_px=9.0, v_max=V_MAX, n_v=41, accel_sigma=20., p_lost=0.05)
    logp = np.full((hmm.S, hmm.V), -np.log(hmm.S * hmm.V))
    logZ = 0.0; s_hat = np.full(K, np.nan)
    for k in range(k0, k1):
        if k > k0: logp = hmm._advect(logp, dt)
        logp = logp + hmm.loglik(UV[k])[:, None]
        m = logp.max(); logp -= m
        w = np.exp(logp); tot = w.sum(); w /= tot
        logZ += m + np.log(tot)
        s_hat[k] = (w.sum(1) * SG).sum()
        logp = np.log(np.maximum(w, 1e-300))
    idx = np.clip(np.rint(np.nan_to_num(s_hat) / DS).astype(int), 0, hmm.S - 1)
    rp = np.linalg.norm(P[idx] - UV, axis=1); rp[np.isnan(s_hat)] = np.nan
    return logZ, rp, s_hat


def viterbi_rp(P):
    hmm = ArcHMM(SG, P, sigma_px=9.0, v_max=V_MAX, n_v=41, accel_sigma=20., p_lost=0.05)
    s_v = hmm.viterbi(UV, dt)
    idx = np.clip(np.rint(s_v / DS).astype(int), 0, hmm.S - 1)
    return np.linalg.norm(P[idx] - UV, axis=1), s_v, idx


# ------------------------------------------------------------ baseline
print("\n[R4] baseline: centerline (r=0), Viterbi under the affine camera (post-fix)")
rp0, s_v0, idx0 = viterbi_rp(A0)
print(f"     reproj med {np.median(rp0):.2f} px | >30px {100*np.mean(rp0>30):.1f}% | "
      f"outbound med {np.median(rp0[OUT_LEG]):.2f} | return med {np.median(rp0[RET]):.2f} "
      f"({100*np.mean(rp0[RET]>30):.1f}%)")

# --------------------------------------- screen: direction x r, fixed s-path
R_GRID = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
DIRS = fib_sphere(400)
AXES = {"+x": [1, 0, 0], "-x": [-1, 0, 0], "+y": [0, 1, 0], "-y": [0, -1, 0], "+z": [0, 0, 1], "-z": [0, 0, -1]}


def screen(g, r, leg):
    P = proj_world(g, r)
    res = np.linalg.norm(P[idx0] - UV, axis=1)
    return np.median(res[leg])


print(f"\n[R4] screen: median residual at the baseline decode's own s-path, "
      f"{len(DIRS)} directions x {len(R_GRID)} radii (fast, no HMM)")
base_out, base_ret = np.median(rp0[OUT_LEG]), np.median(rp0[RET])
tab = np.zeros((len(DIRS), len(R_GRID), 2))
for i, g in enumerate(DIRS):
    for j, r in enumerate(R_GRID):
        tab[i, j, 0] = screen(g, r, OUT_LEG); tab[i, j, 1] = screen(g, r, RET)
best_out = tab[:, :, 0].min(1)                  # best r per direction, outbound
order = np.argsort(best_out)
print(f"     baseline (r=0): outbound {base_out:.2f} px | return {base_ret:.2f} px")
print(f"{'rank':>5}{'direction g (CSV frame)':>26}{'r':>5}{'outbound':>10}{'return':>8}")
for rank, i in enumerate(order[:8]):
    j = int(np.argmin(tab[i, :, 0]))
    print(f"{rank:5d}{str(DIRS[i].round(2)):>26}{R_GRID[j]:5.1f}{tab[i,j,0]:10.2f}{tab[i,j,1]:8.2f}")
print(f"     the six CSV axes (if the model came out of CAD with an axis vertical, one should win):")
for nm, g in AXES.items():
    j = int(np.argmin([screen(g, r, OUT_LEG) for r in R_GRID]))
    print(f"     {nm:>4}   r={R_GRID[j]:.1f}   outbound {screen(g, R_GRID[j], OUT_LEG):.2f}   return {screen(g, R_GRID[j], RET):.2f}")

# ------------------------------------ evidence fit on the OUTBOUND leg (top-3)
print(f"\n[R4] evidence fit on the OUTBOUND leg, top-3 screen directions x r-grid")
lz0_out, _, _ = score(A0, 0, k_turn + 1)
print(f"     r=0 evidence on the fitting leg: {lz0_out:.1f}")
best = (-np.inf, None, None)
for i in order[:3]:
    g = DIRS[i]
    for r in R_GRID:
        lz, _, _ = score(proj_world(g, r), 0, k_turn + 1)
        if lz > best[0]: best = (lz, g, r)
    print(f"     g={g.round(2)}: best r so far {best[2]}  (running best evidence {best[0]:.1f})")
lz_fit, g_hat, r_hat = best
print(f"     best: g = {g_hat.round(3)}, r = {r_hat:.2f} mm, gain over r=0: {lz_fit - lz0_out:+.1f} nats")
_, N1G, N2G = transport_frame(CG)
d_hat = gperp(g_hat)
th_tf = np.degrees(np.arctan2((d_hat * N2G).sum(1), (d_hat * N1G).sum(1)))
print(f"     that direction, expressed in the transport frame U1 used, sweeps "
      f"{np.ptp(np.unwrap(np.radians(th_tf)))*180/np.pi:.0f} deg along the path "
      f"(U1 held it constant; this is why the two are different models)")

# --------------------------------- out-of-sample: RETURN leg, nothing refit
print(f"\n[R4] OUT-OF-SAMPLE: the return leg, offset frozen from the outbound fit")
P_g = proj_world(g_hat, r_hat)
rows = []
for nm, P in [("centerline (r=0)", A0), (f"world-fixed g (R4), r={r_hat:.1f} mm", P_g)]:
    lz, rp, _ = score(P)
    rows.append((nm, np.median(rp[OUT_LEG]), np.median(rp[RET]), 100 * np.mean(rp[RET] > 30), lz))
try:
    v1 = np.load(DATA / "v1_curvature.npz")["rows"]
    for nm, row in zip(["U1 constant, same wall", "U1 constant, opposite wall", "curvature-locked (V1)"], v1[1:]):
        rows.append((nm + "  [V1's table]", row[0], row[1], row[2], row[3]))
except Exception:
    pass
print(f"{'model':<46}{'outbound':>10}{'RETURN':>9}{'ret>30px':>10}{'log Z':>10}")
for nm, a, b, f, lz in rows:
    print(f"{nm:<46}{a:10.2f}{b:9.2f}{f:9.1f}%{lz:10.0f}")

# ------------------------------------------- reverse transfer: fit RETURN, predict OUTBOUND
print(f"\n[R4] reverse transfer: fit on the RETURN leg, predict the OUTBOUND leg")
best_ret = tab[:, :, 1].min(1); order_r = np.argsort(best_ret)
lz0_ret, _, _ = score(A0, k_turn + 1, K)
best_r = (-np.inf, None, None)
for i in order_r[:3]:
    g = DIRS[i]
    for r in R_GRID:
        lz, _, _ = score(proj_world(g, r), k_turn + 1, K)
        if lz > best_r[0]: best_r = (lz, g, r)
lz_fit_r, g_r, r_r = best_r
_, rp_rev, _ = score(proj_world(g_r, r_r))
ang = np.degrees(np.arccos(np.clip(np.dot(g_hat, g_r), -1, 1)))
print(f"     reverse fit: g = {g_r.round(3)}, r = {r_r:.2f} mm, gain {lz_fit_r - lz0_ret:+.1f} nats")
print(f"     scored on ALL frames: outbound med {np.median(rp_rev[OUT_LEG]):.2f} px "
      f"(centerline {rows[0][1]:.2f}), return med {np.median(rp_rev[RET]):.2f} (centerline {rows[0][2]:.2f})")
print(f"     direction agreement forward vs reverse: {ang:.0f} deg apart, r {r_hat:.2f} vs {r_r:.2f} mm "
      f"({'CONSISTENT' if ang < 30 and abs(r_hat - r_r) <= 1.0 else 'INCONSISTENT'} - a real fixed direction "
      f"must be found from either leg)")

# --------------------------------- Viterbi under the fitted model, paired failure view
print(f"\n[R4] Viterbi (the recommended decoder) under each model, all frames")
rp_g, s_vg, idx_g = viterbi_rp(P_g)
for nm, rp in [("centerline", rp0), ("world-fixed g (R4)", rp_g)]:
    print(f"     {nm:<20} reproj med {np.median(rp):5.2f} px | p90 {np.percentile(rp,90):6.2f} "
          f"| >30px {100*np.mean(rp>30):4.1f}% | return med {np.median(rp[RET]):5.2f} "
          f"| return >30px {100*np.mean(rp[RET]>30):4.1f}%")
fail = rp0 > 30
if fail.any():
    print(f"     the {fail.sum()} frames the centerline decode fails on: median rp "
          f"{np.median(rp0[fail]):.1f} -> {np.median(rp_g[fail]):.1f} px; still failing "
          f"{100*np.mean(rp_g[fail]>30):.0f}%")
ep = np.arange(885, 920)
print(f"     junction episode 885-920: median {np.median(rp0[ep]):.1f} -> {np.median(rp_g[ep]):.1f} px, "
      f"max {rp0[ep].max():.1f} -> {rp_g[ep].max():.1f} px")

# ------------------------------------------ variant B: side flips with travel direction
print(f"\n[R4] variant B: g_perp rotated by alpha about the tube axis, sign set by leg")
print(f"     fit alpha on the outbound leg at the fitted (g, r); then predict the return")
print(f"     leg two ways - same alpha (no flip) and -alpha (flip). If flipping wins on the")
print(f"     return leg, direction-of-travel matters; if not, the plain fixed direction stands.")
ALPHAS = np.radians([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180])
bestB = (-np.inf, 0.0)
for a in ALPHAS:
    lz, _, _ = score(proj_world(g_hat, r_hat, a), 0, k_turn + 1)
    if lz > bestB[0]: bestB = (lz, a)
lzB, a_hat = bestB
lz_same, rp_same, _ = score(proj_world(g_hat, r_hat, a_hat), k_turn + 1, K)
lz_flip, rp_flip, _ = score(proj_world(g_hat, r_hat, -a_hat), k_turn + 1, K)
lz_plain, rp_plain, _ = score(P_g, k_turn + 1, K)
print(f"     outbound-fitted alpha = {np.degrees(a_hat):+.0f} deg (gain over alpha=0: {lzB - lz_fit:+.1f} nats)")
print(f"{'return-leg prediction':<34}{'log Z (return)':>16}{'return med px':>15}{'ret>30px':>10}")
for nm, lz, rp in [("plain fixed direction (alpha=0)", lz_plain, rp_plain),
                   (f"same side, alpha={np.degrees(a_hat):+.0f}", lz_same, rp_same),
                   (f"FLIPPED, alpha={-np.degrees(a_hat):+.0f}", lz_flip, rp_flip)]:
    print(f"{nm:<34}{lz:16.1f}{np.nanmedian(rp[RET]):15.2f}{100*np.nanmean(rp[RET]>30):9.1f}%")

# ------------------------------------------------------------- figure + artefacts
bg = imread_u(OUT / "background_median.png")
fig, ax = plt.subplots(1, 3, figsize=(19, 5.5))
a1 = ax[0]
a1.plot(SG, th_tf, lw=1.2)
a1.set_xlabel("arc length s (mm)"); a1.set_ylabel("offset direction in transport frame (deg)")
a1.set_title(f"world-fixed g={g_hat.round(2)} as seen by U1's frame")
a2 = ax[1]; a2.imshow(bg)
a2.plot(A0[:, 0], A0[:, 1], color='cyan', lw=1.2, label="centerline proj")
a2.plot(P_g[:, 0], P_g[:, 1], color='yellow', lw=1.0, ls='--', label=f"world-fixed offset r={r_hat:.1f}")
a2.set_ylim(720, 0); a2.legend(fontsize=8); a2.set_title("projections over the phantom")
a3 = ax[2]; a3.semilogy(np.maximum(rp0, .1), lw=.7, label="centerline")
a3.semilogy(np.maximum(rp_g, .1), lw=.7, label="world-fixed g")
a3.axhline(30, color='r', ls='--'); a3.set_xlabel("frame"); a3.set_ylabel("reproj px")
a3.legend(); a3.set_title("Viterbi residual over time")
plt.tight_layout()
buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=105); buf.seek(0)
arr = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1].copy()
imwrite_u(OUT / "figs" / "r4_gravity_offset.png", arr)
np.savez(DATA / "r4_gravity.npz", g=g_hat, r=r_hat, lz_gain_out=lz_fit - lz0_out,
         g_rev=g_r, r_rev=r_r, lz_gain_rev=lz_fit_r - lz0_ret, angle_fwd_rev=ang,
         rows=np.array([(a, b, f, lz) for _, a, b, f, lz in rows[:2]]),
         alpha=a_hat, lz_plain=lz_plain, lz_same=lz_same, lz_flip=lz_flip,
         rp0=rp0, rp_g=rp_g, s_v0=s_v0, s_vg=s_vg, screen=tab, dirs=DIRS, r_grid=R_GRID)
print("\n[R4] saved out/figs/r4_gravity_offset.png, data/r4_gravity.npz")
