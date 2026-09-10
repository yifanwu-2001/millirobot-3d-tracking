"""V1 - the model U1 named as untested: a curvature-locked wall offset.

U1 rejected a CONSTANT radial offset with a per-leg angle: fitted on the
outbound leg, it made the return leg worse, and a free theta_return scan
peaked at 0 deg (same wall), not the physical pi (opposite wall). U1's own
diagnosis: the displacement T10 measured is concentrated in ~30 mm of curved
vessel, so a single `r` spread over 190 mm is diluted to 0.5 mm, and "a
curvature-dependent offset (large on bends, zero on straights) is the natural
next model and is untested".

That model, with the direction locked to local geometry rather than to a leg:

    X(s) = C(s) + r_max * g(kappa(s)) * ( cos(phi0) Nc(s) + sin(phi0) B(s) )
    g(kappa) = kappa / (kappa + kappa_half)

Nc is the principal normal (toward the centre of curvature), B = T x Nc.
phi0 = 0 rides the INNER wall of a bend, phi0 = pi the OUTER wall; the offset
vanishes on straights and flips side at inflections, continuously, because the
gate goes to zero there. There is NO leg parameter at all: the offset is a
function of place, so it is automatically the same on both passes - which is
what U1's free theta scan (peak at 0 deg) already preferred, while T10's
opposite-wall observation in the mid section is explained by Nc flipping
across the S-bends. One model, both observations.

Protocol (same as U1, same lesson applied): fit (r_max, kappa_half, phi0) on
the OUTBOUND leg by HMM evidence, then predict the RETURN leg out-of-sample
with NO refit. U1's spurious M1-ratio result is guarded against by reporting
both legs' absolute residuals, never a ratio alone. A second transfer
direction (fit on return, predict outbound) is added - a geometry-locked
model should transfer both ways; a noise-absorbing one should not.
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.spatial.transform import Rotation as Rot
from config import DATA, OUT, FPS, imread_u, imwrite_u
from a_centerline import load as load_cl
from k_frame import transport_frame
from e_hmm import ArcHMM

V_MAX = 60.0
s3, C3, T3, kap3 = load_cl(); L = s3[-1]
DS = 0.25
SG = np.arange(0, L + 1e-9, DS)
CG = np.stack([np.interp(SG, s3, C3[:, c]) for c in range(3)], 1)

# ---------------------------------------------------- curvature frame on SG
# centerline.npz gives the analytic tangent on its 0.2 mm grid; differentiate
# THAT (it is smooth) rather than the linearly interpolated CG, then sample.
dT = np.gradient(T3, s3, axis=0)
kap_n = np.linalg.norm(dT, axis=1)
print(f"[V1] numeric curvature vs analytic: max |kap_n - kap|/kap_max = "
      f"{np.max(np.abs(kap_n - kap3)) / kap3.max():.3f}  (should be small)")
win = max(3, int(round(1.0 / (s3[1] - s3[0]))))          # ~1 mm direction smoothing
kap_sm = uniform_filter1d(kap_n, win)
Nc3 = dT / np.clip(kap_n, 1e-9, None)[:, None]
Nc3 = uniform_filter1d(Nc3, win, axis=0)
Nc3 /= np.clip(np.linalg.norm(Nc3, axis=1, keepdims=True), 1e-9, None)

Nc = np.stack([np.interp(SG, s3, Nc3[:, c]) for c in range(3)], 1)
Nc /= np.clip(np.linalg.norm(Nc, axis=1, keepdims=True), 1e-9, None)
Tg = np.stack([np.interp(SG, s3, T3[:, c]) for c in range(3)], 1)
Tg /= np.clip(np.linalg.norm(Tg, axis=1, keepdims=True), 1e-9, None)
Bg = np.cross(Tg, Nc)
kag = np.interp(SG, s3, kap_sm)
print(f"[V1] kappa on grid: median {np.median(kag):.4f}  p90 {np.percentile(kag,90):.4f}  "
      f"max {kag.max():.4f} 1/mm  (min bend radius {1/kag.max():.2f} mm; report says 2.55)")

# ------------------------------------------------------------ affine camera
fin = np.load(DATA / "i_final.npz"); UV, k_turn = fin["UV"], int(fin["k_turn"])
K = len(UV); dt = 1 / FPS
OUT_LEG = np.zeros(K, bool); OUT_LEG[:k_turn + 1] = True
pa = np.load(DATA / "l2_affine.npy")
R_aff = Rot.from_rotvec(pa[:3]).as_matrix(); SC = np.exp(pa[3]); C0 = pa[4:6]
A0 = (CG @ R_aff.T)[:, :2] * SC + C0
D1 = (Nc @ R_aff.T)[:, :2] * SC          # projected offset directions, px per mm
D2 = (Bg  @ R_aff.T)[:, :2] * SC
print(f"[V1] affine camera, scale {SC:.2f} px/mm, tube radius ~3.4 mm")

# U1's fitted constant-offset model, re-scored here for one-code-path comparability
zu = np.load(DATA / "u1_wallhug.npz"); r_u1, th_u1 = float(zu["r"]), float(zu["th"])
_, N1G, N2G = transport_frame(CG)


def proj_curv(r_max, kap_half, phi0):
    g = kag / (kag + kap_half)
    return A0 + r_max * g[:, None] * (np.cos(phi0) * D1 + np.sin(phi0) * D2)


def proj_const(r, th):
    off = r * (np.cos(th) * N1G + np.sin(th) * N2G)
    return A0 + (off @ R_aff.T)[:, :2] * SC


def score(P, frames=None, want_path=False):
    """Causal filter with a static projection; logZ + posterior-mean residuals.

    frames: bool mask of frames whose likelihood counts (fitting uses the
    outbound leg; the advection still runs over all frames, as in U1)."""
    hmm = ArcHMM(SG, P, sigma_px=9.0, v_max=V_MAX, n_v=41, accel_sigma=20., p_lost=0.05)
    logp = np.full((hmm.S, hmm.V), -np.log(hmm.S * hmm.V))
    logZ = 0.0; s_hat = np.zeros(K)
    for k in range(K):
        if k > 0: logp = hmm._advect(logp, dt)
        ll = hmm.loglik(UV[k])
        if frames is not None and not frames[k]:
            ll = np.zeros_like(ll)
        logp = logp + ll[:, None]
        m = logp.max(); logp -= m
        w = np.exp(logp); tot = w.sum(); w /= tot
        logZ += m + np.log(tot)
        ps = w.sum(1); s_hat[k] = (ps * SG).sum()
        logp = np.log(np.maximum(w, 1e-300))
    idx = np.clip(np.rint(s_hat / DS).astype(int), 0, hmm.S - 1)
    rp = np.linalg.norm(P[idx] - UV, axis=1)
    return (logZ, rp, s_hat) if want_path else (logZ, rp)


# ------------------------------------------- fit (r_max, kappa_half, phi0) on OUTBOUND
print(f"\n[V1] fitting on the OUTBOUND leg only ({OUT_LEG.sum()} frames)")
lz0_out, _ = score(A0, frames=OUT_LEG)
print(f"     r=0 evidence on the fitting leg: {lz0_out:.0f}")

def fit_outbound():
    best = (-np.inf, None)
    # stage 1: phi0 at a mid offset / mid gate
    for phi0 in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        lz, _ = score(proj_curv(1.5, 0.10, phi0), frames=OUT_LEG)
        if lz > best[0]: best = (lz, (1.5, 0.10, phi0))
    _, (_, _, phi_b) = best
    # stage 2: (r_max, kappa_half) at the best phi0
    for r_max in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.4):
        for kh in (0.03, 0.06, 0.10, 0.18, 0.30, 0.50):
            lz, _ = score(proj_curv(r_max, kh, phi_b), frames=OUT_LEG)
            if lz > best[0]: best = (lz, (r_max, kh, phi_b))
    _, (r_b, kh_b, _) = best
    # stage 3: refine phi0 finely at the best (r_max, kappa_half)
    for phi0 in np.linspace(phi_b - np.pi / 4, phi_b + np.pi / 4, 13):
        lz, _ = score(proj_curv(r_b, kh_b, phi0), frames=OUT_LEG)
        if lz > best[0]: best = (lz, (r_b, kh_b, phi0))
    return best

lz_fit, (r_hat, kh_hat, phi_hat) = fit_outbound()
side = "INNER wall (toward centre of curvature)" if abs(np.cos(phi_hat)) > 0.7 else \
       ("OUTER wall (away from centre)" if abs(np.cos(phi_hat)) > 0.7 else "oblique")
print(f"     best: r_max = {r_hat:.2f} mm, kappa_half = {kh_hat:.3f} 1/mm "
      f"(half-offset bend radius {1/kh_hat:.1f} mm), phi0 = {np.degrees(phi_hat):.0f} deg -> {side}")
print(f"     evidence gain over r=0 on the fitting leg: {lz_fit - lz0_out:+.1f} nats")

# offset profile: where does it live?
g_hat = kag / (kag + kh_hat)
r_of_s = r_hat * g_hat
print(f"     offset magnitude along path: median {np.median(r_of_s):.2f} mm, "
      f"p90 {np.percentile(r_of_s,90):.2f}, max {r_of_s.max():.2f} mm")
top = np.argsort(r_of_s)[-40:]
print(f"     the 40 largest offsets sit at s = {SG[top].min():.0f}..{SG[top].max():.0f} mm "
      f"(T10's leg-separation region was the curved mid section)")

# --------------------------------- out-of-sample: RETURN leg, nothing refit
print(f"\n[V1] OUT-OF-SAMPLE: the return leg, with the offset function frozen")
RET = ~OUT_LEG
P_curv = proj_curv(r_hat, kh_hat, phi_hat)
rows = []
for nm, P in [("centerline (r=0)", A0),
              ("U1 constant, same wall (r=%.2f, th=%d deg)" % (r_u1, np.degrees(th_u1)),
               proj_const(r_u1, th_u1)),
              ("U1 constant, opposite wall", proj_const(r_u1, th_u1 + np.pi)),
              ("curvature-locked (V1)", P_curv)]:
    lz, rp = score(P)
    rows.append((nm, np.median(rp[OUT_LEG]), np.median(rp[RET]),
                 100 * np.mean(rp[RET] > 30), lz))
print(f"{'model':<46}{'outbound':>10}{'RETURN':>9}{'ret>30px':>10}{'log Z':>10}")
for nm, a, b, f, lz in rows:
    print(f"{nm:<46}{a:10.2f}{b:9.2f}{f:9.1f}%{lz:10.0f}")

# the U1 trap: a ratio whose numerator was damaged is not evidence
print(f"\n[V1] guard against U1's trap (both legs' absolute residuals, not ratios):")
for nm, a, b, f, lz in rows:
    print(f"     {nm:<46} outbound {a:5.2f} px | return {b:5.2f} px")

# second transfer direction: fit on RETURN, predict OUTBOUND
print(f"\n[V1] reverse transfer: fit on the RETURN leg, predict the OUTBOUND leg")
lz0_ret, _ = score(A0, frames=RET)
best_r = (-np.inf, None)
for phi0 in np.linspace(0, 2 * np.pi, 8, endpoint=False):
    lz, _ = score(proj_curv(1.5, 0.10, phi0), frames=RET)
    if lz > best_r[0]: best_r = (lz, (1.5, 0.10, phi0))
_, (_, _, phi_b) = best_r
for r_max in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.4):
    for kh in (0.03, 0.06, 0.10, 0.18, 0.30, 0.50):
        lz, _ = score(proj_curv(r_max, kh, phi_b), frames=RET)
        if lz > best_r[0]: best_r = (lz, (r_max, kh, phi_b))
_, (r_r, kh_r, _) = best_r
for phi0 in np.linspace(phi_b - np.pi / 4, phi_b + np.pi / 4, 13):
    lz, _ = score(proj_curv(r_r, kh_r, phi0), frames=RET)
    if lz > best_r[0]: best_r = (lz, (r_r, kh_r, phi0))
lz_fit_r, (r_hat_r, kh_hat_r, phi_hat_r) = best_r
print(f"     reverse fit: r_max = {r_hat_r:.2f} mm, kappa_half = {kh_hat_r:.3f}, "
      f"phi0 = {np.degrees(phi_hat_r):.0f} deg, gain {lz_fit_r - lz0_ret:+.1f} nats")
P_rev = proj_curv(r_hat_r, kh_hat_r, phi_hat_r)
lz, rp_rev = score(P_rev)
print(f"     scored on ALL frames: outbound median {np.median(rp_rev[OUT_LEG]):.2f} px "
      f"(centerline {rows[0][1]:.2f}), return median {np.median(rp_rev[RET]):.2f} "
      f"(centerline {rows[0][2]:.2f})")
print(f"     direction agreement between the two fits: phi forward {np.degrees(phi_hat):.0f} deg "
      f"vs reverse {np.degrees(phi_hat_r):.0f} deg, r_max {r_hat:.2f} vs {r_hat_r:.2f} mm")

# --------------------------------- offline decode under the best model (recommended output)
print(f"\n[V1] Viterbi (the recommended decoder) under each model, all frames")
def viterbi_rows(P):
    hmm = ArcHMM(SG, P, sigma_px=9.0, v_max=V_MAX, n_v=41, accel_sigma=20., p_lost=0.05)
    s_v = hmm.viterbi(UV, dt)
    idx = np.clip(np.rint(s_v / DS).astype(int), 0, hmm.S - 1)
    rp = np.linalg.norm(P[idx] - UV, axis=1)
    return rp

for nm, P in [("centerline", A0), ("curvature-locked", P_curv)]:
    rp = viterbi_rows(P)
    print(f"     {nm:<20} reproj med {np.median(rp):5.2f} px | p90 {np.percentile(rp,90):6.2f} "
          f"| >30px {100*np.mean(rp>30):4.1f}% | return-leg med {np.median(rp[RET]):5.2f} "
          f"| return >30px {100*np.mean(rp[RET]>30):4.1f}%")

# paired view: what happens to the frames the centerline decode fails on?
rp_c = viterbi_rows(A0); rp_v1 = viterbi_rows(P_curv)
fail = rp_c > 30
if fail.any():
    print(f"\n[V1] the {fail.sum()} frames the centerline decode fails on (rp>30px):")
    print(f"     median rp  centerline {np.median(rp_c[fail]):.1f} px -> curvature {np.median(rp_v1[fail]):.1f} px")
    print(f"     still failing under curvature: {100*np.mean(rp_v1[fail]>30):.0f}%")

# ------------------------------------------------------------- figure + artefacts
bg = imread_u(OUT / "background_median.png")
fig, ax = plt.subplots(1, 3, figsize=(19, 5.5))
a1 = ax[0]
a1.plot(SG, r_of_s, lw=1.5, label="offset r(s)")
a1.fill_between(SG, 0, r_of_s, alpha=.25)
a1.set_xlabel("arc length s (mm)"); a1.set_ylabel("offset (mm)")
a1b = a1.twinx(); a1b.plot(SG, kag, color='r', lw=.8, alpha=.6); a1b.set_ylabel("curvature (1/mm)", color='r')
a1.set_title(f"fitted offset: r_max={r_hat:.2f} mm, k_half={kh_hat:.2f}, phi0={np.degrees(phi_hat):.0f} deg")
a2 = ax[1]; a2.imshow(bg)
a2.plot(A0[:, 0], A0[:, 1], color='cyan', lw=1.2, label="centerline proj")
a2.plot(P_curv[:, 0], P_curv[:, 1], color='yellow', lw=1.0, ls='--', label="curvature-offset proj")
a2.set_ylim(720, 0); a2.legend(fontsize=8); a2.set_title("projections over the phantom")
a3 = ax[2]; a3.semilogy(np.maximum(rp_c, .1), lw=.7, label="centerline")
a3.semilogy(np.maximum(rp_v1, .1), lw=.7, label="curvature-locked")
a3.axhline(30, color='r', ls='--'); a3.set_xlabel("frame"); a3.set_ylabel("reproj px")
a3.legend(); a3.set_title("Viterbi residual over time")
plt.tight_layout()
import io, cv2
buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=105); buf.seek(0)
arr = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1].copy()
imwrite_u(OUT / "figs" / "v1_curvature_offset.png", arr)
print("\n[V1] saved out/figs/v1_curvature_offset.png")
np.savez(DATA / "v1_curvature.npz",
         r_max=r_hat, kap_half=kh_hat, phi0=phi_hat, lz_gain_out=lz_fit - lz0_out,
         r_max_rev=r_hat_r, kap_half_rev=kh_hat_r, phi0_rev=phi_hat_r,
         lz_gain_rev=lz_fit_r - lz0_ret,
         rows=np.array([(a, b, f, lz) for _, a, b, f, lz in rows]),
         SG=SG, r_of_s=r_of_s, kag=kag)
print("[V1] saved data/v1_curvature.npz")
