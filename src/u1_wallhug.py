"""U1 - the off-centerline model K should have tested: a per-leg wall offset.

K1-K5 tested a radial offset ROTATING at 1.17 Hz and correctly rejected it. But
T10 showed what is actually present: the robot rides opposite walls of the same
lumen on the outbound and return passes, up to a full diameter apart, quasi-
statically. That model was never tried.

    X(s, leg) = C(s) + r * ( cos(theta_leg) N1(s) + sin(theta_leg) N2(s) )

with (N1, N2) the parallel-transport frame, `r` constant, and the physical
hypothesis theta_return = theta_outbound + pi (opposite wall). Two global
parameters, not one per frame.

The strong test is out-of-sample and is the one reported: fit (r, theta) on the
OUTBOUND leg alone, flip the angle by pi, and predict the RETURN leg without
ever having seen it. A model that is merely absorbing noise cannot do that.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from config import DATA, OUT, FPS
from a_centerline import load as load_cl
from k_frame import transport_frame
from e_hmm import ArcHMM

V_MAX = 60.0
s3, C3, _, _ = load_cl(); L = s3[-1]
DS = 0.25
SG = np.arange(0, L + 1e-9, DS)
CG = np.stack([np.interp(SG, s3, C3[:, c]) for c in range(3)], 1)
_, N1G, N2G = transport_frame(CG)

fin = np.load(DATA / "i_final.npz"); UV, k_turn = fin["UV"], int(fin["k_turn"])
K = len(UV); dt = 1 / FPS
OUT_LEG = np.zeros(K, bool); OUT_LEG[:k_turn + 1] = True

pa = np.load(DATA / "l2_affine.npy")
R_aff = Rot.from_rotvec(pa[:3]).as_matrix(); SC = np.exp(pa[3]); C0 = pa[4:6]
A0 = (CG @ R_aff.T)[:, :2] * SC + C0
B1 = (N1G @ R_aff.T)[:, :2] * SC
B2 = (N2G @ R_aff.T)[:, :2] * SC
print(f"[U1] affine camera, scale {SC:.2f} px/mm, tube radius ~3.4 mm")


def proj_for(r, th_out, th_ret):
    """Two static projections, one per leg."""
    Po = A0 + r * (np.cos(th_out) * B1 + np.sin(th_out) * B2)
    Pr = A0 + r * (np.cos(th_ret) * B1 + np.sin(th_ret) * B2)
    return Po, Pr


def score(r, th_out, th_ret, frames=None):
    """Run the filter with a leg-dependent projection; return evidence and residual."""
    Po, Pr = proj_for(r, th_out, th_ret)
    hmm = ArcHMM(SG, Po, sigma_px=9.0, v_max=V_MAX, n_v=41, accel_sigma=20., p_lost=0.05)
    logp = np.full((hmm.S, hmm.V), -np.log(hmm.S * hmm.V))
    logZ = 0.0; s_hat = np.zeros(K)
    for k in range(K):
        if k > 0: logp = hmm._advect(logp, dt)
        hmm.P = Po if OUT_LEG[k] else Pr           # swap the measurement model per leg
        ll = hmm.loglik(UV[k])
        if frames is not None and not frames[k]:
            ll = np.zeros_like(ll)                 # excluded frames contribute nothing
        logp = logp + ll[:, None]
        m = logp.max(); logp -= m
        w = np.exp(logp); tot = w.sum(); w /= tot
        logZ += m + np.log(tot)
        ps = w.sum(1); s_hat[k] = (ps * SG).sum()
        logp = np.log(np.maximum(w, 1e-300))
    idx = np.clip(np.rint(s_hat / DS).astype(int), 0, hmm.S - 1)
    P_at = np.where(OUT_LEG[:, None], Po[idx], Pr[idx])
    rp = np.linalg.norm(P_at - UV, axis=1)
    return logZ, rp, s_hat


# ---------------------------------------------------- fit on the OUTBOUND leg only
print(f"\n[U1] fitting (r, theta) on the OUTBOUND leg only ({OUT_LEG.sum()} frames)")
rs = np.arange(0.0, 3.01, 0.25)
ths = np.linspace(0, 2 * np.pi, 16, endpoint=False)
best = None
for r in rs:
    for th in (ths if r > 0 else [0.0]):
        lz, rp, _ = score(r, th, th, frames=OUT_LEG)   # same angle both legs while fitting
        if best is None or lz > best[0]:
            best = (lz, r, th)
    if r == 0.0:
        lz0 = best[0]
lz_fit, r_hat, th_hat = best
print(f"     best: r = {r_hat:.2f} mm ({r_hat*SC:.1f} px), theta = {np.degrees(th_hat):.0f} deg")
print(f"     evidence gain over r=0 on the fitting leg: {lz_fit - lz0:+.1f} nats")

# ------------------------------------------------- out-of-sample: predict the RETURN
print(f"\n[U1] OUT-OF-SAMPLE: predict the return leg. theta_return = theta_out + pi,")
print(f"     nothing about the return leg was used to choose r or theta.")
RET = ~OUT_LEG
rows = []
for name, th_ret in [("centerline (r=0)", None),
                     ("same wall  (theta_ret = theta_out)", th_hat),
                     ("OPPOSITE wall (theta_ret = theta_out + pi)", th_hat + np.pi)]:
    if th_ret is None:
        lz, rp, sh = score(0.0, 0.0, 0.0)
    else:
        lz, rp, sh = score(r_hat, th_hat, th_ret)
    rows.append((name, np.median(rp[OUT_LEG]), np.median(rp[RET]),
                 100 * np.mean(rp[RET] > 30), lz))
print(f"{'model':<44}{'outbound':>10}{'RETURN':>9}{'ret>30px':>10}{'log Z':>10}")
for nm, a, b, f, lz in rows:
    print(f"{nm:<44}{a:10.2f}{b:9.2f}{f:9.1f}%{lz:10.0f}")

base_ret = rows[0][2]; opp_ret = rows[2][2]
print(f"\n[U1] return-leg residual: {base_ret:.2f} -> {opp_ret:.2f} px "
      f"({base_ret/max(opp_ret,1e-9):.2f}x)")
print(f"     M1's held-out gap was the return leg scoring 2.4-2.9x worse than the")
print(f"     outbound. With the opposite-wall model the ratio is "
      f"{opp_ret/max(rows[2][1],1e-9):.2f}x (was {base_ret/max(rows[0][1],1e-9):.2f}x).")

# free theta_return, as a check that pi is really preferred
print(f"\n[U1] check: scan theta_return freely, with r and theta_out fixed from the fit")
print(f"{'theta_ret - theta_out (deg)':>28}{'return resid':>14}{'log Z':>10}")
for dth in np.linspace(0, 2 * np.pi, 9, endpoint=False):
    lz, rp, _ = score(r_hat, th_hat, th_hat + dth)
    mark = "  <== physical prediction" if abs(dth - np.pi) < 1e-6 else ""
    print(f"{np.degrees(dth):28.0f}{np.median(rp[RET]):14.2f}{lz:10.0f}{mark}")

np.savez(DATA / "u1_wallhug.npz", r=r_hat, th=th_hat, rows=np.array(
    [(a, b, f, lz) for _, a, b, f, lz in rows]))
print(f"\n[U1] saved data/u1_wallhug.npz")
