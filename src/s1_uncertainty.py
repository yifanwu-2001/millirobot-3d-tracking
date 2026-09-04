"""S1 - the per-frame uncertainty the report has been missing.

Every sigma quoted so far (Stage I's posterior std, I2's smoothed posterior) is
CONDITIONAL ON THE CAMERA BEING CORRECT. It reflects detection noise and
arc-length ambiguity only. Stage R narrowed the camera family to two models
that no test rejects - pinhole f=516 and the affine model - and showed they
disagree by 0.37 mm median / 2.37 mm p90. That systematic has never appeared in
any reported error bar.

This produces the honest quantity: a 3D trajectory with BOTH components,

    X(t)  +-  sigma_stat(t)    within-model, from the smoothed posterior
          +-  delta_cam(t)     between-model, the surviving family's envelope

They are different in kind and are reported separately rather than summed:
sigma_stat is random and shrinks with more frames or better detection;
delta_cam is systematic and shrinks only with a calibration shot.
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from config import DATA, OUT, FPS
from a_centerline import load as load_cl
from e_hmm import ArcHMM

V_MAX = 60.0
s3, C3, T3, _ = load_cl(); L = s3[-1]
DS = 0.25
SG = np.arange(0, L + 1e-9, DS)
CG = np.stack([np.interp(SG, s3, C3[:, c]) for c in range(3)], 1)
fin = np.load(DATA / "i_final.npz"); UV, k_turn = fin["UV"], int(fin["k_turn"])
K = len(UV); dt = 1 / FPS

# ------------------------------------------------- the two surviving cameras
z10 = np.load(DATA / "c10_ba.npz")
Xc = CG @ Rot.from_rotvec(z10["rv"]).as_matrix().T + z10["t"]
P_516 = Xc[:, :2] / Xc[:, 2:3] * float(z10["f"]) + np.array([float(z10["cx"]), float(z10["cy"])])
pa = np.load(DATA / "l2_affine.npy")
Xa = CG @ Rot.from_rotvec(pa[:3]).as_matrix().T
P_aff = Xa[:, :2] * np.exp(pa[3]) + pa[4:6]
CAMS = {"pinhole f=516": P_516, "affine": P_aff}
print(f"[S1] surviving camera family after Stage R: {list(CAMS)}")


def shift_rows(M, sh):
    out = np.zeros_like(M)
    if sh == 0: return M.copy()
    if sh > 0:
        if sh < M.shape[0]: out[:-sh] = M[sh:]
    else:
        if -sh < M.shape[0]: out[-sh:] = M[:sh]
    return out


def run(P):
    """Viterbi path plus a smoothed per-frame arc-length std."""
    hmm = ArcHMM(SG, P, sigma_px=9.0, v_max=V_MAX, n_v=41, accel_sigma=20., p_lost=0.05)
    S, V = hmm.S, hmm.V
    shift = np.rint(hmm.v * dt / hmm.ds).astype(int)
    alpha = np.zeros((K, S, V), np.float32)
    logp = np.full((S, V), -np.log(S * V))
    for k in range(K):
        if k > 0: logp = hmm._advect(logp, dt)
        logp = logp + hmm.loglik(UV[k])[:, None]
        logp -= logp.max()
        w = np.exp(logp); w /= w.sum()
        alpha[k] = w.astype(np.float32)
        logp = np.log(np.maximum(w, 1e-300))
    beta = np.zeros((K, S, V), np.float32); beta[K - 1] = 1.0
    for k in range(K - 2, -1, -1):
        ll = hmm.loglik(UV[k + 1]); lik = np.exp(ll - ll.max()).astype(np.float32)
        g = beta[k + 1] * lik[:, None]
        bk = np.zeros((S, V), np.float32)
        for j, sh in enumerate(shift):
            bk[:, j] = shift_rows(g, sh) @ hmm.Tv[j, :].astype(np.float32)
        m = bk.max(); beta[k] = bk / m if m > 0 else bk
    gam = alpha * beta
    gam /= np.maximum(gam.sum(axis=(1, 2), keepdims=True), 1e-30)
    ps = gam.sum(axis=2)                      # (K, S) smoothed arc-length marginal
    del alpha, beta, gam
    s_v = hmm.viterbi(UV, dt)
    mu = ps @ SG
    var = ps @ (SG ** 2) - mu ** 2
    sig = np.sqrt(np.maximum(var, 0.0))       # mm of arc length
    idx = np.clip(np.rint(s_v / DS).astype(int), 0, S - 1)
    rp = np.linalg.norm(P[idx] - UV, axis=1)
    return s_v, sig, rp


out = {}
for nm, P in CAMS.items():
    s_v, sig, rp = run(P)
    X = np.stack([np.interp(s_v, SG, CG[:, c]) for c in range(3)], 1)
    out[nm] = dict(s=s_v, sig=sig, X=X, rp=rp)
    print(f"[S1] {nm:<16} reproj med {np.median(rp):5.2f} px   "
          f"sigma_stat med {np.median(sig):5.2f} mm  p90 {np.percentile(sig,90):5.2f} mm")

A, B = list(CAMS)
delta = np.linalg.norm(out[A]["X"] - out[B]["X"], axis=1)
X_ref = 0.5 * (out[A]["X"] + out[B]["X"])
sig_stat = 0.5 * (out[A]["sig"] + out[B]["sig"])

print(f"\n[S1] the two uncertainty components, per frame")
print(f"{'component':<44}{'median':>9}{'p90':>9}{'max':>9}")
print(f"{'sigma_stat  (within-model, random)':<44}{np.median(sig_stat):9.2f}"
      f"{np.percentile(sig_stat,90):9.2f}{sig_stat.max():9.2f}")
print(f"{'delta_cam   (between-model, systematic)':<44}{np.median(delta):9.2f}"
      f"{np.percentile(delta,90):9.2f}{delta.max():9.2f}")
tot = np.sqrt(sig_stat ** 2 + (delta / 2) ** 2)
print(f"{'combined  sqrt(stat^2 + (delta/2)^2)':<44}{np.median(tot):9.2f}"
      f"{np.percentile(tot,90):9.2f}{tot.max():9.2f}")

frac = 100 * np.mean(delta / 2 > sig_stat)
print(f"\n[S1] the camera systematic exceeds the statistical term on "
      f"{frac:.0f}% of frames")
print(f"     -> {'systematic-dominated: a calibration shot is the binding constraint' if frac > 50 else 'statistic-dominated: detection quality is the binding constraint'}")

# where is each component worst?
worst_stat = np.argsort(sig_stat)[-1]
worst_cam = np.argsort(delta)[-1]
print(f"\n[S1] worst statistical frame  {worst_stat:4d} (t={worst_stat/FPS:5.1f}s): "
      f"sigma_stat {sig_stat[worst_stat]:.2f} mm")
print(f"[S1] worst systematic frame   {worst_cam:4d} (t={worst_cam/FPS:5.1f}s): "
      f"delta_cam  {delta[worst_cam]:.2f} mm")

# the deliverable sentence
print(f"\n[S1] DELIVERABLE STATEMENT")
print(f"     Reported 3D position is accurate to about +-{np.median(tot):.2f} mm (median),")
print(f"     +-{np.percentile(tot,90):.2f} mm (p90), worst case {tot.max():.2f} mm, of which")
print(f"     +-{np.median(delta)/2:.2f} mm (median) is an irreducible camera-calibration")
print(f"     systematic and +-{np.median(sig_stat):.2f} mm is estimator noise that more")
print(f"     data or better detection would shrink.")

np.savez(DATA / "s1_uncertainty.npz", X_ref=X_ref, sig_stat=sig_stat, delta=delta,
         tot=tot, **{f"X|{k}": out[k]["X"] for k in out},
         **{f"s|{k}": out[k]["s"] for k in out})
print(f"\n[S1] saved data/s1_uncertainty.npz")
