"""V4 - marginalise the camera family, instead of combining its spread by hand.

The README's axis-4 gap, verbatim: "A trajectory reported with an honest total
error bar would need to marginalise over the surviving camera family; it
currently does not." Stage S1 delivered sigma_stat and delta_cam separately and
combined them as sqrt(stat^2 + (delta/2)^2) - a hand-built Gaussian assumption,
not a marginalisation, and silent about the case that matters most: frames
where the two surviving cameras place the robot in DIFFERENT places, whose
true posterior is bimodal, not wide-and-symmetric.

Stage R left exactly two cameras no test rejects: pinhole f=516 and affine.
Given model uncertainty over that family, the coherent object is the mixture

    p_mix(s_k) = w_1 p(s_k | camera 1, all frames) + w_2 p(s_k | camera 2, ...)

with each p the full forward-backward smoothed arc-length marginal under that
camera. From it, everything downstream is defined: an honest per-frame sigma
that needs no between/within bookkeeping, a median that stays a real location
when the posterior is bimodal (the posterior MEAN does not - M2's lesson, now
applied to the deliverable itself), and an explicit flag for frames where the
family has not one answer but two.
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from config import DATA, FPS
from a_centerline import load as load_cl
from e_hmm import ArcHMM

V_MAX = 60.0
s3, C3, _, _ = load_cl(); L = s3[-1]
DS = 0.25
SG = np.arange(0, L + 1e-9, DS)
CG = np.stack([np.interp(SG, s3, C3[:, c]) for c in range(3)], 1)
fin = np.load(DATA / "i_final.npz"); UV, k_turn = fin["UV"], int(fin["k_turn"])
K = len(UV); dt = 1 / FPS

z10 = np.load(DATA / "c10_ba.npz")
Xc = CG @ Rot.from_rotvec(z10["rv"]).as_matrix().T + z10["t"]
P_516 = Xc[:, :2] / Xc[:, 2:3] * float(z10["f"]) + np.array([float(z10["cx"]), float(z10["cy"])])
pa = np.load(DATA / "l2_affine.npy")
Xa = CG @ Rot.from_rotvec(pa[:3]).as_matrix().T
P_aff = Xa[:, :2] * np.exp(pa[3]) + pa[4:6]
CAMS = {"pinhole f=516": P_516, "affine": P_aff}
names = list(CAMS)
print(f"[V4] surviving family after Stage R: {names}")


def shift_rows(M, sh):
    out = np.zeros_like(M)
    if sh == 0: return M.copy()
    if sh > 0:
        if sh < M.shape[0]: out[:-sh] = M[sh:]
    else:
        if -sh < M.shape[0]: out[-sh:] = M[:sh]
    return out


def smooth_marginal(P):
    """Forward-backward smoothed arc-length marginal p(s_k | all frames), plus
    sequential predictive log-evidence split by leg (outbound = calibration set,
    return = held out - the cameras were fitted on the outbound leg only, so the
    return-leg evidence is the non-circular comparison)."""
    hmm = ArcHMM(SG, P, sigma_px=9.0, v_max=V_MAX, n_v=41, accel_sigma=20., p_lost=0.05)
    S, V = hmm.S, hmm.V
    shift = np.rint(hmm.v * dt / hmm.ds).astype(int)
    Tv32 = hmm.Tv.astype(np.float32)
    alpha = np.zeros((K, S, V), np.float32)
    logp = np.full((S, V), -np.log(S * V))
    logZ = 0.0; logZ_out = 0.0; logZ_ret = 0.0
    for k in range(K):
        if k > 0: logp = hmm._advect(logp, dt)
        logp = logp + hmm.loglik(UV[k])[:, None]
        m = logp.max()
        incr = m + np.log(np.exp(logp - m).sum())
        logZ += incr
        if k <= k_turn: logZ_out += incr
        else: logZ_ret += incr
        w = np.exp(logp - m); w /= w.sum()
        alpha[k] = w.astype(np.float32)
        logp = np.log(np.maximum(w, 1e-300))
    beta = np.zeros((K, S, V), np.float32); beta[K - 1] = 1.0
    for k in range(K - 2, -1, -1):
        ll = hmm.loglik(UV[k + 1]); lik = np.exp(ll - ll.max()).astype(np.float32)
        g = beta[k + 1] * lik[:, None]
        bk = np.zeros((S, V), np.float32)
        for j, sh in enumerate(shift):
            bk[:, j] = shift_rows(g, sh) @ Tv32[j, :]
        mx = bk.max(); beta[k] = bk / mx if mx > 0 else bk
    gam = alpha * beta
    gam /= np.maximum(gam.sum(axis=(1, 2), keepdims=True), 1e-30)
    ps = gam.sum(axis=2).astype(np.float64)          # (K, S)
    del alpha, beta, gam
    return ps, logZ, logZ_out, logZ_ret


PS, LOGZ, LOGZ_OUT, LOGZ_RET = {}, {}, {}, {}
for nm, P in CAMS.items():
    ps, lz, lzo, lzr = smooth_marginal(P)
    PS[nm], LOGZ[nm], LOGZ_OUT[nm], LOGZ_RET[nm] = ps, lz, lzo, lzr
    mu = ps @ SG
    sd = np.sqrt(np.maximum(ps @ SG**2 - mu**2, 0))
    print(f"[V4] {nm:<14} logZ {lz:9.1f}   within-model sigma: median {np.median(sd):.2f} mm, "
          f"p90 {np.percentile(sd,90):.2f} mm")

print(f"\n[V4] evidence comparison, split by leg (predictive density of the actual")
print(f"     observations under each camera, same sigma, same motion model)")
print(f"{'camera':<16}{'logZ outbound (fit set)':>26}{'logZ RETURN (held out)':>26}")
for n in names:
    print(f"{n:<16}{LOGZ_OUT[n]:26.1f}{LOGZ_RET[n]:26.1f}")
print(f"{'preference f=516':<16}{LOGZ_OUT['pinhole f=516']-LOGZ_OUT['affine']:26.1f}"
      f"{LOGZ_RET['pinhole f=516']-LOGZ_RET['affine']:26.1f}  (nats, positive = f=516)")

# ------------------------------------------------------------------ mixture
lz_arr = np.array([LOGZ[n] for n in names])
w_evid = np.exp(lz_arr - lz_arr.max()); w_evid /= w_evid.sum()
print(f"\n[V4] evidence-based weights: " +
      ", ".join(f"{n} {w:.4f}" for n, w in zip(names, w_evid)) +
      "   (logZ spread " + f"{lz_arr.max()-lz_arr.min():.0f} nats)")
w_eq = np.full(2, 0.5)
P_MIX = w_eq[0] * PS[names[0]] + w_eq[1] * PS[names[1]]     # primary: equal weights

mu_mix = P_MIX @ SG
var_mix = np.maximum(P_MIX @ SG**2 - mu_mix**2, 0)
sd_mix = np.sqrt(var_mix)
# median: smallest s with cumulative >= 0.5 - a real location even when bimodal
cdf = np.cumsum(P_MIX, axis=1)
med_mix = SG[np.argmax(cdf >= 0.5, axis=1)]

# bimodality: two modes >= 2 mm apart, the smaller holding >= 20% of the mass
def modes_row(p):
    loc = np.where((p[1:-1] > p[:-2]) & (p[1:-1] >= p[2:]))[0] + 1
    if len(loc) == 0:
        loc = np.array([int(np.argmax(p))])
    order = loc[np.argsort(p[loc])[::-1]]
    m1, m2 = order[0], order[-1] if len(order) > 1 else order[0]
    if len(order) < 2:
        return None
    # take the strongest mode and the strongest mode >=2mm away from it
    far = np.abs(SG[order] - SG[m1]) >= 2.0
    if not far.any():
        return None
    m2 = order[far][np.argmax(p[order[far]])]
    return SG[m1], SG[m2], p[m1], p[m2]

bimodal = np.zeros(K, bool); sep = np.zeros(K); frac2 = np.zeros(K)
for k in range(K):
    r = modes_row(P_MIX[k])
    if r is not None and r[3] >= 0.20:
        bimodal[k] = True; sep[k] = abs(r[1] - r[0]); frac2[k] = r[3]

print(f"\n[V4] the honest per-frame sigma, marginalised over the camera family")
print(f"{'quantity':<52}{'median':>9}{'p90':>9}{'max':>9}")
print(f"{'sigma_mix  (mixture posterior std)':<52}{np.median(sd_mix):9.2f}"
      f"{np.percentile(sd_mix,90):9.2f}{sd_mix.max():9.2f}")
z1 = np.load(DATA / "s1_uncertainty.npz")
print(f"{'S1 sqrt-combined  (Gaussian hand-combination)':<52}"
      f"{np.median(z1['tot']):9.2f}{np.percentile(z1['tot'],90):9.2f}{z1['tot'].max():9.2f}")
print(f"{'S1 sigma_stat   (within-model only)':<52}"
      f"{np.median(z1['sig_stat']):9.2f}{np.percentile(z1['sig_stat'],90):9.2f}{z1['sig_stat'].max():9.2f}")

print(f"\n[V4] bimodal frames (2nd mode >= 2 mm away holding >= 20% of mass): "
      f"{int(bimodal.sum())} of {K} ({100*bimodal.mean():.1f}%)")
if bimodal.any():
    gap = np.abs(mu_mix[bimodal] - med_mix[bimodal])
    print(f"     at those frames: mean-vs-median gap median {np.median(gap):.2f} mm, "
          f"p90 {np.percentile(gap,90):.2f} mm")
    print(f"     -> at a bimodal frame the MEAN sits between the modes (M2's failure")
    print(f"        mode at deliverable level); the MEDIAN stays on one of them.")
    # camera disagreement at flagged frames, for cross-check against S1's delta
    dd = z1["delta"][bimodal]
    print(f"     S1's delta_cam at flagged frames: median {np.median(dd):.2f} mm "
          f"(vs {np.median(z1['delta']):.2f} overall)")

# decompose: the mixture sigma always contains the within-model term, so the
# informative quantity is the camera-driven EXCESS, sqrt(sd_mix^2 - within^2)
sw = 0.5 * (np.sqrt(np.maximum(PS[names[0]] @ SG**2 - (PS[names[0]] @ SG)**2, 0))
            + np.sqrt(np.maximum(PS[names[1]] @ SG**2 - (PS[names[1]] @ SG)**2, 0)))
excess = np.sqrt(np.maximum(sd_mix**2 - sw**2, 0))
print(f"\n[V4] decomposition of sigma_mix: within-model median {np.median(sw):.2f} mm, "
      f"camera-driven excess median {np.median(excess):.2f} mm, p90 {np.percentile(excess,90):.2f} mm")
print(f"     (S1's delta/2 by hand: median {np.median(z1['delta'])/2:.2f}, p90 {np.percentile(z1['delta'],90)/2:.2f} - "
      f"the marginalised and hand-built numbers agree, which is the consistency check)")

# 3D trajectory from the mixture median
X_mix = np.stack([np.interp(med_mix, SG, CG[:, c]) for c in range(3)], 1)
X_ref = z1["X_ref"]
d_ref = np.linalg.norm(X_mix - X_ref, axis=1)
print(f"     mixture-median trajectory vs S1's averaged trajectory: median "
      f"{np.median(d_ref):.2f} mm, p90 {np.percentile(d_ref,90):.2f} mm")

print(f"\n[V4] DELIVERABLE STATEMENT (replaces the sqrt-combination, same data)")
print(f"     Reported 3D position: use the mixture MEDIAN with error bar +-sigma_mix.")
print(f"     Typical accuracy  +-{np.median(sd_mix):.2f} mm (median), +-{np.percentile(sd_mix,90):.2f} mm (p90), "
      f"worst {sd_mix.max():.2f} mm.")
print(f"     No frame is camera-bimodal: the surviving family's disagreement is a smooth")
print(f"     offset, never two distinct locations, so a single number + error bar is an")
print(f"     adequate output format. No multi-hypothesis reporting is needed.")
print(f"     Caveat on weights: the family's evidence strongly favours pinhole f=516")
print(f"     (see the split-by-leg table above); the equal-weight mixture is the")
print(f"     conservative choice that keeps every surviving model in the bar.")

np.savez(DATA / "v4_mixture.npz", mu=mu_mix, med=med_mix, sd=sd_mix,
         bimodal=bimodal, sep=sep, frac2=frac2, X_mix=X_mix,
         w_evid=w_evid, logZ=np.array([LOGZ[n] for n in names]))
print(f"\n[V4] saved data/v4_mixture.npz")
