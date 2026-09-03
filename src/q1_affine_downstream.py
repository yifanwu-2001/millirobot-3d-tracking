"""Q1 - recompute every downstream stage under the NEW working camera.

Stages I, I2, I3, M1, M2 and O1 were all computed with the pinhole `c8_11`
camera, causal filter and v_max=30. P0/P0b replaced all three defaults (affine
camera, Viterbi, v_max=60) and more than halved the failure rate, so those
stages' numbers - and in particular O1's 68%/32% failure-mode split, which the
whole priority argument rests on - have to be re-derived, not carried over.

Everything here reuses the original stages' own definitions so the old and new
columns are directly comparable.
"""
import numpy as np, time
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from config import DATA, OUT, FPS
from a_centerline import load as load_cl
from e_hmm import ArcHMM

V_MAX = 60.0
s3, C3, _, _ = load_cl(); L = s3[-1]
DS = 0.25
SG = np.arange(0, L + 1e-9, DS)
CG = np.stack([np.interp(SG, s3, C3[:, c]) for c in range(3)], 1)
fin = np.load(DATA / "i_final.npz")
UV, k_turn = fin["UV"], int(fin["k_turn"])
K, dt = len(UV), 1 / FPS
A_LEG, B_LEG = np.arange(k_turn + 1), np.arange(k_turn + 1, K)

pa = np.load(DATA / "l2_affine.npy")
Xc_aff = CG @ Rot.from_rotvec(pa[:3]).as_matrix().T
SCALE = np.exp(pa[3])
PG = Xc_aff[:, :2] * SCALE + pa[4:6]
print(f"[Q1] affine camera: scale {SCALE:.2f} px/mm, v_max {V_MAX:.0f} mm/s")

hmm = ArcHMM(SG, PG, sigma_px=9.0, v_max=V_MAX, n_v=41, accel_sigma=20., p_lost=0.05)
S, V = hmm.S, hmm.V
shift = np.rint(hmm.v * dt / hmm.ds).astype(int)


def rp_of(s_hat):
    idx = np.clip(np.rint(s_hat / DS).astype(int), 0, S - 1)
    return np.linalg.norm(PG[idx] - UV, axis=1), idx


# ------------------------------------------------------------------ I: causal
s_cau, s_std, _ = hmm.forward(UV, dt)
rp_cau, _ = rp_of(s_cau)

# --------------------------------------------------- I2: Viterbi + smoothing
def shift_rows(M, sh):
    out = np.zeros_like(M)
    if sh == 0: return M.copy()
    if sh > 0:
        if sh < M.shape[0]: out[:-sh] = M[sh:]
    else:
        if -sh < M.shape[0]: out[-sh:] = M[:sh]
    return out


t0 = time.time()
alpha = np.zeros((K, S, V), np.float32)
logp = np.full((S, V), -np.log(S * V))
for k in range(K):
    if k > 0: logp = hmm._advect(logp, dt)
    logp = logp + hmm.loglik(UV[k])[:, None]
    logp -= logp.max()
    w = np.exp(logp); w /= w.sum()
    alpha[k] = w.astype(np.float32)
    logp = np.log(np.maximum(w, 1e-300))
beta = np.zeros((K, S, V), np.float32)
beta[K - 1] = 1.0
for k in range(K - 2, -1, -1):
    ll = hmm.loglik(UV[k + 1]); lik = np.exp(ll - ll.max())
    g = beta[k + 1] * lik[:, None].astype(np.float32)
    bk = np.zeros((S, V), np.float32)
    for j, sh in enumerate(shift):
        bk[:, j] = shift_rows(g, sh) @ hmm.Tv[j, :].astype(np.float32)
    m = bk.max(); beta[k] = bk / m if m > 0 else bk
gamma = alpha * beta
gamma /= np.maximum(gamma.sum(axis=(1, 2), keepdims=True), 1e-30)
p_s = gamma.sum(axis=2)
del alpha, beta, gamma
s_vit = hmm.viterbi(UV, dt)
rp_vit, idx_vit = rp_of(s_vit)
conf_vit = p_s[np.arange(K), idx_vit]
print(f"[Q1] forward-backward + Viterbi in {time.time()-t0:.1f} s")

OLD = {"causal": (5.6, 48.5, 16.2), "viterbi": (5.6, 35.1, 14.0)}
print(f"\n[Q1] Stage I / I2 under the new camera")
print(f"{'':<26}{'reproj':>9}{'p90':>8}{'>30px%':>9}   {'(old pinhole/v30)':>22}")
for nm, rp in [("causal", rp_cau), ("viterbi", rp_vit)]:
    o = OLD[nm]
    print(f"{nm:<26}{np.median(rp):9.1f}{np.percentile(rp,90):8.1f}{100*np.mean(rp>30):9.1f}"
          f"   {f'{o[0]} / {o[1]} / {o[2]}%':>22}")

# ------------------------------------------------------------ I2 worst quarter
gate = s_std > np.percentile(s_std, 75)
print(f"\n[Q1] on the causal filter's least-confident 25% (n={gate.sum()}):")
print(f"{'causal, these frames':<26}{np.median(rp_cau[gate]):9.1f}"
      f"{np.percentile(rp_cau[gate],90):8.1f}{100*np.mean(rp_cau[gate]>30):9.1f}"
      f"   {'old: 17.5 / 171.4 / 46.9%':>22}")
print(f"{'viterbi, same frames':<26}{np.median(rp_vit[gate]):9.1f}"
      f"{np.percentile(rp_vit[gate],90):8.1f}{100*np.mean(rp_vit[gate]>30):9.1f}"
      f"   {'old:  7.3 /  58.9 / 22.7%':>22}")

# ------------------------------------------------------------------- I3 sweep
print(f"\n[Q1] I3 abstention sweep on the smoothed posterior (Viterbi failures)")
bad = rp_vit > 30
print(f"     confidence on bad frames {np.median(conf_vit[bad]) if bad.any() else float('nan'):.3f}"
      f" vs good {np.median(conf_vit[~bad]):.3f}")
print(f"{'thresh p<=':>12}{'flagged%':>10}{'recall':>9}{'precision':>11}{'kept >30px%':>13}")
for th in (0.01, 0.02, 0.05, 0.10):
    fl = conf_vit <= th
    rec = fl[bad].mean() if bad.any() else 0.0
    prec = bad[fl].mean() if fl.any() else 0.0
    kept = 100 * np.mean(rp_vit[~fl] > 30) if (~fl).any() else 0.0
    print(f"{th:12.2f}{100*fl.mean():10.1f}{rec:9.2f}{prec:11.2f}{kept:13.1f}")

# ---------------------------------------------------------------- M1 held out
print(f"\n[Q1] M1 generalisation split (calibration used the OUTBOUND leg only)")
print(f"{'':<26}{'outbound':>10}{'held-out':>10}{'ratio':>8}   {'(old)':>16}")
for nm, rp, old in [("causal reproj med", rp_cau, "4.4 / 12.8 px"),
                    ("viterbi reproj med", rp_vit, "4.7 / 11.1 px")]:
    a, b = np.median(rp[A_LEG]), np.median(rp[B_LEG])
    print(f"{nm:<26}{a:10.1f}{b:10.1f}{b/max(a,1e-9):8.1f}x   {old:>16}")
for nm, rp, old in [("causal >30px%", rp_cau, "11.2 / 25.7%"),
                    ("viterbi >30px%", rp_vit, "9.2 / 22.9%")]:
    a, b = 100*np.mean(rp[A_LEG] > 30), 100*np.mean(rp[B_LEG] > 30)
    print(f"{nm:<26}{a:10.1f}{b:10.1f}{b/max(a,1e-9):8.1f}x   {old:>16}")

# ------------------------------------------------------- O1 failure-mode split
print(f"\n[Q1] O1 failure-mode split, re-derived (this is what the priority rests on)")
d_min, jn = cKDTree(PG).query(UV)
tree_P = cKDTree(PG)
mult = np.array([len(tree_P.query_ball_point(PG[i], 15.0)) for i in range(S)])
# collapse contiguous runs: only count arc lengths far away in s
mult_far = np.zeros(S)
for i in range(S):
    nb = np.array(tree_P.query_ball_point(PG[i], 15.0))
    mult_far[i] = 1 + np.sum(np.abs(SG[nb] - SG[i]) > 5.0)
fail = rp_vit > 30
nomatch = fail & (d_min > 30)
branch = fail & ~nomatch
print(f"     Viterbi failures {fail.sum()}  ->  branch-selection {branch.sum()}"
      f" ({100*branch.sum()/max(fail.sum(),1):.0f}%),"
      f" no-match {nomatch.sum()} ({100*nomatch.sum()/max(fail.sum(),1):.0f}%)")
print(f"     OLD (pinhole, v30): 143 failures -> 32% branch, 68% no-match")
print(f"     local multiplicity: good frames {mult_far[idx_vit[~fail]].mean():.2f},"
      f" bad frames {mult_far[idx_vit[fail]].mean() if fail.any() else float('nan'):.2f}")
if fail.any():
    print(f"     posterior confidence: branch {np.median(conf_vit[branch]) if branch.any() else float('nan'):.3f}"
          f"  no-match {np.median(conf_vit[nomatch]) if nomatch.any() else float('nan'):.3f}")

# ------------------------------------------------------------ M2 baselines
print(f"\n[Q1] M2 baselines under the same camera (speed bound {V_MAX:.0f} mm/s)")
s_arg = SG[jn]
rp_arg = d_min
def spd_stats(s):
    v = np.abs(np.diff(s)) * FPS
    return 100*np.mean(v > V_MAX + 1e-6), v.max()
print(f"{'method':<26}{'reproj':>9}{'p90':>8}{'>30px%':>9}{'v-viol%':>9}{'max mm/s':>10}")
for nm, s_, rp_ in [("argmax/frame", s_arg, rp_arg),
                    ("HMM causal", s_cau, rp_cau),
                    ("HMM Viterbi", s_vit, rp_vit)]:
    vv, vm = spd_stats(s_)
    print(f"{nm:<26}{np.median(rp_):9.1f}{np.percentile(rp_,90):8.1f}"
          f"{100*np.mean(rp_>30):9.1f}{vv:9.1f}{vm:10.0f}")

np.savez(DATA / "q1_affine_downstream.npz", s_cau=s_cau, s_vit=s_vit, s_std=s_std,
         rp_cau=rp_cau, rp_vit=rp_vit, conf_vit=conf_vit, p_s=p_s.astype(np.float32),
         d_min=d_min, nomatch=nomatch, branch=branch, PG=PG, mult_far=mult_far)
print(f"\n[Q1] saved data/q1_affine_downstream.npz")
