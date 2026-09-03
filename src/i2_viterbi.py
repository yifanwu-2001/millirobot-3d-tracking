"""Stage I2 - offline Viterbi decode, with a real posterior confidence, on the
frames Stage I's causal filter is uncertain about.

Stage I's abstention rule (reject the 25% most uncertain causal-posterior
frames) drops the >30 px failure rate from 16.2% to 6.0%, but it is a REJECT
rule, not a FIX: those frames still have no answer. Viterbi looks at the whole
sequence's likelihood at once, so at a self-crossing it can use evidence from
BOTH sides in time, not just the causal past. This is only valid offline
(post-hoc review, not real-time tracking).

Unlike a spline fit to the causal (s_hat, sigma) point estimates, this
computes a genuine forward-backward SMOOTHED marginal gamma_k(s,v) = p(s_k,v_k
| all 1025 frames), so "confidence" on a previously-abstained frame is an
actual posterior probability, not a smoothness heuristic.
"""
import numpy as np, time
from scipy.spatial import cKDTree
from config import DATA, OUT, FPS
from e_hmm import ArcHMM

z = np.load(DATA / "i_final.npz")
sg, Pg, UV, k_turn = z["sg"], z["Pg"], z["UV"], int(z["k_turn"])
rp_causal, s_std_causal, scale = z["rp"], z["s_std"], float(z["scale"])
K, dt = len(UV), 1 / FPS

hmm = ArcHMM(sg, Pg, sigma_px=9.0, v_max=30., n_v=41, accel_sigma=20., p_lost=0.05)
S, V = hmm.S, hmm.V
shift = np.rint(hmm.v * dt / hmm.ds).astype(int)


def shift_rows(M, sh):
    """out[s, :] = M[s + sh, :], zero-padded past the boundary."""
    out = np.zeros_like(M)
    if sh == 0:
        return M.copy()
    elif sh > 0:
        if sh < M.shape[0]:
            out[:-sh] = M[sh:]
    else:
        if -sh < M.shape[0]:
            out[-sh:] = M[:sh]
    return out


t0 = time.time()
print("[I2] forward pass (storing full alpha history for smoothing)")
alpha = np.zeros((K, S, V))
logp = np.full((S, V), -np.log(S * V))
for k in range(K):
    if k > 0:
        logp = hmm._advect(logp, dt)
    logp = logp + hmm.loglik(UV[k])[:, None]
    logp -= logp.max()
    w = np.exp(logp); w /= w.sum()
    alpha[k] = w
    logp = np.log(np.maximum(w, 1e-300))

print("[I2] backward pass")
beta = np.zeros((K, S, V))
beta[K - 1] = 1.0
Tv = hmm.Tv
for k in range(K - 2, -1, -1):
    ll = hmm.loglik(UV[k + 1])
    lik = np.exp(ll - ll.max())
    g = beta[k + 1] * lik[:, None]
    bk = np.zeros((S, V))
    for j, sh in enumerate(shift):
        gs = shift_rows(g, sh)
        bk[:, j] = gs @ Tv[j, :]
    m = bk.max()
    beta[k] = bk / m if m > 0 else bk

gamma = alpha * beta
gamma /= gamma.sum(axis=(1, 2), keepdims=True)
p_s = gamma.sum(axis=2)  # (K, S) smoothed marginal over arc length
print(f"[I2] forward-backward done in {time.time()-t0:.1f} s")

print("\n[I2] Viterbi (globally optimal path)")
s_vit = hmm.viterbi(UV, dt)
ds = sg[1] - sg[0]
idx_vit = np.clip(np.rint(s_vit / ds).astype(int), 0, S - 1)
Q_vit = Pg[idx_vit]
rp_vit = np.linalg.norm(Q_vit - UV, axis=1)
conf_vit = p_s[np.arange(K), idx_vit]

print(f"\n{'':<28}{'reproj px':>10}{'p90 px':>9}{'>30px%':>9}")
print(f"{'causal (Stage I)':<28}{np.median(rp_causal):10.1f}{np.percentile(rp_causal,90):9.1f}"
      f"{100*np.mean(rp_causal>30):9.1f}")
print(f"{'Viterbi (offline)':<28}{np.median(rp_vit):10.1f}{np.percentile(rp_vit,90):9.1f}"
      f"{100*np.mean(rp_vit>30):9.1f}")

gate = s_std_causal > np.percentile(s_std_causal, 75)
print(f"\n[I2] on the 25% of frames Stage I's causal filter is LEAST confident "
      f"(n={gate.sum()}), where the abstention rule currently just drops them:")
print(f"{'':<28}{'reproj px':>10}{'p90 px':>9}{'>30px%':>9}")
print(f"{'causal, these frames':<28}{np.median(rp_causal[gate]):10.1f}"
      f"{np.percentile(rp_causal[gate],90):9.1f}{100*np.mean(rp_causal[gate]>30):9.1f}")
print(f"{'Viterbi, same frames':<28}{np.median(rp_vit[gate]):10.1f}"
      f"{np.percentile(rp_vit[gate],90):9.1f}{100*np.mean(rp_vit[gate]>30):9.1f}")
print(f"{'  median smoothed conf.':<28}{np.median(conf_vit[gate]):10.3f}"
      f"{'  (peak prob. mass at the Viterbi s on that frame, marginalised over v)':>0}")

still_bad = gate & (rp_vit > 30)
print(f"\n[I2] of the {gate.sum()} abstained frames, Viterbi's own reprojection is "
      f"still >30 px on {still_bad.sum()} ({100*still_bad.sum()/gate.sum():.1f}%); "
      f"median smoothed confidence there: {np.median(conf_vit[still_bad]) if still_bad.any() else float('nan'):.3f} "
      f"vs {np.median(conf_vit[gate & ~still_bad]):.3f} on the ones it fixed -- "
      f"if confidence is reliably lower on the frames it still gets wrong, the posterior "
      f"probability is a usable abstention signal even offline.")

A, B = np.arange(k_turn + 1), np.arange(k_turn + 1, K)
d_img, ia = cKDTree(UV[A]).query(UV[B]); sel = d_img < 8.0
dsr_vit = np.abs(s_vit[B][sel] - s_vit[A][ia[sel]])
print(f"\n[I2] out-and-back repeatability, Viterbi path: median {np.median(dsr_vit):.2f} mm "
      f"p90 {np.percentile(dsr_vit,90):.2f} mm  (causal Stage I reference in i_final.png)")

np.savez(DATA / "i2_viterbi.npz", s_vit=s_vit, rp_vit=rp_vit, conf_vit=conf_vit,
         p_s=p_s.astype(np.float32), gate=gate)
print("\n[I2] saved data/i2_viterbi.npz")
