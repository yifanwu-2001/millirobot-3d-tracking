"""Stage M2 - what does the HMM's motion model actually buy over simpler baselines?

Nothing in the pipeline so far quantifies this. Two baselines, same camera /
projected centerline / observations as Stage I:

  (a) argmax-per-frame: s_k = argmin_s |proj(C(s)) - uv_k|, independently each
      frame. No motion prior at all - this is what you get from registration
      alone, frame by frame. Exposes exactly how much the temporal model is
      doing.

  (b) monotone DTW: globally match the observed track to the projected curve
      under a HARD monotonicity constraint (s non-decreasing on the outbound
      leg, non-increasing on return) via dynamic programming - a common
      2D/3D roadmapping baseline that encodes "the catheter/robot doesn't
      reverse" without a probabilistic velocity model.

Both are scored the same way as Stage I/I2: reprojection error and the
out-and-back repeatability check.
"""
import numpy as np
from scipy.spatial import cKDTree
from config import DATA, FPS
from e_hmm import ArcHMM

z = np.load(DATA / "i_final.npz")
sg, Pg, UV, k_turn = z["sg"], z["Pg"], z["UV"], int(z["k_turn"])
rp_causal, scale = z["rp"], float(z["scale"])
vit = np.load(DATA / "i2_viterbi.npz"); rp_vit = vit["rp_vit"]
K = len(UV)
tree = cKDTree(Pg)

# ---------------------------------------------------------- (a) argmax/frame
d_nn, idx_nn = tree.query(UV)
s_nn = sg[idx_nn]
rp_nn = d_nn

A, B = np.arange(k_turn + 1), np.arange(k_turn + 1, K)
d_img, ia = cKDTree(UV[A]).query(UV[B]); sel = d_img < 8.0


def repeatability(s_hat, tag):
    dsr = np.abs(s_hat[B][sel] - s_hat[A][ia[sel]])
    print(f"    {tag:<32} out-back |ds| median {np.median(dsr):6.2f} mm  p90 {np.percentile(dsr,90):6.2f} mm")
    return dsr


print("[M2] (a) argmax-per-frame (no motion model at all)")
print(f"    reproj px: median {np.median(rp_nn):.1f}  p90 {np.percentile(rp_nn,90):.1f}  "
      f">30px {100*np.mean(rp_nn>30):.1f}%")
repeatability(s_nn, "argmax/frame")

# --------------------------------------------------------------- (b) monotone DTW
# cost[k, i] = ||Pg[i] - UV[k]||, monotone non-decreasing i as k increases within
# each leg (outbound: s increases; return: s decreases -> flip the grid).
print("\n[M2] (b) monotone dynamic-time-warping (hard monotonicity, no velocity model)")


def monotone_dtw(UV_leg, Pg, forward=True):
    """i_k must be non-decreasing in k. g[k][i] = Dm[k,i] + min_{i'<=i} g[k-1][i'],
    i.e. the curve index may jump ahead for free between frames (frames need not
    cover every curve sample) but never go backward. O(n*m) via an explicit
    running prefix-min (not vectorisable as a one-line recurrence without this
    care - an earlier vectorised attempt silently degenerated to i_k = const)."""
    Pu = Pg if forward else Pg[::-1]
    n, m = len(UV_leg), len(Pu)
    Dm = np.linalg.norm(UV_leg[:, None, :] - Pu[None, :, :], axis=-1)  # (n, m)
    pred = np.zeros((n, m), dtype=np.int32)
    prefix_val = np.zeros(m)                       # any start position is free
    prefix_idx = np.arange(m, dtype=np.int32)
    g = None
    for k in range(n):
        g = Dm[k] + prefix_val
        pred[k] = prefix_idx
        best_v, best_i = g[0], 0
        new_val = np.empty(m); new_idx = np.empty(m, dtype=np.int32)
        for i in range(m):
            if g[i] < best_v:
                best_v, best_i = g[i], i
            new_val[i] = best_v; new_idx[i] = best_i
        prefix_val, prefix_idx = new_val, new_idx
    i = int(np.argmin(g))
    path = np.zeros(n, dtype=np.int32)
    for k in range(n - 1, -1, -1):
        path[k] = i
        i = int(pred[k, i])
    return path if forward else (m - 1 - path)


idx_out = monotone_dtw(UV[A], Pg, forward=True)
idx_ret = monotone_dtw(UV[B], Pg, forward=False)
s_dtw = np.concatenate([sg[idx_out], sg[idx_ret]])
Q_dtw = Pg[np.concatenate([idx_out, idx_ret])]
rp_dtw = np.linalg.norm(Q_dtw - UV, axis=1)
print(f"    reproj px: median {np.median(rp_dtw):.1f}  p90 {np.percentile(rp_dtw,90):.1f}  "
      f">30px {100*np.mean(rp_dtw>30):.1f}%")
repeatability(s_dtw, "monotone DTW")

# ------------------------------------------------------------------- summary
print(f"\n{'method':<28}{'reproj med px':>15}{'p90':>8}{'>30px%':>9}{'out-back med mm':>18}")
for tag, rp, s_hat in [("argmax/frame (no model)", rp_nn, s_nn),
                       ("monotone DTW", rp_dtw, s_dtw),
                       ("HMM causal (Stage I)", rp_causal, None),
                       ("HMM Viterbi (Stage I2)", rp_vit, None)]:
    if s_hat is not None:
        dsr = np.median(np.abs(s_hat[B][sel] - s_hat[A][ia[sel]]))
    else:
        s_ref = np.load(DATA / ("i2_viterbi.npz" if "Viterbi" in tag else "i_final.npz"))
        sv = s_ref["s_vit"] if "Viterbi" in tag else s_ref["s_hat"]
        dsr = np.median(np.abs(sv[B][sel] - sv[A][ia[sel]]))
    print(f"{tag:<28}{np.median(rp):15.1f}{np.percentile(rp,90):8.1f}{100*np.mean(rp>30):9.1f}{dsr:18.2f}")

print("\n[M2] READING: argmax-per-frame is the position-only floor (no motion prior, no")
print("     monotonicity). Comparing it to monotone DTW isolates the value of monotonicity")
print("     alone; comparing DTW to the HMM isolates the value of the probabilistic velocity")
print("     model (which additionally handles noise, permits brief non-monotone jitter such")
print("     as the roll wobble, and gives a real posterior instead of a hard assignment).")

# -------------------------------------------------------- is "better" real?
# argmax/frame's reprojection error is close to zero BY CONSTRUCTION (it directly
# minimises point-to-curve distance with no other constraint), and its out-back
# repeatability is not really testing branch selection: a MEMORYLESS pixel->s
# lookup necessarily returns the same s for the same pixel regardless of when it
# is visited, whether or not that s is the physically correct branch. What it
# cannot do is stay CONSISTENT WITH ITSELF frame-to-frame - check implied speed.
print(f"\n[M2] is the low reproj/repeatability real tracking, or an artefact of no memory?")
print(f"     implied |ds/dt| between consecutive frames (mm/s, HMM v_max=30 mm/s):")
print(f"{'method':<28}{'median':>9}{'p95':>8}{'p99':>8}{'max':>9}{'frac>v_max':>12}")
for tag, s_hat in [("argmax/frame", s_nn), ("monotone DTW", s_dtw)]:
    dsdt = np.abs(np.diff(s_hat)) * FPS
    print(f"{tag:<28}{np.median(dsdt):9.1f}{np.percentile(dsdt,95):8.1f}"
          f"{np.percentile(dsdt,99):8.1f}{dsdt.max():9.1f}{100*np.mean(dsdt>30):11.1f}%")
for tag, key, ff in [("HMM causal", "s_hat", "i_final.npz"), ("HMM Viterbi", "s_vit", "i2_viterbi.npz")]:
    s_hat = np.load(DATA / ff)[key]
    dsdt = np.abs(np.diff(s_hat)) * FPS
    print(f"{tag:<28}{np.median(dsdt):9.1f}{np.percentile(dsdt,95):8.1f}"
          f"{np.percentile(dsdt,99):8.1f}{dsdt.max():9.1f}{100*np.mean(dsdt>30):11.1f}%")
print("\n     If argmax/frame's implied speed regularly blows past what the robot can")
print("     physically do, its low reprojection/repeatability numbers are cheap: it is")
print("     teleporting between self-intersection branches frame to frame, and each")
print("     teleport still lands close to SOME point on the curve (small reproj error)")
print("     while coincidentally preserving the pixel->s lookup's determinism (small")
print("     out-back error) without ever producing a physically coherent trajectory.")
print("     That would mean the out-and-back repeatability check, used throughout this")
print("     project as a ground-truth-free validator, cannot by itself distinguish a")
print("     physically real path from a self-consistent-but-teleporting pixel lookup -")
print("     it must be read ALONGSIDE a speed-plausibility check, not alone.")

np.savez(DATA / "m2_baselines.npz", s_nn=s_nn, rp_nn=rp_nn, s_dtw=s_dtw, rp_dtw=rp_dtw)
print("\n[M2] saved data/m2_baselines.npz")
