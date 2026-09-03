"""P1 - are O1's "no good match anywhere" failures a REGISTRATION error, or did
the robot actually leave Path 2?

O1 found that 68% of Viterbi's remaining failures have no good match at ANY arc
length on the projected centerline, and attributed this to registration error.
There is a second explanation it did not separate: the robot briefly entered a
side branch (or Path 2 is locally wrong), in which case no amount of camera
calibration would help and the missing data is the vessel TREE, not intrinsics.

The two hypotheses make different predictions, all testable on data in hand:

                            registration error       genuine excursion
  fraction of frames far    large / systematic       small / episodic
  temporal structure        diffuse                  contiguous episodes
  inside the vessel mask    not necessarily          yes, it is in SOME vessel
  visited twice (out+back)  bad both times           bad once (one-off detour)
  spatial pattern           smooth offset field      isolated, off-curve blobs

C12 already showed one episode is bad outbound (171 px) and fine on the return
(6.1 px) at the same arc length - which a static camera error cannot produce.
This script asks whether that generalises to the whole no-match population.
"""
import numpy as np, cv2, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from config import DATA, OUT, W, H, FPS, imread_u

FAIL_PX = 30.0      # same failure threshold used throughout the project
NOMATCH_PX = 30.0   # "no good match anywhere on the curve"

fin = np.load(DATA / "i_final.npz")
UV, Pg, sg, scale = fin["UV"], fin["Pg"], fin["sg"], float(fin["scale"])
k_turn = int(fin["k_turn"])
vit = np.load(DATA / "i2_viterbi.npz")
s_vit, rp_vit, conf_vit = vit["s_vit"], vit["rp_vit"], vit["conf_vit"]
K = len(UV)
mask = imread_u(OUT / "vessel_mask.png", cv2.IMREAD_GRAYSCALE) > 0

# ---------------------------------------------------------------- 1. distances
# distance from each observation to the ENTIRE projected centerline
d_min, j_min = cKDTree(Pg).query(UV)
fail = rp_vit > FAIL_PX
nomatch = fail & (d_min > NOMATCH_PX)
branch = fail & ~nomatch
print(f"[P1] {K} frames | Viterbi failures (rp>{FAIL_PX:.0f}px): {fail.sum()}")
print(f"     of those: no-match {nomatch.sum()} ({100*nomatch.sum()/max(fail.sum(),1):.0f}%),"
      f" branch-selection {branch.sum()} ({100*branch.sum()/max(fail.sum(),1):.0f}%)")
print(f"     (O1 reported 68% / 32%; this reproduces it from the raw distances)")

# ---------------------------------------- 2. is the camera globally wrong?
print(f"\n[P1] TEST 1 - is the camera globally misregistered?")
print(f"     distance from observation to the whole projected curve, ALL frames:")
print(f"       median {np.median(d_min):.1f} px = {np.median(d_min)/scale:.2f} mm")
print(f"       fraction within {NOMATCH_PX:.0f} px of the curve: {100*np.mean(d_min<=NOMATCH_PX):.1f}%")
print(f"     -> a globally wrong camera would push MOST frames off the curve, not"
      f" {100*np.mean(d_min>NOMATCH_PX):.1f}%.")

# ------------------------------------------- 3. are the no-match frames in a vessel?
ui = np.clip(np.rint(UV[:, 0]).astype(int), 0, W - 1)
vi = np.clip(np.rint(UV[:, 1]).astype(int), 0, H - 1)
inmask = mask[vi, ui]
print(f"\n[P1] TEST 2 - are the no-match observations inside SOME vessel?")
print(f"     inside the vessel mask: all frames {100*inmask.mean():.1f}%,"
      f" no-match frames {100*inmask[nomatch].mean():.1f}%")
print(f"     -> if the robot were mis-detected onto background, this would be low.")

# ------------------------------------------------- 4. temporal structure
def episodes(flag, min_len=3):
    d = np.diff(np.r_[0, flag.astype(int), 0])
    st, en = np.where(d == 1)[0], np.where(d == -1)[0]
    return [(a, b) for a, b in zip(st, en) if b - a >= min_len]

eps = episodes(nomatch)
print(f"\n[P1] TEST 3 - temporal structure of the no-match frames")
print(f"     {len(eps)} contiguous episodes of >=3 frames, covering "
      f"{sum(b-a for a,b in eps)}/{nomatch.sum()} no-match frames "
      f"({100*sum(b-a for a,b in eps)/max(nomatch.sum(),1):.0f}%)")
for a, b in sorted(eps, key=lambda e: e[0] - e[1])[:6]:
    leg = "outbound" if b <= k_turn else ("return" if a > k_turn else "spans turnaround")
    print(f"       frames {a:4d}-{b:4d} ({(b-a)/FPS:5.2f} s, {leg:16s}) "
          f"max off-curve {d_min[a:b].max():6.1f} px = {d_min[a:b].max()/scale:5.1f} mm")

# --------------------------------- 5. was that image location visited twice?
print(f"\n[P1] TEST 4 - was each no-match location visited on BOTH legs?")
A = np.arange(k_turn + 1); B = np.arange(k_turn + 1, K)
tA, tB = cKDTree(UV[A]), cKDTree(UV[B])
other = np.zeros(K)
other[A], _ = tB.query(UV[A])          # for outbound frames: nearest return frame
other[B], _ = tA.query(UV[B])          # and vice versa
print(f"     distance to the nearest observation on the OTHER leg (px):")
print(f"       well-matched frames : median {np.median(other[~fail]):6.1f}")
print(f"       no-match frames     : median {np.median(other[nomatch]):6.1f}")
seen_once = other > 25
print(f"     fraction visited ONLY once (>25 px from any frame on the other leg):")
print(f"       well-matched {100*seen_once[~fail].mean():.0f}%   no-match {100*seen_once[nomatch].mean():.0f}%")
print(f"     -> a static camera error is a property of the LOCATION, so it should")
print(f"        recur on both passes. A one-off detour is visited once.")

# ------------------------------ 6. do the episodes depart and return to the same s?
print(f"\n[P1] TEST 5 - loop signature: does the track leave and rejoin the same place?")
for a, b in sorted(eps, key=lambda e: e[0] - e[1])[:4]:
    pre, post = max(0, a - 4), min(K - 1, b + 3)
    s_before, s_after = s_vit[pre], s_vit[post]
    gap_px = np.linalg.norm(UV[pre] - UV[post])
    print(f"     frames {a:4d}-{b:4d}: s before {s_before:6.1f} mm, after {s_after:6.1f} mm"
          f"  (delta {abs(s_after-s_before):5.1f} mm), image gap {gap_px:5.1f} px")
print(f"     -> a detour returns to nearly the same arc length AND the same pixel;")
print(f"        a latch leaves the arc length far away.")

np.savez(DATA / "p1_excursion.npz", d_min=d_min, nomatch=nomatch, branch=branch,
         fail=fail, inmask=inmask, other=other, eps=np.array(eps) if eps else np.zeros((0, 2)))

# ------------------------------------------------------------------ figure
bg = cv2.cvtColor(imread_u(OUT / "background_median.png"), cv2.COLOR_BGR2RGB)
fig, ax = plt.subplots(1, 3, figsize=(19, 5.6))
ax[0].imshow(bg); ax[0].imshow(mask, alpha=.18, cmap="Blues")
ax[0].plot(Pg[:, 0], Pg[:, 1], color='0.3', lw=1.4, label="Path 2 projected")
ax[0].scatter(UV[~fail, 0], UV[~fail, 1], s=4, c='tab:green', label="matched", alpha=.5)
ax[0].scatter(UV[branch, 0], UV[branch, 1], s=22, c='tab:orange', label=f"branch error ({branch.sum()})")
ax[0].scatter(UV[nomatch, 0], UV[nomatch, 1], s=22, c='tab:red', label=f"no match ({nomatch.sum()})")
ax[0].set_xlim(0, W); ax[0].set_ylim(H, 0); ax[0].legend(fontsize=7)
ax[0].set_title("where do the two failure classes sit?")

ax[1].semilogy(np.maximum(d_min, .1), lw=.8, color='0.4')
ax[1].axhline(NOMATCH_PX, color='r', ls='--', label=f"{NOMATCH_PX:.0f} px")
for a, b in eps: ax[1].axvspan(a, b, color='tab:red', alpha=.25)
ax[1].axvline(k_turn, color='b', ls=':', label="turnaround")
ax[1].set_xlabel("frame"); ax[1].set_ylabel("distance to whole curve (px)")
ax[1].legend(fontsize=7); ax[1].set_title("off-curve episodes over time")

ax[2].hist(other[~fail], bins=40, alpha=.6, density=True, label="matched frames")
ax[2].hist(other[nomatch], bins=40, alpha=.6, density=True, label="no-match frames")
ax[2].set_xlabel("distance to nearest frame on the OTHER leg (px)")
ax[2].set_ylabel("density"); ax[2].legend(fontsize=7)
ax[2].set_title("visited once, or twice?")
plt.tight_layout(); plt.savefig(OUT / "figs" / "p1_excursion.png", dpi=110)
print(f"\n[P1] saved out/figs/p1_excursion.png, data/p1_excursion.npz")
