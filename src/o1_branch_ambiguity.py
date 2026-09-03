"""Stage O1 - on Viterbi's own failures, was the right branch even reachable?

Two checks, both reusing data already computed (no new heavy search):

(1) Self-intersection multiplicity. For every point on the projected centerline,
    count how many OTHER arc-length positions project within a pixel radius -
    a purely geometric property of Path 2 under the fitted camera, independent
    of any observation. Frames where Viterbi's chosen s sits in a
    high-multiplicity zone are, structurally, the ones where a single view
    cannot disambiguate position from the image alone.

(2) Was a closer point on the curve available? M2's argmax/frame baseline
    (`s_nn`) finds the globally nearest curve point to each observation, with
    NO path-continuity constraint at all. On a frame where Viterbi (rp_vit>30)
    fails, if the unconstrained nearest point (rp_nn) is much closer, the
    correct answer was sitting right next to the observation and the model's
    global path constraint is what pushed the estimate away from it - i.e. a
    branch-selection failure. If rp_nn is ALSO bad, no point on the curve
    explains the observation well and the failure is a detection/registration
    problem instead, not a branch ambiguity.
"""
import numpy as np
from scipy.spatial import cKDTree
from config import DATA

z = np.load(DATA / "i_final.npz")
sg, Pg = z["sg"], z["Pg"]
vit = np.load(DATA / "i2_viterbi.npz"); s_vit, rp_vit, conf_vit = vit["s_vit"], vit["rp_vit"], vit["conf_vit"]
m2 = np.load(DATA / "m2_baselines.npz"); s_nn, rp_nn = m2["s_nn"], m2["rp_nn"]

# --------------------------------------------------- (1) geometric multiplicity
print("[O1] (1) self-intersection multiplicity of the projected centerline")
RAD = 15.0  # px
tree = cKDTree(Pg)
pairs = tree.query_pairs(r=RAD, output_type="ndarray")
mult = np.ones(len(sg), int)
if len(pairs):
    ds = np.abs(sg[pairs[:, 0]] - sg[pairs[:, 1]])
    far = pairs[ds > 5.0]                      # only count crossings from a DIFFERENT branch (not adjacent samples)
    for i, j in far:
        mult[i] += 1; mult[j] += 1
print(f"    fraction of centerline samples with >1 competing branch within {RAD:.0f} px: "
      f"{100*np.mean(mult>1):.1f}%")
print(f"    max multiplicity observed: {mult.max()}")

idx_vit = np.clip(np.searchsorted(sg, s_vit), 0, len(sg) - 1)
mult_at_vit = mult[idx_vit]
bad = rp_vit > 30
print(f"\n    mean local multiplicity on GOOD frames (rp_vit<=30): {mult_at_vit[~bad].mean():.2f}")
print(f"    mean local multiplicity on BAD  frames (rp_vit> 30): {mult_at_vit[bad].mean():.2f}")
print(f"    -> {'confirms' if mult_at_vit[bad].mean() > mult_at_vit[~bad].mean() else 'does NOT support'} "
      f"that failures cluster in geometrically ambiguous zones")

# ------------------------------------------------ (2) was the fix within reach?
print(f"\n[O1] (2) on Viterbi's {bad.sum()} failing frames (rp_vit>30 px), was a nearer")
print(f"    curve point available with NO path constraint at all (M2's argmax/frame)?")
d = rp_vit[bad] - rp_nn[bad]
close_alt = rp_nn[bad] < 15.0
print(f"    argmax/frame reproj on these frames: median {np.median(rp_nn[bad]):.1f} px "
      f"(vs Viterbi's {np.median(rp_vit[bad]):.1f} px)")
print(f"    frames where an unconstrained point within 15 px existed: "
      f"{close_alt.sum()}/{bad.sum()} ({100*close_alt.mean():.1f}%)")
print(f"    of those, how far is that point's arc length from Viterbi's own estimate?")
ds_alt = np.abs(s_nn[bad][close_alt] - s_vit[bad][close_alt])
print(f"      median |s_nn - s_vit| = {np.median(ds_alt):.1f} mm  "
      f"(large => genuinely a DIFFERENT branch, not just numerical noise around the same one)")

print(f"\n[O1] READING:")
print(f"    {100*close_alt.mean():.0f}% of Viterbi's failures have a much closer, unconstrained")
print(f"    curve point sitting {np.median(ds_alt):.0f} mm away in arc length - these are branch-")
print(f"    selection failures the geometry could in principle resolve (a second simultaneous")
print(f"    view, per Stage J, is exactly the fix). The remaining "
      f"{100*(1-close_alt.mean()):.0f}% have")
print(f"    no good curve point available for ANY branch, which points at detection/registration")
print(f"    error instead (Stage C's residual), not at branch ambiguity.")

# confidence as a discriminator between the two failure classes
print(f"\n[O1] does the Viterbi posterior confidence separate these two failure classes?")
print(f"    median conf_vit, branch-selection failures : {np.median(conf_vit[bad][close_alt]):.3f}")
print(f"    median conf_vit, no-good-branch failures    : {np.median(conf_vit[bad][~close_alt]):.3f}")

np.savez(DATA / "o1_branch_ambiguity.npz", mult=mult, mult_at_vit=mult_at_vit,
         close_alt=close_alt, ds_alt_full=np.where(bad, np.nan, np.nan))
print("\n[O1] saved data/o1_branch_ambiguity.npz")
