"""P1b - two decisive tests P1 was missing, plus a retraction of P1's TEST 5.

RETRACTION: P1's "loop signature" test asked whether the track leaves and
rejoins the same arc length. It reported deltas of 17-24 mm and concluded
against a detour. That test was confounded: the robot travels ~18-20 mm/s, so
over a 1 s episode s MUST change by about that much whether or not it detoured.
P1 TEST 5 discriminates nothing and its conclusion is withdrawn.

The two tests that do discriminate:

TEST A - same vessel, or a different one?
  Walk a straight line in the image from the observation to the nearest point
  on the projected Path 2. If that segment stays inside the vessel mask, the
  robot is in the SAME tube as the projected curve and the curve is merely
  locally displaced -> registration/geometry error. If the segment crosses
  background, the observation is in a DIFFERENT vessel -> genuine excursion.

TEST B - is a bad location bad on BOTH passes?
  The route is travelled twice. A static camera or geometry error belongs to
  the LOCATION and must recur on both visits. A one-off detour must not. For
  every no-match frame that has a counterpart on the other leg, check whether
  that counterpart is also a no-match.
"""
import numpy as np, cv2
from scipy.spatial import cKDTree
from config import DATA, OUT, W, H, FPS, imread_u

fin = np.load(DATA / "i_final.npz")
UV, Pg, scale, k_turn = fin["UV"], fin["Pg"], float(fin["scale"]), int(fin["k_turn"])
p1 = np.load(DATA / "p1_excursion.npz")
d_min, nomatch, fail = p1["d_min"], p1["nomatch"], p1["fail"]
mask = imread_u(OUT / "vessel_mask.png", cv2.IMREAD_GRAYSCALE) > 0
K = len(UV)
tree_P = cKDTree(Pg)
_, jnear = tree_P.query(UV)

# ------------------------------------------------------------------ TEST A
def segment_in_mask(p0, p1_, m, n=80):
    """Fraction of a straight image segment that lies inside the mask."""
    t = np.linspace(0, 1, n)[:, None]
    pts = p0[None] * (1 - t) + p1_[None] * t
    xi = np.clip(np.rint(pts[:, 0]).astype(int), 0, W - 1)
    yi = np.clip(np.rint(pts[:, 1]).astype(int), 0, H - 1)
    return m[yi, xi].mean()

frac = np.array([segment_in_mask(UV[k], Pg[jnear[k]], mask) for k in range(K)])
print("[P1b] TEST A - is the observation in the SAME tube as the projected curve?")
print("      fraction of the straight path observation->curve that stays in vessel:")
print(f"        matched frames  : median {np.median(frac[~fail]):.3f}")
print(f"        no-match frames : median {np.median(frac[nomatch]):.3f}")
same_tube = frac > 0.95
print(f"      fully inside the vessel (>95%): matched {100*same_tube[~fail].mean():.0f}%,"
      f" no-match {100*same_tube[nomatch].mean():.0f}%")
crosses = frac < 0.80
print(f"      clearly crosses background (<80%): matched {100*crosses[~fail].mean():.0f}%,"
      f" no-match {100*crosses[nomatch].mean():.0f}%")
print("      -> high 'same tube' on no-match frames = the CURVE is displaced, not the robot.")
print("      -> high 'crosses background'          = the robot is in another vessel.\n")

# ------------------------------------------------------------------ TEST B
A = np.arange(k_turn + 1); B = np.arange(k_turn + 1, K)
tA, tB = cKDTree(UV[A]), cKDTree(UV[B])
dA, iA = tB.query(UV[A])      # outbound -> nearest return frame
dB, iB = tA.query(UV[B])      # return   -> nearest outbound frame
partner = np.zeros(K, int); pdist = np.zeros(K)
partner[A] = B[iA]; pdist[A] = dA
partner[B] = A[iB]; pdist[B] = dB

paired = pdist < 15.0          # the same physical spot really was visited twice
nm_paired = nomatch & paired
print("[P1b] TEST B - when a no-match location IS visited twice, is it bad both times?")
print(f"      no-match frames with a counterpart within 15 px: {nm_paired.sum()}/{nomatch.sum()}")
if nm_paired.sum():
    both_bad = nomatch[partner[nm_paired]]
    print(f"      of those, the counterpart is ALSO a no-match: {both_bad.sum()}/{len(both_bad)}"
          f" ({100*both_bad.mean():.0f}%)")
    base = nomatch[partner[paired & ~fail]].mean()
    print(f"      base rate of a no-match counterpart among matched frames: {100*base:.1f}%")
    print(f"      -> {100*both_bad.mean():.0f}% vs {100*base:.1f}% base:"
          f" {'LOCATION-bound (static error)' if both_bad.mean() > 0.5 else 'NOT location-bound (one-off)'}")
    print(f"      counterpart's own off-curve distance: median {np.median(d_min[partner[nm_paired]]):.1f} px"
          f"  vs this frame's {np.median(d_min[nm_paired]):.1f} px")

# --------------------------------------------------- spatial overlap of episodes
print("\n[P1b] do the outbound and return episodes occupy the SAME image region?")
eps = [(544, 570, "outbound"), (779, 818, "return"), (886, 916, "return")]
cen = {}
for a, b, leg in eps:
    c = UV[a:b].mean(0); cen[(a, b, leg)] = c
    print(f"      frames {a:4d}-{b:4d} ({leg:8s}) centroid ({c[0]:6.1f}, {c[1]:6.1f})"
          f"  off-curve median {np.median(d_min[a:b]):5.1f} px")
ks = list(cen)
for i in range(len(ks)):
    for j in range(i + 1, len(ks)):
        d = np.linalg.norm(cen[ks[i]] - cen[ks[j]])
        same = "SAME region" if d < 80 else "different regions"
        print(f"      {ks[i][0]}-{ks[i][1]} vs {ks[j][0]}-{ks[j][1]}: centroids {d:6.1f} px apart -> {same}")

np.savez(DATA / "p1b_discriminate.npz", frac=frac, partner=partner, pdist=pdist)
print("\n[P1b] saved data/p1b_discriminate.npz")
