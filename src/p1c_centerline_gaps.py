"""P1c - if the residual is location-bound and localised, is the CENTERLINE
itself locally wrong rather than the camera?

P1b established the "no good match" failures are location-bound (98% of
twice-visited bad spots are bad on both passes) and confined to 2-3 places,
while the other 90% of frames sit 4.3 px from the projected curve. A wrong
focal length produces a SMOOTH, GLOBAL displacement field - it cannot leave
most of the track at 4 px and two specific spots at 40 px.

A local geometry error can. Path 2.csv has only 71 points over 190 mm, spacing
1.54 to 8.47 mm. A cubic spline through widely-spaced samples cuts corners at
high curvature, displacing the reconstructed centerline from the true vessel
axis by an amount that grows with (spacing^2 x curvature).

Test: locate the bad arc lengths, then check the ORIGINAL CSV sampling and
curvature there against the rest of the path.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import CubicSpline
from config import DATA, CSV
from a_centerline import load as load_cl

fin = np.load(DATA / "i_final.npz")
UV, Pg, sg, scale = fin["UV"], fin["Pg"], fin["sg"], float(fin["scale"])
p1 = np.load(DATA / "p1_excursion.npz")
d_min, nomatch = p1["d_min"], p1["nomatch"]
s3, C3, T3, kap = load_cl()

# arc length of the projected-curve point nearest each observation
_, jnear = cKDTree(Pg).query(UV)
s_at = sg[jnear]
s_bad = s_at[nomatch]
print(f"[P1c] the {nomatch.sum()} no-match frames sit nearest these arc lengths:")
hist, edges = np.histogram(s_bad, bins=19, range=(0, s3[-1]))
bad_bins = [(edges[i], edges[i + 1], hist[i]) for i in range(len(hist)) if hist[i] >= 5]
for a, b, n in bad_bins:
    print(f"      s = {a:6.1f} - {b:6.1f} mm : {n:3d} frames")

# --------------------------------------------- original CSV sampling & curvature
P = np.loadtxt(CSV, delimiter=",", skiprows=2)
seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
s_knot = np.concatenate([[0], np.cumsum(seg)])
print(f"\n[P1c] original CSV: {len(P)} points, spacing min {seg.min():.2f} "
      f"mean {seg.mean():.2f} max {seg.max():.2f} mm")

def local_stats(s0, s1):
    m = (s_knot[:-1] >= s0 - 5) & (s_knot[:-1] <= s1 + 5)
    mk = (s3 >= s0) & (s3 <= s1)
    return (seg[m].max() if m.any() else np.nan,
            seg[m].mean() if m.any() else np.nan,
            kap[mk].max() if mk.any() else np.nan,
            m.sum())

print(f"\n[P1c] sampling and curvature at the bad arc lengths vs the whole path")
print(f"{'region':<22}{'max gap mm':>12}{'mean gap':>10}{'max curv 1/mm':>15}{'corner-cut mm':>15}")
allmax, allmean = seg.max(), seg.mean()
print(f"{'whole path':<22}{allmax:12.2f}{allmean:10.2f}{kap.max():15.4f}"
      f"{allmax**2*kap.max()/8:15.2f}")
for a, b, n in bad_bins:
    g, gm, kk, cnt = local_stats(a, b)
    # chord-to-arc sagitta for a circular arc of curvature k over a chord of length g
    cut = g ** 2 * kk / 8
    print(f"{f's={a:.0f}-{b:.0f} ({n} frames)':<22}{g:12.2f}{gm:10.2f}{kk:15.4f}{cut:15.2f}")

# --------------------------- how much image displacement would a corner cut make?
print(f"\n[P1c] observed off-curve distance at those spots: "
      f"median {np.median(d_min[nomatch]):.1f} px = {np.median(d_min[nomatch])/scale:.2f} mm")
print(f"      (image scale {scale:.2f} px/mm)")

# -------------------------------- direct check: does denser sampling change the curve?
# compare the operational spline against one built from every OTHER knot,
# to estimate how sensitive the reconstruction is to sample density
half = P[::2]
d2 = np.linalg.norm(np.diff(half, axis=0), axis=1)
s2 = np.concatenate([[0], np.cumsum(d2)])
cs2 = CubicSpline(s2, half, axis=0)
u = np.linspace(0, 1, 4000)
A = CubicSpline(s_knot, P, axis=0)(u * s_knot[-1])
B = cs2(u * s2[-1])
dev = np.linalg.norm(A - B, axis=1)
print(f"\n[P1c] SENSITIVITY: rebuild the spline from every OTHER CSV point (36 vs 71)")
print(f"      resulting centerline moves by median {np.median(dev):.2f} mm, "
      f"p90 {np.percentile(dev,90):.2f} mm, max {dev.max():.2f} mm")
print(f"      -> halving the sample density already moves the curve by this much, so the")
print(f"         71-point original carries an error of the same order at its widest gaps.")
print(f"      -> compare with the {np.median(d_min[nomatch])/scale:.1f} mm displacement actually observed.")
np.savez(DATA / "p1c_gaps.npz", s_at=s_at, s_bad=s_bad, dev=dev)
