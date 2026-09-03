"""Stage J plots - what a second view (or a moving one) actually buys."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from config import DATA, OUT

z = np.load(DATA / "j3_pairs.npz")
single, pair = z["single"], z["pair"]

fig, ax = plt.subplots(1, 3, figsize=(18, 5))

ax[0].hist(single, bins=40, alpha=.65, density=True, label=f"single view (median {np.median(single):.2f} mm)")
ax[0].hist(pair, bins=40, alpha=.65, density=True, label=f"view pair (median {np.median(pair):.2f} mm)")
ax[0].set_xlabel("min separation between arc-length-distant path points (mm)")
ax[0].set_ylabel("density"); ax[0].legend(fontsize=8)
ax[0].set_title("J3: ambiguity is about the TYPICAL angle, not the best one")

# partner sweep, hard-coded from the J3 run (see src/j3_pair_observability.py output)
th = np.array([0, 10, 20, 30, 45, 60, 75, 90])
sep = np.array([1.786, 2.371, 2.964, 3.436, 3.939, 4.140, 4.350, 4.300])
ax[1].plot(th, sep, 'o-', lw=2)
ax[1].axhline(sep[0], color='r', ls='--', label="single view")
ax[1].set_xlabel("angle between the two views (deg)")
ax[1].set_ylabel("guaranteed separation (mm)")
ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
ax[1].set_title("J3: gain saturates around 60 deg")

# CRB, from src/j4_identifiability.py output
thc = np.array([0, 10, 20, 30, 45, 60, 90])
sf  = np.array([174.7, 119.3, 105.4, 82.3, 83.3, 72.9, 58.7])
scx = np.array([97.6, 61.0, 51.2, 42.6, 41.9, 35.8, 28.8])
ax[2].plot(thc, sf, 'o-', label="sigma_f (px)", lw=2)
ax[2].plot(thc, scx, 's-', label="sigma_cx (px)", lw=2)
ax[2].set_xlabel("angle between the two views (deg)"); ax[2].set_ylabel("Cramer-Rao std error (px)")
ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
ax[2].set_title("J4: calibration improves ~3x, but stays ill conditioned")

plt.tight_layout(); plt.savefig(OUT / "figs" / "j_multiview.png", dpi=110)
print("saved out/figs/j_multiview.png")
