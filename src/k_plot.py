"""Stage K figures - what modelling the off-centerline motion buys."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import DATA, OUT

fig, ax = plt.subplots(1, 3, figsize=(18, 5))

# --- synthetic: error CDF per model ------------------------------------------
z = np.load(DATA / "k_synth.npz")
for k in z.files:
    e = z[k]
    ax[0].plot(np.sort(e), np.linspace(0, 1, len(e)),
               lw=2, label=f"{k}  (med {np.median(e):.2f} mm)")
ax[0].set_xscale("log"); ax[0].set_xlabel("3D error vs TRUE off-axis position (mm)")
ax[0].set_ylabel("CDF"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
ax[0].set_title("K1 synthetic: (s, sdot) vs (s, sdot, r, phi)")

# --- real: reprojection before / after ---------------------------------------
try:
    r = np.load(DATA / "k2_real.npz")
    for nm, key in [("centerline only", "rp_base"), (f"off-axis r={float(r['r']):.1f} mm", "rp_fit")]:
        v = r[key]
        ax[1].plot(np.sort(v), np.linspace(0, 1, len(v)), lw=2,
                   label=f"{nm}  (med {np.median(v):.1f} px)")
    ax[1].axvline(30, color='r', ls='--', lw=1, label="failure threshold")
    ax[1].set_xscale("log"); ax[1].set_xlabel("reprojection error (px)")
    ax[1].set_ylabel("CDF"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
    ax[1].set_title("K2 real video: reprojection")

    for nm, key in [("centerline only", "dsr_base"), ("off-axis", "dsr_fit")]:
        v = r[key]
        ax[2].plot(np.sort(v), np.linspace(0, 1, len(v)), lw=2,
                   label=f"{nm}  (med {np.median(v):.2f} mm)")
    ax[2].set_xscale("log"); ax[2].set_xlabel("|s_return - s_outbound| (mm)")
    ax[2].set_ylabel("CDF"); ax[2].legend(fontsize=7); ax[2].grid(alpha=.3)
    ax[2].set_title("K2 real video: out-and-back repeatability")
except FileNotFoundError:
    for a in ax[1:]:
        a.text(.5, .5, "run src/k2_real.py first", ha="center", transform=a.transAxes)

plt.tight_layout(); plt.savefig(OUT / "figs" / "k_offaxis.png", dpi=110)
print("saved out/figs/k_offaxis.png")
