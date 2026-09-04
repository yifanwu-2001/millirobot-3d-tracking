"""S2 - the presentation figure set.

The report is 1222 lines. This builds the two figures a 25-minute talk actually
needs: one that answers "how accurate is it", one that shows the evidence
behind the camera choice and the acquisition trade-offs. Everything is redrawn
from saved artefacts, nothing is recomputed.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from config import DATA, OUT, FPS
from a_centerline import load as load_cl

s3, C3, _, _ = load_cl()
u = np.load(DATA / "s1_uncertainty.npz")
X_ref, sig_stat, delta, tot = u["X_ref"], u["sig_stat"], u["delta"], u["tot"]
K = len(X_ref); t = np.arange(K) / FPS

# ============================================================ FIG A: the answer
fig = plt.figure(figsize=(17, 9))
gs = GridSpec(2, 3, figure=fig, height_ratios=[1.25, 1])

ax = fig.add_subplot(gs[0, :2], projection="3d")
ax.plot(C3[:, 0], C3[:, 1], C3[:, 2], color="0.75", lw=1.5, label="vessel centerline (Path 2)")
sc = ax.scatter(X_ref[:, 0], X_ref[:, 1], X_ref[:, 2], c=t, cmap="turbo", s=7)
plt.colorbar(sc, ax=ax, pad=.12, shrink=.7, label="time (s)")
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_zlabel("z (mm)")
ax.set_title("Recovered 3D trajectory from a single 2D view", fontsize=12)
ax.legend(fontsize=8, loc="upper left")
ax.view_init(elev=22, azim=-62)

axb = fig.add_subplot(gs[0, 2])
axb.barh(["combined\n(p90)", "combined\n(median)", "camera\nsystematic", "estimator\nnoise"],
         [np.percentile(tot, 90), np.median(tot), np.median(delta) / 2, np.median(sig_stat)],
         color=["#b3452e", "#d98b52", "#4a7fb5", "#6fae7d"])
for i, v in enumerate([np.percentile(tot, 90), np.median(tot),
                       np.median(delta) / 2, np.median(sig_stat)]):
    axb.text(v + .05, i, f"{v:.2f} mm", va="center", fontsize=10)
axb.set_xlim(0, np.percentile(tot, 90) * 1.35)
axb.set_xlabel("mm"); axb.set_title("Accuracy budget", fontsize=12)

axc = fig.add_subplot(gs[1, :])
axc.fill_between(t, 0, sig_stat, alpha=.55, color="#6fae7d",
                 label=f"estimator noise, random (med {np.median(sig_stat):.2f} mm)")
axc.fill_between(t, sig_stat, sig_stat + delta / 2, alpha=.55, color="#4a7fb5",
                 label=f"camera systematic (med {np.median(delta)/2:.2f} mm)")
axc.set_xlabel("time (s)"); axc.set_ylabel("uncertainty (mm)")
axc.set_title("The two error terms are different in kind: only the blue one needs a calibration shot",
              fontsize=11)
axc.legend(fontsize=9, loc="upper left"); axc.grid(alpha=.25)
axc.set_xlim(0, t[-1])
plt.tight_layout()
plt.savefig(OUT / "figs" / "SUMMARY_A_accuracy.png", dpi=115)
print("saved out/figs/SUMMARY_A_accuracy.png")

# ========================================================== FIG B: the evidence
fig, ax = plt.subplots(2, 3, figsize=(18, 9.5))

# B1 camera family, 2D fit cannot separate them
mods = ["pinhole\nf=1298", "pinhole\nf=516", "affine\n(no depth)"]
offc = [4.3, 3.5, 2.7]; fail = [9.4, 3.8, 3.9]
x = np.arange(3); w = .36
ax[0, 0].bar(x - w/2, offc, w, label="off-curve residual (px)", color="#4a7fb5")
ax[0, 0].bar(x + w/2, fail, w, label="tracking failure (%)", color="#d98b52")
ax[0, 0].set_xticks(x); ax[0, 0].set_xticklabels(mods, fontsize=9)
ax[0, 0].legend(fontsize=8); ax[0, 0].set_title("1. Three cameras fit the same 2D track", fontsize=11)
ax[0, 0].grid(alpha=.25, axis="y")

# B2 but they disagree in 3D
pairs = ["1298 vs 516", "1298 vs affine", "516 vs affine"]
med = [4.25, 5.00, 0.75]; p90 = [7.85, 10.24, 4.75]
ax[0, 1].bar(np.arange(3) - w/2, med, w, label="median", color="#b3452e")
ax[0, 1].bar(np.arange(3) + w/2, p90, w, label="p90", color="#e0a98a")
ax[0, 1].set_xticks(np.arange(3)); ax[0, 1].set_xticklabels(pairs, fontsize=8)
ax[0, 1].set_ylabel("3D disagreement (mm)"); ax[0, 1].legend(fontsize=8)
ax[0, 1].set_title("2. ...but imply different 3D paths", fontsize=11)
ax[0, 1].grid(alpha=.25, axis="y")

# B3 R1 + R2a resolve it
ax[0, 2].axis("off")
tbl = [["", "R1 size cue", "R2a self-consist.", "verdict"],
       ["f=1298", "0.3 s (pass)", "4.9% >5mm  FAIL", "rejected"],
       ["f=516", "1.9 s (pass)", "0.0%  clean", "survives"],
       ["affine", "3.8 s  FAIL", "0.0%  clean", "rejected"]]
T = ax[0, 2].table(cellText=tbl[1:], colLabels=tbl[0], loc="center", cellLoc="center")
T.auto_set_font_size(False); T.set_fontsize(9); T.scale(1, 1.9)
for j in range(4): T[0, j].set_facecolor("#dfe6ee")
for i, c in [(1, "#f6d5cd"), (2, "#d6ecd9"), (3, "#f6d5cd")]:
    for j in range(4): T[i, j].set_facecolor(c)
ax[0, 2].set_title("3. Two tests not graded on the camera's own exam", fontsize=11)

# B4 dose / frame rate
fps = [30, 15, 10, 6, 3, 2]; rp_fps = [5.6, 5.5, 5.4, 5.2, 5.1, 5.2]
sig = [9.0, 10.3, 13.5, 21.9, 45.9]; fail_sig = [16.2, 16.5, 19.5, 42.1, 81.3]
ax[1, 0].plot(fps, rp_fps, "o-", color="#6fae7d", lw=2)
ax[1, 0].set_xlabel("frame rate (fps)"); ax[1, 0].set_ylabel("reproj median (px)", color="#4a7f5a")
ax[1, 0].set_ylim(0, 8); ax[1, 0].invert_xaxis(); ax[1, 0].grid(alpha=.25)
ax[1, 0].set_title("4. Frame rate is nearly free...", fontsize=11)

ax[1, 1].plot(sig, fail_sig, "s-", color="#b3452e", lw=2)
ax[1, 1].set_xlabel("effective detection noise (px)"); ax[1, 1].set_ylabel("failure rate (%)")
ax[1, 1].grid(alpha=.25)
ax[1, 1].set_title("5. ...detection SNR is not", fontsize=11)

# B5 biplane
sep = [0, 10, 30, 90]; catas = [0.6, 0.4, 0.0, 0.0]; p90b = [1.91, 1.56, 1.29, 1.11]
ax[1, 2].bar(np.arange(4) - w/2, catas, w, label="catastrophic >5mm (%)", color="#b3452e")
ax[1, 2].bar(np.arange(4) + w/2, p90b, w, label="p90 error (mm)", color="#4a7fb5")
ax[1, 2].set_xticks(np.arange(4))
ax[1, 2].set_xticklabels([f"{s}°" for s in sep])
ax[1, 2].set_xlabel("angle between the two views")
ax[1, 2].legend(fontsize=8); ax[1, 2].grid(alpha=.25, axis="y")
ax[1, 2].set_title("6. A 2nd view: 30° buys ~everything", fontsize=11)

plt.tight_layout()
plt.savefig(OUT / "figs" / "SUMMARY_B_evidence.png", dpi=115)
print("saved out/figs/SUMMARY_B_evidence.png")
