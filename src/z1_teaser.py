"""Z1 - the teaser: five pipeline steps, one image and one title each.

Deliberately data-free. No numbers, no annotations, no axis labels - the figure
carries the method, the email carries the evidence. Each panel is a real
artefact of the current deliverable (pinhole f=516, tube-aware Viterbi,
v_max=60, radius policy C), not a cartoon.

    Detect      robot in each frame     detected 2D track over the phantom
    Register    camera from track shape projected centerline on that track
    Constrain   3D state becomes s      the same curve, coloured by arc length
    Decode      HMM over (s, sdot)      per-frame likelihood + the decoded path
    Lift        s back to 3D            the recovered 3D trajectory
"""
import numpy as np, cv2, io, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.spatial.transform import Rotation as Rot
from config import DATA, OUT, FPS, imread_u, imwrite_u
from a_centerline import load as load_cl
from e_hmm import ArcHMM

INK, MUT = "#1a1a1a", "#8a949a"
plt.rcParams.update({"figure.dpi": 110})

s3, C3, _, _ = load_cl(); L = s3[-1]
SG = np.arange(0, L + 1e-9, .25)
CG = np.stack([np.interp(SG, s3, C3[:, c]) for c in range(3)], 1)
fin = np.load(DATA / "i_final.npz"); UV, k_turn = fin["UV"], int(fin["k_turn"])
K = len(UV); tt = np.arange(K) / FPS
raw = np.load(DATA / "track2d.npz")["uv"]
v8 = np.load(DATA / "v8_tube.npz"); s_tb, X_tube = v8["s_tb"], v8["X_tube"]
z10 = np.load(DATA / "c10_ba.npz")
Xc = CG @ Rot.from_rotvec(z10["rv"]).as_matrix().T + np.asarray(z10["t"]).ravel()
P_ax = Xc[:, :2] / Xc[:, 2:3] * float(z10["f"]) + np.array([float(z10["cx"]), float(z10["cy"])])
bg = cv2.cvtColor(imread_u(OUT / "background_median.png"), cv2.COLOR_BGR2RGB)

# square crop around everything that must be visible, so all five panels share
# one shape and the row reads as a strip rather than as five different figures
g = np.isfinite(raw[:, 0])
cx0 = min(raw[g, 0].min(), P_ax[:, 0].min()) - 25
cx1 = max(raw[g, 0].max(), P_ax[:, 0].max()) + 25
cy0 = min(raw[g, 1].min(), P_ax[:, 1].min()) - 25
cy1 = max(raw[g, 1].max(), P_ax[:, 1].max()) + 25
side = max(cx1 - cx0, cy1 - cy0)
mx, my = (cx0 + cx1) / 2, (cy0 + cy1) / 2
x0 = int(np.clip(mx - side / 2, 0, 960 - 1)); x1 = int(np.clip(mx + side / 2, 1, 960))
y0 = int(np.clip(my - side / 2, 0, 720 - 1)); y1 = int(np.clip(my + side / 2, 1, 720))
plate = (bg[y0:y1, x0:x1] * .52 + 255 * .48).astype(np.uint8)

fig = plt.figure(figsize=(21.5, 5.35))
W, GAP, Y, Hh = .1735, .0185, .055, .70
steps = [("Detect", "robot in each frame"),
         ("Register", "camera from track shape"),
         ("Constrain", "3D state becomes $s$"),
         ("Decode", "HMM over $(s,\\dot{s})$"),
         ("Lift", "$s$ back to 3D")]
axes = []
for i in range(5):
    left = .012 + i * (W + GAP)
    proj = "3d" if i == 4 else None
    axes.append(fig.add_axes([left, Y, W, Hh], projection=proj))
    fig.text(left, Y + Hh + .085, steps[i][0], fontsize=17, fontweight="bold", color=INK)
    fig.text(left, Y + Hh + .028, steps[i][1], fontsize=11.5, color=MUT)

for i in range(4):
    left = .012 + i * (W + GAP) + W
    fig.patches.append(FancyArrowPatch((left + .0025, Y + Hh / 2), (left + GAP - .0025, Y + Hh / 2),
                                       transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=19, lw=2.0, color="#b9c3c9"))

def bare(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color("#dce3e7")

# ---------------------------------------------------------------- 1 detect
ax = axes[0]; ax.imshow(plate); bare(ax)
ax.scatter(raw[g, 0] - x0, raw[g, 1] - y0, c=tt[g], cmap="viridis", s=6.5, zorder=3)

# ---------------------------------------------------------------- 2 register
ax = axes[1]; ax.imshow(plate); bare(ax)
ax.scatter(raw[g, 0] - x0, raw[g, 1] - y0, color="#9fb0b9", s=5.5, zorder=2)
ax.plot(P_ax[:, 0] - x0, P_ax[:, 1] - y0, color="#c0392b", lw=3.0, zorder=4)

# ---------------------------------------------------------------- 3 constrain
ax = axes[2]; ax.imshow(plate); bare(ax)
ax.scatter(P_ax[:, 0] - x0, P_ax[:, 1] - y0, c=SG, cmap="plasma", s=13, zorder=4)

# ---------------------------------------------------------------- 4 decode
ax = axes[3]; bare(ax)
ax.set_facecolor("#f6f9fb")
ax.fill_between(tt, 0, s_tb, color="#cfe0ec", alpha=.55)
ax.plot(tt[:k_turn + 1], s_tb[:k_turn + 1], color="#0b6fa4", lw=3.2)
ax.plot(tt[k_turn:], s_tb[k_turn:], color="#d97706", lw=3.2)
ax.set_xlim(0, tt[-1]); ax.set_ylim(0, L * 1.04)
ax.set_aspect(tt[-1] / (L * 1.04))

# ---------------------------------------------------------------- 5 lift
ax = axes[4]; ax.set_axis_off()
ax.plot(C3[:, 0], C3[:, 1], C3[:, 2], color="#ccd6dc", lw=2.6, zorder=1)
ax.scatter(X_tube[:, 0], X_tube[:, 1], X_tube[:, 2], c=tt, cmap="viridis",
           s=9, depthshade=False, zorder=3)
ax.set_box_aspect((2.6, 4.4, 2.4), zoom=1.55)
# same base view as the demo clip: on-screen travel direction matches the video
# (right-to-left). Verified by projecting start and turnaround through
# matplotlib's transform: cos = +1.00 here, -0.85 at the earlier azim -58.
ax.view_init(elev=28, azim=152)

fig.text(.012, .935, "Real-time 3D localisation of a magnetic millirobot from a single 2D view",
         fontsize=20, fontweight="bold", color=INK)

# numpy's tofile (inside imwrite_u) trips over this project's non-ASCII path,
# so write the encoded bytes through pathlib instead.
buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=150, facecolor="w"); buf.seek(0)
png = buf.read()
(OUT / "figs" / "TEASER.png").write_bytes(png)
buf2 = io.BytesIO(); plt.savefig(buf2, format="pdf", facecolor="w"); buf2.seek(0)
(OUT / "figs" / "TEASER.pdf").write_bytes(buf2.read())
arr = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
print(f"[Z1] saved out/figs/TEASER.png ({arr.shape[1]}x{arr.shape[0]}) and TEASER.pdf")
