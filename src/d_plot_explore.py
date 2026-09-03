import numpy as np, cv2, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import OUT, imread_u

r = np.load(OUT / "explore_diff.npy")
bg = cv2.cvtColor(imread_u(OUT / "background_median.png"), cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
ax[0].imshow(bg)
sc = ax[0].scatter(r[:, 2], r[:, 3], c=r[:, 0], cmap="turbo", s=8)
plt.colorbar(sc, ax=ax[0], label="frame")
ax[0].set_title("raw argmax-of-difference trajectory (no gating)")
ax[1].plot(r[:, 0], r[:, 4]); ax[1].set_yscale("log")
ax[1].set_xlabel("frame"); ax[1].set_ylabel("moving area (px, log)")
ax[1].set_title("moving-pixel area  -> spikes = other motion / occlusion")
plt.tight_layout(); plt.savefig(OUT / "figs" / "d_explore.png", dpi=110)
print("saved out/figs/d_explore.png")

# how jumpy is the raw argmax? (a proxy for how much gating we need)
d = np.linalg.norm(np.diff(r[:, 2:4], axis=0), axis=1)
print(f"frame-to-frame jump px: median {np.median(d):.1f}  p90 {np.percentile(d,90):.1f}  max {d.max():.1f}")
print(f"fraction of frames jumping >50 px: {100*np.mean(d>50):.1f}%")
