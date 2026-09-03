import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import cv2
from config import OUT, DATA, imread_u

z = np.load(DATA / "track2d.npz"); uv, conf, area = z["uv"], z["conf"], z["area"]
bg = cv2.cvtColor(imread_u(OUT / "background_median.png"), cv2.COLOR_BGR2RGB)
vm = imread_u(OUT / "vessel_mask.png", cv2.IMREAD_GRAYSCALE)
ok = ~np.isnan(uv[:, 0]); k = np.arange(len(uv))

fig, ax = plt.subplots(2, 2, figsize=(16, 11))
ax[0,0].imshow(bg); sc = ax[0,0].scatter(uv[ok,0], uv[ok,1], c=k[ok], cmap="turbo", s=6)
plt.colorbar(sc, ax=ax[0,0], label="frame"); ax[0,0].set_title("Stage D: detected 2D trajectory")
ax[0,1].imshow(bg); ax[0,1].imshow(vm, alpha=.35, cmap="Blues"); ax[0,1].set_title("vessel mask (42% of image)")
ax[1,0].plot(k[ok], uv[ok,0], '.', ms=2, label="u (px)"); ax[1,0].plot(k[ok], uv[ok,1], '.', ms=2, label="v (px)")
ax[1,0].legend(); ax[1,0].set_xlabel("frame"); ax[1,0].set_title("2D coordinates vs time  (outliers = vertical spikes)")
j = np.full(len(uv), np.nan); j[1:] = np.linalg.norm(np.diff(uv, axis=0), axis=1)
ax[1,1].semilogy(k, j, '.', ms=3); ax[1,1].axhline(30, color='r', ls='--', label="30 px gate")
ax[1,1].legend(); ax[1,1].set_xlabel("frame"); ax[1,1].set_ylabel("jump px"); ax[1,1].set_title("frame-to-frame jump")
plt.tight_layout(); plt.savefig(OUT / "figs" / "d_track2d.png", dpi=105)

bad = np.where(j > 30)[0]
print(f"frames with jump >30 px: {len(bad)} -> {bad[:40]}")
print(f"lost frames: {np.where(~ok)[0]}")
