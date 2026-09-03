"""Stage D2 - clean the raw 2D track: reject outliers, fill gaps, find the turnaround.

The robot traverses the vessel OUT and then BACK along the same path, so the
2D trajectory is a curve traced twice. We split it at the turnaround so the
outbound leg can be used for correspondence-based registration (Stage C).
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.signal import medfilt
from config import DATA, OUT

z = np.load(DATA / "track2d.npz"); uv = z["uv"].copy(); conf = z["conf"]
N = len(uv); k = np.arange(N)

# --- 1. outlier rejection: median filter residual --------------------------
ok = ~np.isnan(uv[:, 0])
uvf = uv.copy()
for c in range(2):
    v = np.interp(k, k[ok], uv[ok, c])
    uvf[:, c] = medfilt(v, 9)
resid = np.linalg.norm(np.nan_to_num(uv - uvf, nan=0.0), axis=1)
outlier = ok & (resid > 25)
print(f"[D2] outliers rejected: {outlier.sum()}  frames {np.where(outlier)[0]}")
good = ok & ~outlier

# --- 2. interpolate gaps, light smoothing ----------------------------------
uvc = np.stack([np.interp(k, k[good], uv[good, c]) for c in range(2)], 1)
uvs = np.stack([medfilt(uvc[:, c], 5) for c in range(2)], 1)

# --- 3. cumulative 2D path length; turnaround = its maximum -----------------
step = np.concatenate([[0], np.linalg.norm(np.diff(uvs, axis=0), axis=1)])
# signed progress: project motion onto a smoothed direction to detect reversal
from scipy.ndimage import uniform_filter1d
sm = uniform_filter1d(uvs, 31, axis=0)
d2 = np.gradient(sm, axis=0)
# reference direction = start -> extreme point
i_far = int(np.argmax(np.linalg.norm(uvs - uvs[0], axis=1)))
print(f"[D2] farthest-from-start frame = {i_far}  at {uvs[i_far].round(1)}  (dist {np.linalg.norm(uvs[i_far]-uvs[0]):.0f} px)")

# robust turnaround: the frame maximising distance-from-start, on a smoothed signal
dist0 = uniform_filter1d(np.linalg.norm(uvs - uvs[0], axis=1), 21)
k_turn = int(np.argmax(dist0))
print(f"[D2] turnaround frame k* = {k_turn}  ({k_turn/30:.1f} s)")
print(f"[D2] outbound = frames 0..{k_turn} ({k_turn+1}), return = {k_turn+1}..{N-1} ({N-1-k_turn})")

# --- 4. how well do outbound and return retrace each other? ----------------
# for each return point, distance to the nearest outbound point (2D)
A = uvs[:k_turn+1]; B = uvs[k_turn+1:]
dmin = np.array([np.min(np.linalg.norm(A - b, axis=1)) for b in B])
print(f"[D2] return-vs-outbound retrace error px: median {np.median(dmin):.1f}  p90 {np.percentile(dmin,90):.1f}  max {dmin.max():.1f}")
print(f"     -> this is a FREE self-consistency check: the pipeline must map both legs to the same arc lengths.")

np.savez(DATA / "track2d_clean.npz", uv=uvs, good=good, k_turn=k_turn,
         path_len_px=step.cumsum()[-1])

fig, ax = plt.subplots(1, 3, figsize=(18, 5))
ax[0].plot(k, uvs[:, 0], label="u"); ax[0].plot(k, uvs[:, 1], label="v")
ax[0].axvline(k_turn, color='r', ls='--', label=f"turnaround k*={k_turn}")
ax[0].legend(); ax[0].set_xlabel("frame"); ax[0].set_title("cleaned 2D track")
ax[1].plot(k, dist0); ax[1].axvline(k_turn, color='r', ls='--')
ax[1].set_xlabel("frame"); ax[1].set_ylabel("px"); ax[1].set_title("distance from start (smoothed)")
ax[2].plot(A[:, 0], A[:, 1], '-', lw=2, label="outbound")
ax[2].plot(B[:, 0], B[:, 1], '--', lw=2, label="return")
ax[2].invert_yaxis(); ax[2].legend(); ax[2].set_aspect('equal'); ax[2].set_title("out vs return retrace")
plt.tight_layout(); plt.savefig(OUT / "figs" / "d2_clean.png", dpi=110)
print("[D2] saved data/track2d_clean.npz, out/figs/d2_clean.png")
