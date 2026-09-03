"""Stage D3 - separate the robot's NET progress along the vessel from the
periodic wobble induced by the rotating magnetic field.

Why this matters for registration, not just for physics: the wobble inflates the
measured 2D arc length (2674 px observed vs <=1397 px geometrically possible at
the fitted scale), and arc-length resampling of a wobbling track oversamples the
wobble. Both corrupt Stage C.
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter, welch
from config import DATA, OUT, FPS

z = np.load(DATA / "track2d_clean.npz"); uv = z["uv"]; k_turn = int(z["k_turn"])
N = len(uv); k = np.arange(N)

# --- smooth trend vs wobble ---------------------------------------------------
trend = savgol_filter(uv, 41, 2, axis=0)         # ~1.4 s window, quadratic
wob = uv - trend
amp = np.linalg.norm(wob, axis=1)
print(f"[D3] wobble amplitude px: median {np.median(amp):.2f}  p90 {np.percentile(amp,90):.2f}  max {amp.max():.2f}")

# --- spectrum of the wobble ---------------------------------------------------
for c, nm in enumerate("uv"):
    fr, Pw = welch(wob[:, c], fs=FPS, nperseg=256)
    pk = fr[np.argmax(Pw[1:]) + 1]
    print(f"[D3] wobble spectrum {nm}: peak at {pk:.2f} Hz  ({FPS/pk:.1f} frames/cycle)")

# --- arc length before vs after ----------------------------------------------
def plen(A): return float(np.linalg.norm(np.diff(A, axis=0), axis=1).sum())
print(f"\n[D3] 2D path length, outbound leg:")
print(f"     raw track      {plen(uv[:k_turn+1]):8.0f} px")
print(f"     savgol trend   {plen(trend[:k_turn+1]):8.0f} px")
for w in (21, 41, 61, 81):
    tr = savgol_filter(uv, w, 2, axis=0)
    print(f"     savgol w={w:3d}     {plen(tr[:k_turn+1]):8.0f} px")

# --- speed profile along the path --------------------------------------------
spd = np.r_[0, np.linalg.norm(np.diff(trend, axis=0), axis=1)] * FPS   # px/s
spd_s = uniform_filter1d(spd, 15)
print(f"\n[D3] net speed px/s: median {np.median(spd_s):.1f}  p90 {np.percentile(spd_s,90):.1f}  max {spd_s.max():.1f}")
print(f"     fraction of frames nearly stationary (<10 px/s): {100*np.mean(spd_s<10):.0f}%")

np.savez(DATA / "track2d_trend.npz", trend=trend, wobble=wob, k_turn=k_turn, speed=spd_s)

fig, ax = plt.subplots(1, 3, figsize=(18, 5))
ax[0].plot(k, uv[:, 0], lw=.7, label="raw u"); ax[0].plot(k, trend[:, 0], lw=2, label="trend u")
ax[0].set_xlim(150, 500); ax[0].legend(); ax[0].set_xlabel("frame"); ax[0].set_title("wobble vs net progress (zoom)")
fr, Pw = welch(wob[:, 0], fs=FPS, nperseg=256)
ax[1].semilogy(fr, Pw); ax[1].set_xlabel("Hz"); ax[1].set_ylabel("PSD"); ax[1].set_title("wobble spectrum (u)")
ax[2].plot(k, spd_s); ax[2].axvline(k_turn, color='r', ls='--', label="turnaround")
ax[2].set_xlabel("frame"); ax[2].set_ylabel("px/s"); ax[2].legend(); ax[2].set_title("net speed along path")
plt.tight_layout(); plt.savefig(OUT / "figs" / "d3_wobble.png", dpi=110)
print("[D3] saved data/track2d_trend.npz, out/figs/d3_wobble.png")
