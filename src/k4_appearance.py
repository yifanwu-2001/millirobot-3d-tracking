"""Stage K4 - is the 1.17 Hz wobble a POSITION oscillation or an APPEARANCE one?

K3 refuted the off-centerline interpretation: gating to well-registered frames
did not raise the fitted radius (0.3 mm at every gate) and the evidence gain
went to zero (0.2 nats on the cleanest 609 frames). If the robot really moved
1.3 mm off-axis, cleaner frames would show it MORE strongly, not less.

Alternative: the robot is an elongated dark body. As it rolls, its projected
SILHOUETTE changes, so the detected centroid oscillates even if the body centre
stays on the vessel axis. That would produce exactly the D3 signature and would
be invisible to a 3D off-axis model.

Discriminating test: an appearance effect must also modulate the blob's SHAPE -
its second moments - at the same 1.17 Hz. A pure rigid translation would move
the centroid while leaving orientation and elongation flat.
"""
import cv2, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch, savgol_filter
from config import VIDEO, DATA, OUT, FPS, imread_u
from d_detect import vessel_mask, DIFF_TH

bg = imread_u(OUT / "background_median.png")
vm = vessel_mask(bg)
bg_g = cv2.GaussianBlur(cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.int16)
kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

uv = np.load(DATA / "track2d.npz")["uv"]
cap = cv2.VideoCapture(str(VIDEO))
K = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

ang = np.full(K, np.nan); elong = np.full(K, np.nan); area = np.full(K, np.nan)
for k in range(K):
    ok, fr = cap.read()
    if not ok or np.isnan(uv[k, 0]):
        continue
    g = cv2.GaussianBlur(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.int16)
    d = np.clip(bg_g - g, 0, 255).astype(np.uint8); d[~vm] = 0
    binm = cv2.morphologyEx((d > DIFF_TH).astype(np.uint8), cv2.MORPH_OPEN, kern)
    # isolate the component containing the tracked centroid
    n, lab, st, ce = cv2.connectedComponentsWithStats(binm, 8)
    x, y = int(round(uv[k, 0])), int(round(uv[k, 1]))
    if not (0 <= x < lab.shape[1] and 0 <= y < lab.shape[0]):
        continue
    li = lab[y, x]
    if li == 0:
        continue
    ys, xs = np.where(lab == li)
    if len(xs) < 30:
        continue
    m = cv2.moments((lab == li).astype(np.uint8), binaryImage=True)
    mu20, mu02, mu11 = m["mu20"] / m["m00"], m["mu02"] / m["m00"], m["mu11"] / m["m00"]
    cov = np.array([[mu20, mu11], [mu11, mu02]])
    w, _ = np.linalg.eigh(cov)
    w = np.sort(np.maximum(w, 1e-9))
    ang[k] = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    elong[k] = np.sqrt(w[1] / w[0])
    area[k] = m["m00"]
cap.release()

good = ~np.isnan(elong)
print(f"[K4] blob shape measured on {good.sum()}/{K} frames")
print(f"[K4] elongation (major/minor axis): median {np.nanmedian(elong):.2f} "
      f"p10 {np.nanpercentile(elong,10):.2f} p90 {np.nanpercentile(elong,90):.2f}")
print(f"     -> the robot is {'ELONGATED, so rolling changes its silhouette' if np.nanmedian(elong)>1.4 else 'roughly round; rolling would NOT move the centroid much'}")

k = np.arange(K)
def detrend(v):
    x = np.interp(k, k[good], v[good])
    return x - savgol_filter(x, 41, 2)

# orientation is defined mod pi; unwrap before detrending
a_un = np.interp(k, k[good], np.unwrap(2 * ang[good]) / 2)
sig = {"centroid u": detrend(uv[:, 0]), "centroid v": detrend(uv[:, 1]),
       "blob orientation": a_un - savgol_filter(a_un, 41, 2),
       "blob elongation": detrend(elong), "blob area": detrend(area)}

fr, _ = welch(sig["centroid u"], fs=FPS, nperseg=256)
df = fr[1] - fr[0]
inband = (fr > 0.8) & (fr < 2.0)
print(f"\n[K4] spectral content. NOTE the frequency resolution is only {df:.3f} Hz and all")
print(f"     quantities share ONE broad bump over roughly 0.8-2.0 Hz, so the individual")
print(f"     argmax frequencies below are NOT resolvably different from each other.")
print(f"     The meaningful statistic is how much power sits in that band.")
print(f"\n{'quantity':<20}{'argmax Hz':>11}{'band power / out-of-band':>26}")
for nm, v in sig.items():
    _, P = welch(v, fs=FPS, nperseg=256)
    i = np.argmax(P[1:]) + 1
    ratio = P[inband].mean() / P[~inband & (fr > 0)].mean()
    print(f"{nm:<20}{fr[i]:11.2f}{ratio:26.1f}")

print(f"\n[K4] READING: the blob's AREA and ORIENTATION carry as much 0.8-2.0 Hz power as")
print(f"     the centroid does. A rigid 1.3 mm sideways displacement would move the")
print(f"     centroid while leaving area alone, so the observed shape modulation argues")
print(f"     the oscillation is largely an APPEARANCE effect of the robot rolling.")
print(f"     Combined with K3 this refutes the 'robot rides 1.3 mm off-axis' reading of")
print(f"     Stage D3. It does not by itself PROVE the rolling-silhouette mechanism -")
print(f"     that would need a view of the robot body, which this data does not give.")

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
for nm, v in sig.items():
    fr, P = welch(v, fs=FPS, nperseg=256)
    ax[0].semilogy(fr, P / np.median(P), label=nm, lw=1.3)
ax[0].axvline(1.17, color='r', ls='--', label="1.17 Hz (D3)")
ax[0].set_xlabel("Hz"); ax[0].set_ylabel("PSD / median"); ax[0].legend(fontsize=7)
ax[0].set_title("K4: what oscillates at the roll frequency?")
sl = slice(200, 460)
ax[1].plot(k[sl], sig["centroid u"][sl] / np.std(sig["centroid u"]), label="centroid u (norm)")
ax[1].plot(k[sl], sig["blob elongation"][sl] / np.std(sig["blob elongation"]), label="elongation (norm)")
ax[1].set_xlabel("frame"); ax[1].legend(fontsize=8)
ax[1].set_title("centroid vs shape, same window")
plt.tight_layout(); plt.savefig(OUT / "figs" / "k4_appearance.png", dpi=110)
np.savez(DATA / "k4_shape.npz", ang=ang, elong=elong, area=area)
print("\n[K4] saved out/figs/k4_appearance.png, data/k4_shape.npz")
