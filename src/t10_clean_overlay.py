"""T10 - the honest version of the red-vs-yellow overlay.

T9 drew the RAW track, both legs together. The raw track carries the 1.17 Hz
roll wobble, which spreads points across the tube and made the cloud look like a
second, longer route around the bend. Comparing that cloud's arc length to a
smooth curve is the exact error Stage D3 was written to prevent, and it produced
a bogus '3.25x longer route' claim. This plots the DE-WOBBLED track, with the
outbound and return legs separated.
"""
import cv2, numpy as np
from scipy.spatial import cKDTree
from config import DATA, OUT, imread_u, imwrite_u

bg = imread_u(OUT / "background_median.png")
PG = np.load(DATA / "q1_affine_downstream.npz")["PG"]
tr = np.load(DATA / "track2d_trend.npz")["trend"]
raw = np.load(DATA / "track2d.npz")["uv"]
k_turn = int(np.load(DATA / "i_final.npz")["k_turn"])
d, _ = cKDTree(PG).query(tr)

x0, y0, x1, y1 = 250, 470, 570, 660
Z = 3
crop = cv2.resize(bg[y0:y1, x0:x1], ((x1 - x0) * Z, (y1 - y0) * Z), interpolation=cv2.INTER_CUBIC)
P = lambda p: (int((p[0] - x0) * Z), int((p[1] - y0) * Z))

for i in range(len(PG) - 1):
    if x0 <= PG[i, 0] < x1 and y0 <= PG[i, 1] < y1 and x0 <= PG[i+1, 0] < x1 and y0 <= PG[i+1, 1] < y1:
        cv2.line(crop, P(PG[i]), P(PG[i+1]), (0, 220, 255), 3)

inbox = lambda A: (A[:, 0] >= x0) & (A[:, 0] < x1) & (A[:, 1] >= y0) & (A[:, 1] < y1)
mr = inbox(raw) & ~np.isnan(raw[:, 0])
for p in raw[mr]:
    cv2.circle(crop, P(p), 2, (200, 200, 200), -1)          # grey: raw, wobbling
out = np.zeros(len(tr), bool); out[:k_turn+1] = True
for m, col in [(inbox(tr) & out, (0, 0, 255)), (inbox(tr) & ~out, (255, 0, 255))]:
    for p in tr[m]:
        cv2.circle(crop, P(p), 4, col, -1)

for txt, y in [("yellow = Path 2 projected", 34),
               ("grey = raw detections (carry the 1.17 Hz roll wobble)", 66),
               ("red = de-wobbled OUTBOUND    magenta = de-wobbled RETURN", 98)]:
    cv2.putText(crop, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 4)
    cv2.putText(crop, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
imwrite_u(OUT / "figs" / "t10_clean_mid.png", crop)

m = inbox(tr)
print(f"[T10] mid region, de-wobbled: n={m.sum()}  median {np.median(d[m]):.1f} px  "
      f"p90 {np.percentile(d[m],90):.1f} px  max {d[m].max():.1f} px")
mo, mb = m & out, m & ~out
print(f"      outbound n={mo.sum()} median {np.median(d[mo]):.1f} px   "
      f"return n={mb.sum()} median {np.median(d[mb]):.1f} px")
print(f"[T10] compare T9's raw-track figure: median 16.5 px, p90 39.7 px")
print("[T10] saved out/figs/t10_clean_mid.png")
