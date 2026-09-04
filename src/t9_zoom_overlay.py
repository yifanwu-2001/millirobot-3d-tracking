"""T9 - zoom in on the red-vs-yellow disagreement instead of judging it from a
thumbnail. Draws the detections and the projected centerline over the plate at
3x, in the band where the p90 offset is worst."""
import cv2, numpy as np
from scipy.spatial import cKDTree
from config import DATA, OUT, W, H, imread_u, imwrite_u

bg = imread_u(OUT / "background_median.png")
PG = np.load(DATA / "q1_affine_downstream.npz")["PG"]
uv = np.load(DATA / "track2d.npz")["uv"]; ok = ~np.isnan(uv[:, 0])
U = uv[ok]
d, _ = cKDTree(PG).query(U)

for tag, (x0, y0, x1, y1) in {
        "mid": (250, 470, 570, 660),
        "left": (110, 450, 330, 660),
        "right": (600, 230, 960, 500)}.items():
    crop = bg[y0:y1, x0:x1].copy()
    Z = 3
    crop = cv2.resize(crop, ((x1 - x0) * Z, (y1 - y0) * Z), interpolation=cv2.INTER_CUBIC)
    seg = [(i, i + 1) for i in range(len(PG) - 1)
           if x0 <= PG[i, 0] < x1 and y0 <= PG[i, 1] < y1
           and x0 <= PG[i + 1, 0] < x1 and y0 <= PG[i + 1, 1] < y1]
    for a, b in seg:
        p = ((PG[a, 0] - x0) * Z, (PG[a, 1] - y0) * Z)
        q = ((PG[b, 0] - x0) * Z, (PG[b, 1] - y0) * Z)
        cv2.line(crop, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), (0, 220, 255), 3)
    m = (U[:, 0] >= x0) & (U[:, 0] < x1) & (U[:, 1] >= y0) & (U[:, 1] < y1)
    for p, dd in zip(U[m], d[m]):
        c = (0, 0, 255) if dd < 15 else (255, 0, 255)
        cv2.circle(crop, (int((p[0] - x0) * Z), int((p[1] - y0) * Z)), 4, c, -1)
    if m.sum():
        cv2.putText(crop, f"{tag}: n={m.sum()}  median {np.median(d[m]):.1f}px  p90 {np.percentile(d[m],90):.1f}px",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 0), 4)
        cv2.putText(crop, f"{tag}: n={m.sum()}  median {np.median(d[m]):.1f}px  p90 {np.percentile(d[m],90):.1f}px",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
    imwrite_u(OUT / "figs" / f"t9_{tag}.png", crop)
    print(f"[T9] {tag}: n={m.sum()}, median {np.median(d[m]):.1f} px, "
          f"p90 {np.percentile(d[m],90):.1f} px, max {d[m].max():.1f} px")
print("[T9] yellow = projected Path 2, red = detection within 15px, MAGENTA = further")
