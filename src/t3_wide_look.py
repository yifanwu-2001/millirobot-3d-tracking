"""T3 - large native-resolution crops, and a consecutive-frame strip, so the
object can be judged by eye instead of by a projection statistic."""
import cv2, numpy as np
from config import VIDEO, DATA, OUT, imwrite_u

uv = np.load(DATA / "track2d.npz")["uv"]
cap = cv2.VideoCapture(str(VIDEO))

# (a) big context crops at native resolution
R = 200
for k in (300, 620, 900):
    cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, f = cap.read()
    if not ok: continue
    x, y = int(uv[k, 0]), int(uv[k, 1])
    x0, y0 = max(0, x - R), max(0, y - R)
    crop = f[y0:min(720, y + R), x0:min(960, x + R)].copy()
    cv2.circle(crop, (x - x0, y - y0), 30, (0, 255, 255), 2)
    imwrite_u(OUT / "figs" / f"t3_wide_f{k}.png", crop)
    print(f"[T3] saved t3_wide_f{k}.png  ({crop.shape[1]}x{crop.shape[0]} native px)")

# (b) consecutive-frame strip: does it translate ALONG the tube?
k0 = 300
tiles = []
for k in range(k0, k0 + 40, 8):
    cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, f = cap.read()
    if not ok or np.isnan(uv[k, 0]): continue
    x, y = int(uv[k, 0]), int(uv[k, 1])
    x0, y0 = max(0, x - 110), max(0, y - 110)
    c = f[y0:y0 + 220, x0:x0 + 220].copy()
    if c.shape[:2] != (220, 220): continue
    cv2.circle(c, (x - x0, y - y0), 26, (0, 255, 255), 2)
    cv2.putText(c, f"{k}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
    tiles.append(c)
if tiles:
    imwrite_u(OUT / "figs" / "t3_strip.png", np.hstack(tiles))
    print(f"[T3] saved t3_strip.png ({len(tiles)} consecutive-ish frames)")
cap.release()
