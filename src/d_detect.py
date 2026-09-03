"""Stage D - 2D robot detection in the video.

The camera is STATIC (verified: background frames are identical), so a temporal
median is an exact background plate and simple differencing beats any learned
detector here, at ~2 ms/frame on CPU.

The robot is DARK against the orange-filled lumen, so we score "how much darker
than background", restrict to the vessel mask, and pick the best connected
component with a motion gate around the previous estimate.

Output: data/track2d.npz
    uv    (N,2)  detected pixel position (nan where lost)
    conf  (N,)   detection confidence 0..1
    area  (N,)   blob area px
"""
import cv2, numpy as np
from config import VIDEO, OUT, DATA, imwrite_u

DIFF_TH   = 22      # grey levels darker than background
MIN_AREA  = 60      # px
MAX_AREA  = 4000    # px  (bigger => fluid front / lighting change, not the robot)
GATE_PX   = 70      # max plausible frame-to-frame motion


def vessel_mask(bg):
    """Orange lumen mask from the background plate."""
    hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = ((h < 25) | (h > 170)) & (s > 40) & (v > 40)
    m = cv2.morphologyEx(m.astype(np.uint8) * 255, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    return m > 0


def build_background(cap, N, n=80):
    idx = np.linspace(0, N - 1, n).astype(int)
    buf = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, f = cap.read()
        if ok: buf.append(f)
    return np.median(np.stack(buf), axis=0).astype(np.uint8)


def run(verbose=True):
    cap = cv2.VideoCapture(str(VIDEO))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bg = build_background(cap, N)
    imwrite_u(OUT / "background_median.png", bg)
    vm = vessel_mask(bg)
    imwrite_u(OUT / "vessel_mask.png", (vm * 255).astype(np.uint8))
    if verbose:
        print(f"[D] {N} frames, vessel mask covers {100*vm.mean():.1f}% of image")

    bg_g = cv2.GaussianBlur(cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.int16)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    uv = np.full((N, 2), np.nan); conf = np.zeros(N); area = np.zeros(N)
    prev = None
    for k in range(N):
        ok, f = cap.read()
        if not ok: break
        g = cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.int16)
        d = np.clip(bg_g - g, 0, 255).astype(np.uint8)
        d[~vm] = 0
        binm = cv2.morphologyEx((d > DIFF_TH).astype(np.uint8), cv2.MORPH_OPEN, kern)
        n_lab, lab, stats, cent = cv2.connectedComponentsWithStats(binm, 8)

        best, best_score = None, -1e9
        for i in range(1, n_lab):
            a = stats[i, cv2.CC_STAT_AREA]
            if a < MIN_AREA or a > MAX_AREA: continue
            cx, cy = cent[i]
            strength = float(d[lab == i].mean())
            score = strength + 0.02 * a
            if prev is not None:
                dist = np.hypot(cx - prev[0], cy - prev[1])
                if dist > GATE_PX: score -= 3.0 * (dist - GATE_PX)   # soft gate
            if score > best_score:
                best_score, best = score, (cx, cy, a, strength)
        if best is not None:
            uv[k] = best[:2]; area[k] = best[2]
            conf[k] = np.clip((best[3] - DIFF_TH) / 60.0, 0, 1)
            prev = best[:2]
    cap.release()

    lost = np.isnan(uv[:, 0])
    if verbose:
        print(f"[D] detected {N-lost.sum()}/{N} frames ({100*(1-lost.mean()):.1f}%)")
        j = np.linalg.norm(np.diff(uv[~lost], axis=0), axis=1)
        print(f"[D] frame-to-frame jump px: median {np.median(j):.1f}  p95 {np.percentile(j,95):.1f}  max {j.max():.1f}")
        print(f"[D] blob area px: median {np.median(area[~lost]):.0f}  p5 {np.percentile(area[~lost],5):.0f}  p95 {np.percentile(area[~lost],95):.0f}")
        print(f"[D] confidence: median {np.median(conf[~lost]):.2f}")
    np.savez(DATA / "track2d.npz", uv=uv, conf=conf, area=area)
    return uv, conf, area


if __name__ == "__main__":
    run()
