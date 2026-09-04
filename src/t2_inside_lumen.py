"""T2 - is the tracked object inside the lumen, or moving over/outside it?

Stage D's vessel mask is dilated 15 px and closed 9 px on purpose: it was a
detection ROI, deliberately generous. Testing "is the robot in a vessel"
against THAT mask is close to meaningless - it would pass an object sliding
over the phantom's outer surface, or a shadow cast by an external driving
magnet, just as happily as one inside the lumen.

This rebuilds a TIGHT lumen mask (no dilation, eroded instead) and asks how
much of the trajectory actually sits inside it, plus how deep inside.
"""
import cv2, numpy as np
from config import OUT, DATA, W, H, imread_u, imwrite_u

bg = imread_u(OUT / "background_median.png")
hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
raw = (((h < 25) | (h > 170)) & (s > 40) & (v > 40)).astype(np.uint8) * 255
tight = cv2.morphologyEx(raw, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
imwrite_u(OUT / "figs" / "t2_tight_mask.png", tight)
loose = imread_u(OUT / "vessel_mask.png", cv2.IMREAD_GRAYSCALE) > 0
T = tight > 0
print(f"[T2] mask coverage: loose (Stage D ROI) {100*loose.mean():.1f}%, tight {100*T.mean():.1f}%")

# distance INTO the lumen: positive inside, 0 outside
dist_in = cv2.distanceTransform(T.astype(np.uint8), cv2.DIST_L2, 5)

uv = np.load(DATA / "track2d.npz")["uv"]
ok = ~np.isnan(uv[:, 0])
xi = np.clip(np.rint(uv[ok, 0]).astype(int), 0, W - 1)
yi = np.clip(np.rint(uv[ok, 1]).astype(int), 0, H - 1)
inside_loose = loose[yi, xi]
inside_tight = T[yi, xi]
depth = dist_in[yi, xi]

print(f"\n[T2] detected robot position, {ok.sum()} frames")
print(f"     inside the LOOSE mask : {100*inside_loose.mean():5.1f}%   (Stage D / P1 quoted this)")
print(f"     inside the TIGHT lumen: {100*inside_tight.mean():5.1f}%")
print(f"     depth into the lumen  : median {np.median(depth):.1f} px, "
      f"p10 {np.percentile(depth,10):.1f}, max {depth.max():.1f}")

# how wide is a tube? sample the distance transform along its skeleton
sk = cv2.ximgproc.thinning(T.astype(np.uint8)*255) if hasattr(cv2, "ximgproc") else None
if sk is not None:
    r = dist_in[sk > 0]
    print(f"     tube half-width (mask skeleton): median {np.median(r):.1f} px")
else:
    r = dist_in[T]
    print(f"     distance-to-edge over all lumen px: median {np.median(r):.1f}, "
          f"p90 {np.percentile(r,90):.1f} px  (p90 ~ tube half-width)")
    hw = np.percentile(r, 90)
    print(f"     -> a centred object would sit ~{hw:.0f} px from the wall; "
          f"observed median is {np.median(depth):.1f} px")

# random control: how often would a RANDOM point in the image land inside?
rng = np.random.default_rng(0)
rx = rng.integers(0, W, 20000); ry = rng.integers(0, H, 20000)
print(f"\n[T2] control: a uniformly random image point falls in the tight lumen "
      f"{100*T[ry, rx].mean():.1f}% of the time")
print(f"     the tracked object does so {100*inside_tight.mean():.1f}% of the time")

frac_out = 100 * (1 - inside_tight.mean())
print(f"\n[T2] VERDICT INPUT: {frac_out:.1f}% of detections are OUTSIDE the tight lumen.")
np.savez(DATA / "t2_inside.npz", inside_tight=inside_tight, depth=depth)
