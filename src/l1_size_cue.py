"""Stage L1 - can APPARENT SIZE break the f/depth degeneracy C7-C10 found?

C7-C10: image scale (f/Z_med) is pinned to ~3%, but f and Z_med separately are
not - the valley is flat from f=600 to f=40000 (C7) and two different solvers
land on f=1298 (c8_11) vs f=516 (c10_ba) at nearly the same residual. Position
alone cannot resolve this. SIZE can: a fixed-size object at depth Z subtends
angle proportional to 1/Z, so as the robot travels the path (known depth
extent, ~64 mm peak-to-peak under the fitted view), its apparent size should
modulate by a known amount IF f (and hence Z_med) is known. Measuring that
modulation and solving backwards for Z_med is a second, independent equation -
exactly what position-only fitting is missing.

Two channels:
  (1) the robot blob (already unbiased - k4_appearance.py measures it from raw
      image differencing, and a direct check (dilating the vessel mask 15->35 px
      changes robot area on only 1/1001 frames) confirms mask morphology does not
      clip the robot silhouette).
  (2) vessel tube width. K4/K5's own vessel_mask.png is DELIBERATELY dilated by
      15px + closed by 9px (d_detect.py) so the robot detector's ROI is generous -
      that is fine for detection, but using that same binary mask to measure tube
      WIDTH would add a large, roughly constant pixel offset that compresses the
      relative modulation. So this stage re-measures width from the CONTINUOUS
      HSV "orangeness" field on the background plate directly (sub-pixel edge
      crossings), bypassing the dilated mask entirely.

Both channels are regressed as  log(size) = a - b * depth(s),  b = 1/Z_med > 0
required for a genuine depth cue. The predictions of the three candidate
cameras already in `data/` (c8_11 f=1298, c10_ba f=516, m1 refit f=1826) are
compared against the measured slope with its uncertainty.
"""
import numpy as np, cv2
from config import DATA, OUT, W, H, imread_u

# ------------------------------------------------------------- shared geometry
fin = np.load(DATA / "i_final.npz")
R, t, sg, Cg = fin["R"], fin["t"], fin["sg"], fin["Cg"]
scale = float(fin["scale"])
zg = (Cg @ R.T + t)[:, 2]
z_med = float(np.median(zg))
print(f"[L1] fitted view (c8_11): depth range {zg.min():.0f}..{zg.max():.0f} mm "
      f"(ptp {np.ptp(zg):.1f} mm), Z_med {z_med:.0f} mm, image scale {scale:.2f} px/mm")

cands = {"c8_11  (f=1298)": 1297.65, "c10_ba (f= 516)": 516.29}
try:
    m1 = np.load(DATA / "m1_holdout.npz")
    cands["m1 refit (f=1826)"] = float(np.exp(m1["p_full"][6]))
except FileNotFoundError:
    pass


def slope_ci(depth, logsize):
    """OLS slope of logsize on depth with a 95% CI; returns (b, se)."""
    x = depth - depth.mean(); y = logsize - logsize.mean()
    b = (x @ y) / (x @ x)
    resid = y - b * x
    n = len(x)
    se = np.sqrt((resid @ resid) / max(n - 2, 1) / (x @ x))
    return b, se


def report(tag, depth, logsize):
    g = np.isfinite(depth) & np.isfinite(logsize)
    b, se = slope_ci(depth[g], logsize[g])
    rho = np.corrcoef(depth[g], logsize[g])[0, 1]
    print(f"\n[{tag}]  n={g.sum()}  corr(depth, log size) = {rho:+.3f}")
    print(f"    measured slope  {b:+.6f} +- {se:.6f} /mm   (need <0 for a real depth cue; "
          f"|b|/se = {abs(b)/se:.1f} sigma from zero)")
    if b < 0:
        Z = -1 / b; Zlo = -1 / (b - 1.96*se) if (b-1.96*se) < 0 else np.inf
        Zhi = -1 / (b + 1.96*se) if (b+1.96*se) < 0 else np.inf
        print(f"    => implied Z_med {Z:7.0f} mm  [95% CI {min(Zlo,Zhi):.0f} .. {max(Zlo,Zhi):.0f}]  "
              f"=> f = {scale*Z:6.0f} px  [{scale*min(Zlo,Zhi):.0f} .. {scale*max(Zlo,Zhi):.0f}]")
    for name, f in cands.items():
        Zc = f / scale
        pred = -1 / Zc
        sig = abs(b - pred) / se
        print(f"    candidate {name:<20} predicts slope {pred:+.6f}  "
              f"-> {sig:5.1f} sigma from measured   {'[REJECTED >3 sigma]' if sig>3 else ('[consistent]' if sig<2 else '[marginal]')}")
    return b, se


# =========================================================== channel 1: robot
sh = np.load(DATA / "k4_shape.npz"); area, elong = sh["area"], sh["elong"]
vit = np.load(DATA / "i2_viterbi.npz"); s_vit, rp_vit = vit["s_vit"], vit["rp_vit"]
ok = (area > 0) & np.isfinite(s_vit) & (rp_vit < 20)          # trust well-registered frames only
dz = np.interp(s_vit, sg, zg) - z_med
w_px = np.sqrt(area / np.maximum(elong, 1e-6))                 # minor axis (diameter-like)
print(f"\n{'='*70}\nCHANNEL 1: robot blob ({ok.sum()}/{len(ok)} frames, reproj<20px)")
report("robot minor-axis width", np.where(ok, dz, np.nan), np.where(ok, np.log(np.maximum(w_px, .1)), np.nan))
report("robot sqrt(area)", np.where(ok, dz, np.nan), np.where(ok, .5*np.log(np.maximum(area, 1)), np.nan))

# ================================================= channel 2: vessel tube width
# Continuous "orangeness" field (NOT the dilated binary mask) so sub-pixel edge
# crossings are unbiased by morphological dilate/close.
bg = imread_u(OUT / "background_median.png")
hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV).astype(np.float32)
h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
hue_dist = np.minimum(np.abs(h - 0), np.abs(h - 180))          # orange ~ hue 0
orangeness = np.clip(1 - hue_dist / 25, 0, 1) * np.clip(s / 60, 0, 1) * np.clip(v / 60, 0, 1)
orangeness = cv2.GaussianBlur(orangeness, (3, 3), 0)            # denoise only, no dilation

Pg = fin["Pg"]
Tt = np.gradient(Pg, axis=0); Tt /= np.maximum(np.linalg.norm(Tt, axis=1, keepdims=True), 1e-9)
Nn = np.stack([-Tt[:, 1], Tt[:, 0]], 1)
off = np.arange(-60, 60.05, 0.25)


def sample(field, pts):
    xi, yi = pts[:, 0], pts[:, 1]
    x0, y0 = np.floor(xi).astype(int), np.floor(yi).astype(int)
    fx, fy = xi - x0, yi - y0
    x0 = np.clip(x0, 0, W-2); y0 = np.clip(y0, 0, H-2)
    v00, v10 = field[y0, x0], field[y0, x0+1]
    v01, v11 = field[y0+1, x0], field[y0+1, x0+1]
    return (v00*(1-fx)*(1-fy) + v10*fx*(1-fy) + v01*(1-fx)*fy + v11*fx*fy)


width_px = np.full(len(Pg), np.nan)
TH = 0.5
for i in range(len(Pg)):
    prof = sample(orangeness, Pg[i][None, :] + off[:, None] * Nn[i][None, :])
    c = len(off) // 2
    if prof[c] < TH:
        continue
    a = c
    while a > 0 and prof[a-1] >= TH: a -= 1
    b_ = c
    while b_ < len(off)-1 and prof[b_+1] >= TH: b_ += 1
    # subpixel: linear interp of the crossing on each side
    la = off[a] - (TH - prof[a]) / (prof[a-1] - prof[a] + 1e-9) * (off[a]-off[a-1]) if a > 0 else off[a]
    lb = off[b_] + (TH - prof[b_]) / (prof[b_+1] - prof[b_] + 1e-9) * (off[b_+1]-off[b_]) if b_ < len(off)-1 else off[b_]
    width_px[i] = lb - la

good = np.isfinite(width_px) & (width_px > 5) & (width_px < 90)
print(f"\n{'='*70}\nCHANNEL 2: vessel tube width, unbiased sub-pixel edge (not the dilated mask)")
print(f"    measured on {good.sum()}/{len(Pg)} centerline samples")
print(f"    median {np.nanmedian(width_px[good]):.1f} px = {np.nanmedian(width_px[good])/scale:.2f} mm  "
      f"(cf. dilated-mask estimate ~9.5 mm - this should be smaller / more physically plausible)")
print(f"    p5..p95: {np.nanpercentile(width_px[good],5):.1f} .. {np.nanpercentile(width_px[good],95):.1f} px "
      f"(ratio {np.nanpercentile(width_px[good],95)/np.nanpercentile(width_px[good],5):.2f}x)")
zg_s = zg - z_med
report("vessel tube width (unbiased)", np.where(good, zg_s, np.nan),
       np.where(good, np.log(np.maximum(width_px, 1)), np.nan))
print("\n    NOTE: this channel is confounded with anatomical taper (the vessel genuinely")
print("    narrows/widens along its real length) - a nonzero slope here is NOT proof of a")
print("    depth cue by itself. It is reported as a cross-check on channel 1, not standalone.")

np.savez(DATA / "l1_size_cue.npz", width_px=width_px, w_robot=w_px, dz_robot=dz, ok_robot=ok)
print("\n[L1] saved data/l1_size_cue.npz")
