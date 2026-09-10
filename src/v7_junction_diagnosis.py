"""V7 - junction diagnosis: why the coordinate table cannot fix the 29.6-30.4 s mismatch.

The reviewer's point: "I already gave you the coordinate table (Path 2) - the
vessel constraint is right there." It is, and the whole method is built on it:
every reported 3D position lies exactly ON the Path-2 curve. The question V7
answers is why, at t = 29.6-30.4 s, the reported position (on the curve) is
still ~46 px from where the robot visibly is, and what the table can and
cannot do about it.

Diagnostics, all from data already in the repository:
  1. local scale at the junction stretch (f=516 perspective) - to convert the
     45.8 px mismatch into mm honestly;
  2. raw CSV point spacing there (is the table too coarse to describe the
     bend? P1c asked this for the OLD camera's bad spots, not this one);
  3. outbound vs return pass through the SAME stretch - does the curve fit
     one pass and not the other?;
  4. where the closest curve point to the return detections actually is
     (routing mistake vs lateral displacement).
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from config import DATA, CSV, FPS
from a_centerline import load as load_cl

s3c, C3, _, kap = load_cl()
SG = np.arange(0, s3c[-1] + 1e-9, 0.25)
CG = np.stack([np.interp(SG, s3c, C3[:, c]) for c in range(3)], 1)
z10 = np.load(DATA / "c10_ba.npz")
R = Rot.from_rotvec(z10["rv"]).as_matrix(); t = z10["t"]; f = float(z10["f"])
Xc = CG @ R.T + t
Z = Xc[:, 2]
P516 = Xc[:, :2] / Z[:, None] * f + np.array([float(z10["cx"]), float(z10["cy"])])
fin = np.load(DATA / "i_final.npz"); UV, k_turn = fin["UV"], int(fin["k_turn"])

scale_med = f / np.median(Z)
sc_loc = f / Z[(SG > 105) & (SG < 125)]
print(f"[V7] median depth {np.median(Z):.1f} mm -> median scale {scale_med:.2f} px/mm")
print(f"[V7] junction stretch s=105-125 is FARTHER (Z = {Z[(SG>105)&(SG<125)].min():.1f}"
      f"-{Z[(SG>105)&(SG<125)].max():.1f} mm): local scale {sc_loc.mean():.2f} px/mm")
print(f"[V7] so the worst mismatch, 45.8 px, is {45.8/sc_loc.mean():.1f} mm in-plane "
      f"(tube diameter ~6.8 mm)")

# --- is the table too coarse here? -------------------------------------------
P_raw = np.loadtxt(CSV, delimiter=",", skiprows=2)
d_raw = np.linalg.norm(np.diff(P_raw, axis=0), axis=1)
s_raw = np.concatenate([[0], np.cumsum(d_raw)])
idx = np.where((s_raw > 100) & (s_raw < 130))[0]
print(f"\n[V7] raw CSV points with chord-s in 100-130 mm: indices {idx.min()}-{idx.max()}, "
      f"spacing {d_raw[idx].min():.2f}-{d_raw[idx].max():.2f} mm")
kg = np.interp(SG, s3c, kap)
m = (SG > 100) & (SG < 130)
print(f"[V7] bend radius there: {1/kg[m].max():.1f} mm (global sharpest bend is "
      f"{1/kg.max():.1f} mm at s = {SG[np.argmax(kg)]:.0f} mm - the junction is NOT the sharpest)")
# chord-vs-arc corner cut from that spacing at that curvature: ~d^2/(8R)
d_mean = d_raw[idx].mean()
print(f"[V7] spline corner-cut at that spacing and curvature ~ d^2/(8R) = "
      f"{d_mean**2 / (8 / kg[m].max()):.2f} mm - far too small to matter")

# --- outbound vs return through the SAME stretch ------------------------------
def dmin(P, uv):
    return np.linalg.norm(P - uv, axis=1).min()

d_out = [dmin(P516, UV[k]) for k in range(460, 500) if np.isfinite(UV[k, 0])]
d_ret = [dmin(P516, UV[k]) for k in range(885, 920) if np.isfinite(UV[k, 0])]
print(f"\n[V7] detection-to-curve distance through s~105-125:")
print(f"     OUTBOUND pass (frames 460-500): median {np.median(d_out):5.1f} px "
      f"({np.median(d_out)/sc_loc.mean():.1f} mm), max {np.max(d_out):.1f} px")
print(f"     RETURN   pass (frames 885-920): median {np.median(d_ret):5.1f} px "
      f"({np.median(d_ret)/sc_loc.mean():.1f} mm), max {np.max(d_ret):.1f} px")
print(f"     -> the curve FITS the outbound pass; the return pass is displaced by "
      f"~one tube diameter")

# --- routing mistake or lateral displacement? ---------------------------------
print(f"\n[V7] closest curve point to each return detection (is the model even on "
      f"the right stretch?):")
for k in (890, 897, 905):
    dd = np.linalg.norm(P516 - UV[k], axis=1)
    j = int(np.argmin(dd))
    print(f"     frame {k} (t={k/FPS:.1f} s): closest point at s = {SG[j]:5.1f} mm, "
          f"d = {dd[j]:.1f} px - the SAME stretch the filter already reports")

print(f"""
[V7] READING. The coordinate table IS the constraint - every 3D output lies on
     it, and the outbound pass confirms the axis is right here to ~1.6 mm.
     What the table does not contain is the tube CROSS-SECTION: the estimator
     has no degree of freedom for where inside the lumen the robot presses.
     On the return pass through the junction the robot rides the far wall - a
     full diameter from the axis side the outbound pass took - and no smooth
     offset model tried so far survives its own out-of-sample test (K2-K4
     rotating, U1 per-leg constant, V1 curvature-locked). A fourth model fitted
     to these 25 frames could never be validated: there is no third pass. The
     honest output is therefore what V6 renders: the axis position with the
     error bar widened to +-9 mm during the episode, plus the flag that the
     constraint itself, not the camera or the estimator, is what fails there.
     Fixing it needs cross-section information the table cannot provide: the
     robot's diameter, the lumen surface (STL), or a second simultaneous view.
""")
np.savez(DATA / "v7_junction.npz", sc_loc=sc_loc.mean(), d_out=np.array(d_out),
         d_ret=np.array(d_ret))
print("[V7] saved data/v7_junction.npz")
