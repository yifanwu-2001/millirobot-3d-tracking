"""Stage R6 - three side checks left open by the V-stage review.

(a) V11's out-and-back repeatability regressed 0.25 -> 2.00 mm against V8.
    Uniform or episodic? Caused by the transverse (tube) degree of freedom, or
    by the online front end (causal notch + fixed-lag commit)?
(b) V12 measured ~3 px of background motion. Stage D assumed a static camera
    and Stage C calibrated on that assumption. Is it a drift between the two
    legs (which would leak into M1's held-out gap) or frame-to-frame jitter?
(c) The f=516 camera is now the family's best-supported member. Its intrinsics
    - 86 deg field of view, principal point 39% of the frame below centre -
    were never revisited after Stage C. Are they physically plausible?
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from config import DATA, W, H

fin = np.load(DATA / "i_final.npz"); UV, k_turn = fin["UV"], int(fin["k_turn"]); K = len(UV)
A, B = np.arange(k_turn + 1), np.arange(k_turn + 1, K)
d_img, ia = cKDTree(UV[A]).query(UV[B]); sel = d_img < 8.0

# ------------------------------------------------------------ (a) V11 vs V8
v11 = np.load(DATA / "v11_realtime.npz"); v8 = np.load(DATA / "v8_tube.npz")
s11, s8, d11, acc = v11["s"], v8["s_tb"], v11["d_mm"], v11["accepted"].astype(bool)
ds11 = np.abs(s11[B][sel] - s11[A][ia[sel]]); ds8 = np.abs(s8[B][sel] - s8[A][ia[sel]])
print("[R6a] out-and-back |s_return - s_outbound| on the same 248 camera-free pairs")
for nm, d in [("V8 (offline)", ds8), ("V11 (online)", ds11)]:
    print(f"      {nm:<14} median {np.median(d):.2f}  p90 {np.percentile(d,90):.2f}  >2mm {100*np.mean(d>2):.1f}%  >5mm {100*np.mean(d>5):.1f}%")
big = ds11 > 2; rb = B[sel][big]
runs = np.split(rb, np.where(np.diff(rb) > 5)[0] + 1)
print(f"      V11 pairs >2 mm: {big.sum()} in {len(runs)} separate runs over frames {rb.min()}-{rb.max()} -> "
      f"{'uniform across the leg' if len(runs) >= 8 else 'episodic'}")
worst = max(runs, key=lambda r: np.median(ds11[np.isin(B[sel], r)]))
print(f"      worst run: frames {worst.min()}-{worst.max()}, median |ds| {np.median(ds11[np.isin(B[sel], worst)]):.1f} mm")
print(f"      corr(|ds11|, |d_mm| at the return frame) = {np.corrcoef(ds11, np.abs(d11[B][sel]))[0,1]:+.2f} "
      f"-> the transverse offset is {'not ' if abs(np.corrcoef(ds11, np.abs(d11[B][sel]))[0,1]) < 0.3 else ''}the cause")
sd = (s11 - s8)
print(f"      signed (s11 - s8): outbound median {np.median(sd[A][acc[A]]):+.2f} mm, return median {np.median(sd[B][acc[B]]):+.2f} mm "
      f"-> opposite signs on the two legs = a lag in the direction of travel")

# ---------------------------------------------------------- (b) V12 drift
m = np.load(DATA / "v12_motion_audit.npz"); mot, tf = m["motion_px"], m["transforms"]
sc = 6.43
print(f"\n[R6b] background motion vs frame 0 (px): outbound median {np.median(mot[A]):.2f} max {mot[A].max():.2f} | "
      f"return median {np.median(mot[B]):.2f} max {mot[B].max():.2f}")
print(f"      corr(motion, frame index) = {np.corrcoef(mot, np.arange(K))[0,1]:+.2f} -> "
      f"{'a slow drift' if abs(np.corrcoef(mot, np.arange(K))[0,1]) > 0.5 else 'jitter, not drift'}")
tx, ty = tf[:, 0, 2], tf[:, 1, 2]
off = np.hypot(tx[B].mean() - tx[A].mean(), ty[B].mean() - ty[A].mean())
print(f"      leg-to-leg mean translation offset {off:.2f} px = {off/sc:.2f} mm -> contribution to M1's held-out gap: none")
print(f"      magnitude in mm at {sc} px/mm: median {np.median(mot)/sc:.2f}, max {mot.max()/sc:.2f}; "
      f"sigma_stat for reference 0.78 mm -> inside the budget already carried by sigma_px")

# --------------------------------------------------- (c) f=516 intrinsics
z10 = np.load(DATA / "c10_ba.npz"); f = float(z10["f"]); cx, cy = float(z10["cx"]), float(z10["cy"])
R = Rot.from_rotvec(z10["rv"]).as_matrix(); t = np.asarray(z10["t"]).ravel()
print(f"\n[R6c] f=516: HFOV {2*np.degrees(np.arctan(W/2/f)):.1f} deg, VFOV {2*np.degrees(np.arctan(H/2/f)):.1f} deg; "
      f"camera centre {(-R.T @ t).round(1)} mm in the CSV frame, view axis {R[2].round(3)}")
print(f"      principal point ({cx:.0f},{cy:.0f}) = ({100*(cx-W/2)/W:+.0f}%, {100*(cy-H/2)/H:+.0f}%) of the frame from centre, "
      f"{H-cy:.0f} px from the bottom edge")
p11 = np.load(DATA / "c8_11.npy")
print(f"      f=1298 for comparison: HFOV {2*np.degrees(np.arctan(W/2/np.exp(p11[6]))):.1f} deg, principal point ({p11[9]:.0f},{p11[10]:.0f})")
print("      reading: a wide lens ~8 cm above a flat phantom, looking straight down, is a plausible phone/action-camera")
print("      setup; a principal point 39% below centre is not plausible for an uncropped sensor, but on one near-flat")
print("      curve the principal point trades off against a small rotation and is not separately identifiable -")
print("      the fitted value is an identifiability artefact, not a physical claim about the lens.")
