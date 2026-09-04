"""Stage R2a - out-and-back arc-length agreement, scored per camera.

Q2 ranked the camera family on off-curve distance, reprojection and held-out
reprojection. All three are the same kind of measurement: distance from the
observation to THAT camera's own projected curve. A camera is being graded on
its own exam, so those three columns are three views of one piece of evidence,
not three independent ones (the same blind spot M2 identified for reprojection).

This test is not of that kind. The robot traverses the vessel out and back, so
most physical locations are visited twice. Pairs of frames are matched by
proximity of the RAW DETECTED 2D TRACK - an observation, with no camera model
involved - and then each camera is asked whether it assigns the same arc length
to both visits. A camera can place its curve beautifully close to the
observations (low off-curve) and still label the same physical spot
inconsistently across the two passes.

What it can and cannot do:
  - CAN rule a camera out: systematic disagreement between two visits to one
    place is a real defect, whichever camera causes it.
  - CANNOT confirm a camera: a projection that is warped CONSISTENTLY gives the
    same wrong answer both times and passes. This detects inconsistency, not
    bias. (Warping does leak in weakly through the motion prior, because the two
    passes traverse the same geometry at different speeds and in opposite
    directions, so ambiguities get resolved differently.)
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from config import DATA, FPS
from a_centerline import load as load_cl
from e_hmm import ArcHMM

V_MAX = 60.0
s3, C3, _, _ = load_cl(); L = s3[-1]
DS = 0.25
SG = np.arange(0, L + 1e-9, DS)
CG = np.stack([np.interp(SG, s3, C3[:, c]) for c in range(3)], 1)
fin = np.load(DATA / "i_final.npz"); UV, k_turn = fin["UV"], int(fin["k_turn"])
K = len(UV)
A_LEG, B_LEG = np.arange(k_turn + 1), np.arange(k_turn + 1, K)


def pin(rv, t, f, cx, cy):
    Xc = CG @ Rot.from_rotvec(rv).as_matrix().T + t
    return Xc[:, :2] / Xc[:, 2:3] * f + np.array([cx, cy])


cams = {}
p11 = np.load(DATA / "c8_11.npy")
cams["pinhole f=1298 (c8_11)"] = pin(p11[:3], p11[3:6], np.exp(p11[6]), p11[9], p11[10])
z10 = np.load(DATA / "c10_ba.npz")
cams["pinhole f=516 (c10_ba)"] = pin(z10["rv"], z10["t"], float(z10["f"]),
                                     float(z10["cx"]), float(z10["cy"]))
pa = np.load(DATA / "l2_affine.npy")
cams["affine, no depth (l2)"] = (CG @ Rot.from_rotvec(pa[:3]).as_matrix().T)[:, :2] * np.exp(pa[3]) + pa[4:6]

# ---- camera-independent pairing: match return frames to outbound frames by
# ---- proximity of the raw detected track. No camera enters this step.
tree_A = cKDTree(UV[A_LEG])
d_img, ia = tree_A.query(UV[B_LEG])

# self-intersection multiplicity of the *observed* track, also camera-free:
# how many outbound frames sit within 8 px of this return frame
n_near = np.array([len(tree_A.query_ball_point(u, 8.0)) for u in UV[B_LEG]])

print(f"[R2a] pairing is camera-free: {len(B_LEG)} return frames matched to the "
      f"outbound leg by raw-track proximity")
for thr in (3.0, 8.0):
    print(f"      image-match < {thr:.0f} px: {int((d_img < thr).sum())} pairs")

print(f"\n[R2a] |s_return - s_outbound| at the same physical place, Viterbi v_max={V_MAX:.0f}\n")
print(f"{'camera':<26}{'match':>7}{'n':>6}{'median':>9}{'p90':>9}{'>2mm':>8}{'>5mm':>8}{'max':>9}")
res = {}
for nm, P in cams.items():
    hmm = ArcHMM(SG, P, sigma_px=9.0, v_max=V_MAX, n_v=41, accel_sigma=20., p_lost=0.05)
    s_v = hmm.viterbi(UV, 1 / FPS)
    res[nm] = s_v
    for thr in (3.0, 8.0):
        sel = d_img < thr
        ds_ = np.abs(s_v[B_LEG][sel] - s_v[A_LEG][ia[sel]])
        print(f"{nm if thr == 3.0 else '':<26}{thr:6.0f}p{sel.sum():6d}{np.median(ds_):9.2f}"
              f"{np.percentile(ds_, 90):9.2f}{100*np.mean(ds_ > 2):7.1f}%{100*np.mean(ds_ > 5):7.1f}%"
              f"{ds_.max():9.2f}")

# ---------------------------------------------------- stratify by ambiguity
print(f"\n[R2a] split by whether the 2D track is locally self-intersecting")
print(f"      (n_near = how many outbound frames lie within 8 px of that return frame;")
print(f"       >6 means the track doubles back through that pixel neighbourhood)\n")
sel = d_img < 8.0
amb = sel & (n_near > 6)
clr = sel & (n_near <= 6)
print(f"{'camera':<26}{'clear n':>9}{'median':>9}{'ambig n':>9}{'median':>9}{'ratio':>8}")
for nm, s_v in res.items():
    dc = np.abs(s_v[B_LEG][clr] - s_v[A_LEG][ia[clr]])
    da = np.abs(s_v[B_LEG][amb] - s_v[A_LEG][ia[amb]])
    ratio = np.median(da) / max(np.median(dc), 1e-9)
    print(f"{nm:<26}{clr.sum():9d}{np.median(dc):9.2f}{amb.sum():9d}{np.median(da):9.2f}{ratio:8.1f}x")

print(f"\n{'':-<78}")
print("[R2a] reading: this column is INDEPENDENT of Q2's three off-curve/reproj")
print("      columns, so it is a genuine second axis of evidence on camera choice.")
print("      A camera with a large median here mislabels the same physical location")
print("      differently on two visits, which no amount of good curve-fitting excuses.")

np.savez(DATA / "r2a_outback_percamera.npz", d_img=d_img, ia=ia, n_near=n_near,
         **{f"s|{k}": v for k, v in res.items()})
print("\n[R2a] saved data/r2a_outback_percamera.npz")
