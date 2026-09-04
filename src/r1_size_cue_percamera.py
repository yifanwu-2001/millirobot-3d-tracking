"""Stage R1 - redo L1's size cue as a proper per-camera model-selection test.

L1 IS INVALID AS PUBLISHED. It regressed log(vessel width) on the depth profile
of ONE camera (`c8_11`, f=1298) and then compared candidate focal lengths by
rescaling that single profile. But the candidates do not share a view direction:
c8_11's and the affine model's view axes are 46 deg apart and their depth
profiles along the path are ANTI-correlated (r = -0.86). So L1 asked "given
c8_11's geometry, which f fits the size modulation" - a question whose answer is
conditional on the camera Q2 later showed to be the outlier (worst on every
tracking column). The 9.8-sigma rejection of f=516 does not survive that.

The right test, done here: every camera gets scored against ITS OWN geometry.
For each candidate,
  - project the centerline with that camera -> its own image curve,
  - measure vessel width along THAT curve (sampling locations differ per camera),
  - compute that camera's own depth profile z_cam(s),
  - regress log(width) on z_cam and compare the measured slope to what that
    camera itself predicts:
        pinhole  -> slope = -1/Z_med   (a fixed-diameter tube at depth Z
                                        subtends width proportional to 1/Z)
        affine   -> slope = 0          (no depth term in the projection at all)
A camera is supported if its own size-cue slope matches its own prediction.

The taper confound from L1 still applies (a real vessel does narrow along its
length), so this is model SELECTION between candidates under a shared confound,
not an absolute measurement of perspective strength.
"""
import numpy as np, cv2
from scipy.spatial.transform import Rotation as Rot
from config import DATA, OUT, W, H, imread_u
from a_centerline import load as load_cl

s3, C3, _, _ = load_cl(); L = s3[-1]
ds = 0.25
sg = np.arange(0, L + 1e-9, ds)
Cg = np.stack([np.interp(sg, s3, C3[:, c]) for c in range(3)], 1)

# ---------------------------------------------------------------- the cameras
cams = {}

p8 = np.load(DATA / "c8_11.npy")
cams["pinhole f=1298 (c8_11)"] = dict(
    kind="pinhole", R=Rot.from_rotvec(p8[:3]).as_matrix(), t=p8[3:6],
    f=float(np.exp(p8[6])), c=np.array([p8[9], p8[10]]))

z10 = np.load(DATA / "c10_ba.npz")
cams["pinhole f=516 (c10_ba)"] = dict(
    kind="pinhole", R=Rot.from_rotvec(z10["rv"]).as_matrix(), t=np.asarray(z10["t"]).ravel(),
    f=float(z10["f"]), c=np.array([float(z10["cx"]), float(z10["cy"])]))

pa = np.load(DATA / "l2_affine.npy")
cams["affine, no depth (l2)"] = dict(
    kind="affine", R=Rot.from_rotvec(pa[:3]).as_matrix(), t=None,
    scale=float(np.exp(pa[3])), c=np.array([pa[4], pa[5]]))


def project(cam, X):
    Xc = X @ cam["R"].T
    if cam["kind"] == "pinhole":
        Xc = Xc + cam["t"]
        z = Xc[:, 2]
        ok = z > 1e-3
        Q = np.full((len(X), 2), np.nan)
        Q[ok] = Xc[ok, :2] / z[ok, None] * cam["f"] + cam["c"]
        return Q, z
    Q = Xc[:, :2] * cam["scale"] + cam["c"]
    return Q, Xc[:, 2]          # depth is geometrically defined even if unused


# ------------------------------------------- continuous vessel "orangeness"
bg = imread_u(OUT / "background_median.png")
hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV).astype(np.float32)
h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
hue_dist = np.minimum(np.abs(h - 0), np.abs(h - 180))
orange = np.clip(1 - hue_dist / 25, 0, 1) * np.clip(s / 60, 0, 1) * np.clip(v / 60, 0, 1)
orange = cv2.GaussianBlur(orange, (3, 3), 0)
off = np.arange(-60, 60.05, 0.25)
TH = 0.5


def sample(field, pts):
    xi, yi = pts[:, 0], pts[:, 1]
    x0 = np.clip(np.floor(xi).astype(int), 0, W - 2)
    y0 = np.clip(np.floor(yi).astype(int), 0, H - 2)
    fx, fy = xi - x0, yi - y0
    return (field[y0, x0]*(1-fx)*(1-fy) + field[y0, x0+1]*fx*(1-fy)
            + field[y0+1, x0]*(1-fx)*fy + field[y0+1, x0+1]*fx*fy)


def widths_along(Q):
    """Sub-pixel vessel width on the normal at each projected sample."""
    T = np.gradient(Q, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-9)
    N = np.stack([-T[:, 1], T[:, 0]], 1)
    out = np.full(len(Q), np.nan)
    for i in range(len(Q)):
        if not np.isfinite(Q[i]).all():
            continue
        if not (0 <= Q[i, 0] < W and 0 <= Q[i, 1] < H):
            continue
        prof = sample(orange, Q[i][None, :] + off[:, None] * N[i][None, :])
        c0 = len(off) // 2
        if prof[c0] < TH:
            continue
        a = c0
        while a > 0 and prof[a-1] >= TH: a -= 1
        b = c0
        while b < len(off)-1 and prof[b+1] >= TH: b += 1
        la = off[a] - (TH-prof[a])/(prof[a-1]-prof[a]+1e-9)*(off[a]-off[a-1]) if a > 0 else off[a]
        lb = off[b] + (TH-prof[b])/(prof[b+1]-prof[b]+1e-9)*(off[b+1]-off[b]) if b < len(off)-1 else off[b]
        out[i] = lb - la
    return out


print("[R1] per-camera size cue: each model scored against its OWN geometry\n")
print(f"{'camera':<26}{'valid':>7}{'width med':>11}{'measured slope':>18}{'predicted':>12}{'sigma off':>11}")
results = {}
for name, cam in cams.items():
    Q, z = project(cam, Cg)
    wid = widths_along(Q)
    good = np.isfinite(wid) & (wid > 5) & (wid < 90)
    if good.sum() < 30:
        print(f"{name:<26}{good.sum():>7}   too few valid samples - projected curve misses the vessel")
        continue
    zc = z - np.median(z)
    x = zc[good] - zc[good].mean()
    y = np.log(wid[good]); y = y - y.mean()
    b = (x @ y) / (x @ x)
    r = y - b * x
    se = np.sqrt((r @ r) / max(len(x) - 2, 1) / (x @ x))
    pred = 0.0 if cam["kind"] == "affine" else -1.0 / float(np.median(z))
    nsig = abs(b - pred) / se
    results[name] = dict(b=b, se=se, pred=pred, nsig=nsig, n=int(good.sum()),
                         wmed=float(np.median(wid[good])), zmed=float(np.median(z)))
    print(f"{name:<26}{good.sum():>7}{np.median(wid[good]):11.1f}"
          f"{f'{b:+.5f}+-{se:.5f}':>18}{pred:12.5f}{nsig:10.1f}s")

print(f"\n{'':-<80}")
print("[R1] reading: a camera is SELF-CONSISTENT if its measured size-cue slope")
print("     matches what its own geometry predicts (low sigma). This is the test")
print("     L1 should have run; L1 instead held c8_11's rotation fixed for all")
print("     candidates, and c8_11's depth profile is anti-correlated (r=-0.86)")
print("     with the affine model's, so L1's verdict was an artefact of that choice.\n")
if results:
    best = min(results, key=lambda k: results[k]["nsig"])
    for name, rr in sorted(results.items(), key=lambda kv: kv[1]["nsig"]):
        verdict = ("SELF-CONSISTENT" if rr["nsig"] < 2 else
                   "marginal" if rr["nsig"] < 3 else "INCONSISTENT with its own geometry")
        print(f"     {name:<26} {rr['nsig']:5.1f} sigma   {verdict}")
    print(f"\n     best-supported by the size cue: {best}")
    print("     (taper confound unchanged from L1: all rows share it, so this ranks")
    print("      candidates against each other rather than measuring perspective absolutely)")

np.savez(DATA / "r1_size_cue_percamera.npz",
         **{f"{k}_{f}": v[f] for k, v in results.items() for f in ("b", "se", "pred", "nsig")})
print("\n[R1] saved data/r1_size_cue_percamera.npz")
