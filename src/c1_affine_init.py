"""Stage C1 - weak-perspective (affine) initialisation, and a first answer to
"does Path 2.csv actually correspond to the path travelled in this video?"

Idea: on the OUTBOUND leg the robot moves monotonically from one end of the
centerline to the other. Matching normalised 2D arc length to normalised 3D arc
length therefore gives an approximate correspondence with NO camera knowledge.
With correspondences, the 2x4 affine camera  u = A X + b  is a closed-form
least-squares solve (8 unknowns). The residual tells us whether the two files
describe the same path.
"""
import numpy as np
from config import DATA, OUT
from a_centerline import load as load_cl

s3, C3, T3, _ = load_cl()
z = np.load(DATA / "track2d_clean.npz"); uv = z["uv"]; k_turn = int(z["k_turn"])
out2d = uv[:k_turn + 1]

# arc-length parameterise the 2D outbound polyline
d2 = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(out2d, axis=0), axis=1))])
print(f"[C1] 2D outbound path length {d2[-1]:.0f} px over {len(out2d)} frames")
print(f"[C1] 3D centerline length    {s3[-1]:.1f} mm over {len(s3)} pts")
print(f"[C1] => nominal scale {d2[-1]/s3[-1]:.2f} px/mm (weak-perspective average)")

M = 400
t = np.linspace(0, 1, M)
U = np.stack([np.interp(t * d2[-1], d2, out2d[:, c]) for c in range(2)], 1)   # 2D, normalised


def fit_affine(X, U):
    """u = A X + b  (2x3 A, 2-vector b) by least squares. Returns A|b, rms px."""
    G = np.hstack([X, np.ones((len(X), 1))])            # (M,4)
    P, *_ = np.linalg.lstsq(G, U, rcond=None)           # (4,2)
    pred = G @ P
    return P, float(np.sqrt(((pred - U) ** 2).sum(1).mean())), pred


print("\n[C1] testing both traversal directions and partial-coverage hypotheses")
print(f"{'hypothesis':<34}{'rms px':>9}{'rms/pathlen':>13}")
best = None
for name, flip in [("robot goes s=0 -> s=L", False), ("robot goes s=L -> s=0", True)]:
    for cov in (1.00, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50):
        # 3D points covering a fraction `cov` of the centerline
        if flip: ss = s3[-1] * (1 - t * cov)
        else:    ss = s3[-1] * (t * cov)
        X = np.stack([np.interp(ss, s3, C3[:, c]) for c in range(3)], 1)
        P, rms, _ = fit_affine(X, U)
        tag = f"{name}, cover {cov:.0%}"
        print(f"{tag:<34}{rms:9.2f}{rms/d2[-1]:13.4f}")
        if best is None or rms < best[0]:
            best = (rms, name, cov, flip, P, X)

rms, name, cov, flip, P, X = best
print(f"\n[C1] BEST: {name}, coverage {cov:.0%},  rms {rms:.2f} px  ({100*rms/d2[-1]:.2f}% of path length)")
np.savez(DATA / "c1_affine.npz", P=P, flip=flip, cov=cov, rms=rms, U=U, X=X, t=t)
