"""Stage J3 - which PAIR of C-arm angles should you use?

Single-view metric (Stage G): a pair of arc lengths far apart on the path but
close in the projection is an ambiguity. With two views the ambiguity survives
only if the two arc lengths collide in BOTH projections, so the separation
delivered by a view pair is

    score(v1,v2) = min over (i,j) with |s_i - s_j| > d_min  of  max(sep_v1, sep_v2)

Evaluating this exactly over all view pairs is done by first collecting the
point pairs that are dangerously close in at least one direction: any pair
outside that set is farther than the threshold in EVERY direction, so its max
is above threshold and it cannot be the minimiser.
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from config import DATA, OUT
from a_centerline import load as load_cl

NDIR, M, ARC_MIN = 150, 140, 5.0
s3, C3, _, _ = load_cl(); L = s3[-1]
ss = np.linspace(0, L, M)
Q3 = np.stack([np.interp(ss, s3, C3[:, c]) for c in range(3)], 1)


def fib(n):
    i = np.arange(n) + .5
    phi = np.arccos(1 - 2 * i / n); th = np.pi * (1 + 5 ** .5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def sep_matrix(v):
    v = v / np.linalg.norm(v)
    a = np.array([0, 0, 1.]) if abs(v[2]) < .9 else np.array([1., 0, 0])
    e1 = np.cross(v, a); e1 /= np.linalg.norm(e1); e2 = np.cross(v, e1)
    P = Q3 @ np.stack([e1, e2], 1)
    return np.linalg.norm(P[:, None] - P[None], axis=-1)


dirs = fib(NDIR)
iu, ju = np.triu_indices(M, 1)
mask = np.abs(ss[iu] - ss[ju]) > ARC_MIN
iu, ju = iu[mask], ju[mask]
print(f"[J3] {NDIR} view directions, {M} path samples, {len(iu)} arc-length-distant point pairs")

S = np.stack([sep_matrix(v)[iu, ju] for v in dirs])     # (NDIR, npairs), mm in the projection plane
single = S.min(1)
b, w = int(np.argmax(single)), int(np.argmin(single))
print(f"[J3] SINGLE view separation (mm): best {single[b]:.2f} at {dirs[b].round(3)}"
      f"   median {np.median(single):.2f}   worst {single[w]:.3f}")

# exact: evaluate every view pair over every arc-length-distant point pair
best = (-1., None); worst = (1e9, None); vals = []
for i in range(NDIR - 1):
    sc = np.maximum(S[i], S[i+1:]).min(1)          # (NDIR-i-1,)
    vals.append(sc)
    j = int(np.argmax(sc))
    if sc[j] > best[0]: best = (float(sc[j]), (i, i + 1 + j))
    k = int(np.argmin(sc))
    if sc[k] < worst[0]: worst = (float(sc[k]), (i, i + 1 + k))
vals = np.concatenate(vals)
print(f"\n[J3] view PAIR separation (mm): best {best[0]:.2f}   median {np.median(vals):.2f}"
      f"   worst {worst[0]:.3f}")
i, j = best[1]
ang = np.degrees(np.arccos(np.clip(abs(dirs[i] @ dirs[j]), -1, 1)))
print(f"     best pair: {dirs[i].round(3)} and {dirs[j].round(3)}  (separated {ang:.0f} deg)")
print(f"     -> single-view best {single[b]:.2f} mm  vs  pair best {best[0]:.2f} mm"
      f"   = {best[0]/single[b]:.1f}x better")
print(f"     -> single-view MEDIAN {np.median(single):.2f} mm vs pair MEDIAN {np.median(vals):.2f} mm"
      f"   = {np.median(vals)/np.median(single):.1f}x better")
print(f"     -> a badly chosen PAIR ({worst[0]:.3f} mm) is still worse than a well chosen single view")

print(f"\n[J3] partner sweep for the ACTUAL fitted view axis")
V0 = np.array([-0.086, 0.382, -0.920]); V0 /= np.linalg.norm(V0)
AX = np.array([-0.176, -0.983, 0.052]); AX /= np.linalg.norm(AX)
S0 = sep_matrix(V0)[iu, ju]
print(f"{'separation angle':>18}{'pair sep mm':>13}{'gain':>8}")
base = float(S0.min())
print(f"{'single view':>18}{base:13.3f}{'1.0x':>8}")
for th in (10, 20, 30, 45, 60, 75, 90):
    v2 = Rot.from_rotvec(np.radians(th) * AX).apply(V0)
    S2 = sep_matrix(v2)[iu, ju]
    sc = float(np.maximum(S0, S2).min())
    print(f"{th:15d} deg{sc:13.3f}{sc/base:7.1f}x")
np.savez(DATA / "j3_pairs.npz", dirs=dirs, single=single, pair=vals, best=best[0])
