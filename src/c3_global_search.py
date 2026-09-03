"""Stage C3 - correspondence-free global search for the viewing geometry.

The 11-DOF projective DLT of C2 is over-parameterised on a space curve and
converges to cameras that put points behind the image plane. Here we instead
search a PHYSICALLY VALID model, and we do it without any correspondence:

  1. orthographic stage (6 DOF: view dir 2, in-plane rot 1, scale 1, shift 2)
     - view dir and in-plane rot by exhaustive grid
     - scale + shift removed analytically by normalising centroid & RMS radius
     - score = trimmed one-sided Chamfer, so the 3D path may be LONGER than the
       observed track (the robot may not traverse all of it)
  2. the top-K orthographic hypotheses seed a full perspective refinement (C4).

If even the best alignment leaves a large residual, that is itself the answer:
Path 2.csv would not be the path travelled in this video.
"""
import numpy as np
from scipy.spatial import cKDTree
from config import DATA, OUT
from a_centerline import load as load_cl

NDIR, NROT = 600, 36
M2D, M3D = 160, 480
TRIM = 0.90            # ignore worst 10% of point distances (occlusion / detector noise)


def fib_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n); th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def norm_curve(A):
    A = A - A.mean(0)
    r = np.sqrt((A ** 2).sum(1).mean())
    return A / r, r


def load_curves():
    s3, C3, _, _ = load_cl()
    z = np.load(DATA / "track2d_trend.npz"); uv = z["trend"]; k_turn = int(z["k_turn"])
    o = uv[:k_turn + 1]
    d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(o, axis=0), axis=1))])
    t = np.linspace(0, 1, M2D)
    U = np.stack([np.interp(t * d[-1], d, o[:, c]) for c in range(2)], 1)
    ss = np.linspace(0, s3[-1], M3D)
    X = np.stack([np.interp(ss, s3, C3[:, c]) for c in range(3)], 1)
    return U, X, ss


def run():
    U, X, ss = load_curves()
    Un, Ur = norm_curve(U)
    print(f"[C3] 2D track: {M2D} pts, RMS radius {Ur:.1f} px")
    print(f"[C3] 3D path : {M3D} pts, length {ss[-1]:.1f} mm")
    print(f"[C3] searching {NDIR} view directions x {NROT} in-plane rotations = {NDIR*NROT} hypotheses\n")

    dirs = fib_sphere(NDIR)
    angs = np.linspace(0, 2 * np.pi, NROT, endpoint=False)
    cs, sn = np.cos(angs), np.sin(angs)
    ntrim = int(TRIM * M2D)

    results = []
    for di, v in enumerate(dirs):
        a = np.array([0, 0, 1.0]) if abs(v[2]) < 0.9 else np.array([1.0, 0, 0])
        e1 = np.cross(v, a); e1 /= np.linalg.norm(e1); e2 = np.cross(v, e1)
        Pj = X @ np.stack([e1, e2], 1)                       # (M3D,2) orthographic
        for ri in range(NROT):
            R = np.array([[cs[ri], -sn[ri]], [sn[ri], cs[ri]]])
            Q = Pj @ R.T
            Qn, _ = norm_curve(Q)
            tree = cKDTree(Qn)
            dist, _ = tree.query(Un)
            score = float(np.sort(dist)[:ntrim].mean())      # trimmed one-sided Chamfer
            results.append((score, di, ri))
    results.sort()
    print(f"[C3] trimmed Chamfer (normalised units, 1.0 = RMS radius of the track)")
    print(f"     best {results[0][0]:.4f}   p1 {results[len(results)//100][0]:.4f}"
          f"   median {results[len(results)//2][0]:.4f}   worst {results[-1][0]:.4f}")
    print(f"     best in PIXELS: {results[0][0]*Ur:.1f} px\n")
    print(f"{'rank':>5}{'score':>9}{'px':>8}   view direction")
    for r in range(10):
        sc, di, ri = results[r]
        print(f"{r:5d}{sc:9.4f}{sc*Ur:8.1f}   {dirs[di].round(3)}  rot {np.degrees(angs[ri]):6.1f} deg")

    top = np.array([(sc, di, ri) for sc, di, ri in results[:40]])
    np.savez(DATA / "c3_global_trend.npz", dirs=dirs, angs=angs, top=top,
             U=U, X=X, ss=ss, Ur=Ur, best_score=results[0][0])
    return results, dirs, angs, U, X, Ur


if __name__ == "__main__":
    run()
