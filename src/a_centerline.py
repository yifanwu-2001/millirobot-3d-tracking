"""Stage A - centerline preprocessing.

Input : Path 2.csv  (71 unevenly spaced XYZ points, mm)
Output: data/centerline.npz  with
        s   (M,)   arc length, uniformly spaced
        C   (M,3)  spline-resampled centerline
        T   (M,3)  unit tangent
        kap (M,)   curvature 1/mm
Also builds a KD-tree at query time (see load()).
"""
import numpy as np
from scipy.interpolate import CubicSpline
from config import CSV, DATA, OUT

DS = 0.20  # mm, resampling step


def build(ds=DS, verbose=True):
    P = np.loadtxt(CSV, delimiter=",", skiprows=2)
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s0 = np.concatenate([[0.0], np.cumsum(d)])

    cs = CubicSpline(s0, P, axis=0)
    # re-parameterise: spline arc length != chord arc length, so integrate |C'|
    fine = np.linspace(0, s0[-1], 20000)
    Pf = cs(fine)
    true_s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(Pf, axis=0), axis=1))])
    M = int(np.floor(true_s[-1] / ds)) + 1
    s = np.arange(M) * ds
    t_of_s = np.interp(s, true_s, fine)          # map true arc length -> spline param
    C = cs(t_of_s)

    d1 = cs(t_of_s, 1); d2 = cs(t_of_s, 2)
    sp = np.linalg.norm(d1, axis=1, keepdims=True)
    T = d1 / np.clip(sp, 1e-9, None)
    kap = np.linalg.norm(np.cross(d1, d2), axis=1) / np.clip(sp[:, 0] ** 3, 1e-12, None)

    if verbose:
        print(f"[A] raw pts {len(P)}  chord length {s0[-1]:.2f} mm  spline length {true_s[-1]:.2f} mm")
        print(f"[A] raw spacing  min {d.min():.2f}  mean {d.mean():.2f}  max {d.max():.2f} mm")
        print(f"[A] resampled to {M} pts @ {ds} mm")
        print(f"[A] curvature 1/mm: median {np.median(kap):.4f}  p95 {np.percentile(kap,95):.4f}  max {kap.max():.4f}")
        print(f"[A] min radius of curvature: {1/kap.max():.2f} mm")
        print(f"[A] bbox extent mm: {(C.max(0)-C.min(0)).round(2)}")

    np.savez(DATA / "centerline.npz", s=s, C=C, T=T, kap=kap, P_raw=P)
    return s, C, T, kap


def load():
    z = np.load(DATA / "centerline.npz")
    return z["s"], z["C"], z["T"], z["kap"]


if __name__ == "__main__":
    build()
