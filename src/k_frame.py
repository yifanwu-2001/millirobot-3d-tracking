"""Rotation-minimising (parallel transport) frame along the centerline.

The Frenet normal flips sign wherever curvature passes through zero, and this
path has 951 samples with curvature ranging over two orders of magnitude, so a
Frenet frame would inject spurious 180 deg jumps into the roll phase. Parallel
transport has no such singularity.

Returns N1, N2 with (T, N1, N2) right-handed and orthonormal at every sample.
"""
import numpy as np


def transport_frame(C):
    n = len(C)
    T = np.gradient(C, axis=0)
    T /= np.clip(np.linalg.norm(T, axis=1, keepdims=True), 1e-12, None)
    N1 = np.zeros_like(C)
    # seed with any vector not parallel to T[0]
    a = np.array([0., 0., 1.]) if abs(T[0, 2]) < .9 else np.array([1., 0., 0.])
    v = np.cross(T[0], a); N1[0] = v / np.linalg.norm(v)
    for i in range(1, n):
        # rotate the previous normal by the rotation taking T[i-1] to T[i]
        b = np.cross(T[i-1], T[i]); s = np.linalg.norm(b)
        if s < 1e-12:
            N1[i] = N1[i-1]
        else:
            b /= s
            th = np.arctan2(s, float(np.dot(T[i-1], T[i])))
            k = np.cross(b, N1[i-1])
            N1[i] = (N1[i-1] * np.cos(th) + k * np.sin(th)
                     + b * float(np.dot(b, N1[i-1])) * (1 - np.cos(th)))
        N1[i] -= T[i] * float(np.dot(T[i], N1[i]))          # re-orthogonalise
        N1[i] /= np.linalg.norm(N1[i])
    N2 = np.cross(T, N1)
    return T, N1, N2


def check(C, T, N1, N2, verbose=True):
    o1 = np.abs((T * N1).sum(1)).max(); o2 = np.abs((T * N2).sum(1)).max()
    o3 = np.abs((N1 * N2).sum(1)).max()
    nn = np.abs(np.linalg.norm(N1, axis=1) - 1).max()
    twist = np.degrees(np.arccos(np.clip((N1[:-1] * N1[1:]).sum(1), -1, 1)))
    if verbose:
        print(f"[K-frame] orthogonality max |T.N1|={o1:.2e} |T.N2|={o2:.2e} |N1.N2|={o3:.2e}")
        print(f"[K-frame] |N1| deviation from 1: {nn:.2e}")
        print(f"[K-frame] step-to-step normal rotation: max {twist.max():.3f} deg "
              f"(a Frenet frame would show 180 deg flips)")
    return max(o1, o2, o3, nn)
