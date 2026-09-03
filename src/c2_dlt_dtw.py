"""Stage C2 - full projective camera by alternating DLT <-> monotone DTW.

No intrinsics assumed: we estimate the full 3x4 camera matrix P directly by DLT.
Correspondence between the ordered 2D outbound track and the ordered 3D
centerline is re-established each iteration by Dynamic Time Warping, which
enforces monotonicity (the robot cannot go backwards during the outbound leg)
while allowing arbitrary speed variation and foreshortening.
"""
import numpy as np
from config import DATA, OUT
from a_centerline import load as load_cl


def normalize2d(U):
    m = U.mean(0); sc = np.sqrt(2) / np.linalg.norm(U - m, axis=1).mean()
    T = np.array([[sc, 0, -sc * m[0]], [0, sc, -sc * m[1]], [0, 0, 1]])
    return T, np.c_[U, np.ones(len(U))] @ T.T


def normalize3d(X):
    m = X.mean(0); sc = np.sqrt(3) / np.linalg.norm(X - m, axis=1).mean()
    T = np.eye(4); T[:3, :3] *= sc; T[:3, 3] = -sc * m
    return T, np.c_[X, np.ones(len(X))] @ T.T


def dlt(X, U):
    """Full projective 3x4 camera from >=6 correspondences (Hartley-normalised DLT)."""
    T2, Un = normalize2d(U); T3, Xn = normalize3d(X)
    n = len(X); A = np.zeros((2 * n, 12))
    for i in range(n):
        Xi = Xn[i]; u, v, w = Un[i]
        A[2*i]     = np.r_[np.zeros(4), -w * Xi,  v * Xi]
        A[2*i + 1] = np.r_[ w * Xi, np.zeros(4), -u * Xi]
    _, S, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)
    P = np.linalg.inv(T2) @ P @ T3
    return P / P[2, 3], S[-2] / S[-1] if S[-1] > 0 else np.inf


def project(P, X):
    h = np.c_[X, np.ones(len(X))] @ P.T
    z = h[:, 2:3]
    return h[:, :2] / np.where(np.abs(z) < 1e-9, 1e-9, z), z[:, 0]


def dtw_corr(A, B, band=None):
    """Monotone correspondence between ordered 2D polylines A (n) and B (m).
    Returns index pairs. O(n*m) - fine at these sizes."""
    n, m = len(A), len(B)
    D = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)
    acc = np.full((n + 1, m + 1), np.inf); acc[0, 0] = 0
    for i in range(1, n + 1):
        lo, hi = 1, m
        if band:
            c = int(i * m / n); lo, hi = max(1, c - band), min(m, c + band)
        prev = acc[i - 1]; cur = acc[i]
        for j in range(lo, hi + 1):
            cur[j] = D[i - 1, j - 1] + min(prev[j], cur[j - 1], prev[j - 1])
    # backtrack
    i, j, pairs = n, m, []
    while i > 0 and j > 0:
        pairs.append((i - 1, j - 1))
        step = np.argmin([acc[i-1, j], acc[i, j-1], acc[i-1, j-1]])
        if step == 0: i -= 1
        elif step == 1: j -= 1
        else: i -= 1; j -= 1
    return np.array(pairs[::-1]), acc[n, m] / max(n, m)


def run(cov=1.0, flip=False, M2=300, iters=12, verbose=True):
    s3, C3, _, _ = load_cl()
    z = np.load(DATA / "track2d_clean.npz"); uv = z["uv"]; k_turn = int(z["k_turn"])
    out2d = uv[:k_turn + 1]
    d2 = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(out2d, axis=0), axis=1))])
    t = np.linspace(0, 1, M2)
    U = np.stack([np.interp(t * d2[-1], d2, out2d[:, c]) for c in range(2)], 1)

    ss = s3[-1] * ((1 - t * cov) if flip else (t * cov))
    X3 = np.stack([np.interp(ss, s3, C3[:, c]) for c in range(3)], 1)  # ordered 3D, same count

    idx = np.arange(M2)
    pairs = np.stack([idx, idx], 1)          # identity init
    hist = []
    for it in range(iters):
        P, cond = dlt(X3[pairs[:, 1]], U[pairs[:, 0]])
        proj, zc = project(P, X3)
        pairs, _ = dtw_corr(U, proj, band=max(20, M2 // 6))
        rms = float(np.sqrt(((U[pairs[:, 0]] - proj[pairs[:, 1]]) ** 2).sum(1).mean()))
        hist.append(rms)
        if verbose: print(f"   iter {it:2d}  rms {rms:7.2f} px   cond {cond:8.2e}   depth sign {'ok' if (zc>0).all() or (zc<0).all() else 'MIXED'}")
        if it > 2 and abs(hist[-2] - hist[-1]) < 1e-3: break
    return P, rms, pairs, U, X3, proj


if __name__ == "__main__":
    print("[C2] alternating DLT <-> DTW, testing direction & coverage hypotheses\n")
    best = None
    for flip in (False, True):
        for cov in (1.00, 0.90, 0.80, 0.70):
            print(f"-- {'s=L->s=0' if flip else 's=0->s=L'}, coverage {cov:.0%}")
            P, rms, pairs, U, X3, proj = run(cov, flip, verbose=True)
            if best is None or rms < best[1]:
                best = (P, rms, pairs, U, X3, proj, cov, flip)
            print()
    P, rms, pairs, U, X3, proj, cov, flip = best
    print(f"[C2] BEST: {'s=L->s=0' if flip else 's=0->s=L'}, coverage {cov:.0%}, rms {rms:.2f} px")
    np.savez(DATA / "c2_camera.npz", P=P, rms=rms, cov=cov, flip=flip, U=U, X3=X3, proj=proj)
