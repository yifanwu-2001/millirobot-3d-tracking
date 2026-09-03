"""Stage C5 - where does Path 2.csv disagree with the observed 2D track?

Uses a MONOTONE (DTW) correspondence rather than nearest-neighbour Chamfer, so
the residual can be attributed to a specific stretch of arc length instead of
being hidden by the track matching a convenient nearby part of the curve.
Also checks physical plausibility of the recovered camera.
"""
import numpy as np, cv2, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
from config import DATA, OUT, W, H, imread_u
from a_centerline import load as load_cl

z = np.load(DATA / "c4_camera.npz")
rv, t, f, C0 = z["rv"], z["t"], float(z["f"]), z["c"]
U, X = z["U"], z["X"]
s3, C3, T3, _ = load_cl()
ss = np.linspace(0, s3[-1], len(X))

# --- near-orthographic solution is the physically consistent one; recompute it -
# (rerun of the f0=40000 seed, stored below by c4 as the best only if it won)
def project(X, rv, t, f, c=C0):
    Xc = X @ Rot.from_rotvec(rv).as_matrix().T + t
    return Xc[:, :2] / Xc[:, 2:3] * f + c, Xc[:, 2]

Q, zc = project(X, rv, t, f)
print(f"[C5] camera in use: f={f:.0f}px  t_z={t[2]:.0f}mm  depth range {np.ptp(zc):.1f}mm")
print(f"[C5] visible width at nearest depth: {2*zc.min()*np.tan(np.arctan(W/2/f)):.0f} mm"
      f"   (model bbox is 28 x 139 x 20 mm)")

# --- monotone DTW residual vs arc length ------------------------------------
def dtw(A, B):
    n, m = len(A), len(B)
    D = np.linalg.norm(A[:, None] - B[None], axis=-1)
    acc = np.full((n+1, m+1), np.inf); acc[0, 0] = 0
    for i in range(1, n+1):
        acc[i, 1:] = D[i-1]
        for j in range(1, m+1):
            acc[i, j] = D[i-1, j-1] + min(acc[i-1, j], acc[i, j-1], acc[i-1, j-1])
    i, j, pr = n, m, []
    while i > 0 and j > 0:
        pr.append((i-1, j-1))
        k = int(np.argmin([acc[i-1, j], acc[i, j-1], acc[i-1, j-1]]))
        if k == 0: i -= 1
        elif k == 1: j -= 1
        else: i -= 1; j -= 1
    return np.array(pr[::-1])

pairs = dtw(U, Q)
res = np.linalg.norm(U[pairs[:, 0]] - Q[pairs[:, 1]], axis=1)
s_at = ss[pairs[:, 1]]
print(f"\n[C5] DTW-monotone residual px: median {np.median(res):.1f}  p90 {np.percentile(res,90):.1f}  max {res.max():.1f}")

# residual binned by arc length
nb = 20
edges = np.linspace(0, ss[-1], nb+1)
print(f"\n[C5] residual by arc length (mm along Path 2, s=0 is the CSV first row)")
print(f"{'s range mm':>14}{'median px':>11}{'p90 px':>9}{'n':>6}")
binmed = np.zeros(nb)
for b in range(nb):
    m = (s_at >= edges[b]) & (s_at < edges[b+1])
    if m.sum() == 0: binmed[b] = np.nan; continue
    binmed[b] = np.median(res[m])
    flag = "  <== MISMATCH" if binmed[b] > 25 else ""
    print(f"{edges[b]:6.0f}-{edges[b+1]:5.0f}{binmed[b]:11.1f}{np.percentile(res[m],90):9.1f}{m.sum():6d}{flag}")

# --- foreshortening of the ACTUAL recovered view -----------------------------
Rm = Rot.from_rotvec(rv).as_matrix(); vaxis = Rm[2]
T_at = np.stack([np.interp(ss, s3, T3[:, c]) for c in range(3)], 1)
T_at /= np.linalg.norm(T_at, axis=1, keepdims=True)
cosang = np.abs(T_at @ vaxis)
print(f"\n[C5] foreshortening at the ACTUAL view axis {vaxis.round(3)}:")
print(f"     |cos(tangent, ray)|  mean {cosang.mean():.2f}  p90 {np.percentile(cosang,90):.2f}  max {cosang.max():.2f}")
print(f"     fraction of path badly foreshortened (|cos|>0.9): {100*np.mean(cosang>0.9):.1f}%")
print(f"     fraction moderately  (|cos|>0.7): {100*np.mean(cosang>0.7):.1f}%")

fig, ax = plt.subplots(1, 3, figsize=(19, 5.5))
bg = cv2.cvtColor(imread_u(OUT / "background_median.png"), cv2.COLOR_BGR2RGB)
ax[0].imshow(bg); ax[0].plot(U[:,0], U[:,1], 'b-', lw=3, label="detected")
sc = ax[0].scatter(Q[:,0], Q[:,1], c=ss, cmap="viridis", s=9, label="Path 2 projected")
plt.colorbar(sc, ax=ax[0], label="arc length s (mm)")
ax[0].set_xlim(0,W); ax[0].set_ylim(H,0); ax[0].legend(fontsize=8); ax[0].set_title("perspective fit")
ax[1].plot(s_at, res, '.', ms=3); ax[1].plot(edges[:-1]+np.diff(edges)/2, binmed, 'r-o', lw=2, label="binned median")
ax[1].axhline(10, color='g', ls='--'); ax[1].set_xlabel("arc length s (mm)"); ax[1].set_ylabel("residual px")
ax[1].legend(); ax[1].set_title("DTW residual vs arc length")
ax[2].plot(ss, cosang); ax[2].axhline(.9, color='r', ls='--', label="severe foreshortening")
ax[2].axhline(.7, color='orange', ls='--', label="moderate")
ax[2].set_xlabel("arc length s (mm)"); ax[2].set_ylabel("|cos(tangent, view ray)|")
ax[2].legend(); ax[2].set_title("foreshortening along the path, at the ACTUAL view")
plt.tight_layout(); plt.savefig(OUT / "figs" / "c5_diagnose.png", dpi=105)
print("\n[C5] saved out/figs/c5_diagnose.png")
