"""Stage I - final end-to-end run with the best available camera (C8, 11 params),
plus every validation metric that does not need external ground truth.
"""
import numpy as np, cv2, time, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from config import DATA, OUT, W, H, FPS, imread_u
from a_centerline import load as load_cl
from e_hmm import ArcHMM

s3, C3, _, _ = load_cl(); L = s3[-1]
p = np.load(DATA / "c8_11.npy")
rv, t, f, sa, sb, cx, cy = p[:3], p[3:6], np.exp(p[6]), p[7], p[8], p[9], p[10]
R = Rot.from_rotvec(rv).as_matrix()
print(f"[I] camera: f={f:.0f} px  principal ({cx:.0f},{cy:.0f})  view axis {R[2].round(3)}")
print(f"[I] traversed interval from registration: s = {sa:.1f} .. {sb:.1f} mm")

def proj(X):
    Xc = X @ R.T + t
    return Xc[:, :2] / Xc[:, 2:3] * f + np.array([cx, cy]), Xc[:, 2]

ds = 0.25
sg = np.arange(0, L + 1e-9, ds)
Cg = np.stack([np.interp(sg, s3, C3[:, c]) for c in range(3)], 1)
Pg, zg = proj(Cg)
scale = f / np.median(zg)
print(f"[I] image scale {scale:.2f} px/mm   camera {zg.min():.0f}..{zg.max():.0f} mm   depth range {np.ptp(zg):.1f} mm")

zt = np.load(DATA / "track2d_trend.npz"); UV = zt["trend"]; k_turn = int(zt["k_turn"])
hmm = ArcHMM(sg, Pg, sigma_px=9.0, v_max=30., n_v=41, accel_sigma=20., p_lost=0.05)
t0 = time.time(); s_hat, s_std, v_hat = hmm.forward(UV, 1/FPS); ms = 1000*(time.time()-t0)/len(UV)
X_hat = np.stack([np.interp(s_hat, sg, Cg[:, c]) for c in range(3)], 1)
Q_hat, _ = proj(X_hat)
rp = np.linalg.norm(Q_hat - UV, axis=1)

print(f"\n[I] RESULTS (causal / real-time filter, {len(UV)} frames)")
print(f"    speed                 {ms:.2f} ms/frame = {1000/ms:.0f} fps   (need 30) -> {int(1000/ms/30)}x headroom")
print(f"    reprojection error    median {np.median(rp):.1f} px = {np.median(rp)/scale:.2f} mm")
print(f"                          p90 {np.percentile(rp,90):.1f} px   p99 {np.percentile(rp,99):.1f} px")
print(f"    frames > 30 px        {100*np.mean(rp>30):.1f}%   (was 18.0% with the C7 camera)")
print(f"    posterior sigma       median {np.median(s_std):.2f} mm   p90 {np.percentile(s_std,90):.2f} mm")

A, B = np.arange(k_turn+1), np.arange(k_turn+1, len(UV))
d_img, ia = cKDTree(UV[A]).query(UV[B]); sel = d_img < 8.0
dsr = np.abs(s_hat[B][sel] - s_hat[A][ia[sel]])
print(f"\n[I] OUT-AND-BACK REPEATABILITY  (n={sel.sum()} overlapping frames, independent check)")
print(f"    |s_return - s_outbound|  median {np.median(dsr):.2f} mm   p90 {np.percentile(dsr,90):.2f} mm")
print(f"    fraction agreeing within 2 mm: {100*np.mean(dsr<2):.0f}%   within 5 mm: {100*np.mean(dsr<5):.0f}%")

corr = np.corrcoef(s_std, np.log10(np.maximum(rp, .1)))[0,1]
print(f"\n[I] self-awareness  corr(posterior sigma, log reprojection error) = {corr:.2f}")
gate = s_std > np.percentile(s_std, 75)
print(f"    if we reject the 25% most uncertain frames, error > 30 px drops from "
      f"{100*np.mean(rp>30):.1f}% to {100*np.mean(rp[~gate]>30):.1f}%")

np.savez(DATA / "i_final.npz", s_hat=s_hat, s_std=s_std, X_hat=X_hat, Q_hat=Q_hat,
         UV=UV, Pg=Pg, sg=sg, Cg=Cg, rp=rp, k_turn=k_turn, scale=scale, R=R, t=t, f=f)

bg = cv2.cvtColor(imread_u(OUT / "background_median.png"), cv2.COLOR_BGR2RGB)
fig = plt.figure(figsize=(18, 10))
a1 = fig.add_subplot(2, 3, 1); a1.imshow(bg)
a1.plot(Pg[:,0], Pg[:,1], color='0.35', lw=1.2, label="Path 2 projected")
a1.plot(UV[:,0], UV[:,1], 'b-', lw=2.2, label="detected")
sc = a1.scatter(Q_hat[:,0], Q_hat[:,1], c=rp, cmap="inferno_r", s=6, vmin=0, vmax=40)
plt.colorbar(sc, ax=a1, label="reproj px"); a1.set_xlim(0,W); a1.set_ylim(H,0)
a1.legend(fontsize=8); a1.set_title("final overlay")
a2 = fig.add_subplot(2, 3, 2)
a2.plot(s_hat, lw=1.4); a2.fill_between(np.arange(len(s_hat)), s_hat-2*s_std, s_hat+2*s_std, alpha=.3)
a2.axvline(k_turn, color='r', ls='--'); a2.set_xlabel("frame"); a2.set_ylabel("s (mm)")
a2.set_title("recovered arc length +- 2 sigma")
a3 = fig.add_subplot(2, 3, 3)
a3.hist(dsr, bins=40); a3.set_xlabel("|s_return - s_out| (mm)"); a3.set_title("out-and-back repeatability")
a4 = fig.add_subplot(2, 3, 4, projection="3d")
a4.plot(C3[:,0], C3[:,1], C3[:,2], color='0.6', lw=1, label="Path 2")
sc4 = a4.scatter(X_hat[:,0], X_hat[:,1], X_hat[:,2], c=np.arange(len(X_hat)), cmap="turbo", s=4)
a4.set_title("recovered 3D trajectory"); a4.legend(fontsize=8)
a5 = fig.add_subplot(2, 3, 5); a5.semilogy(np.maximum(rp,.1), lw=.7); a5.axhline(30, color='r', ls='--')
a5.set_xlabel("frame"); a5.set_ylabel("reproj px"); a5.set_title("residual over time")
a6 = fig.add_subplot(2, 3, 6)
zz = np.load(DATA / "f_synth.npz")
for k in zz.files: a6.plot(np.sort(zz[k]), np.linspace(0,1,len(zz[k])), label=k, lw=1.3)
a6.set_xscale("log"); a6.set_xlabel("3D error (mm)"); a6.set_ylabel("CDF")
a6.legend(fontsize=6); a6.set_title("synthetic: 3D error by view direction"); a6.grid(alpha=.3)
plt.tight_layout(); plt.savefig(OUT / "figs" / "i_final.png", dpi=105)
print("\n[I] saved out/figs/i_final.png")
