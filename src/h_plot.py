import numpy as np, cv2, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from config import DATA, OUT, W, H, FPS, imread_u

z = np.load(DATA / "h_result.npz")
s_hat, s_std, UV, Q_hat, Pg, sg = z["s_hat"], z["s_std"], z["UV"], z["Q_hat"], z["Pg"], z["sg"]
reproj, k_turn = z["reproj"], int(z["k_turn"])
k = np.arange(len(s_hat))
bg = cv2.cvtColor(imread_u(OUT / "background_median.png"), cv2.COLOR_BGR2RGB)

bad = reproj > 30
print(f"[H] frames with reprojection > 30 px: {bad.sum()}/{len(bad)} ({100*bad.mean():.0f}%)")
# contiguous bad runs
d = np.diff(np.r_[0, bad.astype(int), 0]); st, en = np.where(d == 1)[0], np.where(d == -1)[0]
runs = sorted(zip(en - st, st, en), reverse=True)[:6]
print(f"[H] longest failure runs (len, frames): {[(int(a), int(b), int(c)) for a,b,c in runs]}")

fig, ax = plt.subplots(2, 2, figsize=(17, 11))
ax[0,0].imshow(bg)
ax[0,0].plot(Pg[:,0], Pg[:,1], '-', color='0.4', lw=1, label="Path 2 projected")
ax[0,0].plot(UV[:,0], UV[:,1], 'b-', lw=2.5, label="detected (de-wobbled)")
sc = ax[0,0].scatter(Q_hat[:,0], Q_hat[:,1], c=reproj, cmap="inferno_r", s=7, vmin=0, vmax=60)
plt.colorbar(sc, ax=ax[0,0], label="reprojection err px")
ax[0,0].set_xlim(0,W); ax[0,0].set_ylim(H,0); ax[0,0].legend(fontsize=8)
ax[0,0].set_title("estimate reprojected (colour = error)")

ax[0,1].plot(k, s_hat, lw=1.5, label="s estimate")
ax[0,1].fill_between(k, s_hat-2*s_std, s_hat+2*s_std, alpha=.3, label="+-2 sigma")
ax[0,1].axvline(k_turn, color='r', ls='--', label="turnaround")
ax[0,1].set_xlabel("frame"); ax[0,1].set_ylabel("arc length s (mm)")
ax[0,1].legend(fontsize=8); ax[0,1].set_title("recovered arc length")

ax[1,0].semilogy(k, np.maximum(reproj, .1), lw=.8); ax[1,0].axhline(30, color='r', ls='--')
ax[1,0].axvline(k_turn, color='r', ls=':')
ax[1,0].set_xlabel("frame"); ax[1,0].set_ylabel("reprojection px"); ax[1,0].set_title("where the fit breaks")

ax[1,1].plot(k, s_std, lw=1); ax[1,1].set_xlabel("frame"); ax[1,1].set_ylabel("posterior std (mm)")
ax[1,1].set_title("filter's own uncertainty -- does it KNOW when it is lost?")
ax2 = ax[1,1].twinx(); ax2.semilogy(k, np.maximum(reproj,.1), color='r', alpha=.4, lw=.7)
ax2.set_ylabel("reproj px", color='r')
corr = np.corrcoef(s_std, np.log10(np.maximum(reproj, .1)))[0,1]
ax[1,1].text(.02,.95, f"corr(sigma, log err) = {corr:.2f}", transform=ax[1,1].transAxes)
print(f"[H] correlation between posterior sigma and log reprojection error: {corr:.2f}")
print(f"    -> {'filter is self-aware of failure' if corr>0.4 else 'filter is OVERCONFIDENT when wrong'}")
plt.tight_layout(); plt.savefig(OUT / "figs" / "h_endtoend.png", dpi=105)
print("[H] saved out/figs/h_endtoend.png")
