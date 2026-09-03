import numpy as np, cv2, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from config import DATA, OUT, imread_u

z = np.load(DATA / "c3_global.npz")
dirs, angs, top, U, X, Ur = z["dirs"], z["angs"], z["top"], z["U"], z["X"], float(z["Ur"])
bg = cv2.cvtColor(imread_u(OUT / "background_median.png"), cv2.COLOR_BGR2RGB)


def ortho(v, ang):
    a = np.array([0, 0, 1.0]) if abs(v[2]) < 0.9 else np.array([1.0, 0, 0])
    e1 = np.cross(v, a); e1 /= np.linalg.norm(e1); e2 = np.cross(v, e1)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    return (X @ np.stack([e1, e2], 1)) @ R.T


Uc = U.mean(0); Urms = np.sqrt(((U - Uc) ** 2).sum(1).mean())
fig, ax = plt.subplots(2, 3, figsize=(19, 11))
for i, axi in enumerate(ax.ravel()):
    sc, di, ri = top[i]; di, ri = int(di), int(ri)
    Q = ortho(dirs[di], angs[ri])
    Qc = Q.mean(0); Qr = np.sqrt(((Q - Qc) ** 2).sum(1).mean())
    Qi = (Q - Qc) / Qr * Urms + Uc                       # map into pixels
    axi.imshow(bg)
    axi.plot(U[:, 0], U[:, 1], 'b-', lw=3, alpha=.75, label="detected 2D track")
    axi.plot(Qi[:, 0], Qi[:, 1], 'lime', lw=2, label="Path 2.csv, orthographic")
    axi.plot(Qi[0, 0], Qi[0, 1], 'go', ms=9); axi.plot(Qi[-1, 0], Qi[-1, 1], 'rs', ms=9)
    axi.set_title(f"rank {i}: {sc*Ur:.1f} px  dir {dirs[di].round(2)}", fontsize=10)
    axi.set_xlim(0, 960); axi.set_ylim(720, 0)
    if i == 0: axi.legend(loc="lower right", fontsize=9)
plt.tight_layout(); plt.savefig(OUT / "figs" / "c3_overlay.png", dpi=100)
print("saved out/figs/c3_overlay.png  (green circle = s=0 end, red square = s=L end)")
