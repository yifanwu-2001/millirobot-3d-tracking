"""Stage M3 - hyperparameter sensitivity sweep on the HMM (sigma_px, v_max,
accel_sigma, p_lost, n_v). Nothing in the pipeline previously checked whether
Stage I's numbers depend on tuning to this one video, vs. being robust across a
reasonable range - important because there is no second video to tune-then-test
on. One parameter varied at a time around the Stage I defaults
(sigma_px=9, v_max=30, n_v=41, accel_sigma=20, p_lost=0.05).
"""
import numpy as np
from config import DATA, FPS
from e_hmm import ArcHMM

z = np.load(DATA / "i_final.npz")
sg, Pg, UV = z["sg"], z["Pg"], z["UV"]
DEF = dict(sigma_px=9.0, v_max=30.0, n_v=41, accel_sigma=20.0, p_lost=0.05)


def run(**kw):
    p = dict(DEF); p.update(kw)
    hmm = ArcHMM(sg, Pg, **p)
    s_hat, s_std, v_hat = hmm.forward(UV, 1 / FPS)
    Q = np.stack([np.interp(s_hat, sg, Pg[:, c]) for c in range(2)], 1)
    rp = np.linalg.norm(Q - UV, axis=1)
    return np.median(rp), np.percentile(rp, 90), 100 * np.mean(rp > 30)


base = run()
print(f"[M3] default: sigma_px=9 v_max=30 n_v=41 accel_sigma=20 p_lost=0.05")
print(f"     -> reproj med {base[0]:.1f} px  p90 {base[1]:.1f}  >30px {base[2]:.1f}%\n")

sweeps = {
    "sigma_px":     [4, 6, 9, 12, 15, 20],
    "v_max":        [15, 20, 25, 30, 40, 60],
    "accel_sigma":  [5, 10, 15, 20, 30, 50],
    "p_lost":       [0.005, 0.02, 0.05, 0.10, 0.20, 0.35],
    "n_v":          [11, 21, 31, 41, 61, 81],
}
print(f"{'param':<12}{'value':>10}{'reproj med px':>15}{'p90':>8}{'>30px%':>9}{'delta vs default':>18}")
for name, vals in sweeps.items():
    for val in vals:
        med, p90, bad = run(**{name: val})
        tag = "  <- default" if val == DEF[name] else ""
        print(f"{name:<12}{val:>10}{med:15.1f}{p90:8.1f}{bad:9.1f}{100*(med/base[0]-1):+17.0f}%{tag}")
    print()

print("[M3] READING: if reproj med stays within roughly +-20% of the default across each")
print("     row, Stage I's numbers are not fragile to this specific tuning - a genuine")
print("     concern with only one video and no held-out set to tune on.")
