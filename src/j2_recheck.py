"""Stage J2 recheck - the 3-seed sweep suggested that FASTER C-arm rotation makes
tracking worse, non-monotonically (0% catastrophic at 2 deg/s, 5.9% at 5 deg/s,
1.6% at 20 deg/s). That pattern is a strong claim on 3 seeds. Rerun with 8 seeds
and a randomised sweep start phase, and report the spread, not just the mean.
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from config import DATA, FPS
from j_multiview import run, V0, AXIS, rot_about

SEEDS = range(8)
N = 900
print("[J2-recheck] 8 seeds, randomised sweep start phase, mean +- std across seeds\n")
print(f"{'case':<30}{'med mm':>16}{'p90 mm':>16}{'>5mm %':>18}")
for w in (0, 2, 5, 10, 20, 40):
    rs = []
    for sd in SEEDS:
        ph = np.random.default_rng(sd).uniform(0, 360)
        fn = (lambda k, w=w, ph=ph: [rot_about(V0, AXIS, ph + w * k / FPS)])
        rs.append(run(fn, n_frames=N, seed=sd, static=(w == 0)))
    a = np.array(rs)
    tag = f"sweep {w} deg/s ({w*N/FPS:.0f} deg)"
    print(f"{tag:<30}{f'{a[:,0].mean():.2f} +- {a[:,0].std():.2f}':>16}"
          f"{f'{a[:,1].mean():.2f} +- {a[:,1].std():.2f}':>16}"
          f"{f'{a[:,3].mean():.1f} +- {a[:,3].std():.1f}':>18}")
