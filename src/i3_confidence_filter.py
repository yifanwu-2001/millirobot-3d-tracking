"""Stage I3 - turn I2's smoothed posterior into an actual abstention rule for
the offline Viterbi trajectory, and check whether p > 0.02 is a good cut.

I2 showed median confidence was 0.084 on frames Viterbi fixed and 0.012 on
frames it still got wrong - suggestive, but two medians do not establish that
a single threshold cleanly separates 1025 individual frames. This scores a
threshold sweep against Viterbi's own reprojection error (>30 px = "bad") as
ground truth, across the FULL sequence, not just the 25% Stage I already
flagged - the confidence signal might catch failures outside that group too,
or miss some inside it.
"""
import numpy as np
from config import DATA

z = np.load(DATA / "i2_viterbi.npz")
s_vit, rp_vit, conf, gate = z["s_vit"], z["rp_vit"], z["conf_vit"], z["gate"]
K = len(rp_vit)
bad = rp_vit > 30

print(f"[I3] {K} frames, {bad.sum()} ({100*bad.mean():.1f}%) are >30 px on the Viterbi path")
print(f"[I3] confidence on bad frames:  median {np.median(conf[bad]):.3f}  "
      f"p75 {np.percentile(conf[bad],75):.3f}")
print(f"[I3] confidence on good frames: median {np.median(conf[~bad]):.3f}  "
      f"p25 {np.percentile(conf[~bad],25):.3f}")

print(f"\n{'threshold':>10}{'flagged %':>11}{'recall':>9}{'precision':>11}"
      f"{'kept >30px%':>13}{'kept median px':>16}")
for thr in (0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10):
    flag = conf <= thr
    tp = (flag & bad).sum(); fn = (~flag & bad).sum(); fp = (flag & ~bad).sum()
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    kept = ~flag
    print(f"{thr:10.3f}{100*flag.mean():11.1f}{recall:9.2f}{precision:11.2f}"
          f"{100*np.mean(bad[kept]):13.1f}{np.median(rp_vit[kept]):16.1f}")

THR = 0.02
flag = conf <= THR
kept = ~flag
print(f"\n[I3] at the proposed threshold p<={THR}: flags {100*flag.mean():.1f}% of all "
      f"{K} frames ({flag.sum()} frames), of which {100*np.mean(bad[flag]):.1f}% were "
      f"actually >30 px (precision) and it catches {100*np.mean(flag[bad]):.1f}% of all "
      f"bad frames (recall).")
print(f"[I3] on the KEPT {kept.sum()} frames: median {np.median(rp_vit[kept]):.1f} px, "
      f"p90 {np.percentile(rp_vit[kept],90):.1f} px, >30px {100*np.mean(bad[kept]):.1f}% "
      f"(unfiltered Viterbi: median {np.median(rp_vit):.1f}, >30px {100*bad.mean():.1f}%)")

np.savez(DATA / "i3_confidence_filter.npz", s_vit=s_vit, conf=conf, flag_low_conf=flag,
         threshold=THR)
print(f"\n[I3] saved data/i3_confidence_filter.npz  "
      f"(s_vit = recommended offline trajectory, flag_low_conf = frames to exclude/mark "
      f"in downstream quantitative analysis)")
