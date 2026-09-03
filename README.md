# 2D video → 3D vessel localisation of a magnetic millirobot

Full experiment log, stage by stage: [`report.md`](report.md). This file is
the task brief and a 7-axis evaluation of the current state of the work.

## Task

A take-home exercise: given a 2D video of a magnetic milli-robot navigating a
cerebral-artery flow phantom, and a CSV of XYZ points describing the 3D vessel
centerline it travelled, map the robot's 2D image position back to its 3D
location inside the vessel model, in a way that would work for real-time
tracking. The wider motivation is tracking a device in real time from 2D
fluoroscopy alone, given the vascular anatomy in advance.

Inputs available: a 2D video of the robot (`Video.mp4`, 1025 frames, 960x720,
30 fps) and 71 XYZ points along the 3D vessel centerline (`Path 2.csv`, 190 mm
of arc length). No camera intrinsics, no robot dimensions, no second view, no
ground-truth 3D trajectory.

> **Note on inputs.** The video and the centerline CSV are not redistributed in
> this repository, and neither are the derived `data/` and `out/` artefacts.
> Every script reads them from the repository root via `src/config.py`; drop
> your own `Video.mp4` and `Path 2.csv` there and the pipeline runs end to end.
> All numbers quoted in `report.md` were produced from the original inputs.

## Current best configuration

Stages P0/P1 changed the working defaults. Everything below the "7 axes"
heading was written against the OLD defaults (pinhole camera, causal filter,
`v_max`=30) and its numbers are superseded where they conflict with this table:

| | old default | current best |
|---|---|---|
| camera | pinhole `c8_11` (`f`=1298, 11 param) | **affine `l2_affine`** (8 param, no depth term) |
| decode | causal filter | **Viterbi** (0.0% speed violations by construction) |
| `v_max` | 30 mm/s | **60 mm/s** |
| reprojection median / p90 | 5.6 / 48.5 px | **3.5 / 18.3 px** |
| failure rate (>30 px) | 16.2% | **3.9%** |
| held-out (return-leg) failure rate | 25.7% | **9.3%** |
| speed violations | 6.2% (max 1442 mm/s) | **0.0%** (max 60 mm/s) |

The unresolved item is a systematic, not a failure rate. Three mutually
incompatible cameras fit this 2D track; their 3D trajectories spread by a
**median of 3.3 mm (p90 5.6 mm)**, and nothing in the data selects between
them — the size cue (L1) and the tracking metrics point opposite ways. See
Stages P and Q in `report.md`.

Stage Q also retired three earlier conclusions that turned out to be artefacts
of the old camera: I2's 2.4x offline-smoothing win (now 1.3 px causal vs 1.5 px
Viterbi on the same frames), I3's abstention rule (recall 0.27 -> 0.05), and
O1's "failures avoid ambiguous zones" — which **reverses** (bad-frame
multiplicity 1.00 -> 3.80 vs good 1.22), partially restoring the case for a
second simultaneous view.

## Evaluation — 7 axes

Scored against the evidence in [`report.md`](report.md); each verdict is
followed by the specific findings behind it. Numbers quoted below the old
defaults are marked where Stage P supersedes them.

### 1. Geometric accuracy

**Estimator-limited accuracy is excellent (sub-mm); system-limited accuracy
is capped by an unresolved camera ambiguity of several mm.** In synthetic
conditions with real geometry and measured noise, median 3D error is 0.47 mm,
and depth error comes out 3-4x *smaller* than in-plane error once the
centerline constraint is applied — the estimator itself is not the
bottleneck (Stage F). On the real video, focal length is not identifiable
from a single centerline in a single view (Stage C7-C10: a 40x range in `f`
changes cost by <15%), and two equally-plausible camera models recover 3D
trajectories that disagree by a median of 4.2 mm, growing past 10 mm against
an orthographic extreme (Stage C11, C13) — roughly 9x the estimator's own
synthetic error. A size-based cue (Stage L1, unbiased sub-pixel vessel-width
regression) narrows this by rejecting one of the two candidate cameras at 9.8
sigma, and a depth-free affine camera fits the 2D track *better* than any
perspective model tried (Stage L2) — meaning the 2D data cannot even confirm
perspective is the right model family. Net: geometric accuracy is
*algorithmically* sound but *practically* capped at several mm by missing
calibration data, not by the estimator.

### 2. Reprojection self-consistency

**Self-consistent and stable in aggregate, but the raw metric is a weak
validator on its own and degrades notably out-of-sample.** *(Superseded by
Stage P: the affine camera cuts the held-out failure rate from 19.8% to 9.3%
and the whole-track off-curve distance from 4.3 px to 2.7 px.)* In-sample median
reprojection is 5.6 px (0.85 mm), tightening to p90 35.1 px offline (Stage
I/I2). Split by calibration-set vs. held-out leg (Stage M1) the honest
generalisation gap is 2.4-2.9x (4.4-4.7 px in-sample vs. 11-13 px held-out).
More importantly, reprojection error alone cannot be trusted as a
self-consistency check: a memoryless nearest-point baseline achieves *lower*
reprojection error than the full model by teleporting between
self-intersection branches (Stage M2) — so this axis must be read jointly
with axis 6 (physical plausibility), not in isolation. Of the errors that
remain after offline decoding, only 32% have a better-matching alternative
anywhere on the projected curve; 68% have no good match in *any* branch
(Stage O1) — i.e. most residual inconsistency is a registration/detection
miss, not a self-crossing decoding failure.

### 3. Temporal robustness

**Strong under acquisition-rate reduction; failures are episodic rather than
a chronic drift; real-time (causal) mode is meaningfully weaker than offline
review.** *(Stage P sharpens this: the episodes are location-bound — 98% of
twice-visited bad spots fail on both passes — and they were caused by the
pinhole fit itself, not by excursion, sampling, distortion or deformation.)* Subsampling the real track from 30 fps to 2 fps leaves median
reprojection essentially flat (Stage N1) — the motion model absorbs larger
inter-frame gaps for free, a genuinely favourable property for low-dose,
low-pulse-rate fluoroscopy. The dominant single-video failure (86 px p90
residual) is not sustained degradation but one isolated ~1.7 s
mis-registration episode, confirmed by visiting the same physical location
twice (out-and-back) with a 28x error difference between visits (Stage C12).
A rotating single detector does not substitute for a second *simultaneous*
view — temporal spread of viewpoint buys nothing measurable (Stage J2),
confirming that simultaneity, not time-accumulation, is what resolves
ambiguity. Offline forward-backward smoothing recovers 2.4x on the causal
filter's worst quartile using only already-collected data (Stage I2) — real-
time tracking inherits the full failure rate; offline review does
substantially better.

### 4. Uncertainty representation

**A well-calibrated, useful signal within the model — with two open gaps: it
is a weak *global* abstention rule, and it does not currently account for
camera-model uncertainty at all.** The causal filter's posterior sigma
correlates with log-error (r=0.50); rejecting the most-uncertain 25% of
frames cuts the failure rate from 16.2% to 6.0% (Stage I). The offline
forward-backward posterior is a genuine probability, not a smoothness
heuristic, and separates fixed-vs-still-wrong frames sharply within the
flagged subset (0.084 vs 0.012, Stage I2) — and separately, separates two
distinct failure classes discovered later (branch-ambiguity vs. no-good-match,
0.224 vs. 0.041, Stage O1). As a *global*, unconditional abstention threshold
it is considerably weaker (27% recall at 47% precision at the recommended
cutoff, Stage I3) and should not be oversold as a general filter. The larger
gap: none of this uncertainty currently reflects the ~4.2 mm camera-model
ambiguity from axis 1 — reported sigma is conditional on the (unidentified)
camera being correct.

### 5. Parameter sensitivity

**The HMM's own hyperparameters are robust and reasonably well-tuned (one
free improvement found); the dominant sensitivity in the whole pipeline is
external, in the camera model, not internal to the estimator.** A one-at-a-
time sweep over `sigma_px`, `accel_sigma`, `p_lost`, `n_v` moves the median
reprojection by well under 15% across wide ranges (Stage M3) — not fragile to
tuning-on-one-video. `v_max` is the exception and is currently under-set:
raising it from 30 to 60 mm/s is a free -21% median error and -6.5 points on
the failure rate. All of this is dwarfed by sensitivity to the unidentified
camera intrinsic `f` (axis 1): a 40x range in assumed `f` changes 2D cost by
under 15% (Stage C7) while moving the recovered 3D trajectory by several mm —
and even a second (biplane) view only shrinks that uncertainty ~3x, not away
(Stage J4). Parameter sensitivity inside the estimator is a solved problem;
sensitivity to the unmeasured external geometry is not.

### 6. Physical plausibility

**The 3D output respects the strongest physical constraint (on-centerline
motion) by construction; the causal point-estimate summary does not respect
a second, weaker constraint (bounded speed) that the same model encodes.** A
candidate 3D-displacement explanation for the observed 1.17 Hz / 8.6 px image
wobble was tested and decisively falsified across four independent checks —
wrong recovered magnitude, model-evidence collapse under a cleaner-data gate
(the opposite of what a real effect predicts), shape modulation exceeding
centroid modulation at the same frequency, and no improvement from a
detection-side fix (Stages K1-K5) — the wobble is a rolling-silhouette
appearance artefact, not the robot leaving the vessel axis. Separately, the
causal filter's posterior-*mean* trajectory implies frame-to-frame speeds up
to 1442 mm/s — 48x the robot's own assumed 30 mm/s cap — because reporting
the mean of an occasionally bimodal posterior can jump between modes; only
the Viterbi (globally-decoded) path is exactly speed-consistent by
construction (Stage M2). A direct physical cross-check (recovered speed vs.
the known 1.17 Hz roll rate under a no-slip assumption) was attempted and is
blocked by missing data — no robot dimensions, no device metadata in the
video or CSV.

### 7. Computational efficiency

**Comfortably real-time on CPU alone, with large headroom, including under a
multi-view extension; the only added cost (offline smoothing) is
intentionally not real-time.** The causal filter runs at 0.62 ms/frame (1608
fps), a 53x real-time margin at the clinical 30 fps rate; detection is ~2
ms/frame (Stage I). Extending to N simultaneous views sums per-view
log-likelihoods inside the same per-frame update, so cost scales linearly in
views, not combinatorially (Stage J). The one-time correspondence-free view
search costs ~90 s single-threaded and runs once, not per-frame. Offline
Viterbi + forward-backward smoothing (Stage I2) has the same per-step cost as
the causal filter but needs a full second pass and stores history across the
whole sequence — by design an offline tool, not a real-time one. No GPU is
used anywhere in this pipeline; one would only become necessary if the static-
background detector (Stage D) were replaced by a learned segmenter for real,
moving-anatomy fluoroscopy.
