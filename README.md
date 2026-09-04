# 2D video → 3D vessel localisation of a magnetic millirobot

**Start here: [`SUMMARY.md`](SUMMARY.md)** — the argument, in a few minutes.

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

## Approach in one paragraph

A 2D image point back-projects to a ray, so the image alone cannot fix a 3D
position. The vessel geometry supplies what is missing: the robot is *on the
centerline*, so its entire 3D state collapses to one scalar — arc length `s`.
The problem becomes 1-D estimation ("which `s` projects to this pixel?"), not
3-D reconstruction. Where the projected centerline crosses itself, several `s`
explain the same pixel, and a motion model over `(s, ds/dt)` picks the
reachable one. Everything else in `report.md` is about what that leaves
unresolved.

## Current configuration

| | |
|---|---|
| camera | affine, 8 param, no depth term (`l2_affine`) |
| decode | Viterbi, offline-optimal, exactly speed-bounded |
| `v_max` | 60 mm/s |
| reprojection median / p90 | 3.5 / 18.3 px |
| failure rate (>30 px) | 3.9% |
| held-out (return-leg) failure rate | 9.3% |
| speed violations | 0.0% |
| throughput | 0.62 ms/frame causal (53x real-time at 30 fps) |

## Evaluation — 7 axes

Scored against the evidence in [`report.md`](report.md). Stages P-R replaced
the working camera, decoder and `v_max`, and retracted several earlier
conclusions; the numbers below are the post-R state throughout.

### 1. Geometric accuracy

**Sub-mm in principle, and the camera ambiguity that used to dominate it has
been reduced to roughly the estimator's own noise floor.** In synthetic
conditions with real geometry and measured noise, median 3D error is 0.47 mm,
with depth error 3-4x *smaller* than in-plane error once the centerline
constraint is applied — the quantity unobservable in raw back-projection
becomes the best-determined one (Stage F). On the real video the limiting
factor was never the estimator but the camera: focal length is not
identifiable from one centerline in one view (a 40x range in `f` changes 2D
cost by <15%, Stage C7), and three mutually incompatible cameras fit the same
track. Stage R resolved most of that with two tests that do not grade a camera
on its own projection — the per-camera size cue rejects the affine model at
3.8 sigma, and out-and-back arc-length agreement rejects `f`=1298 as the only
model that ever labels one physical location two ways by >5 mm. Dropping
`f`=1298 collapses the family's 3D spread from **3.25 mm to 0.37 mm median
(p90 5.57 → 2.37 mm)**. Remaining honest caveat: the two surviving models are
physically incompatible in *kind* — perspective demonstrably exists (R1), yet
the depth-free affine model is what tracks best — so the pipeline is running a
projection it has evidence against, and only an external intrinsic settles
which physically-correct model also tracks well.

### 2. Reprojection self-consistency

**Good and much improved, but the metric is structurally weak on its own and
was actively misleading before Stage P.** Reprojection median is 3.5 px with
p90 18.3 px and a 3.9% failure rate, against 5.6 / 48.5 / 16.2% under the old
pinhole camera (Stage Q1). Two structural limits stand regardless of the
numbers. First, a memoryless nearest-point baseline still achieves *lower*
reprojection error than the tracker (2.7 px vs 3.5 px) while teleporting
between branches, so low reprojection is not evidence of correct tracking and
must be read jointly with axis 6 (Stage M2, re-confirmed post-swap in Q1).
Second, all of Q2's camera-ranking columns measure distance to *that camera's
own* projected curve — a camera graded on the exam it wrote — which is why
Stage R's two independent discriminators were needed to actually choose one.
Held-out generalisation is 7.0 px / 9.3% on the return leg, which never enters
calibration; the in-sample-to-held-out ratio widened to 3.7x even as both
absolute numbers improved, a real overfitting signal since the camera is fitted
on the outbound leg alone.

### 3. Temporal robustness

**Strong under acquisition-rate reduction; the residual failures are
location-bound rather than a chronic drift.** Subsampling the real track from
30 fps to 2 fps leaves median reprojection essentially flat (Stage N1) — the
motion model widens its transition kernel to match the gap, which is a
favourable property for the low-pulse-rate regime real fluoroscopy actually
runs in. Detection noise, not frame rate, is what costs accuracy (failure rate
16.2% → 42.1% → 81.3% as effective sigma goes 9 → 22 → 46 px), which argues
for spending a dose budget on per-pulse image quality over frames per second.
The failures that remain are tied to *places*, not moments: 98% of
twice-visited bad locations fail on both passes against a 0.0% base rate
(Stage P1), and the cause turned out to be pinhole mis-projection rather than
excursion, undersampling, distortion or phantom deformation — all four were
tested and refuted (P1c/P1d/P1e), and swapping to the affine camera drops
those frames' distance-to-curve from 40.9 px to 3.2 px. A rotating single
detector does not substitute for a second *simultaneous* view (Stage J2):
simultaneity, not angular coverage accumulated over time, is what resolves
ambiguity.

### 4. Uncertainty representation

**The weakest axis, and the one that got worse under scrutiny.** Two
uncertainty tools banked earlier did not survive re-derivation under the
corrected camera (Stage Q): I2's 2.4x offline-smoothing gain on the worst
quartile is gone (1.3 px causal vs 1.5 px Viterbi on the same frames — the
gain was an artefact of the mis-specified camera), and I3's abstention rule
collapsed from 0.27 recall / 0.47 precision to 0.05 / 0.06, because with only
40 failures left in 1025 frames there is very little for a global threshold to
catch. That rule should be dropped rather than re-tuned. The smoothed
posterior still separates good from bad frames in aggregate (0.123 vs 0.061),
so the signal is real — it is the *decision rule* built on it that no longer
pays. The structural gap also remains: reported per-frame sigma reflects
observation noise conditional on an assumed camera, and still does not
propagate the camera-family systematic (now ±0.37 mm median, ±2.37 mm p90)
into the output. A trajectory reported with an honest total error bar would
need to marginalise over the surviving camera family; it currently does not.

### 5. Parameter sensitivity

**Estimator-internal parameters are robust; one was mistuned and fixing it
was free; the dominant sensitivity was external and is now largely resolved.**
A one-at-a-time sweep over `sigma_px`, `accel_sigma`, `p_lost` and `n_v` moves
median reprojection by well under 15% across wide ranges (Stage M3) — not
fragile to having been tuned on a single video. `v_max` was the exception and
was set too tight: raising it 30 → 60 mm/s cut the held-out failure rate
25.7% → 19.5% and the speed-violation rate 6.2% → 2.1%, because part of the
violations were the cap itself forcing the filter to jump (M3, confirmed P0).
The camera model remains a far larger lever than any of these — swapping
pinhole → affine more than halved the held-out failure rate (19.8% → 9.3%) —
but it is no longer an *unbounded* one: Stage R reduced the surviving spread to
0.37 mm median. A second view would only shrink intrinsic uncertainty ~3x and
would not remove it (Stage J4); an external calibration still beats both.

### 6. Physical plausibility

**The output respects the strongest physical constraint by construction, and
the one violation found has a free fix that is now the default.** A candidate
3D-displacement explanation for the 1.17 Hz / 8.6 px image wobble was
falsified across four independent checks — wrong recovered magnitude,
model-evidence *shrinking* as frames get cleaner (the opposite of what a real
effect predicts), shape modulating more strongly than the centroid at the same
frequency, and no benefit from a detection-side fix (Stages K1-K5). The wobble
is a rolling-silhouette appearance artefact, not the robot leaving the vessel
axis; this is the project's most thoroughly cross-validated physical
conclusion. Separately, the causal filter's posterior-*mean* trajectory implied
frame-to-frame speeds up to 1442 mm/s — 48x the robot's own cap — because the
mean of an occasionally bimodal posterior can jump between modes; Viterbi is
exactly speed-bounded by construction (0.0%, Stage M2) and is now the default
for that reason. Stage R1 adds a physical result of its own: measured vessel
width modulates with depth in a way that is self-consistent for both pinhole
models and inconsistent with the affine model at 3.8 sigma, i.e. **real
perspective is present in this scene** even though the depth-free model tracks
better. One physical cross-check remains blocked by missing data: recovered
speed against the known 1.17 Hz roll rate under a no-slip assumption needs the
robot's dimensions, which are not available.

### 7. Computational efficiency

**Comfortably real-time on CPU, with headroom that survives the multi-view
extension; the only expensive additions are offline by design.** The causal
filter runs at 0.62 ms/frame (1608 fps), a 53x margin at the clinical 30 fps
rate, with detection at ~2 ms/frame (Stage I). Extending to N simultaneous
views sums per-view log-likelihoods inside the same per-frame update, so cost
grows linearly in views rather than combinatorially — biplane tracking stays
real-time (Stage J). The one-time correspondence-free view search costs ~90 s
single-threaded and runs once at setup. Viterbi decoding, now the recommended
output, needs the whole sequence and so is retrospective-only; the causal
filter remains available for genuine real-time use at the cost of the speed
violations described in axis 6. No GPU is used anywhere; one becomes necessary
only if the static-background detector (Stage D) is replaced by a learned
segmenter for real, moving-anatomy fluoroscopy — and inference for a small
segmentation net at 30 fps is itself feasible on CPU.

## What would move this furthest

1. **A checkerboard shot through the same optics.** The two surviving camera
   models disagree about whether perspective exists at all; an external
   intrinsic is the only thing that settles it. The stake is smaller than it
   was (0.37 mm, not 3.3 mm) but the ambiguity is now qualitative rather than
   quantitative, which is arguably worse.
2. **The full 3D vessel geometry** (STL or all-branch centerlines), so the
   whole vessel tree constrains the camera instead of one curve.
3. **A second simultaneous view** (~30 degrees suffices). Stage Q raised this
   item's expected value again — remaining failures now concentrate in
   self-intersection zones by 3.1x — though that reversal rests on 40 failures
   and deserves re-establishing on a proper multiplicity measure.
