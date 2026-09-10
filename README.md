# 2D video → 3D vessel localisation of a magnetic millirobot

**Start here: [`SUMMARY.md`](SUMMARY.md)** — the argument, in a few minutes.

Full experiment log, stage by stage: [`report.md`](report.md). The V-stage
improvements and their review: [`IMPROVEMENTS.md`](IMPROVEMENTS.md),
[`V12_ALIGNMENT.md`](V12_ALIGNMENT.md), [`V13_3D_LIMITS.md`](V13_3D_LIMITS.md).
This file is the task brief, the deliverable, and a 7-axis evaluation of the
current state of the work.

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
> All numbers quoted were produced from the original inputs.

## Approach in one paragraph

A 2D image point back-projects to a ray, so the image alone cannot fix a 3D
position. The vessel geometry supplies what is missing: the robot is *inside
the vessel*, so its 3D state collapses to arc length `s` along the centerline
plus, where the axis alone is wrong, a bounded transverse offset inside the
lumen. The problem becomes 1-D estimation ("which `s` projects to this
pixel?") with a small latent, not 3-D reconstruction. Where the projected
centerline crosses itself, a motion model over `(s, ds/dt)` picks the
reachable candidate. Everything else in `report.md` is about what that leaves
unresolved.

## Deliverable

One trajectory is the answer; the others are variants of it.

| | |
|---|---|
| **deliverable** | V8 tube-aware Viterbi decode (`data/v8_tube.npz`), audited frame by frame in V6 |
| camera | pinhole `f`=516; the depth-free affine model is an equal-weight partner in the error bar, not a rejected alternative (R5) |
| vessel constraint | the lumen, not the axis: transverse offset bounded by the tube radius measured on the current projection, 3.0 mm floor where the image cannot measure, nothing imputed (V8b, policy C) |
| decoder | Viterbi, offline; exactly speed-bounded by construction |
| reprojection median / p90 | 3.38 / 10.28 px |
| failure rate (> 30 px) | 0.9% |
| held-out (return-leg) | 5.43 px, 1.1% failures — the number to quote for unseen data |
| speed violations | 0.0% |
| out-and-back repeatability | 0.25 mm |
| 3D error bar | ±0.98 mm median / ±2.61 mm p90, the camera-family marginal (V4); widened where the observation contradicts the axis (V6, 64 frames, max ±6.0 mm); where the tube offset is active, an along-ray interval no single view closes (V13) |
| anatomical audit | reported position on a fluid-filled lumen 95.8%, on any tube structure 99.8%, off all structure 2 frames (worst 0.34 mm) (V6) |
| online variant | V11: 0.40 s latency, 7.3 ms/frame, 1.07% failures (return leg 1.13%), at the cost of a ~0.75 mm travel-direction lag (R6a) |
| visual | V12 stabilised overlay video, `out/figs/V12_stabilized_aligned.mp4` |

## Evaluation — 7 axes

Scored against the evidence in `report.md` and `IMPROVEMENTS.md`. Stages P-R
and V replaced the camera, the decoder, the vessel constraint and `v_max`, and
retracted several earlier conclusions; the state below is post-V8b, after
every Viterbi-dependent stage was re-run with the fixed backtrack.

### 1. Geometric accuracy

**Sub-mm on the axis; the camera ambiguity that once dominated is now ±0.4 mm
median; the largest remaining uncertainty is the along-ray depth of the tube
offset, which is a property of single-view geometry, not of the estimator.**
In synthetic conditions with real geometry and measured noise, median 3D
error is 0.47 mm, with depth error 3-4x *smaller* than in-plane error once the
centerline constraint is applied (Stage F). On the real video the camera was
the limit: three mutually incompatible cameras fit the same track and no 2D
residual chooses (C7-C13, L2). Two tests that do not grade a camera on its own
projection settled most of it — the per-camera size cue rejects the affine
model at 3.8 sigma (real perspective is measurable, R1), and bug-immune
held-out evidence rejects `f`=1298 by 266 nats, diffusely (R3). Dropping
`f`=1298 collapses the family's 3D spread from **3.25 to 0.37 mm median (p90
5.49 → 2.31)**. The offset then trades depth for anatomy: on the axis model
depth is the best-determined coordinate but the model is wrong at one
junction on one pass (V6/V7); on the tube model the anatomy is respected and
the along-ray component of the offset is unobservable inside a radius the
image does not measure there (V13, V8b). The deliverable takes the tube model
and reports the interval.

### 2. Reprojection self-consistency

**Good, and much improved by the lumen constraint — but the metric certifies
nothing on its own.** Reprojection is 3.38 px median / 10.28 px p90 with 0.9%
failures, against 5.6 / 48.5 / 16.2% under the original pinhole-axis
configuration (Q1, V8b). The held-out return leg is 5.43 px / 1.1%, a 2.3x
in-sample-to-held-out ratio that T10/V2 trace mostly to the return leg's
physical displacement inside the lumen, not to overfitting. Two structural
limits stand. A memoryless nearest-point baseline still scores *better* on
reprojection (3.46 px) while exceeding 60 mm/s on 1.5% of frames (max 592) —
low reprojection is not evidence of tracking (M2, re-confirmed in V5). And
every camera-ranking column in Q2 measures distance to that camera's own
curve — a camera graded on the exam it wrote — which is why R3's evidence,
which integrates over the whole posterior, was needed to choose.

### 3. Temporal robustness

**Strong under acquisition-rate reduction; residual failures are bound to one
place on one pass; the online variant pays a lag.** Subsampling the real track
from 30 to 2 fps leaves median reprojection flat (3.63 → 3.49 px, V5 under the
current camera; N1 originally) — the motion model widens its kernel to match
the gap. Detection noise is what costs: failure 3.9% → 6.6% → 30.2% as the
effective sigma goes 9 → 13.5 → 22 px, so a dose budget belongs in per-pulse
quality, not pulse rate. The one large residual episode (frames 889-913, the
return pass through the junction, 45.8 px under the axis model) is not a
drift but a place — 98% of twice-visited bad locations fail on both passes
(P1) — and the tube-aware decode halves it (max 30.3 px) without inventing a
radius (V8b). A rotating single detector does not substitute for a second
*simultaneous* view (J2). The online decoder (V11) keeps 1.07% failures at
0.40 s latency but trails the robot by a signed −0.25 / +0.50 mm on the two
legs — a causal-filter lag, uniform across the leg and unrelated to the tube
offset (R6a), recoverable with a lag-compensated readout.

### 4. Uncertainty representation

**The camera systematic is now inside the error bar; two earlier tools were
retracted; the honest open gap is an along-ray interval conditional on an
assumed radius.** The per-frame bar is the camera-family marginal posterior
(V4): `sigma_mix` ±0.98 mm median / ±2.61 mm p90, decomposing into 0.78 mm
estimator noise and 0.38 mm camera systematic, with zero camera-bimodal
frames — a single number and bar is an adequate format. The bar self-widens
where the observation contradicts the axis (V6: 64 frames, max ±6.0 mm), so
the output never looks more confident than the geometry allows. Two tools
banked earlier did not survive re-derivation under the corrected camera and
are withdrawn: I2's offline-smoothing gain and I3's abstention rule (Q). The
remaining gap is structural: wherever the tube offset is active, the position
along the viewing ray is bounded only by a tube radius that at the junction is
a 3.0 mm floor, not a measurement (V13, V8b). V13's counterexample — two
trajectories 6 mm apart with identical projections, both inside the radius
and speed limits — is the correct statement of that limit.

### 5. Parameter sensitivity

**Estimator-internal parameters are robust; the one sensitive geometric
choice is whether *any* transverse freedom exists at the junction, not how
much.** Sweeps over `sigma_px`, `accel_sigma`, `p_lost`, `n_v` move median
reprojection by well under 15% (M3); `v_max` was under-set and raising it to
60 mm/s was free (M3, P0); sigma=9 stays not because it is the detection noise
(2.7 px) but because it absorbs the unmodelled ~19 px return-leg offset (V2).
The camera pair that survives differs by 0.37 mm median. The radius policy
for the tube constraint is insensitive to its value — imputed 5.3 mm,
interpolated 4.2, floored 3.0 all give 0.7-1.3% failures — but granting no
freedom at the unmeasured junction reverts to axis-only numbers (3.4%, V8b).

### 6. Physical plausibility

**The on-axis assumption is falsified for one junction on one pass; four
smooth models of the displacement are rejected; gravity cannot be tested from
this viewpoint; the decoder respects the lumen and the speed bound.** The
1.17 Hz image wobble is a rolling-silhouette artefact, not a 3D excursion —
falsified four ways (K1-K5). The quasi-static wall-hugging displacement is
real (T10, V6, V7: up to a tube diameter on the return pass through the
junction), and rotating (K), per-leg constant (U1), curvature-locked (V1) and
world-fixed direction (R4) offset models all fail out-of-sample transfer.
R4 adds the structural point: the working cameras look straight down the CSV
z-axis, so a gravity offset is along the viewing ray and invisible. The tube
decoder absorbs the displacement as a latent bounded by the lumen without
explaining it. Viterbi is exactly speed-bounded (0.0% > 60 mm/s); measured
perspective is real (R1); the reported trajectory lies on tube-like structure
for 99.8% of frames, worst 0.34 mm (V6). Still blocked without the robot's
dimensions: the roll-slip check. `f`=516's principal point, 39% of the frame
off centre, is an identifiability artefact of a single near-flat curve, not a
physical claim (R6c).

### 7. Computational efficiency

**Comfortably real-time on CPU; the lumen constraint and online decoder add
milliseconds, not orders of magnitude.** Causal filter 0.67 ms/frame (50x
real time at 30 fps); offline Viterbi 6.68 ms/frame; the tube-marginalised
likelihood adds a 7-point sum per state (V8, 4.99 ms/frame decoder); the
genuinely online V11 runs end to end at 7.3 ms/frame (138 fps) with a
12-frame lag; V12's registration adds ~5.0 ms/frame. Setup view search ~90 s,
once. Multi-view extension stays linear in views (J). No GPU anywhere; one
becomes necessary only if the static-background detector is replaced by a
learned segmenter for moving-anatomy fluoroscopy.

## What would move this furthest

1. **A checkerboard shot through the same optics.** Separates the remaining
   camera pair (0.37 mm) and makes the principal-point question physical.
   Smallest stake it has ever had; still the cheapest single action.
2. **The lumen surface (STL) or tube diameters.** The junction radius bounding
   the deliverable's offset is an assumption; a surface makes it a
   measurement and turns V13's conditional interval into a bounded one.
3. **A second simultaneous view (~30°).** No longer for 2D branch errors —
   there are none left (V3) — but because it is the only observation that
   resolves the along-ray depth of the tube offset, now the largest stated
   uncertainty in the deliverable.
4. **The robot's dimensions.** Fixes the offset magnitude, unblocks the
   roll-slip check, and lets the size cue be read absolutely.
