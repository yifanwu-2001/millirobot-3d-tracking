# Improvements — Stage V (four new experiments, all-new files, nothing existing changed)

> **Post-review corrections (Stages R3-R6 and V8b in `report.md`).** Read
> these before the stage write-ups below; where they conflict, these stand.
>
> 1. **R2a is retracted.** The Viterbi backtracking fix in V11 removed the
>    out-and-back tail that was the stated reason for excluding `f`=1298
>    (post-fix: 0.0% > 5 mm for every camera). The exclusion now rests on R3 —
>    bug-immune held-out predictive evidence, `f`=1298 worse by 266 nats,
>    diffuse across the sequence — and on Q2's five columns. P0, Q2, V3, V5 and
>    V6 were re-run after the fix and are unchanged; the ±0.37 mm headline
>    survives on a corrected footing.
> 2. **V4's camera switch is withdrawn.** The 56.6-nat preference for `f`=516
>    over affine is real but not diffuse: 49.5 nats come from twelve frames in
>    two bursts — one of degraded detection (frames 824-843), one self-crossing
>    of the projected curve (855-869) — neither in the junction (R5). The two
>    cameras remain the equal-weight family V4's own mixture uses; `f`=516 is
>    the working projection for the tube decoder, not a rejected-alternative
>    verdict on affine.
> 3. **V8's junction radius was never measured.** V13 was right: L1's widths
>    were sampled on the old camera and imputed at the junction. V8b shows the
>    gain survives with widths measured on the current projection and a 3.0 mm
>    floor where the image cannot measure (failure 0.9%, held-out 1.1%,
>    junction max 30.3 px) — and that *no* transverse freedom there reverts to
>    axis-only numbers. `v8_tube_tracker.py` now uses that policy (C).
> 4. **V11's out-and-back regression (0.25 → 2.00 mm) is a front-end lag**, a
>    signed −0.25 / +0.50 mm on the two legs, uniform across the return leg,
>    uncorrelated with the tube offset (R6a) — not the tube model.
> 5. **V12's background motion is jitter, not drift**: 0.00 mm between legs,
>    no trend with time, inside the 0.78 mm statistical term (R6b).
> 6. **A fourth offset family is rejected**: a world-fixed direction (gravity
>    / field gradient, R4) fails reverse transfer and leaves the junction
>    untouched. Gravity specifically is untestable from this viewpoint — the
>    camera looks straight down z, so a vertical offset is along the ray.
> 7. The 7-axis numbers in "Stage V5" below are the pre-V8 axis-only
>    configuration; the current deliverable numbers are in `README.md`.

This file is a follow-on to [`report.md`](report.md) and [`README.md`](README.md).
It adds Stages **V1–V5**, targeting exactly the items those documents
themselves flag as open. No existing code or document was modified; every new
artefact is `src/v*.py`, `data/v*.npz`, `out/figs/v1_*.png`,
`out/figs/V_results_animation.mp4` (local asset), and this file.

Reproduce: with `Video.mp4` and `Path 2.csv` at the root and the original
pipeline's `data/` artefacts rebuilt,

```
python src/v1_curvature_offset.py    # V1 the offset model U1 named as untested
python src/v2_sigma_retune.py        # V2 is sigma_px=9 still the right noise width?
python src/v3_multiplicity.py        # V3 the proper multiplicity measure R2a asked for
python src/v4_mixture_posterior.py   # V4 marginalise the camera family properly
python src/v5_results_video.py       # V5 the improved configuration measured + results video
python src/v6_onvessel_verification.py # V6 per-frame audit on real video frames + re-render
python src/v7_junction_diagnosis.py  # V7 what Path 2 can and cannot fix at the junction
python src/v8_tube_tracker.py        # V8 the vessel constraint upgraded from axis to TUBE
python src/v9_vessel_structure.py    # V9 vessel visibility: what segments, what cannot, all made visible
```

All four reuse the existing machinery unmodified (`config.py`, `a_centerline.py`,
`e_hmm.py`, `k_frame.py`) and read the existing `data/` artefacts.

---

## Executive summary

| stage | question it answers | verdict |
|---|---|---|
| V1 | the curvature-locked wall offset U1 called "the natural next model and untested" | **tested, not adopted** — passes the forward out-of-sample test U1's model failed, but fails reverse transfer, hits the physical bound, and does not fix the failure frames |
| V2 | is `sigma_px=9` still right, now that the observation is the de-wobbled trend? | **negative, with the reason found** — 9 px is absorbing the unmodelled ~19 px return-leg wall-hugging offset, not detection noise; the value survives with a corrected rationale |
| V3 | do failures concentrate in self-intersection zones, on a *proper* multiplicity measure? | **Q1's mechanism is absent** — 0/40 remaining failures are branch-selection errors; the 3.1x co-location is a curvature confound and threshold-fragile; the second view is demoted again |
| V4 | marginalise the surviving camera family into the error bar (README axis-4 gap) | **adopted** — plus a new finding: held-out HMM evidence prefers `f`=516 over affine by 56.6 nats, a third non-circular camera discriminator |
| V5 | measure the improved configuration end to end + render the results video | done — see "Stage V5" for the 7-axis numbers and `out/figs/V_results_animation.mp4` |

Net effect on the recommended configuration: **unchanged** (affine + Viterbi +
`v_max`=60 + `sigma_px`=9), with two upgrades: the per-frame uncertainty should
be quoted as V4's marginalised `sigma_mix` (it numerically confirms S1's
hand-built combination), and the working *camera* now has four independent lines
of evidence pointing at pinhole `f`=516 — see V4.

---

## V1 — the curvature-locked wall offset, tested

U1 rejected a constant radial offset with a per-leg angle, and ended with:
"a curvature-dependent offset (large on bends, zero on straights) is the natural
next model and is **untested**." V1 tests it.

**Model.** `X(s) = C(s) + r_max · g(κ(s)) · (cos φ0 · Nc(s) + sin φ0 · B(s))`
with `g(κ) = κ/(κ+κ_half)`, Nc the principal normal (toward the centre of
curvature), B = T×Nc. φ0 = 0 rides the inner wall of a bend, φ0 = π the outer
wall; the offset vanishes on straights and flips side at inflections
continuously. There is **no leg parameter**: the offset is a function of place,
so it is automatically the same on both passes — which is what U1's free
θ_return scan (peak at 0°) already preferred — while T10's opposite-wall
observation in the mid section is explained by Nc flipping across the S-bends.
One model, both prior observations.

**Protocol** (same as U1, both lessons applied): fit (`r_max`, `κ_half`, `φ0`)
on the outbound leg by HMM evidence; predict the return leg with nothing refit;
report both legs' absolute residuals (U1's ratio trap); add the reverse
transfer direction (fit on return, predict outbound), which a geometry-locked
model should pass and a noise-absorbing one should not.

| model | outbound med px | RETURN med px | ret >30px | log Z (all) |
|---|---|---|---|---|
| centerline (r=0) | **1.85** | 6.92 | 8.5% | −7714 |
| U1 constant, same wall | 3.46 | **5.50** | 7.3% | −7606 |
| U1 constant, opposite wall | 4.19 | 8.55 | 9.9% | −7898 |
| curvature-locked (V1) | 2.43 | 6.34 | 7.9% | −7639 |

The curvature-locked model **passes the test U1's model was built for and
failed**: return leg 6.92 → 6.34 px out-of-sample, return >30px 8.5% → 7.9%,
and under Viterbi the overall failure rate drops 3.9% → 3.4% (median 3.48 →
3.09 px). But three things refuse to let it be adopted:

1. **Reverse transfer fails.** Fitted on the return leg (+68.6 nats evidence
   gain), it makes the *fitting* leg worse (6.92 → 7.33 px) and the outbound
   leg worse too (1.85 → 2.56 px). Evidence and residual moving in opposite
   directions is U1's count-3 signature: the marginal likelihood is gaining
   from how probability mass is spread, not from a better fit.
2. **`r_max` hits the grid bound** (3.4 mm ≈ the tube radius) in both fit
   directions — the model wants more offset than the gate shape allows, i.e.
   the saturating form is fighting the data.
3. **The offset is in the wrong place.** The largest fitted offsets sit at
   s = 118–190 mm; T10's observed leg-separation region spans s = 77.75–140.25
   mm, and the median offset there (0.65 mm) barely exceeds the global median
   (0.55 mm). The model is not describing the displacement T10 actually saw.

And the decisive practical point: of the 40 frames the centerline decode fails
on, **80% still fail** under the curvature model (median residual 39.1 → 38.5 px).

**Verdict: not adopted.** A third offset-family model is now rejected or at
best marginal on this data (K2–K4 rotating, U1 constant per-leg, V1
curvature-locked). The wall-hugging displacement T10 found is real, but it is
not described by any smooth offset model tried so far; whatever describes it
must explain why it concentrates in the curved mid section yet flips
inconsistently between passes.

---

## V2 — `sigma_px` re-derived for the observation actually being used

`sigma_px = 9` was set in Stage I to absorb the 8.6 px / 1.17 Hz roll wobble.
But the observation stream fed to the HMM was switched to the de-wobbled
savgol trend in that same stage, and every later stage (P0, Q, S1, U1) uses
that stream. The noise σ was sized for is no longer in the data — the trend's
own off-curve scatter is **2.7 px median**. M3's robustness sweep (<15%) was
run under the mis-specified pinhole camera and scored on median reprojection
only; under the affine camera, where the remaining failures are the branch-ish
tails a sharper likelihood should resolve, it was never re-run.

Premise confirmed, conclusion refuted — and the reason is the finding:

| σ_px (Viterbi) | overall >30px | return med px | return >30px |
|---|---|---|---|
| 2 | 19.0% | 7.10 | 33.1% |
| 3 | 13.4% | 7.06 | 22.9% |
| 4 | 6.5% | 7.05 | 11.9% |
| 5 | 4.4% | 7.06 | 10.7% |
| 6 | 4.0% | 7.02 | 9.6% |
| 9 (default) | **3.9%** | **7.01** | **9.3%** |
| 12 | 3.9% | 7.04 | 9.3% |

Narrowing σ makes everything *worse*, monotonically below 9. The failure-mode
split says why: at σ=3 failures explode to 137 frames and 103 of them become
"weak-match" frames — the sharp likelihood treats the genuinely-displaced
return-leg observations as outliers instead of pulling `s` toward them. **σ=9
is doing the job of absorbing the unmodelled ~19 px return-leg wall-hugging
offset (T10), not detection noise.** That is also exactly why V1's offset
model, which absorbs part of the same mismatch, helped the return leg.

**Verdict: keep σ=9, with a corrected rationale** — and a sharper statement of
the real fix: the observation model needs to describe the wall-hugging offset,
not a wider Gaussian. Until a validated offset model exists (V1 is the third
failure), σ=9 is the honest width.

---

## V3 — the proper multiplicity measure, and the second view demoted again

Q1 reported that remaining failures concentrate in self-intersection zones by
3.1x (mean multiplicity 1.22 good vs 3.80 bad), which re-promoted the second
simultaneous view to priority #3. The report itself flagged the evidence as
thin: 40 failures, and R2a's attempted re-check used an invalid proxy. V3
builds the measure R2a asked for and re-tests the claim.

**Measure.** `m(s) = #{s' : |s'−s| ≥ 10 mm, ‖proj(s')−proj(s)‖ ≤ 15 px}` on the
affine projection — other *locations*, not neighbouring samples (10 mm is five
frames of travel at `v_max`; neighbours are excluded), thresholded at the
likelihood's own scale. ρ ∈ {10,15,20} px and d_min ∈ {5,10,15} mm are swept.

**What the measure shows.**

1. **The affine projection is almost never self-approaching**: m = 0 on 96% of
   the curve at ρ=15 px (99% at ρ=10). Self-intersection is a rare, localised
   property of this projection, not a pervasive condition.
2. **Co-location survives, weakly**: mean m at failure frames 0.70 vs 0.18 at
   all frames — 3.86x, permutation p = 0.031 on n = 40. But the ratio is
   threshold-fragile: it swings from 0.63x (failures *avoid* congested zones)
   to 25.6x across the ρ/d_min table, because m is nonzero on so few samples
   that a handful of frames move it.
3. **The mechanism is absent.** Classifying the 40 failures: **strict
   branch-selection errors = 0** (no failing frame has a genuinely good
   alternative match ≥14 mm away in s); 28/40 match *nothing* on the curve
   within 30 px, in any branch — and their mean multiplicity is **0.00**: they
   are not near self-intersections at all. The 12 "weak-match" frames carry the
   entire co-location signal (mean m 2.33). Under O1's looser criterion
   ("closer than Viterbi's own answer") the branch class is 18%, not the 30–32%
   earlier stages quoted.

**Reading.** Bends are where (a) the return leg hugs the opposite wall (T10)
and (b) the projection approaches itself. Q1's 3.1x conflated the two. What a
second simultaneous view resolves is ambiguity *between candidates that both
match well*; here there are none — the failures are observation-model failures
(wall-hugging + camera systematic), which V2 and V1 probed from the likelihood
side. **The second view is demoted from priority #3 back to a lower rank**; the
top asks (calibration shot, full vessel geometry) are unchanged, and the
highest-value *algorithmic* ask is now a validated model of the wall-hugging
offset — three attempts have failed (K, U1, V1).

---

## V4 — the camera family marginalised, and a new camera discriminator

README axis 4 (verbatim): "A trajectory reported with an honest total error bar
would need to marginalise over the surviving camera family; it currently does
not." Stage S1 combined `sigma_stat` and `delta_cam` by hand as
`sqrt(stat² + (Δ/2)²)`. V4 replaces the hand-combination with the coherent
object: the equal-weight mixture of the two full forward-backward smoothed
arc-length posteriors (the only two cameras no test rejects).

**Consistency: the two routes agree.** `sigma_mix` median 0.98 / p90 2.61 /
max 4.54 mm vs S1's 0.98 / 2.71 / 4.60 mm; the camera-driven excess of the
mixture (median 0.38, p90 2.29 mm) matches S1's Δ/2 (0.37 / 2.37 mm). The
hand-built Gaussian bookkeeping was not distorting anything — but the mixture
is the principled version, and it answers the question the hand-combination
could not:

**Zero frames are camera-bimodal.** With a 2 mm / 20%-mass criterion, not one
of 1025 frames splits into two locations. The surviving family's disagreement
is a smooth per-frame offset, never a branch flip — so a single number + error
bar is an adequate output format, and no multi-hypothesis reporting is needed.
(The deliverable point estimate should be the mixture **median**, which stays
on a real location if a bimodal frame ever does appear — the posterior *mean*
does not, M2's lesson applied to the deliverable itself.)

**The new finding: held-out evidence ranks the camera family.** Each camera's
sequential predictive log-evidence — the probability it assigns to the actual
observation stream under the same σ and motion model, a likelihood *ratio*
rather than a residual graded on the camera's own curve — splits cleanly:

| camera | logZ outbound (fit set) | logZ RETURN (held out) |
|---|---|---|
| pinhole `f`=516 | −4756.9 | **−2849.3** |
| affine | −4807.9 | −2905.9 |
| preference for `f`=516 | **+51.0 nats** | **+56.6 nats** |

The preference holds on the held-out leg, so it is not a calibration-set
artefact. This is a third, non-circular discriminator, and it lands the same
way the other two did:

| | R1 (size cue) | R2a (self-consistency) | V4 (held-out evidence) |
|---|---|---|---|
| pinhole `f`=1298 | passes | **fails** | (rejected in R2a; not re-run here) |
| pinhole `f`=516 | passes | cleanest tail | **best by ~56 nats** |
| affine | **fails at 3.8σ** | clean | worse by ~56 nats |

*Post-review (R3, R5): the R2a column is retracted — after the Viterbi fix every
camera is clean on it. `f`=1298 was then scored on the same held-out evidence
and loses by −266.5 nats, diffusely, so the row's conclusion stands on that
instead. The ~56-nat `f`=516-vs-affine gap is real but comes from twelve frames
in two bursts (one of degraded detection, one self-crossing), so it does not
justify promoting `f`=516 over affine.*

With affine now also disfavoured on tracking evidence, the README's residual
dilemma — "R1 shows real perspective exists, yet the pipeline runs a projection
it has evidence against" — is resolved in the direction the physics asked for:
**pinhole `f`=516 is the model that both respects the measured perspective and
now wins the predictive score, with the affine model retained inside the error
bar** (the mixture keeps it; the 0.75 mm median f=516-vs-affine disagreement
bounds what the choice is worth). The checkerboard remains the top ask — it is
still what would settle the family by measurement — but the working default
camera should switch from affine to `f`=516.

---

## What this changes

1. **Uncertainty output (adopted).** Quote V4's marginalised `sigma_mix` per
   frame with the mixture median as the point estimate; report that no frame is
   camera-bimodal. Headline numbers are unchanged: ±0.98 mm median, ±2.61 mm
   p90, 4.54 mm worst case.
2. **Working camera (recommended change).** affine → pinhole `f`=516, on four
   converging lines (R1, R2a tail, Q2's held-out columns, V4's held-out
   evidence) — *withdrawn post-review: R2a is retracted and V4's gap is twelve
   frames (R5). `f`=516 stays the working projection for the tube decoder;
   affine remains an equal partner in the error bar, not a rejected model.* Q2 already carries this camera's head-to-head numbers: held-out
   6.2 px / 7.9% failures vs affine's 7.0 px / 9.3%.
3. **Second view (demoted).** V3 removes the mechanism Q1 relied on; item #3 of
   the "what would move this furthest" list drops below a validated
   wall-hugging observation model.
4. **σ=9 (kept, rationale corrected).** It is the width of the *unmodelled
   wall offset*, not of detection noise — which is the concrete, quantified
   argument (19 px median on the return leg, V2) for why the next modelling
   effort should go there.
5. **Wall-hugging offset models (three strikes).** K2–K4 (rotating), U1
   (constant per-leg), V1 (curvature-locked) — all rejected on out-of-sample
   or consistency grounds. The displacement is real but resists every smooth
   offset family tried; a per-branch or data-driven (learned) offset prior is
   the remaining idea, and it should be tested with V1's reverse-transfer
   protocol, which proved to be the sharpest of the guards.

---

## Stage V5 — the improved configuration, measured end to end

`src/v5_results_video.py` measures the recommended configuration as a whole and
renders the results animation (`out/figs/V_results_animation.mp4`, local asset —
it renders the phantom background, same rule as s3):

    left panel : the 2D view — lumen tint, projected centerline under the
                 recommended camera (dashed across unfilled segments), the
                 detection trail, and the reported position projected back
    right panel: the marginalised 3D position (V4 mixture median) with its
                 per-frame sigma_mix bar along the vessel tangent

### Configuration

| | README (before) | Stage V (after) |
|---|---|---|
| camera | affine, no depth | **pinhole `f`=516** (R1 + Q2 + V4; R2a later retracted, R3 substitutes) |
| decode | Viterbi, offline-optimal | unchanged |
| `v_max` | 60 mm/s | unchanged |
| `sigma_px` | 9 | 9 — kept, rationale corrected (V2) |
| uncertainty | S1 hand-combination | **V4 camera-family mixture** (median + `sigma_mix`) |
| headline reproj med / p90 | 3.5 / 18.3 px | 3.78 / 16.59 px |
| failure rate (>30 px) | 3.9% | 3.8% |
| held-out (return leg) fail | 9.3% | **7.9%** (6.23 px) |
| speed violations | 0.0% | 0.0% |
| throughput | 0.62 ms/frame causal | 0.66 ms/frame causal (51x real time); Viterbi 4.99 ms/frame offline |
| 3D error bar | ±0.98 / ±2.71 / 4.60 mm | **±0.98 / ±2.61 / 4.54 mm** |

### The 7 axes, quantified under the improved configuration

**1. Geometric accuracy.** Reported position is the V4 mixture median with
`sigma_mix` = **±0.98 mm median, ±2.61 mm p90, 4.54 mm worst case**, decomposing
into 0.78 mm estimator noise and 0.38 mm camera systematic (the f=516↔affine
family bound: 0.75 mm median / 4.75 mm p90). The estimator's own noise floor in
synthetic conditions is 0.47 mm median with depth error 3–4× *smaller* than
in-plane (Stage F, unchanged). The camera ambiguity is now a *bound* (a second
surviving model) rather than an unresolved error, and the primary camera
respects the measured perspective (R1) that the affine model contradicted.

**2. Reprojection self-consistency.** Viterbi under `f`=516: median 3.78 px,
p90 16.59 px, 3.8% failures; held-out return leg 6.23 px / 7.9% (in-sample:held-out
= 2.4×, the M1 gap, now mostly the return leg's physical wall displacement per
T10/V2). The M2 blind spot is re-checked under the same camera: the memoryless
argmax baseline scores 3.46 px — *better* than the tracker — but exceeds 60 mm/s
on 1.5% of frames (max 592 mm/s): reprojection alone still certifies nothing.

**3. Temporal robustness.** Causal filter, N1's exact protocol: median
reprojection **flat 30 → 2 fps** (3.63 → 3.49 px; failure 3.9% → 2.9%), while
detection noise is what costs: effective σ 9 → 13.5 → 21.9 px gives
3.63 → 11.70 → 22.39 px (failures 3.9% → 6.6% → 30.2%). Spend dose on per-pulse
quality, not pulse rate — the conclusion carries over to the improved camera.
One new caveat: the offline Viterbi decoder quantises at low pulse rates (41
velocity bins × a 0.5 s gap = 22.5 mm displacement granularity; 2 fps med
55.4 px) — irrelevant at the clinical 30 fps, and fixable in principle with a
velocity grid scaled to the gap.

**4. Uncertainty representation.** The README's weakest axis, now addressed:
the per-frame error bar is the camera-family **marginal** posterior (V4), not a
conditional one with a hand-built systematic added. The marginalised and
hand-built numbers agree to 0.1 mm (consistency check passed), zero frames are
camera-bimodal (so a single number + bar is an adequate output format), and the
decomposition (0.78 statistical / 0.38 camera) says which lever shrinks which
term.

**5. Parameter sensitivity.** Unchanged from M3 (σ, `accel_sigma`, `p_lost`,
`n_v` each move the median <15%), plus V2's re-derivation of σ under the affine
camera and the trend observation: flat 6–12, degrading below 6 for a *physical*
reason (the width absorbs the unmodelled wall offset), and `v_max` 30→60 remains
the one free win already taken.

**6. Physical plausibility.** 0.0% of frames exceed 60 mm/s (Viterbi is
speed-bounded by construction; measured max exactly 60.0). Anatomical
plausibility (Stage T's metric): the projected centerline lies on tube-like
structure for **98.8%** of samples (worst deviation 5.2 px ≈ 1 mm), detections
99.8%; the K4 rolling-silhouette conclusion and the R1 perspective finding
stand, and the primary camera no longer contradicts the latter. The honest open
item: the quasi-static wall-hugging offset remains unmodelled (K, U1, V1 all
rejected); σ=9 is what absorbs it (V2), quantified at ~19 px on the return leg.

**7. Computational efficiency.** Causal filter 0.66 ms/frame (1521 fps, 51×
real time at 30 fps); Viterbi 4.99 ms/frame offline (the recommended decode is
now cheap enough to run after every acquisition, not just for review). Setup
view search ~90 s, once. Multi-view extension remains linear in views (Stage J).
No GPU anywhere.

---

## Stage V6 — audited against the actual video, and re-rendered on it

The reviewer challenged the result twice with the same two points: the robot's
trajectory in the source video does not match the animation, and a stretch of
the drawn route is outside the vessel. Both challenges deserved better than
the s3-style animation could give: it rendered the BACKGROUND plate, not the
video, so the robot and the estimate could never be seen together, and the
fluid-free tube segments are nearly invisible against the pale background —
an overlay that *reads* as "outside the vessel" is a rendering failure even
when the metric says otherwise. `src/v6_onvessel_verification.py` audits and
fixes both.

**A. Per-frame audit of the reported trajectory** (V4 mixture median, projected
through the f=516 camera, tested against the Stage-T masks on every frame):

| | on a fluid-filled lumen | on any tube structure | off all structure |
|---|---|---|---|
| reported position (1025 frames) | 92.4% | **98.9%** | 1.1% (11 frames, worst 5.2 px = 0.81 mm) |
| the robot detection itself | 98.9% | 99.8% | — |

Every stretch where the reported position leaves the filled lumen —
t = 0–0.4 s, 15.8–16.2 s, 29.3–30.6 s, 33.7–34.1 s (s ≈ 5–6 mm and 106–121 mm) —
sits on an **unfilled, fluid-free tube segment** (78–100% on tube structure
within each stretch): real vessels the orange mask cannot see, exactly the
Stage-T6 finding, now localised in time so the video can be scrubbed to them.
`out/figs/v6_onvessel_verification.png` shows native-resolution, contrast-
enhanced video crops at those stretches with the overlays.

**B. The "trajectory mismatch" is real, and now localised.** In 25 frames —
**frames 889–913, t = 29.6–30.4 s, s = 109–116 mm, one single ~0.9 s episode** —
the detection is >30 px from BOTH surviving cameras' projected curves (worst
45.8 px = 7.9 mm): the robot is in the filled tube below the junction while
the centerline route passes through the pale connecting segment above it. This
is not a camera-choice issue (25 of the 28 return-leg >30 px frames fail under
both cameras alike) — it is the **centerline constraint itself being violated**
by the wall-hugging return leg at the bend junction: the displacement T10
measured, which the three offset models (K2–K4, U1, V1) all failed to capture.
It is the single largest honest defect in the current output, and it is
concentrated in one scrubable moment, not spread through the track.

**C. Two fixes, both in the new `out/figs/V_results_animation.mp4`:**

1. **Rendered on real video frames.** The robot, the raw detection (×), and
   the reported position (●) are on screen together, every frame. The whole
   vasculature is tinted (filled lumen blue; other tube-like structure grey —
   with the Stage-T7 permissiveness caveat), so an unfilled segment can no
   longer read as empty space; the projected route is dashed where it crosses
   one.
2. **An honest, self-widening error bar.** `sigma_reported = sqrt(sigma_mix² +
   max(0, rp_v/SC − r0)²)`, where `rp_v` is the primary camera's own Viterbi
   residual and `r0` its median floor (0.65 mm): when the observation itself
   contradicts the centerline constraint, the bar widens by exactly that
   excess. 112 frames are widened (median 1.01 mm, p90 3.04 mm, max 9.00 mm at
   the 889–913 episode), and the video draws the bar as a dashed circle around
   the reported dot — at the violating moment the circle reaches the robot,
   so the output never looks more confident than it is. Caveat stated plainly:
   this is a post-hoc diagnostic inflation (the filter itself does not know);
   its significance is that it identifies the *centerline assumption*, not the
   camera or the estimator, as the binding constraint at those frames.

Artefacts: `src/v6_onvessel_verification.py`, `data/v6_onvessel.npz`,
`out/figs/v6_onvessel_verification.png`, `out/figs/v6_frame897_wide.png`,
and the re-rendered `out/figs/V_results_animation.mp4` (local asset — it
contains source-video frames).

---

## Stage V7 — "the coordinate table is given": what Path 2 can and cannot fix

The reviewer's follow-up: the centreline table (Path 2) was provided, so why
can the 29.6–30.4 s mismatch not be fixed with it? `src/v7_junction_diagnosis.py`
answers with four checks on data already in the repository:

1. **The table is not too coarse at the junction.** Raw CSV spacing there is
   1.56–2.73 mm; the spline corner-cut at that spacing and the local bend
   radius (4.2 mm) is ~0.13 mm. The junction is not even the sharpest bend of
   the path (2.6 mm radius at s = 163 mm). The table describes this stretch fine.
2. **The axis is right.** The OUTBOUND pass through the same stretch (frames
   460–500) fits the projected curve to a median of 8.5 px = 1.6 mm.
3. **The RETURN pass is displaced by ~one tube diameter**: median 38.1 px
   (7.2 mm), max 45.8 px (8.7 mm at the local scale of 5.25 px/mm — the
   stretch is farther than median depth, so the honest local scale, not the
   5.80 px/mm median, converts pixels).
4. **It is not a routing mistake.** The closest curve point to each return
   detection is at s = 114–118 mm — the same stretch the filter already
   reports. Right place of the vessel, wrong side of the lumen.

**Reading.** Path 2 is the tube AXIS — one curve, no cross-section. The whole
method places the robot ON it, and outbound evidence confirms the axis is
correct to ~1.6 mm here. What the table cannot express is where inside the
lumen the robot presses: on the return pass through the junction it rides the
far wall, a full diameter from the axis side the outbound pass took. Three
offset models have been fitted and honestly rejected on out-of-sample tests
(K2–K4 rotating, U1 per-leg constant, V1 curvature-locked); a fourth fitted to
these 25 frames could never be validated — there is no third pass. The honest
output is therefore V6's: the axis position (correct stretch of vessel) with
the error bar widened to ±9 mm and flagged during the episode. Removing the
offset needs cross-section information the table cannot provide: the robot's
diameter, the lumen surface (STL), or a second simultaneous view.

Artefacts: `src/v7_junction_diagnosis.py`, `data/v7_junction.npz`.

---

## Stage V8 — the vessel constraint upgraded from an AXIS to a TUBE (no new data)

V7 ended by saying the junction mismatch needs cross-section information the
table cannot provide. That was too pessimistic: the table gives the axis, and
the IMAGE already gives the tube width around it (Stage L1's vessel-width
profile). "Use the vessel as the constraint" should mean the LUMEN — a tube of
measured radius around the axis — not the axis line. V8 adds exactly that
degree of freedom as a per-frame latent state, with no new data and no new
physical parameters to fit:

- **d_k**, the signed offset from the axis along the projected curve's image
  normal, bounded by **R_tube(s)** measured from the L1 width profile
  (NaN-filled, clipped to 3–6 mm; median 4.8 mm, 5.3 mm at the junction).
- The **s-decode** runs on the tube-marginalised likelihood
  `log p(uv|s) = logsumexp_d [log N(uv; P(s)+d·n(s), σ) + log N(d; 0, σ_d0)]`,
  so an observation can be explained by any point of the lumen while the
  N(0, 1.2 mm) prior keeps d on the axis wherever the axis fits. Viterbi on
  (s, v) is otherwise unchanged and remains exactly speed-bounded.
- **d is decoded by an exact 1-D chain DP** given the s-path, with a 1.5
  mm/frame smoothness kernel (the robot sweeps wall-to-wall in ~0.3 s).
- The **3D position** is C(s) + δ, δ the minimum-norm 3D offset whose f=516
  projection equals the image offset (2×3 Jacobian pseudo-inverse at the axis
  point), clamped to R_tube(s).

| metric | axis-only (was default) | **TUBE-aware (V8)** |
|---|---|---|
| reprojection med / p90 | 3.78 / 16.59 px | 3.78 / **10.82 px** |
| failure rate (>30 px) | 3.8% | **0.6%** (6 frames) |
| outbound (guard — must not degrade) | 2.65 px | 2.63 px ✓ |
| RETURN leg med / fail | 6.23 px / 7.9% | **5.86 px / 1.7%** |
| junction episode 885–920 med / max | 38.1 / 45.8 px | **26.1 / 33.4 px** |
| out-and-back repeatability | 0.25 mm | 0.25 mm ✓ |
| speed violations | 0.0% (≤60 mm/s) | 0.0% (≤60 mm/s) ✓ |
| decoded offset activity | — | \|d\|>2 mm on 1% of frames; d → −3.5 mm at the junction |

The offset state stays silent wherever the axis fits (median |d| = 0.00 mm on
both legs) and activates only where the axis constraint fails — the behaviour
the three rejected parametric models (K, U1, V1) were trying to hard-code.

**Re-audit with the V8 deliverable** (V6 re-run): reported positions on a
filled lumen 92.4% → **95.9%**, on any tube structure 98.9% → **99.9%**, off
all structure 11 frames → **1 frame** (0.78 mm). The t = 0–0.4 s and
33.7–34.1 s off-lumen stretches disappear entirely; the remaining two are the
unfilled segments (91–100% on tube). The constraint-violation inflation thins
from 112 frames (max ±9.0 mm) to 61 frames (max ±5.8 mm). The results video is
re-rendered with the tube-aware trajectory: at the junction the reported dot
now moves to the wall with the robot, and the residual there (max 33 px ≈
6.3 mm, mostly the detection sitting on the tube edge during the fast wall
sweep plus the widened junction chamber) is covered by the widened bar instead
of being invisible.

**Caveats, stated.** The minimum-norm δ is the shortest 3D offset consistent
with the image offset — a depth component along the viewing ray is invisible,
so the true offset can be longer (bounded by R_tube); R_tube inherits the L1
width measurement's edge-gradient bias; and the 6 remaining >30 px frames are
where the observation itself sits outside the measured lumen — the widened bar
is the honest output there, as everywhere the model runs out of geometry.

**Net effect on the recommended configuration**: pinhole `f`=516 + Viterbi +
`v_max`=60 + `sigma_px`=9 + **tube-aware decode (V8)** + V4-mixture/`V6`-style
error bars. Headline: failure rate 3.9% → **0.6%**, held-out 7.9% → **1.7%**,
p90 18.3 → **10.8 px**, with the physical constraint upgraded from the axis to
the lumen.

Artefacts: `src/v8_tube_tracker.py`, `data/v8_tube.npz`,
`out/figs/v8_tube_tracker.png`, and the re-rendered
`out/figs/V_results_animation.mp4`.

---

## Stage V9 — "can't you segment the vessels?" What segments, what cannot, and making it visible

The reviewer's challenge: the route still *looks* like it leaves the vessel —
can't the vessels be segmented? V9 measures what the image actually supports
(`src/v9_vessel_structure.py`):

1. **Fluid-filled lumen: segmentable, and segmented.** The dye is orange
   (hue 7–12), cleanly separable, 31.0% of the frame — this is what Stage D/T
   always used.
2. **Fluid-free (transparent) tubes: NOT segmentable by appearance — and that
   is a property of the imaging, not the algorithm.** Measured directly: the
   pale tube interior is gray 216–241 where the background is 228–235
   (indistinguishable), and the phantom sits in a clear acrylic block whose
   supports look the same. The only signature is the WALL LINES (60–100
   gray-level dips), which are detected (blackhat at tube scale, 5.6% of
   pixels) but which cannot tell "silicone tube" from "acrylic support" by
   appearance. No segmenter can separate two materials that image identically.
3. **What the route does against everything that IS measurable:**
   - projected route (Path 2, f=516): **98.8%** of samples inside tube-like
     structure, worst excursion **5.4 px = 0.93 mm**, and only 5 of 761 samples
     more than 2 px off;
   - reported trajectory (V8 tube-aware): **99.9%** inside, worst 0.93 mm.

The stretches that prompted the challenge are a **pale vertical connecting
tube** at t ≈ 15.9 s and t ≈ 30.0 s (s ≈ 106–121 mm) — a real, fluid-free
vessel that the robot traverses on BOTH passes. Under the raw contrast it is
invisible; under CLAHE its two wall lines and the robot inside it are plain
(`out/figs/v9_unfilled_verification.png`, native-resolution crops with the
wall lines drawn). "The route is outside the vessel" was therefore a
contrast-visibility artefact, not a geometry error.

**Rendering fix (in the results video):** frames are now CLAHE
contrast-enhanced, the filled lumen tinted blue, tube-like structure grey, the
detected wall lines drawn as faint dark lines, and the route drawn solid inside
filled lumen / dashed inside unfilled segments — so every stretch the route
takes is verifiable by eye. Remaining honest item (unchanged from V6/V8): on
the return pass through the junction the robot rides the lower wall; the
tube-aware estimate follows it to the measured lumen boundary and the widened
bar (61 frames, max ±5.8 mm) covers the rest.

**Rendering fix (in the results video), revised after the reviewer's fourth
challenge.** The first V9 render was itself a failure: a CLAHE wash plus a flat
grey structure tint made the whole phantom uniform, so the network could not be
told from the background — the reviewer was right that the route "looked"
outside the vessel. The final render drops all fill washes: natural colours, a
mild CLAHE, and the **detected wall lines drawn as bright white lines** (all
tubes, filled or not), with the route outlined in dark under yellow — solid
inside filled lumen, dashed inside unfilled segments. At animation scale the
eye can now trace walls around the route everywhere; zoomed evidence at the
junction is in `out/figs/v10_zoom_t30.png` / `v10_zoom_t15.png`.

**The complete route audit, final numbers.** Over all 761 route samples, the
only samples outside the structure mask are at s = 114.5–116.5 mm (the junction
mouth) — five samples >2 px, worst **5.4 px = 0.93 mm**, less than half a wall
width, on a 2 mm stretch of the whole 190 mm route. The reported trajectory is
99.9% inside, worst 0.93 mm. There is no stretch of the route outside the
vasculature; what reads that way is the transparent connector, whose walls are
now drawn.

Artefacts: `src/v9_vessel_structure.py`, `data/v9_vessel_structure.npz`,
`out/figs/v9_unfilled_verification.png`, `out/figs/v10_zoom_t30.png`,
`out/figs/v10_zoom_t15.png`, `out/figs/v10_render_prototype.png`, and the
re-rendered `out/figs/V_results_animation.mp4`.

---

## Stage V11 — genuinely online, tube-aware tracking

V8 fixed the largest geometric defect, but it was not a real-time method: its
input was a centred 41-frame Savitzky–Golay trend (future leakage of about 0.7
s at the window edge) and both `s` and `d` were decoded after seeing all 1025
frames. V11 replaces those components while retaining the measured lumen as
the constraint.

### Method

1. Raw connected-component detections are processed in arrival order. A hard
   innovation gate rejects 18 lost/glitch frames, and a second-order causal
   notch at the measured 1.171875 Hz roll frequency suppresses the silhouette
   wobble. No centred temporal filter is used.
2. A 12-frame receding-horizon Viterbi tracker estimates `(s, ds/dt)` using a
   tube-marginalised likelihood. Once a state is published, later hypotheses
   must descend from that committed position, so future evidence cannot edit
   already emitted history. The output delay is 0.40 s at 30 fps.
3. A second short dynamic program estimates signed image-normal position in
   the lumen. Its robust normal-plus-uniform prior retains mass at the wall;
   V8's narrow Gaussian systematically stopped one grid bin short during the
   wall-hugging episode.
4. The transverse displacement is lifted with the pinhole projection
   Jacobian's pseudoinverse. This is the minimum-norm 3D point consistent with
   the image. The CSV separately reports the remaining unobserved radial bound;
   a single view cannot determine displacement along the projection nullspace.

The 12-frame lag is not cosmetic. A sweep over 0, 3, 6, 9, 12, 15 and 20 frames
showed that this projection needs at least 12 frames to avoid committing to the
wrong initial/self-overlapping branch when the initial 3D location is not
supplied. A zero-lag deployment is possible if the insertion/initial branch is
known, but should not be advertised for this unconstrained-start experiment.

### Results

The offline trend is used only as a common evaluation reference; V11 never
reads it during tracking.

| metric | V8 (full-video offline) | V11 (online, raw input) |
|---|---:|---:|
| algorithmic latency | whole sequence | **0.40 s** |
| compute | 4.99 ms/frame decoder only | **7.26 ms/frame end-to-end tracking (138 fps)** |
| residual median / p90 | 3.38 / 10.28 px | 5.05 / 18.29 px |
| frames >30 px | 0.9% | **1.07%** |
| return-leg frames >30 px | 1.1% | **1.13%** |
| junction 885–919 median / max | 25.9 / 30.3 px | **22.7 / 30.6 px** |
| max axial speed / violations | 60 mm/s / 0% | **60 mm/s / 0%** |
| out-and-back arc-length median | 0.25 mm | 2.00 mm |
| reported points on tube-like structure | 99.8% (V6) | not re-audited |

*Post-review: both columns are the radius-policy-C re-run (widths measured on
the current projection, 3 mm floor where unmeasured — `report.md` V8b). Under
the imputed 5.3 mm junction radius originally used, V11 read 5.09 / 17.94 px,
0.68% failures, return leg 0.0%, junction 12.2 / 19.1 px. The junction residual
roughly doubled once the radius there stopped being invented, and V11's edge
over V8 on that segment shrank to ~3 px median; the rest of the reading below
stands. The out-and-back regression is a front-end lag, not the tube model (R6a).*

The fair reading is that V11 trades some typical-frame smoothness and
out-and-back repeatability for bounded latency and live inputs, while improving
the exact wall-hugging segment that motivated the geometric upgrade. There is
still no external 3D ground truth, so the residual and structure-mask gains do
not prove absolute 3D accuracy. The output explicitly preserves that caveat as
`unseen_radial_bound_mm` (about one lumen radius unless the decoded transverse
offset already consumes that radius).

### Correctness fix in the shared baseline

V11 development exposed a bug in `ArcHMM.viterbi`: the forward recursion
advected position with the previous velocity, but backtracking subtracted the
new velocity. That can join states which were never adjacent in the forward
graph. `src/e_hmm.py` now backtracks with `shift[v_previous]`; the corrected V8
numbers above were regenerated after this fix. The changes are small for V8,
but the fix is required before claiming a hard physical speed constraint.

Reproduce:

```
python src/v8_tube_tracker.py       # regenerated corrected offline comparator
python src/v11_realtime_tube.py     # NPZ, CSV and summary figure
python src/v11_realtime_tube.py --render  # plus auditable 2D + 3D video
```

Artefacts: `src/v11_realtime_tube.py`, `data/v11_realtime.npz`,
`out/v11_realtime_tracking.csv`, `out/figs/v11_realtime.png`, and
`out/figs/V11_realtime_tracking.mp4`.
