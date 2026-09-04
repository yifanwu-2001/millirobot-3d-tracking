# Experiment report — 2D video → 3D vessel localisation of a magnetic millirobot

Full stage-by-stage log for the 7-day task (see `README.md` for the task brief
and the 7-dimension evaluation summary; this file is the underlying evidence).
Everything runs on CPU; no GPU is needed anywhere in this pipeline (see
"Compute" at the bottom).

```
python src/a_centerline.py     # A  centerline preprocessing
python src/d_detect.py         # D  2D detection in the video
python src/d2_clean.py         # D2 outlier rejection, turnaround detection
python src/d3_wobble.py        # D3 separate net progress from field-driven wobble
python src/c3_global_search.py # C3 correspondence-free global view search
python src/c8_distortion.py    # C8 camera-model ablation
python src/c9_calibrate.py     # C9 single-view calibration experiments
python src/c10_ba.py           # C10 damped free-PP refit: solver crash vs structural degeneracy
python src/c11_f_sensitivity.py # C11 decisive: does the unidentifiable f actually move the 3D output?
python src/c12_residual_diagnosis.py # C12 is the 86px residual arc-length- or time-clustered?
python src/c13_f_infinity_sweep.py   # C13 extends C11 with I2 + the orthographic (f=inf) limit
python src/f_synthetic.py      # F  synthetic validation of the estimator
python src/i_final.py          # I  final end-to-end run + figures
python src/i2_viterbi.py       # I2 offline Viterbi + forward-backward smoothing
python src/i3_confidence_filter.py   # I3 threshold-sweep the smoothed posterior as an abstention rule
python src/i4_map_filter.py          # I4 causal posterior mean vs mode (negative)
python src/j3_pair_observability.py  # J3 which PAIR of C-arm angles
python src/j_multiview.py            # J1/J2 biplane + rotating C-arm tracking
python src/j2_recheck.py             # J2 recheck, 8 seeds
python src/j4_identifiability.py     # J4 Cramer-Rao bound on calibration
python src/j_plot.py                 # J  figures
python src/k_offaxis.py              # K1 off-centerline model, synthetic
python src/k2_real.py                # K2 off-centerline model fit on the real video (negative)
python src/k3_gated.py               # K3 refit on registration-clean frames only (falsifies K2's own hypothesis)
python src/k4_appearance.py          # K4 is the wobble a position or a blob-shape effect? (appearance)
python src/k5_axis_track.py          # K5 does tracking the axis-midpoint instead of centroid help? (negative)
python src/k_plot.py                 # K  figures
python src/m1_holdout.py             # M1 out-of-sample validation: calibrate on outbound leg, score on return
python src/m2_baselines.py           # M2 what does the motion model buy over argmax/frame and monotone DTW?
python src/m3_hyperparam_sweep.py    # M3 HMM hyperparameter sensitivity, one video / no held-out tuning set
python src/l1_size_cue.py            # L1 apparent-size regression: a second equation for the f/depth degeneracy
python src/l2_affine_camera.py       # L2 does a depth-free (affine) camera fit the 2D data at all?
python src/n1_dose_framerate.py      # N1 frame-rate / detection-noise ablation on the real track
python src/o1_branch_ambiguity.py    # O1 are Viterbi's failures actually self-intersection branch errors?
python src/p1_excursion.py           # P1  did the robot leave Path 2? (no - location-bound)
python src/p1b_discriminate.py       # P1b same tube or another? bad on both passes?
python src/p1c_centerline_gaps.py    # P1c is the CSV undersampled at the bad spots? (no)
python src/p1d_radial.py             # P1d is it radial lens distortion? (no)
python src/p1e_rigidity.py           # P1e did the phantom deform? (not supported)
python src/p0_headtohead.py          # P0  every camera under ONE composite metric
python src/p0b_failure_classes.py    # P0b does affine fix P1's no-match class? (yes)
python src/q1_affine_downstream.py   # Q1  re-derive I/I2/I3/M1/M2/O1 under the new camera
python src/q2_camera_family.py       # Q2  all cameras + the 3D spread to quote
python src/r1_size_cue_percamera.py  # R1  size cue redone per camera - retracts L1, rejects affine
python src/r2a_outback_percamera.py  # R2a out-and-back agreement per camera - rejects f=1298
```

Stages P, Q and R supersede parts of what precedes them. Where an earlier
stage's number conflicts with a later one, the later stands; every retracted
claim is flagged where it was made (Stage L's banner, Q's three retractions,
Q2's superseded headline).

## The idea

A 2D image point back-projects to a ray, so the robot's 3D position is not
determined by the image alone. The vessel geometry supplies the missing
constraint: the robot is *on the centerline*, so its 3D state is a single
scalar — arc length `s`. The problem becomes

> given pixel `u_k`, find `s_k` such that `proj(C(s_k))` is close to `u_k`

which is 1-D estimation, not 3-D reconstruction. Where the projected centerline
self-intersects, several `s` explain the same pixel; a motion model over
`(s, sdot)` picks the reachable one. That is the whole pipeline.

## What each stage measured

**A — centerline.** 71 CSV points into a 190.1 mm spline, resampled at 0.2 mm.
Minimum radius of curvature 2.55 mm: strongly tortuous, as expected for
cerebral anatomy.

**D — detection.** The camera is static, so a temporal median is an exact
background plate; the robot is dark against the orange lumen. 1011/1025 frames
(98.6%), median frame-to-frame jump 4.5 px, ~2 ms/frame. No learned detector
needed *for this video* — see "What changes for real fluoroscopy".

**D2 — structure.** The robot goes out and comes back, turning around at frame
670 (22.3 s). The two legs retrace the same 2D curve to a median of 2.0 px.
This gives a ground-truth-free validation metric used throughout.

**D3 — wobble.** The detected centroid oscillates at **1.17 Hz** (in both u and
v: a rotating-field signature), median amplitude **8.6 px**. This is not
cosmetic: it inflated the measured 2D path length from 1172 px to 2674 px,
which had been corrupting the registration until it was removed.

> **Correction.** I originally divided that 8.6 px by the image scale, called
> the result "1.2 mm off-centerline", and treated it as a physical displacement
> and as the error floor. Stages K3 and K4 refute that reading — see
> "K — modelling the robot OFF the centerline" below. The measured quantity is a centroid
> oscillation in the image; its physical origin is not established, and the
> evidence now points away from a rigid off-axis displacement. Every place the
> "1.2 mm off-centerline" figure was used as a *simulation assumption*
> (Stages F, J, K1) is still internally valid, but it is an assumption about a
> hypothetical robot, not a measurement of this one.

**C — registration (the hard part).** Correspondence-free global search over
600 view directions x 36 in-plane rotations gives a sharp minimum
(best 9.8 px vs median 0.42 in normalised units — an order of magnitude), which
confirms **Path 2.csv is the route in this video**.

Refinement then ran into the real obstacle:

| camera model | free params | trimmed residual |
|---|---|---|
| pinhole, principal point at image centre | 9 | 9.37 px |
| + free principal point | 11 | **4.92 px** |
| + radial distortion k1,k2 | 13 | 5.71 px |

Freeing the principal point halves the residual — so the camera model, not the
path, was the limitation.

> **Correction (found while building C10 below).** The "48 px off centre,
> consistent with a cropped video" claim that used to sit here described the
> **13-parameter** model (`c8_13.npy`, with distortion), not the 11-parameter
> free-principal-point model (`c8_11.npy`) that Stages I, K2, K5 and C9 all
> actually load as "the best available camera." `c8_11`'s own fitted principal
> point is `(348, 574)` against an image centre of `(480, 360)` — **~251 px
> off centre**, roughly a quarter of the image width, not 48 px. A cropped
> video easily explains 48 px; it does not comfortably explain 251 px. This
> doesn't overturn anything downstream (every stage that uses `c8_11` was
> already using the real, large offset — only the prose describing it was
> quoting the wrong model's number), but the "consistent with a cropped video"
> framing was too confident and is retracted.

**But the camera is not identifiable from one centerline in one view:**

- fixing `f` on a grid from 800 to 40000 px (50x) changes the cost only from
  8.8 to 11.3 px — focal length is essentially free;
- what *is* pinned down is image scale, 7.06 to 7.48 px/mm (about 3%), and the
  view axis to roughly 14 degrees;
- `cv2.calibrateCamera` with a free principal point diverges immediately
  (principal point runs to (1923, -19), then aborts);
- with the principal point *fixed*, `f` does become identifiable and more
  correspondences do help — bootstrap 5-95% spread narrows 3.0x to 1.2x going
  from 20 to 250 points.

So the situation is precisely pinned: fitting well **requires** a free
principal point, and freeing it **makes calibration degenerate**. More points
cannot break that; a second view or a known intrinsic can.

**C10 — is the `cv2.calibrateCamera` divergence a solver problem or a structural
one?** Re-ran the free-principal-point refit as a bounded, robust
`scipy.optimize.least_squares` (soft_l1 loss, `f`/`cx`/`cy` bounded to the image
+ a plausible focal range) seeded from the stable fixed-PP fit, with a
Tikhonov prior pulling `(cx, cy)` toward the image centre (`src/c10_ba.py`):

| lambda (prior strength) | final f | final principal point | trimmed residual |
|---|---|---|---|
| 0 (undamped) | 521 px | (348, 651) | 4.62 px |
| 1e-3 (best) | 516 px | (347, 643) | 4.62 px |
| 1e-1 (strong) | 363 px | (348, 517) | 4.78 px |

Two results, one expected and one not. **Expected:** a bounded solver does not
explode the way `cv2.calibrateCamera` did — it converges smoothly every time,
confirming that part of C9's divergence was solver behaviour, not a total
absence of any stable optimum. **Not expected, and more informative:** this
"stable" optimum (4.62 px, matching or beating C8's 4.92 px) sits at `f`≈520,
nearly 2.5x smaller than `c8_11`'s own `f`=1298 — a *third*, substantially
different camera model with equally good fit. Three optimizers (C8's bounded
Nelder-Mead, C9's `cv2.calibrateCamera`, C10's damped least-squares) now
disagree on `f` by more than an order of magnitude (7388 fixed-PP / 1298 vs
521 free-PP) while agreeing to within ~0.3 px on trimmed residual. The
Tikhonov prior barely moves `cx` at all across a 10,000x sweep in lambda,
while `cy` and `f` slide together — exactly the flat, coupled valley C7/J4
already quantified, now visible as literally different converged answers
rather than only as a wide confidence interval. **So: yes, damping fixes the
crash; no, it does not fix calibration** — which is what the CRB
(sigma_f=175 px in the best case) already predicted. A Jacobian-based
parameter uncertainty from this one fit (sigma_f=94, sigma_cx=18 px) comes out
*tighter* than that CRB, which is a red flag rather than good news: a local
linearization at one of several very different optima cannot see the other
optima, so it understates the true uncertainty. Trust the CRB, not this
number.

**C11 — decisive: does the unidentifiable `f` actually move the 3D output, or
is it a nuisance parameter?** C10 leaves two camera models (`f`=1298, the one
every other stage uses, vs `f`=516, C10's damped fit) that both explain the 2D
track about equally well. That does not automatically mean it matters — the
estimator's output is `X(s) = C(s)`, a fixed physical map from the *camera-
independent* centerline, so the real question is only whether the two models'
HMM fits recover the same `s_hat(k)` from the same pixels. `src/c11_f_sensitivity.py`
feeds the identical real track through both:

| | model A (f=1298) | model B (f=516) |
|---|---|---|
| own reprojection, median / p90 | 5.6 / 48.5 px | 4.6 / 47.5 px |

Both fits look equally defensible in 2D — this is exactly the ambiguity C7-C10
described. In 3D it is not a nuisance parameter:

| | median | p90 | max |
|---|---|---|---|
| arc-length disagreement, `abs(s_A - s_B)` | 4.20 mm | 9.97 mm | 58.0 mm |
| 3D trajectory disagreement, `abs(X_A - X_B)` | **4.18 mm** | 8.45 mm | 51.9 mm |

And it is not confined to the self-crossing region either (median 6.83 mm near
the turnaround vs 4.06 mm elsewhere, with the *elsewhere* p90 actually higher,
9.44 vs 7.30 mm) — it is systematic across the whole trajectory, not a
localized artefact. **Median disagreement between two equally-plausible camera
models is 4.2 mm — roughly 9x the 0.47 mm median error the same estimator
achieves in Stage F's synthetic validation with realistic noise, and about
3.5x the debunked 1.2 mm off-centerline term K2-K4 spent five stages ruling
out.** The unidentified camera intrinsic is the single largest source of
disagreement anywhere in this project, larger than everything Stages D
through K addressed combined. This is the strongest, most concrete evidence
yet for why the checkerboard shot is the top item below: it is not closing a
theoretical gap, it is the thing actually setting how far off the reported 3D
trajectory could be.

**C13 — does the disagreement keep growing, or does it saturate?** C11 has two
points (f=1298, f=516); `src/c13_f_infinity_sweep.py` adds a third at the
extreme end, f -> infinity (orthographic projection: same rotation as model A,
image scale fixed at its fitted px/mm, no perspective at all), and reruns both
the causal filter and I2's offline Viterbi for all three:

| pair | causal 3D disagreement (med / p90) | Viterbi 3D disagreement (med / p90) |
|---|---|---|
| A (f=1298) vs B (f=516) | 4.18 / 8.45 mm | 4.50 / 8.22 mm |
| A vs C (orthographic) | 10.44 / 64.02 mm | 12.60 / 64.07 mm |
| B vs C (orthographic) | 6.64 / 66.47 mm | 9.50 / 66.61 mm |

Disagreement grows with distance from the fitted region rather than
saturating - but note model C's own fit is far worse than A or B (median
reproj 64.9 px vs their 4.6-5.6 px, C's rotation was reused from model A
rather than independently re-optimised for the orthographic case), so this
comparison is a lower bound on how bad an *equally good-fitting* orthographic
model would look, not a properly controlled third point. The trustworthy part
of C13 is qualitative: A and B are both plausible fits and already disagree by
4.2 mm; nothing here suggests that gap closes.

**C12 — is the raw-track 86 px residual tied to a vessel LOCATION or to a TIME
window?** K2's baseline (centerline-only, raw wobbling track) has p90=86 px.
Binning that residual by arc length looked bad at first (worst bin s=86-95 mm,
median 163 px), but the same test binned by frame index was just as bad
(worst window frames 461-522, median 162 px) — suspicious, since a single
episode visited once produces both signatures simultaneously. The real test is
whether the *same location* is bad on *both* passes of the out-and-back route
(`src/c12_residual_diagnosis.py` isolates this): s=86-95 mm scores median
171 px on the outbound visit (frames 471-522) and **6.1 px** on the return
visit (frames 934-944) — the same few millimetres of vessel, visited twice,
28x apart in error. That rules out a fixed camera/geometry problem at that
location (which would show up both times) and confirms it is a single ~1.7 s
episode. Detection quality is normal throughout the window (confidence 1.0,
blob area typical, no dropped frames) — nothing wrong with what was seen, just
with which arc length the filter assigned it. This is a clean, real-data
instance of exactly the "wrong-branch latch" failure mode Stage I names, not a
new problem.

**I4 — is the causal filter's point estimate (posterior mean) the right
summary, or should it report the posterior mode instead?** `ArcHMM.forward()`
returns `s_hat = (p(s) * s).sum()`, the mean of the filtered marginal. At a
genuinely bimodal posterior the mean can land in the valley between two real
candidates — a location with near-zero probability and no physical meaning —
while the mode always names an actual candidate. Tested side by side on the
same real track without touching `e_hmm.py` (`src/i4_map_filter.py`):

| estimator | reproj px (med) | p90 px | >30 px % |
|---|---|---|---|
| posterior mean (Stage I) | 5.6 | 48.5 | 16.2 |
| posterior mode (I4) | 5.5 | 46.6 | 15.8 |

Barely moves the needle — mean and mode disagree by >3 mm on only 3 of 1025
frames (0.3%). The reason is informative: the *causal filtered* marginal is
almost always effectively unimodal, even when wrong, because the motion prior
suppresses an alternate branch's probability within a few frames of the filter
committing to one side. Bimodality is a property of the *smoothed* (whole-
sequence) posterior, not the causal one — which is exactly why I2's offline
forward-backward pass helps substantially (2.4x on the worst quarter) while
this same-information, still-causal change barely does anything. Mean vs mode
was not the lever; using the future was.

**E — estimator.** HMM over `(s, sdot)`, 761 arc-length bins x 41 velocity
bins. Forward filtering is causal (real-time); Viterbi is available for offline
reference. Outlier-robust likelihood with a uniform component.

**F — synthetic validation** (known truth, real geometry, measured noise:
9 px detection + 1.2 mm off-centerline at 1.17 Hz):

| view direction | 3D err med/p90/max (mm) | depth med/p90 | in-plane med/p90 |
|---|---|---|---|
| fitted view | 0.47 / 1.25 / 2.87 | 0.12 / 0.49 | 0.43 / 1.15 |
| best-observability | 0.49 / 1.26 / 2.72 | 0.09 / 0.41 | 0.46 / 1.18 |
| worst-observability | 0.66 / 1.88 / 12.90 | 0.23 / 1.41 | 0.50 / 1.28 |
| along the path's long axis | 1.04 / 4.56 / 20.78 | 0.87 / 4.18 | 0.56 / 1.60 |

**The headline: once constrained to the centerline, depth error is 3-4x smaller
than in-plane error.** The quantity that is unobservable in raw back-projection
becomes the best-determined one. Failures are confined to degenerate views,
which is an argument for choosing the C-arm angle deliberately.

**I — real end-to-end.** 0.62 ms/frame = 1608 fps, **53x real-time headroom**
at 30 fps. Median reprojection 5.6 px (0.85 mm). Out-and-back repeatability
median 2.53 mm. But 16.2% of frames exceed 30 px: the estimator latches onto
the wrong arc length where the projected path diverges from the observed track.

Usefully, the filter *knows*: corr(posterior sigma, log error) = 0.50, and
**rejecting the 25% most uncertain frames drops the failure rate from 16.2% to
6.0%.** A clinical system should abstain rather than report a confident wrong
position.

**I2 — offline Viterbi + forward-backward smoothing on the abstained frames.**
Rejecting the 25% most uncertain frames throws them away; it does not fix
them. Offline (not real-time), the full sequence's likelihood is available at
once, so a self-crossing that looks ambiguous from the causal past alone can
be resolved using both directions in time. `src/i2_viterbi.py` runs the
existing `ArcHMM.viterbi()` globally-optimal decode plus a forward-backward
smoother (`gamma_k(s,v) = p(s_k,v_k | all 1025 frames)`, giving a genuine
posterior probability, not a smoothness heuristic, at every frame):

| | reproj px (med) | p90 px | >30 px % |
|---|---|---|---|
| causal (Stage I) | 5.6 | 48.5 | 16.2 |
| Viterbi (offline) | 5.6 | 35.1 | 14.0 |

The gain concentrates exactly where it should — on the 25% of frames Stage I's
causal filter already flagged as least confident:

| on the 256 most-uncertain frames | reproj px (med) | p90 px | >30 px % |
|---|---|---|---|
| causal, these frames | 17.5 | 171.4 | 46.9 |
| Viterbi, same frames | 7.3 | 58.9 | 22.7 |

Median error on the previously-worst quarter drops 2.4x (17.5 to 7.3 px) with
zero new data, purely from using the observations that were already collected.
It does not fix everything: 22.7% of even the offline decode is still >30 px
there. But *within that previously-uncertain group*, the smoothed posterior
discriminates the two outcomes — median confidence 0.084 on the frames it
fixed vs 0.012 on the frames still wrong — so unlike a post-hoc spline over
the point estimates (which cannot see which branch is right), this signal
carries real information there. Out-and-back repeatability improves alongside
it: median 1.00 mm vs Stage I's causal 2.53 mm. This is an offline-only gain —
Viterbi needs the whole sequence and cannot run causally at 30 fps — but for
retrospective review (or any non-real-time use of this data) the Viterbi path
is the recommended trajectory over the causal filter's.

**I3 — does that confidence signal work as a general abstention rule?** Not as
cleanly as the two medians above suggest — those were computed *within* the
already-uncertain 256 frames. Scored against Viterbi's own >30 px failures
across all 1025 frames (`src/i3_confidence_filter.py`), confidence on bad
frames is 0.074 vs 0.131 on good ones — a real but much smaller gap than
0.012 vs 0.084:

| threshold p <= | frames flagged | recall | precision | kept set >30 px % |
|---|---|---|---|---|
| 0.01 | 5.2% | 0.22 | 0.60 | 11.4% |
| 0.02 | 7.9% | 0.27 | 0.47 | 11.1% |
| 0.05 | 19.4% | 0.43 | 0.31 | 9.9% |
| 0.10 | 34.8% | 0.64 | 0.25 | 7.8% |

At `p<=0.02` the filter only catches 27% of the actually-bad frames while
mislabelling more than half of what it does flag (47% precision) — a global
threshold is a much weaker tool than the subset-specific numbers in I2
implied. It still buys something (kept-set >30 px drops 14.0% to 11.1% at
essentially no cost, since only 7.9% of frames are dropped), but calling it an
outlier *filter* overstates it; it is a mild, low-cost trim. Getting to
recall > 0.5 means flagging a third of the video (`p<=0.10`), at which point
it is closer to bulk rejection than confidence-based curation. Recommendation:
use `p<=0.02` as a cheap default trim on the Viterbi trajectory, but don't
present it as having resolved the tail — the 86 px registration error (C, C10)
is still the thing actually limiting how good "good" gets.

## J — biplane and rotating C-arm

Single-view tracking failed by latching onto the wrong arc length where the
projected centerline self-intersects. A second view attacks exactly that: two
arc lengths that collide in one projection almost never collide in both. The
HMM extends to N views by summing log-likelihoods (views are conditionally
independent given `s`), and the projection is allowed to change every frame,
which is what a rotating C-arm is.

**J3 — which pair of angles?** Generalising the single-view separation metric
to a pair, `score(v1,v2) = min over distant point pairs of max(sep1, sep2)`,
evaluated exactly over 150 directions x 140 path samples:

| | best angle | typical angle | worst angle |
|---|---|---|---|
| single view | 4.34 mm | 0.82 mm | 0.021 mm |
| view pair | 4.83 mm | **3.33 mm** | 0.202 mm |

The gain is not in the best case (1.1x) but in the **typical** case (4.0x). A
second view makes an arbitrary angle choice nearly as good as an optimal single
view — you stop having to get the angle right. But a badly chosen *pair* is
still 10x worse than a well chosen single view, so biplane widens the good
region rather than removing the need to choose.

For the view actually fitted in Stage C, adding a partner gains 1.3x at 10 deg,
2.2x at 45 deg, 2.4x at 75 deg — saturating around 60 deg.

**J1 — does that translate into tracking accuracy?** Same synthetic harness as
Stage F, 3 seeds:

| configuration | med mm | p90 mm | max mm | >5 mm |
|---|---|---|---|---|
| single view | 0.66 | 1.91 | 7.47 | 0.6% |
| biplane 10 deg | 0.53 | 1.56 | 6.83 | 0.4% |
| biplane 30 deg | 0.46 | 1.29 | 2.76 | **0.0%** |
| biplane 90 deg | 0.45 | 1.11 | 2.49 | **0.0%** |

The median improves only 1.5x, because median error is set by the injected 1.2 mm
off-centerline wobble in this synthetic harness (an assumption later found
unsupported on real data — see the D3 correction and the K section), which no
amount of viewing geometry removes. **The real
gain is the tail: max error 7.47 to 2.49 mm, and catastrophic failures to
zero.** And it saturates by 30 deg, not 90 — the motion model absorbs the
residual ambiguity, so tracking needs less angular diversity than the static
observability metric of J3 suggests. Practically: a modest angular offset buys
almost everything an orthogonal biplane does.

**J2 — can a rotating single detector substitute for biplane?** No. Sweeping
the view direction during acquisition, 8 seeds with randomised start phase
(`src/j2_recheck.py`):

| sweep rate | med mm | p90 mm | >5 mm |
|---|---|---|---|
| 0 deg/s (static) | 0.59 ± 0.09 | 1.74 ± 0.36 | 1.6 ± 1.5% |
| 2 deg/s | 0.64 ± 0.08 | 2.35 ± 1.66 | 2.2 ± 3.8% |
| 5 deg/s | 0.72 ± 0.15 | 2.30 ± 0.70 | 3.8 ± 2.5% |
| 10 deg/s | 0.68 ± 0.06 | 1.90 ± 0.26 | 3.1 ± 2.4% |
| 40 deg/s | 0.64 ± 0.03 | 1.93 ± 0.27 | 2.2 ± 1.5% |

Every row overlaps within one standard deviation: rotation buys nothing
measurable, against biplane's clean drop to 0.0% catastrophic. (A first pass
with only 3 seeds appeared to show a non-monotonic "fast rotation is worse"
effect; the 8-seed rerun shows that was noise. Worth stating because the 3-seed
table looked convincing.)

The reason is that **simultaneity is what matters, not angular coverage.** Two
views at the same instant constrain one 3D point. Two views at different
instants constrain two different points, because the robot has moved in
between. This is exactly why rotational angiography reconstructs static anatomy
well yet cannot replace biplane for real-time tracking of a moving device.

**J4 — does a second view fix the calibration degeneracy?** Cramer-Rao standard
errors on the shared intrinsics, assuming *exact* correspondences (the best case
the single view can possibly have):

| configuration | sigma_f | sigma_cx | Fisher conditioning |
|---|---|---|---|
| single view | 175 px | 98 px | 2.9e-11 |
| biplane 30 deg | 82 px | 43 px | 1.3e-10 |
| biplane 90 deg | 59 px | 29 px | 2.9e-10 |

A second view buys about 3x on focal length and principal point, and 10x on
conditioning — but neither configuration is well conditioned. Note the
single-view bound (17% on f) is far *better* than the 50x flat valley actually
measured in C7, because the bound assumes correspondences the real pipeline does
not have. **Biplane reduces the calibration problem; it does not remove it. A
one-off calibration shot still beats both.**

## K — modelling the robot OFF the centerline (tested on real data — falsified)

Stage J showed a second view fixes the *tail* of the error distribution but not
the median, and D3's image-space wobble (8.6 px, 1.17 Hz) was originally read as
the robot riding ~1.2 mm off the vessel centerline. K1 asks: if that's true, does
modelling it help? K2-K4 then ask the question that actually matters: *is it
true, for this robot?* It is not — see K2-K4 below. The `(r, phi)` machinery
built for K1 is correct and reusable, but on this video it fits `r` to zero
regardless of how the fit is constrained, and the "improvement" it does find
uses the same failure mode as the underlying registration error, not a
distinct physical effect.

```
X(s, r, phi) = C(s) + r * ( cos(phi) N1(s) + sin(phi) N2(s) )
```

with `(N1, N2)` a parallel-transport frame (`k_frame.py`; a Frenet normal flips
180 deg at curvature zeros and would inject false jumps into the roll phase —
the transport frame's largest step-to-step rotation is 3.0 deg). The roll phase
advances *deterministically* at a fixed rate, `phi_k = phi0 + omega k / fps`, so
the state space does **not** grow from 761x41 to 761x41x7x12 (1.9 M states,
which would not run at 30 fps) — only two extra global parameters `(r, phi0)`
are fit by maximising the HMM's own model evidence `log Z`.

**K1 — synthetic, method check only.** Truth `r` = 1.2 mm, roll 1.17 Hz, 6 px
detection noise, 4 seeds — i.e. *assume* the off-axis hypothesis and check
whether the estimator can recover it. Scoring is against the injected
off-axis position, not the centerline, since that is what the model claims to
predict here.

| model | med mm | p90 mm | max mm | >5 mm |
|---|---|---|---|---|
| centerline only | 1.36 | 2.23 | 6.10 | 0.3% |
| oracle (true r, phi0) | 0.53 | 1.48 | 3.77 | 0.0% |
| **estimated by grid search** | **0.55** | 1.50 | 3.74 | 0.0% |

When a true off-axis radius exists, the fit reaches it almost exactly (`r`
recovered in all four seeds, evidence gain 283 nats: log Z -6283.8 at r=0 vs
-6000.8 at r=1.2 mm). **This shows the machinery works, nothing about whether
this real robot actually rides off-axis** — that is a separate empirical
question, answered next.

**K2 — fit on the real video: essentially no support.** Same fit, real track:

| model | reproj px | p90 px | >30px % | out-back mm | log Z |
|---|---|---|---|---|---|
| centerline only | 7.7 | 86.0 | 20.9% | 14.46 | -9368 |
| off-axis, fitted `r`=0.3 mm | 7.7 | 85.7 | 21.1% | 14.35 | -9352 |

Three problems. (1) The fitted radius, 0.3 mm, does not match the ~1.2-1.3 mm
D3 predicted from the image-space wobble — the two routes to "off-axis
distance" disagree by 4x. (2) The evidence gain is 16 nats, against 283 nats
for the same fit on synthetic data with a real effect present — an 18x
shortfall, and every other metric (reprojection, tail rate, repeatability) is
flat to noise. (3) The fit is not even stable: rerunning at a different grid
resolution returns `r`=0.6 mm instead of 0.3 mm — the optimum is not
constrained, it is wandering in a shallow bowl. A roll-rate scan does find its
evidence optimum at 1.17 Hz (-9380 vs -9411/-9420 at neighbouring rates), so the
model is picking up something at the right frequency; it just cannot pin down
an amplitude for it.

**K3 — decisive test: refit using only well-registered frames.** If (2) is
because registration error (median-scale ~86 px = 13 mm) is swamping a real
~1.3 mm signal, then restricting to the cleanest frames should let `r` rise
toward the D3 prediction and the evidence gain should grow. Gating on
reprojection error from the centerline-only fit:

| gate (reproj <) | frames kept | fitted `r` | evidence gain (nats) |
|---|---|---|---|
| none (all 1025) | 1025 | 0.3 mm | 16.3 |
| 40 px | 844 | 0.3 mm | 13.9 |
| 25 px | 796 | 0.3 mm | 11.4 |
| 15 px | 720 | 0.3 mm | 9.1 |
| 10 px | 609 | 0.3 mm | 0.2 |

`r` does not move off 0.3 mm at *any* gate, and the evidence gain shrinks
toward zero as the frames get cleaner — the opposite of what a real off-axis
signal predicts. This rules out "registration noise is hiding the effect" and
points to the model being wrong for this data, not under-powered.

**K4 — why: it's an appearance effect, not a position effect.** An elongated
robot rolling in place changes its projected silhouette without moving its
body centre; that would produce exactly D3's centroid oscillation while being
invisible to a 3D off-axis model. Discriminator: a rigid translation moves the
centroid but cannot change blob area, so if the shape modulates at least as
much as the centroid does, appearance — not position — is the better
explanation. Measured band power (0.8-2.0 Hz, a single shared bump — the
frequency resolution here, 0.117 Hz, cannot resolve separate peaks in that
range) relative to out-of-band power:

| quantity | band power ratio |
|---|---|
| centroid u / v | 4.3x / 4.2x |
| blob orientation | 10.1x |
| blob area | 12.6x |

Area and orientation are modulated *more* strongly than the centroid is. Since
rigid translation cannot touch area, this is decisive against the off-axis
reading: the D3 wobble is dominated by the rolling robot's changing silhouette,
not by its 3D position leaving the vessel axis. (This does not itself prove the
rolling-silhouette mechanism end-to-end — that needs a view of the robot body,
which this video does not give — but it rules out the rigid-translation
account that K1/K2 assumed.)

**K5 — tried a cheaper detection point to dodge the appearance effect: no
change.** If the wobble is the centroid moving as the silhouette's mass
redistributes during roll, the *midpoint* of the blob's `minAreaRect` (defined
by the extremal extent along the principal axis, not by where the mass sits
within that extent) was a candidate to be less sensitive to it — cheap to
compute, no new data (`src/k5_axis_track.py`). It measures the same thing:

| quantity | band power ratio, centroid | band power ratio, axis-midpoint |
|---|---|---|
| u | 4.3x | 4.3x |
| v | 4.2x | 4.2x |

The two tracking points differ by a median of 0.43 px (p90 1.03 px) across
1001 frames — for this blob shape they are essentially the same point, so
refitting the centerline-only HMM on the axis-midpoint track changes nothing
(reproj 7.7 to 7.7 px, log Z -9368 to -9370). The premise was that roll
redistributes mass *within* a fixed silhouette outline; what K4 actually found
is that the outline's *extent* itself changes as the robot rolls (area
modulates 12.6x), which shifts a rectangle fit to that outline by the same
amount as it shifts the centroid. A detection-side fix would need a feature
that is insensitive to the outline's extent, not just to the mass distribution
inside it — this one wasn't.

So the error terms are:

| error term | size | status |
|---|---|---|
| catastrophic wrong-branch latch | up to 20 mm | fixed by a second simultaneous view (J) |
| detection + registration error | p90 86 px ~ 13 mm on real video | dominant real-data term; needs calibration, better geometry (C) |
| ~~off-centerline offset~~ | ~~~1.2 mm~~ | **not supported (K2-K4)** — D3's wobble is best explained as a 2D appearance artefact of robot roll, not a 3D radial displacement; modelling `(r, phi)` gives no measurable benefit on this video |

## L — can appearance break the f/depth degeneracy?

> **RETRACTED IN PART (Stage R1).** L1's headline — "the vessel-width cue
> rejects `f`=516 at 9.8 sigma and favours the large-`f` family" — **is not
> valid and is withdrawn.** L1 regressed log(width) on the depth profile of a
> single camera (`c8_11`) and then compared candidates by rescaling that one
> profile. But the candidates do not share a view direction: `c8_11`'s and the
> affine model's view axes are **46 degrees apart** and their depth profiles
> along the path are **anti-correlated (r = -0.86)**, so the sign of the fitted
> slope depends on whose geometry you assume. L1 therefore asked "given
> `c8_11`'s geometry, which `f` fits?" — a question whose answer is conditional
> on the camera Stage Q2 later showed to be the outlier. Re-run per camera
> (Stage R1), **both pinhole models are self-consistent and the cue cannot
> separate them**; what it does reject is the affine model. The L2 result below
> is unaffected — it involves no depth profile. Read L1's method, not its
> verdict.

C7-C13 established that position data alone cannot identify `f` (flat valley,
40 stable px). Position is not the only signal in the video: a fixed-size
object's apparent SIZE falls off as `1/Z`, so as the robot travels the path's
known ~64 mm depth range, its image size should modulate by an amount that
depends on `Z_med` — and `f = scale * Z_med`, with `scale` already pinned to
3%. This is a second equation the C-stages never used.

**L1 — regress log(apparent size) on the depth each candidate camera predicts.**
Two channels, `src/l1_size_cue.py`:

| channel | slope vs 0 | f=1298 (c8_11) | f=516 (c10_ba) | f=1826 (this stage's own refit) |
|---|---|---|---|---|
| robot blob width | 0.1 sigma (flat) | 10.0 sigma [reject] | 24.8 sigma [reject] | 7.1 sigma [reject] |
| robot sqrt(area) | 3.6 sigma | 12.2 sigma [reject] | 36.0 sigma [reject] | 7.6 sigma [reject] |
| vessel tube width (sub-pixel, unbiased) | 6.0 sigma | **0.3 sigma [consistent]** | **9.8 sigma [reject]** | **1.6 sigma [consistent]** |

The robot channel is a bust — it rejects every candidate, including the one
every downstream stage uses. That is not evidence against `c8_11`; it is K4's
roll-driven appearance noise (12.6x band power) drowning out the much smaller
depth-modulation signal in the same channel. The vessel-width channel avoids
this (the vessel does not roll) and is measured from the CONTINUOUS HSV
"orangeness" field on the background plate with sub-pixel edge crossings — not
from `vessel_mask.png`, which is deliberately dilated 15 px + closed 9 px for
Stage D's detection ROI and would otherwise compress the relative modulation.
It gives a real, 6-sigma slope that is **consistent with the large-`f` family
(c8_11, and this stage's own 1826 px refit) and rejects C10's f=516 at 9.8
sigma.** Caveat stated in the script and repeated here: vessel width is
confounded with genuine anatomical taper, so this is corroborating evidence,
not proof — but it is the first data-driven argument, on this video, for
preferring one of C10's two equally-2D-plausible cameras over the other.

**L2 — does the 2D data need perspective at all?** A weak-perspective (affine)
camera has no `1/Z` term whatsoever — 8 free params (rotation, one global
scale, principal point, traversed interval), vs. the pinhole family's 9-13:

| camera model | free params | trimmed residual |
|---|---|---|
| pinhole, fixed principal point | 9 | 9.37 px |
| pinhole, free principal point | 11 | 4.92 px |
| pinhole, + radial distortion | 13 | 5.71 px |
| **weak-perspective (affine), no depth term** | **8** | **3.81 px** |

The affine model has FEWER parameters than the free-PP pinhole and fits
BETTER. The 2D track does not merely fail to identify `f` — it does not need
`f`, or any perspective term, to be explained at all. Combined with L1's
vessel-width lean and C11/C13's finding that different cameras nonetheless
imply real, non-saturating 3D disagreement (4.2-10+ mm), the full picture is:
2D residual cannot select a camera model; the (weak, corroborating) size cue
leans toward the large-`f` family used throughout the pipeline; and the choice
still matters for absolute 3D position. All three conclusions point the same
way as the existing #1 recommendation below — nothing here replaces a real
calibration shot, but L1's vessel-width result is a reason for somewhat more
confidence in `c8_11` as the working default until one exists.

## M — out-of-sample validation, baselines, and hyperparameter robustness

Nothing before this checked whether Stage I's numbers hold up on data the
pipeline never fit, how much the motion model buys over simpler methods, or
whether the HMM's tuning is fragile to this one video.

**M1 — was the camera fit ever checked on held-out data?** Yes, already,
implicitly: Stage C's ICP registration (C1-C10) uses ONLY the outbound leg's
2D shape (`track2d_trend.trend[:k_turn+1]`); the return leg never enters
calibration. Splitting Stage I/I2's own reprojection error by leg
(`src/m1_holdout.py`) makes that split explicit for the first time:

| | outbound (calibration set) | return (held out) | ratio |
|---|---|---|---|
| causal, reproj med / >30px% | 4.4 px / 11.2% | 12.8 px / 25.7% | 2.9x / 2.3x |
| Viterbi, reproj med / >30px% | 4.7 px / 9.2% | 11.1 px / 22.9% | 2.4x / 2.5x |

A real, honest generalisation gap, not previously reported. A second check —
refit the same 11-param camera on only HALF the outbound leg — asks whether
this is overfitting to a short calibration arc: the half-leg fit reaches a
LOW cost on its own subset (comparable to the full fit) but then fails
catastrophically even on the immediately adjacent second half of the SAME leg
(270 px median, vs. the full fit's 6.4 px on the entire held-out return leg).
More calibration data clearly stabilises the fit (consistent with C9's own
bootstrap finding), so the *degree* of degeneracy is data-quantity dependent —
but the fact that even the FULL-outbound fit still shows a real 2.9x gap on
return means some of it is a genuine leg-specific effect too (plausibly K4's
appearance/roll difference between outbound and return), not pure
underfitting.

**M2 — what does the HMM's motion model actually buy?** Two baselines against
the same camera/centerline/observations as Stage I (`src/m2_baselines.py`):
(a) argmax-per-frame — independently nearest curve point each frame, no motion
prior at all; (b) monotone dynamic-time-warping — hard non-decreasing arc
length within each leg, no probabilistic velocity model.

| method | reproj med px | p90 | >30px% | out-back med mm |
|---|---|---|---|---|
| argmax/frame (no model) | 4.3 | 27.0 | 9.3% | 0.25 |
| monotone DTW | 4.7 | 27.2 | 9.3% | 0.25 |
| HMM causal (Stage I) | 5.6 | 48.5 | 16.2% | 2.53 |
| HMM Viterbi (Stage I2) | 5.6 | 35.1 | 14.0% | 1.00 |

Counter-intuitive at first: the memoryless baselines beat the HMM on both
metrics used throughout this project to validate it. The reason is that both
metrics have a blind spot for a memoryless method: reprojection error is
trivially small when nothing stops the estimate from jumping to whichever
curve point is nearest *this frame*, and out-and-back repeatability is
close to guaranteed for a time-independent pixel-to-arc-length lookup (the
same pixel always maps to the same answer, correct branch or not — it isn't
really testing branch selection for a stateless method). The tell is implied
speed: argmax/frame's frame-to-frame `|ds/dt|` exceeds the physical 30 mm/s
bound on 5.0% of frames (max 600 mm/s — it is teleporting between
self-intersection branches, not tracking). More surprising: **the causal HMM
(Stage I)'s own posterior-MEAN point estimate does this too, and worse** (6.2%
of frames, max 1442 mm/s) — reporting the mean of an occasionally-bimodal
posterior can jump between modes faster than a stateless nearest-point lookup
ever does. **Viterbi is the only estimator here that is exactly
speed-bounded by construction** (0.0% of frames exceed v_max, since it follows
a single path through the discrete velocity grid) — a genuine advantage over
posterior-mean tracking that I4's mean-vs-mode comparison (which only looked
at accuracy, not physical plausibility) did not surface. Practical reading:
**the out-and-back repeatability check used throughout this project cannot,
by itself, distinguish real tracking from a teleporting lookup — it needs to
be read alongside a speed-plausibility check**, and any report of the causal
filter's trajectory should note that its point estimate is not
velocity-consistent even though the underlying model is.

**M3 — is Stage I's tuning fragile?** One parameter at a time around the
defaults (`sigma_px=9, v_max=30, n_v=41, accel_sigma=20, p_lost=0.05`),
`src/m3_hyperparam_sweep.py`. `sigma_px`, `accel_sigma`, `p_lost`, and `n_v`
all move the median reprojection by well under 15% across a wide sweep —
Stage I's headline numbers are not fragile to those choices. `v_max` is the
exception, and not a reassuring one: dropping it to 15 mm/s roughly doubles
the failure rate (30.7% vs 16.2%), while RAISING it to 60 mm/s (double the
current cap) *improves* both median error (-21%) and failure rate (9.7%) —
i.e. the current `v_max=30` may be actively too conservative for this video,
not just an arbitrary-but-harmless choice. Worth revisiting as a free
improvement.

## N — dose and frame-rate ablation

Directly answers the "What changes for real fluoroscopy" section below with
numbers instead of a qualitative note: clinical fluoroscopy trades pulse rate
and per-pulse exposure (dose) for radiation exposure. `src/n1_dose_framerate.py`
subsamples the real 30 fps track in time and injects extra Gaussian detection
noise, re-running the causal HMM on each:

| target fps | reproj med px | >30px% | | extra noise (px) | eff. sigma | reproj med px | >30px% |
|---|---|---|---|---|---|---|---|
| 30 (baseline) | 5.6 | 16.2% | | 0 (baseline) | 9.0 | 5.6 | 16.2% |
| 15 | 5.5 | 16.0% | | 5 | 10.3 | 8.6 | 16.5% |
| 10 | 5.4 | 16.4% | | 10 | 13.5 | 14.5 | 19.5% |
| 6 | 5.2 | 16.1% | | 20 | 21.9 | 27.0 | 42.1% |
| 3 | 5.1 | 15.5% | | 45 | 45.9 | 57.0 | 81.3% |
| 2 | 5.2 | 17.4% | | | | | |

**Frame rate barely matters — median reprojection is flat from 30 fps down to
2 fps**, because the motion model simply widens its transition kernel to match
the larger inter-frame gap. **Detection noise (dose) is what actually costs
accuracy**, degrading smoothly and then sharply past ~sigma=15-20 px. The
practical implication for a real fluoroscopy deployment is the opposite of the
naive assumption: cutting pulse rate aggressively is nearly free for this
estimator, so a dose budget is much better spent keeping per-pulse image
quality (and hence detection SNR) high than on acquiring more frames per
second.

## O — are Viterbi's remaining failures actually branch-ambiguity errors?

The README's own framing has assumed the dominant failure mode is projected
self-intersection (two arc lengths landing on the same pixel). `src/o1_branch_ambiguity.py`
tests that assumption directly on Viterbi's 143 remaining failures (rp>30px),
instead of continuing to assert it.

**(1) Geometric multiplicity does not predict failure.** Labelling every
centerline sample by how many OTHER arc lengths project within 15 px of it (a
property of the geometry alone, no observations involved): mean local
multiplicity on Viterbi's *good* frames is 1.32, on its *bad* frames 1.00 —
if anything, failures avoid the geometrically ambiguous zones rather than
concentrating in them.

**(2) Was a closer point even available?** Comparing each failing frame's
Viterbi answer to M2's unconstrained argmax/frame answer: only **32% (46/143)**
of Viterbi's failures have a much closer curve point sitting a further 14 mm
away in arc length — genuine branch-selection failures, the kind a second
simultaneous view (Stage J) is built to fix. **The other 68% have no good
match anywhere on the curve for that observation, in any branch** — that is a
detection/registration failure (Stage C's residual), not an ambiguity the
motion model or a second view could resolve. The Viterbi posterior confidence
tells the two classes apart too (median 0.224 on the branch-selection group vs
0.041 on the no-good-match group).

This revises the priority argument for Stage J's second view: it targets
roughly a third of the real remaining failures, not the majority. It does not
change item #1 below — if anything it reinforces it, since two-thirds of what
is left is exactly the registration/calibration error a real intrinsic
calibration would shrink, and the self-intersection failures a second view
targets were already the smaller share even before this test.

## P — chasing the localised residual, and a camera swap that closes it

O1 attributed 68% of Viterbi's remaining failures to "registration error" without
separating what kind. P1 tested four concrete alternatives and refuted all four;
P0 then found the answer was a fifth possibility none of them covered.

**P1 — the failures are location-bound, not a detour.** The 95 no-match frames
form just 3 contiguous episodes of ~1 s, all 100% inside the vessel mask, while
90.7% of all frames sit within 30 px of the projected curve (median 4.3 px) — so
the camera is globally fine and only specific places fail. Of the no-match
locations that the out-and-back route visits twice, **98% (58/59) are bad on
BOTH passes** against a 0.0% base rate, and the outbound episode (frames
544-570) and a return episode (779-818) have centroids only 19.9 px apart — the
same spot, failing both times. A one-off excursion into a side branch cannot do
that; the error belongs to the location.

*Retracted:* P1's first "loop signature" test asked whether the track leaves and
rejoins the same arc length, measured 17-24 mm deltas, and concluded against a
detour. That test was confounded — at ~18-20 mm/s the robot must move that far
during a 1 s episode regardless — and discriminates nothing. Its conclusion is
withdrawn; TEST B above is what actually settles it.

**P1c/P1d/P1e — three more hypotheses, three refutations.**

| hypothesis | discriminator | result |
|---|---|---|
| centerline undersampled there | predicted spline corner-cut vs observed | 0.18-0.35 mm vs **6.22 mm**; and the bad spots are sampled *denser* than average (max gap 2.2-3.0 mm vs the path's 8.47 mm). Halving the CSV density moves the curve only 0.38 mm median / 0.68 mm max |
| radial lens distortion | residual vs image radius | corr = **-0.02**; the outermost radius band has the *lowest* residual (1.9 px, 0% failures). Refit with the principal point pinned so k1/k2 must work alone: **18.19 px**, far worse than free-PP's 4.92 px. Free PP + distortion is 4.92 px — distortion adds exactly nothing |
| phantom deformed in its mount | knee in residual vs allowed perturbation | none: 9.08 -> 8.48 -> 7.70 -> 6.87 -> 5.19 px for <=2/5/8/15 deg. Improvement is *proportional to the freedom granted*, the signature of overfitting, not of a real displacement |

An unconstrained refit of the bad stretch alone does reach 2.5 px (2.5x better),
but it needs a camera 50.5 deg and 830 px of focal away from the global one —
that is M1's under-constrained-subset effect, not evidence of anything.

**P0/P0b — the answer: the pinhole fit itself was wrong there.** L2 had already
found a depth-free affine camera fits the 2D track better (3.81 px) than any
pinhole (4.92 px) with fewer parameters, but never fed it through the tracker.
Doing so resolves P1 completely. Off-curve distance is a property of the CAMERA
alone, with no estimator involved:

| camera | off-curve median | p90 | >30 px |
|---|---|---|---|
| pinhole f=1298 (`c8_11`, used by every prior stage) | 4.3 px | 27.0 px | 9.3% |
| affine, no depth term (`l2_affine`) | **2.7 px** | **17.9 px** | **2.7%** |

And specifically on the 95 frames P1 could not explain: their distance to the
projected curve falls from **median 40.9 px under pinhole to 3.2 px under
affine**, with only 28% still unmatched. So the localised 6 mm residual was
camera-model mis-specification — and pointedly **not** focal length, since the
affine model has none. The 11-parameter fit's 251 px principal-point offset was
locally warping the projection.

**P0 — every camera under one composite metric.** Following M2's finding that
reprojection and out-and-back repeatability each have a blind spot, no row is
judged on one number:

| camera / decode / v_max | reproj | p90 | fail% | v-viol% | max mm/s | out-back | HELD-OUT | held fail% |
|---|---|---|---|---|---|---|---|---|
| pinhole, causal, 30 (the old default) | 5.6 | 48.5 | 16.2 | 6.2 | 1442 | 2.53 | 12.8 | 25.7 |
| pinhole, viterbi, 60 | 4.7 | 27.3 | 9.4 | **0.0** | 60 | 0.25 | 7.5 | 19.8 |
| affine, causal, 60 | 3.2 | 18.8 | 3.6 | 1.3 | 583 | 0.16 | 6.9 | 8.5 |
| **affine, viterbi, 60** | **3.5** | **18.3** | **3.9** | **0.0** | 60 | 0.25 | **7.0** | **9.3** |

Three confirmations. **M3's `v_max` finding is real and bigger than reported** —
30 to 60 mm/s cuts the held-out failure rate 25.7% to 19.5% causal, and also
drops the causal speed-violation rate 6.2% to 2.1%, because some violations were
the too-tight cap forcing the filter to jump. **M2's Viterbi result holds under
every camera** — 0.0% speed violations by construction, always. And **the affine
camera more than halves the held-out failure rate**, 19.8% to 9.3%.

Recommended working configuration: **affine camera + Viterbi + `v_max`=60**.
It gives up 0.3 px and 0.3 points of failure rate against the causal variant and
buys exact speed consistency, which is the M2 principle applied.

**What this does NOT fix.** Affine removes the *parameter* ambiguity (there is no
`f` to choose) but not the *systematic*: the affine and pinhole 3D trajectories
still differ by a **median of 5.00 mm (p90 10.24 mm)** — the same magnitude as
C11's 4.2 mm. L1's 6.0-sigma vessel-width slope says real perspective exists in
this scene, so the affine model, despite fitting and tracking better, is
physically incomplete and its absolute 3D output is biased by an unknown amount.
The checkerboard remains priority #1, now for a cleaner reason: not to fix the
tracking failures (affine already does) but to resolve which member of a
four-model family — `f`=516, 1298, 1826, or affine — the absolute 3D trajectory
should be read from.

*Caveat:* `c10_ba.npz` does not store a parameter vector in the shape this
comparison expects, so the `f`=516 model is absent from the P0 table. That is a
gap in the sweep, not a result.

## Q — what the camera swap does to everything computed under the old one

Stages I, I2, I3, M1, M2 and O1 were all computed with the pinhole `c8_11`
camera, causal filter and `v_max`=30. P0 replaced all three. Re-deriving those
stages is not bookkeeping: three of their conclusions do not survive.

**Q1 — the headline numbers, old vs new.**

| | old (pinhole, causal, v30) | new (affine, v60) |
|---|---|---|
| causal reproj med / p90 / >30px | 5.6 / 48.5 / 16.2% | **3.2 / 18.8 / 3.6%** |
| Viterbi reproj med / p90 / >30px | 5.6 / 35.1 / 14.0% | **3.5 / 18.3 / 3.9%** |
| held-out return leg, reproj / >30px | 11.1 px / 22.9% | **7.0 px / 9.3%** |

**Retraction 1 — I2's offline-smoothing win is gone.** Under the old camera,
Viterbi cut the median error on the causal filter's worst quarter 2.4x
(17.5 -> 7.3 px), and that was banked as "one real, zero-data win". Under the
new camera the same 256 frames score **1.3 px causal vs 1.5 px Viterbi** —
offline smoothing is now marginally *worse* there. The 2.4x was an artefact of
the mis-specified camera, not a property of the estimator. Note also that the
worst-quarter median (1.3 px) is now *below* the overall median (3.2 px): the
causal filter's `s_std` no longer selects the bad frames at all.

**Retraction 2 — I3's abstention rule no longer works.** Smoothed confidence
still separates (0.061 on bad frames vs 0.123 on good), but at the recommended
`p<=0.02` cutoff recall falls from 0.27 to **0.05** and precision from 0.47 to
**0.06**. With only 40 failures left in 1025 frames there is very little for a
global threshold to catch. The rule should be dropped, not re-tuned.

**Retraction 3 — O1's failure-mode finding reverses, and this moves a
priority.** O1 measured local self-intersection multiplicity at 1.32 on good
frames and 1.00 on bad ones, and concluded failures *avoid* geometrically
ambiguous zones — which is what demoted the second view to item #3. Re-derived
under the new camera: **good frames 1.22, bad frames 3.80.** Once the camera is
no longer mis-projecting, the failures that remain are concentrated in
self-intersection zones by a factor of 3.1x — exactly what a second
simultaneous view is built to resolve. The branch/no-match split itself barely
moves (30%/70% vs 32%/68%), so "no good match anywhere" still describes most of
them; but those now sit in congested geometry rather than scattered anywhere.
*Caveat: 40 failures, split 12/28. These are small samples and the confidence
figures attached to them (branch 0.035, no-match 0.090) also flip sign versus
O1's, which is itself a reason not to lean hard on either version.*

**Unchanged — M2 holds under the new camera.** The memoryless baseline still
beats the HMM on reprojection (argmax 2.7 px vs Viterbi 3.5 px) and still
teleports (1.1% of frames above the speed cap, max 652 mm/s) while Viterbi is
exactly speed-bounded (0.0%, max 60). The gap narrowed a lot — argmax's
violation rate fell from 5.0% to 1.1%, because with a well-specified camera the
nearest curve point is usually the right one — but the ranking and the reason
for preferring Viterbi are the same.

**M1's generalisation gap widened in ratio while improving in absolute terms**:
held-out reprojection 12.8 -> 6.9 px and held-out failure rate 25.7% -> 8.5%,
but the in-sample side improved more (1.8 px, 1.0%), so the ratio went from
2.9x to 3.7x (and 2.3x to 8.1x on failure rate). The absolute held-out number
is what to quote; the widened ratio is a real overfitting signal, since the
affine camera was fitted on the outbound leg alone.

**Q2 — the full camera family, and the number to quote.** With C10's `f`=516
included (it stores `rv`/`t`/`f`/`cx`/`cy` separately, which is why P0's table
missed it), all three mutually incompatible cameras score as:

| camera | off-curve med | reproj | fail% | held-out | held fail% |
|---|---|---|---|---|---|
| pinhole `f`=1298 (`c8_11`) | 4.3 px | 4.7 | 9.4 | 7.5 px | 19.8 |
| pinhole `f`=516 (`c10_ba`) | 3.5 px | 3.8 | 3.8 | **6.2 px** | **7.9** |
| affine, no depth (`l2`) | **2.7 px** | **3.5** | 3.9 | 7.0 px | 9.3 |

**`f`=1298 — the model every stage before P0 used — is the outlier**, worst on
every column. And the two better models agree with each other to a median of
**0.75 mm**, while both disagree with `f`=1298 by 4.25-5.00 mm:

| pair | median | p90 | max |
|---|---|---|---|
| `f`=1298 vs `f`=516 | 4.25 mm | 7.85 | 17.36 |
| `f`=1298 vs affine | 5.00 mm | 10.24 | 19.35 |
| `f`=516 vs affine | **0.75 mm** | 4.75 | 8.95 |

That creates a conflict worth stating rather than resolving by preference:
**L1's vessel-width cue rejected `f`=516 at 9.8 sigma and favoured the large-`f`
family, but `f`=516 tracks better than `f`=1298 on every metric and agrees with
the affine model to 0.75 mm.** Size-cue evidence and tracking evidence point
opposite ways. A calibration shot settles it; nothing in this data does.

> **This conflict was not real (Stage R1).** L1's 9.8 sigma came from scoring
> every candidate against `c8_11`'s depth profile; scored against their own,
> both pinholes are self-consistent and the cue does not separate them at all.
> There is nothing here for tracking evidence to contradict. What the corrected
> cue *does* reject is the affine model — see Stage R.

Per-frame spread across the whole family (max deviation from their mean):
**median 3.25 mm, p90 5.57 mm, max 12.21 mm.** Excluding `f`=516 on L1's
evidence gives 2.50 / 5.12 / 9.68 mm, but given the conflict above that
exclusion is not safe, so the full-family figure is the honest one.

> **SUPERSEDED BY STAGE R.** The "conflict" above was an artefact of L1's
> method (see the retraction banner on Stage L), and the exclusion that turns
> out to be justified is the opposite one: two independent non-circular tests
> (R1, R2a) both point away from `f`=1298, and dropping it collapses the family
> spread from 3.25 mm to **0.37 mm** median. The number to quote is in Stage R
> below, not here.

## R — two tests that do not grade a camera on its own exam

Every column in Q2 — off-curve distance, reprojection, held-out reprojection —
measures the distance from an observation to *that camera's own projected
curve*. A camera is graded on the exam it wrote, so those columns are three
views of one piece of evidence, not three independent ones (the same blind spot
M2 identified for reprojection). Stage R adds two discriminators that are not
of that form, and they resolve the family.

**R1 — the size cue, redone per camera.** A fixed-diameter tube at depth `Z`
subtends an image width proportional to `1/Z`, so every camera makes a
*testable prediction* about how vessel width should modulate along the path:
a pinhole predicts slope `-1/Z_med` against its own depth profile, and the
affine model, having no depth term at all, predicts exactly zero. Each camera
is scored against its own geometry — its own projected curve for the width
sampling, its own depth profile for the regression (`src/r1_size_cue_percamera.py`):

| camera | valid samples | measured slope | its own prediction | off by | verdict |
|---|---|---|---|---|---|
| pinhole `f`=1298 | 572 | -0.00485 +- 0.00081 | -0.00506 | 0.3 sigma | self-consistent |
| pinhole `f`=516 | 594 | -0.01531 +- 0.00216 | -0.01123 | 1.9 sigma | self-consistent |
| **affine, no depth** | 624 | **+0.00246 +- 0.00064** | **0** | **3.8 sigma** | **inconsistent** |

Two findings. **The cue cannot separate the two pinholes** — both land on their
own predictions, which is what kills L1's 9.8-sigma claim. **What it does
reject is the affine model**, at 3.8 sigma: a real, non-zero width modulation
is measured along the affine model's own projected curve, and the affine model
predicts none. So **real perspective exists in this scene** and the affine
camera, despite fitting and tracking best, is physically incomplete — exactly
the caveat Stage P raised, now with a number. Note the measured slopes track
the predictions *across* models (predictions differ 2.2x, measurements differ
3.2x, same direction), which a pure taper artefact would not do; the taper
confound is shared by all rows and so cancels in the ranking, though it still
prevents reading any single row as an absolute measurement of perspective.

**R2a — out-and-back arc-length agreement, per camera.** The robot visits most
locations twice. Return frames are matched to outbound frames by proximity of
the **raw detected 2D track** — an observation, with no camera in the loop —
and each camera is then asked whether it assigns the same arc length to both
visits (`src/r2a_outback_percamera.py`). A camera can place its curve beautifully
close to the observations and still label one physical spot two different ways:

| camera | p90 (match <3 px) | >2 mm | >5 mm | max |
|---|---|---|---|---|
| pinhole `f`=1298 | 1.20 mm | 6.3% | **4.9%** | 6.25 mm |
| pinhole `f`=516 | 0.75 mm | 0.0% | **0.0%** | 1.50 mm |
| affine, no depth | 0.50 mm | 1.4% | **0.0%** | 3.25 mm |

All three medians sit at 0.25 mm — the arc-length grid step — so the bulk of
pairs agree exactly and the discriminating signal is entirely in the tail.
**`f`=1298 is the only camera that ever disagrees with itself by more than
5 mm** (4.9% of tight-matched pairs, worst case 6.25 mm). That is a defect no
amount of good curve-fitting excuses, and it is measured on an axis Q2 never
touched.

*Limits, stated plainly.* R2a can rule a camera out but cannot confirm one: a
projection warped *consistently* returns the same wrong label both times and
passes. (Warping does leak in weakly through the motion prior, since the two
passes cross the same geometry in opposite directions at different speeds.) The
`f`=1298 tail rests on 7 pairs out of 143 — suggestive, not decisive on its
own; its weight comes from corroborating the five Q2 columns that already
ranked `f`=1298 last. And R2a's attempted stratification by self-intersection
was **invalid and is not reported above**: the proxy used ("how many outbound
frames lie within 8 px") mostly measures where the robot moved slowly, since
consecutive frames are only ~4.5 px apart, so it does not identify
self-intersections at all. That sub-test needs a real geometric multiplicity
measure, not this one.

**What the two tests do together.** They are orthogonal, and each eliminates a
different model:

| | R1 (physical, size cue) | R2a (self-consistency) | Q2's five columns (circular) |
|---|---|---|---|
| pinhole `f`=1298 | best (0.3 sigma) | **fails** — only model >5 mm | **worst on every column** |
| pinhole `f`=516 | passes (1.9 sigma) | cleanest tail | best held-out |
| affine | **fails (3.8 sigma)** | clean | best own-curve residual |

`f`=516 is the only model that fails nothing. The honest reading of `f`=1298 is
"opposed by six measurements, mildly favoured by one weak one" — R1 does prefer
it, but R1's separation between the two pinholes (0.3 vs 1.9 sigma, both
passing, under a shared taper confound) is far weaker than R2a's tail
separation, and every other axis is against it.

**The number to quote (replaces Q2's).** Dropping `f`=1298 from the family:

| family | median | p90 | max |
|---|---|---|---|
| all three models (Q2's figure) | 3.25 mm | 5.57 mm | 12.21 mm |
| **excluding `f`=1298 (R1 + R2a)** | **0.37 mm** | **2.37 mm** | **4.48 mm** |

> **Revised headline: absolute 3D position from this video is uncertain by
> about +-0.4 mm (median) / +-2.4 mm (p90) from the unresolved camera choice —
> not +-3.3 mm.** Nearly the whole of Q2's systematic was one model that two
> independent tests now reject. This does not make the calibration shot
> unnecessary — it is what would let the remaining pair be checked rather than
> argued — but it moves the camera ambiguity from "the dominant error term in
> the project" to roughly the level of the estimator's own noise floor.

## S — the per-frame uncertainty, with the camera term included

Every sigma reported before this point (Stage I's posterior std, I2's smoothed
posterior) is conditional on the camera being correct: it covers detection
noise and arc-length ambiguity only. Stage R left two models that no test
rejects, so the systematic between them is measurable and belongs in the error
bar (`src/s1_uncertainty.py`, running the full forward-backward smoother under
each surviving camera):

| component | median | p90 | max |
|---|---|---|---|
| `sigma_stat` within-model, random | 0.78 mm | 0.95 mm | 3.27 mm |
| `delta_cam` between-model, systematic | 0.75 mm (+-0.37) | 4.75 mm (+-2.37) | 8.95 mm |
| **combined** | **0.98 mm** | **2.71 mm** | **4.60 mm** |

They are reported separately rather than summed because they behave
differently: `sigma_stat` shrinks with better detection or more frames,
`delta_cam` shrinks only with a calibration shot.

**The camera systematic exceeds the statistical term on only 30% of frames.**
At the median this is noise-dominated — detection noise (0.78 mm) is roughly
twice the camera systematic (0.37 mm) — and the systematic takes over only in
the tail (+-2.37 vs +-0.95 mm at p90). That sharpens rather than weakens the
calibration argument: **a checkerboard buys worst-case guarantees, not typical
accuracy.** For a device that must not be reported in the wrong vessel, the
tail is the number that matters, but it should be quoted as such.

> **Deliverable statement.** Reported 3D position is accurate to about
> **+-0.98 mm (median), +-2.71 mm (p90), 4.60 mm worst case**, of which
> +-0.37 mm (median) is an irreducible camera-calibration systematic and
> +-0.78 mm is estimator noise that better detection would shrink.

`src/s2_summary_figs.py` and `src/s3_animation.py` build the presentation
assets from these artefacts. Both write into `out/figs/`, which is not
committed - the animation renders frames of the source video.

## Honest status

The estimator was never the limiting factor, and after Stages P-R it is not
the largest uncertainty either. Current working configuration is **affine
camera + Viterbi + `v_max`=60**: reprojection median 3.5 px, failure rate
3.9%, held-out (return-leg) failure rate 9.3%, and 0.0% speed violations by
construction.

**On the camera ambiguity, which was the headline problem for most of this
project.** Three mutually incompatible cameras fit the same 2D track, and no
2D residual can choose between them (C7-C13, L2). Stage R broke that with two
discriminators that do not grade a camera on its own projection: the
per-camera size cue (R1) rejects the affine model at 3.8 sigma, showing real
perspective exists in the scene; out-and-back arc-length agreement (R2a)
rejects `f`=1298, the only model that ever labels one physical location two
ways by more than 5 mm. With `f`=1298 dropped, the family's 3D spread falls
from **3.25 mm to 0.37 mm median (p90 5.57 -> 2.37 mm)**. The residual
camera ambiguity is now comparable to the estimator's own noise floor rather
than 9x larger than it.

That reframes the checkerboard from "the thing setting how wrong the answer
could be" to "the thing that would let the last two candidates be checked
instead of argued." Still the top ask, but for a smaller stake, and the
project's central claim no longer depends on getting it.

Stage J changes the ordering of what to ask for. A second *simultaneous* view
removes catastrophic tracking failures outright and needs only ~30 degrees of
separation; a rotating single detector does not help at all. But biplane only
improves calibration by ~3x, so it does not substitute for knowing the
intrinsics.

**The zero-data wins that survived.** Two of the three banked earlier did not
(see Stage Q): I2's 2.4x offline-smoothing gain and I3's abstention rule were
both artefacts of the mis-specified pinhole camera and are withdrawn. What
holds up: **report the Viterbi path, not the causal posterior mean** — the
mean can imply speeds up to 1442 mm/s at bimodal moments, worse than a
memoryless nearest-point lookup, while Viterbi is exactly speed-bounded by
construction (M2, confirmed under every camera in P0); **raise `v_max` from
30 to 60 mm/s**, which cuts the held-out failure rate 25.7% -> 19.5% and the
speed-violation rate 6.2% -> 2.1% (M3, P0); and **switch the camera from
pinhole to affine for the tracking front end**, which more than halves the
held-out failure rate, 19.8% -> 9.3% (P0). None of these needed new data.

To unblock, in order of value:

1. **A checkerboard shot through the same optics.** Still first, but the case
   changed in Stage R. It is no longer needed to bound a 3-4 mm systematic —
   R1 and R2a already reduced that to 0.37 mm median by eliminating `f`=1298.
   It is needed because the two surviving models are physically incompatible
   in *kind*: R1 shows real perspective exists (rejecting the affine model at
   3.8 sigma), yet the affine model is what tracks best, so the pipeline is
   currently using a projection it has evidence against, and only an external
   intrinsic can say what the physically-correct model that also tracks well
   actually looks like. C10 confirmed a better solver cannot substitute.
2. **The full 3D vessel geometry (STL or all-branch centerlines).** Lets the
   whole vessel tree constrain the camera instead of a single curve, which is
   what published 2D/3D roadmapping methods actually do.
3. **A second simultaneous view**, if the hardware allows it. ~30 degrees of
   separation is enough; orthogonal biplane adds little beyond that. Stage Q
   *raised* this item's expected payoff back up: O1's original finding (that
   failures avoid geometrically ambiguous zones, which had demoted it) reverses
   under the corrected camera — remaining failures now concentrate in
   self-intersection zones by 3.1x. Caveat: that reversal rests on 40 failures
   split 12/28, and the attempted re-check of it in R2a used an invalid
   self-intersection proxy, so the evidence here is thinner than the other
   items and worth re-establishing on a proper multiplicity measure.
4. **A direct view of the robot body (deprioritised).** Vessel-diameter data
   was previously listed here to turn the D3 wobble into a modelled radial
   offset. K2-K4 found no evidence for that offset on real data — the wobble
   looks like a rolling-silhouette artefact instead (K4). K5 tried to dodge it
   on the detection side alone (track the blob's axis-midpoint instead of its
   centroid) at zero data cost, and it made no difference — the outline's
   extent moves, not just the mass within it, so any real fix needs a view
   that resolves the robot's cross-section, not vessel diameters and not a
   cleverer 2D feature.

Also worth carrying forward, independent of new hardware: **M1's out-of-sample
gap (calibration-set 4.4-4.7 px vs held-out return-leg 11-13 px, a real 2.4-2.9x)
is the number to quote for expected accuracy on unseen data**, not the
calibration-set residual C reports — a distinction this project did not
previously draw.

## What changes for real fluoroscopy

The geometry is unchanged; the front end is not. Background subtraction works
here because the camera is static and the scene is rigid. Under fluoroscopy the
C-arm moves, the anatomy breathes and pulses, and contrast washes in and out, so
stage D becomes a learned segmentation (U-Net class) and stage C becomes
per-frame 2D/3D registration with motion compensation. Stages A/E/F carry over
unchanged — and the C-arm angle becomes a variable you can *choose*, which is
what the observability analysis is for.

**N puts numbers on the dose/rate tradeoff this implies.** Subsampling the
real track shows median reprojection is flat from 30 fps down to 2 fps (the
motion model absorbs the larger gaps for free) while injected detection noise
degrades accuracy smoothly and then sharply past ~15-20 px effective sigma
(failure rate 16.2% -> 42.1% -> 81.3% at sigma 9 -> 22 -> 46 px). For a fixed
radiation budget, this argues for spending it on per-pulse image quality
(detection SNR) over pulse rate — the opposite of the naive "more frames is
better" assumption, at least for a tracker built on a strong motion prior like
this one.

## Compute

No GPU required. The whole pipeline is CPU-only: detection ~2 ms/frame, the
filter 0.62 ms/frame. The heaviest step is the global view search (21600
hypotheses, ~90 s single-threaded) and it runs once.

A GPU only becomes necessary when stage D is replaced by a learned segmenter
for real fluoroscopy — training a U-Net on annotated sequences. Inference for a
small segmentation net at 30 fps is also feasible on CPU.

## Layout

```
src/       pipeline stages, prefixed by stage letter
data/      intermediate .npz artefacts   (not in the repo - regenerated)
out/       background plate, vessel mask, figures (not in the repo - regenerated)
out/figs/  d_track2d, d2_clean, d3_wobble, c3_overlay, c5_diagnose,
           i_final, j_multiview, k_offaxis, k4_appearance, k5_axis_track,
           c11_f_sensitivity, c12_residual_diagnosis, p1_excursion
```

`data/` and `out/` are derived entirely from the two input files and are not
committed - the figures render frames of the source video, so publishing them
would redistribute the input. Supply `Video.mp4` and `Path 2.csv` at the
repository root and run the stages in the order listed at the top to rebuild
everything.

Note: `src/config.py` provides `imread_u`/`imwrite_u` — OpenCV's own `imread`/
`imwrite` silently fail on the non-ASCII path this project lives under.
