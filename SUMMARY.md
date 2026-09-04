# Real-time 3D tracking of a millirobot from a single 2D view

A short read. The stage-by-stage evidence is in [`report.md`](report.md); this
is the argument.

---

## 1. The idea

A pixel back-projects to a ray, so a 2D image cannot fix a 3D position. But the
vessel geometry is known in advance, and the robot is inside the vessel — so
its 3D state is not three numbers, it is **one**: arc length `s` along the
centerline.

That turns an ill-posed reconstruction into a 1-D estimation problem:

> given pixel `u_k`, find `s_k` such that `project(C(s_k)) ≈ u_k`

Where the projected centerline crosses itself, several `s` explain the same
pixel. A motion model over `(s, ṡ)` picks the reachable one. An HMM over that
state, decoded with Viterbi, is the whole method.

## 2. Does it work

| | |
|---|---|
| tracking failure rate | **3.9%** of 1025 frames |
| physically impossible speeds | **0.0%** (Viterbi is speed-bounded by construction) |
| speed | **0.62 ms/frame — 53× real time** at 30 fps, CPU only, no GPU |
| **3D accuracy** | **±0.98 mm median, ±2.71 mm p90, 4.60 mm worst case** |

The accuracy figure has two parts that behave differently
([`SUMMARY_A_accuracy.png`](out/figs/SUMMARY_A_accuracy.png)):

- **±0.78 mm estimator noise** — random, shrinks with better detection
- **±0.37 mm camera systematic** — irreducible without a calibration shot

At the median this is *noise*-dominated; the systematic only takes over in the
tail (±2.37 vs ±0.95 mm at p90). So calibration buys **worst-case guarantees**,
not typical accuracy — which is the honest way to put it to a clinician.

## 3. Six things that were not obvious at the start

**The vessel constraint inverts the difficulty.** Depth is the unobservable
direction in raw back-projection. Once the robot is pinned to the centerline,
depth error comes out **3–4× smaller than in-plane error** (0.12 vs 0.43 mm).
The hardest quantity becomes the best-determined one.

**The obvious metrics have a blind spot.** A memoryless nearest-point lookup
*beats* the full model on both reprojection error and out-and-back
repeatability — by teleporting between self-intersection branches (implied
speeds to 652 mm/s against a 30 mm/s robot). Neither metric penalises a
physically impossible path. Any trajectory has to be judged on reprojection
**and** speed-consistency **and** held-out generalisation together.

**The camera is not identifiable from one centerline in one view.** Three
mutually incompatible cameras fit the same 2D track: focal length ranges over
50× with under 15% change in cost, and a depth-free affine model with *fewer*
parameters fits *better* than any pinhole. Their 3D trajectories disagreed by
up to 5 mm. Resolving this needed two tests that don't grade a camera on its
own exam — a physical size cue and out-and-back self-consistency — which
eliminated one model each and cut the ambiguity from ±3.3 mm to ±0.37 mm
([`SUMMARY_B_evidence.png`](out/figs/SUMMARY_B_evidence.png), panels 1–3).

**Simultaneity matters, angular coverage does not.** A second *simultaneous*
view removes catastrophic failures outright, and 30° of separation buys
essentially everything a 90° biplane does. A single detector *rotating* over
the same angles buys nothing measurable. Two views at one instant constrain one
3D point; two views at different instants constrain two different points,
because the robot moved. This is why rotational angiography reconstructs static
anatomy but cannot replace biplane for tracking a moving device.

**For dose, spend on image quality, not frame rate.** Median accuracy is flat
from 30 fps down to **2 fps** — the motion model absorbs the larger gaps for
free — while detection noise degrades accuracy smoothly and then sharply
(16% → 42% → 81% failure at σ = 9 → 22 → 46 px). The naive assumption is
backwards: cut pulse rate, keep per-pulse SNR.

**One appealing hypothesis was wrong, and testing it properly mattered.** The
detected robot oscillates at 1.17 Hz — a rotating-field signature — and reading
that as "the robot rides 1.2 mm off the vessel axis" seemed obvious enough to
build a model around. Four independent checks killed it: the fitted radius was
4× too small, the model evidence *collapsed* when restricted to cleaner frames
(the opposite of what a real effect does), and the blob's **area** modulates
more strongly than its centroid — which a rigid translation cannot cause. It is
a rolling-silhouette appearance artefact. The error budget was wrong until it
was tested.

## 4. What would move the needle, in order

1. **A checkerboard through the same optics.** One photograph. It collapses the
   remaining camera ambiguity and converts the ±2.4 mm p90 systematic into a
   measured quantity. Nothing in the data can substitute for it — an 8-parameter
   depth-free camera fits the 2D track better than any perspective model, so the
   images cannot even confirm which model *family* is right.
2. **A second simultaneous view, ~30°.** Targets the self-intersection failures
   that remain after the camera is fixed. Not a substitute for (1): biplane
   improves calibration only ~3×, and does not remove it.
3. **The full vessel tree** (STL or all-branch centerlines), which would let the
   whole anatomy constrain the camera rather than a single curve.

## 5. What I would do differently

The camera model was the single largest error source for most of this work, and
I spent a long time refining the estimator on top of a mis-specified one —
several conclusions (an offline-smoothing gain, an abstention rule, a
failure-mode split) turned out to be artefacts of it and had to be retracted
once the camera was fixed. The lesson I would carry into the real fluoroscopy
setting: **validate the projection model against something outside its own
residual before trusting anything downstream of it.**

## 6. Where the detail is

| | |
|---|---|
| [`report.md`](report.md) | full stage-by-stage log, including everything that failed |
| [`README.md`](README.md) | 7-axis evaluation of the current state |
| `src/` | one script per stage, prefixed by stage letter |

Reproduce: put `Video.mp4` and `Path 2.csv` at the repository root and run the
stages in the order listed at the top of `report.md`. CPU only, no GPU.
