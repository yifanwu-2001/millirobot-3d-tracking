# Real-time 3D tracking of a millirobot from a single 2D view

A short read. The stage-by-stage evidence is in [`report.md`](report.md) and
[`IMPROVEMENTS.md`](IMPROVEMENTS.md); this is the argument.

---

## 1. The idea

A pixel back-projects to a ray, so a 2D image cannot fix a 3D position. But the
vessel geometry is known in advance, and the robot is inside the vessel — so
its 3D state is not three free numbers. It is **one**: arc length `s` along the
centerline — plus, where the axis alone is wrong, a small offset bounded by the
lumen.

That turns an ill-posed reconstruction into a 1-D estimation problem:

> given pixel `u_k`, find `s_k` such that `project(C(s_k)) ≈ u_k`

Where the projected centerline crosses itself, several `s` explain the same
pixel. A motion model over `(s, ṡ)` picks the reachable one. An HMM over that
state, decoded with Viterbi, with the observation marginalised over the tube
cross-section, is the whole method.

## 2. Does it work

| | |
|---|---|
| tracking failure rate | **0.9%** of 1025 frames (held-out return leg: 1.1%) |
| physically impossible speeds | **0.0%** (Viterbi is speed-bounded by construction) |
| speed | **6.7 ms/frame offline; 7.3 ms/frame genuinely online at 0.4 s latency** — CPU only, no GPU |
| **3D accuracy on the axis** | **±0.98 mm median, ±2.61 mm p90**, of which ±0.38 mm is the camera |
| where the robot leaves the axis | the in-plane position follows the lumen wall; the along-ray depth is an interval, not a point |

The accuracy figure has two parts that behave differently:

- **±0.78 mm estimator noise** — random, shrinks with better detection
- **±0.38 mm camera systematic** — irreducible without a calibration shot

At the median this is *noise*-dominated; the systematic only takes over in the
tail. So calibration buys **worst-case guarantees**, not typical accuracy —
which is the honest way to put it to a clinician.

## 3. Seven things that were not obvious at the start

**The vessel constraint inverts the difficulty.** Depth is the unobservable
direction in raw back-projection. Once the robot is pinned to the centerline,
depth error comes out **3–4× smaller than in-plane error** (0.12 vs 0.43 mm).
The hardest quantity becomes the best-determined one — *on the axis*.

**The obvious metrics have a blind spot.** A memoryless nearest-point lookup
*beats* the full model on reprojection error and on out-and-back repeatability
— by teleporting between branches at implied speeds up to 592 mm/s. Neither
metric penalises a physically impossible path. Any trajectory has to be judged
on reprojection **and** speed-consistency **and** held-out generalisation
together.

**The camera is not identifiable from one centerline in one view, and the way
to choose one is to stop grading it on its own residual.** Three mutually
incompatible cameras fit the same track: focal length ranges over 50× with
under 15% change in cost, and a depth-free affine model with *fewer*
parameters fits *better* than any pinhole. Their 3D trajectories disagreed by
up to 5 mm. Two tests that do not use a camera's own projection resolved most
of it — a physical size cue (vessel width must fall off as 1/depth, which the
affine model cannot express: rejected at 3.8σ) and bug-immune held-out
predictive evidence (the third camera loses by 266 nats). That cut the
ambiguity from **±3.3 mm to ±0.37 mm**. A first version of this argument used
an out-and-back consistency test instead; its verdict turned out to be an
artefact of a decoder bug and was retracted — see point seven.

**The robot does leave the axis, and no smooth model of how has survived.**
On the return pass through one junction it rides the far wall, a full tube
diameter from the axis. Four offset models — rotating with the field, constant
per leg, locked to curvature, a fixed world direction such as gravity — were
each fitted on one leg and tested on the other, and each failed. Gravity in
particular is untestable here: the camera looks straight down, so a vertical
offset is along the viewing ray and invisible. The fix that works is not a
model of the displacement but a **degree of freedom** for it: the observation
is marginalised over the lumen cross-section, so the decoder can follow the
robot to the wall. Failures drop from 3.8% to 0.9%, and the gain does not
depend on a radius the image cannot measure — a conservative 3 mm floor at the
junction gives the same result as the imputed 5.3 mm that was first used.

**That freedom has a price, and it is the honest headline limit.** Once the
robot may sit anywhere inside the tube, its position *along the viewing ray*
is no longer pinned — two trajectories 6 mm apart can have identical
projections while both obeying the radius and speed bounds. On the axis, depth
was the best-determined coordinate; off the axis, it is an interval. A second
simultaneous view is the only observation that closes it.

**Simultaneity matters, angular coverage does not.** A second *simultaneous*
view removes catastrophic failures outright, and 30° of separation buys
essentially everything a 90° biplane does. A single detector *rotating* over
the same angles buys nothing measurable, because two views at different
instants constrain two different points. This is why rotational angiography
reconstructs static anatomy but cannot replace biplane for tracking a moving
device.

**For dose, spend on image quality, not frame rate — and re-run everything
when the decoder changes.** Median accuracy is flat from 30 fps down to
**2 fps** while detection noise degrades it sharply (3.9% → 30% failures at
σ = 9 → 22 px): cut pulse rate, keep per-pulse SNR. Separately: a backtracking
bug in the Viterbi decoder was found late. Every conclusion built on decoded
paths was re-run; all survived except one — the out-and-back tail that had
been the stated reason for excluding a camera. That exclusion now rests on
evidence that never touched the buggy path. The lesson generalises: a
conclusion that lives in the tail of a decoded statistic needs a second,
decoder-independent witness before it is banked.

## 4. What would move the needle, in order

1. **A checkerboard through the same optics.** One photograph. It separates
   the two surviving cameras (0.37 mm apart) and turns the principal-point
   question into a physical fact. Smallest stake it has ever had; still the
   cheapest single action.
2. **The lumen surface (STL), or tube diameters.** The junction radius that
   bounds the reported offset is a 3 mm floor, not a measurement. A surface
   makes the along-ray interval a bounded one.
3. **A second simultaneous view, ~30°.** Not for branch errors — there are
   none left — but because it is the only thing that resolves depth once the
   robot is off the axis. Biplane improves calibration only ~3×, so it does
   not replace item 1.
4. **The robot's dimensions.** Fixes the offset magnitude and unblocks a
   roll-slip cross-check that is otherwise untestable.

## 5. What I would do differently

The camera model was the single largest error source for most of this work,
and I spent a long time refining the estimator on top of a mis-specified one —
several conclusions (an offline-smoothing gain, an abstention rule, a
failure-mode split) turned out to be artefacts of it and had to be retracted
once the camera was fixed. Then a decoder bug took out one more, and the
correction only held because a second, independent witness for the same
conclusion existed. Two lessons I would carry into the real fluoroscopy
setting: **validate the projection model against something outside its own
residual before trusting anything downstream of it**, and **never let a
headline rest on a single decoded statistic's tail**.

## 6. Where the detail is

| | |
|---|---|
| [`report.md`](report.md) | full stage-by-stage log, including everything that failed and every retraction |
| [`IMPROVEMENTS.md`](IMPROVEMENTS.md) | the V-stage work (tube constraint, online decoder, alignment) and its review |
| [`README.md`](README.md) | the deliverable and a 7-axis evaluation of the current state |
| `src/` | one script per stage, prefixed by stage letter |

Reproduce: put `Video.mp4` and `Path 2.csv` at the repository root and run the
stages in the order listed at the top of `report.md`. CPU only, no GPU.
