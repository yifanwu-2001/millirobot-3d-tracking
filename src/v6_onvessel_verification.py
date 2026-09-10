"""V6 - audit the reported trajectory against the ACTUAL video, frame by frame,
and re-render the results video on real frames.

The reviewer's challenge, twice now: "the robot's trajectory in my video does
not match your animation, and one stretch is outside the vessel." Stage T
answered the second challenge for the whole projected CURVE (f=516: 98.8% on
tube-like structure, worst 5.2 px) - but the s3-style animation still rendered
the BACKGROUND plate, not the video, so the reviewer could never see the robot
and the estimate together, and the fluid-free tube segments were nearly
invisible against the pale background: an honest overlay that still *reads* as
"outside the vessel" is a rendering failure, not just a metric footnote.

This stage does three things:

  A. per-frame audit: the reported 3D position (V4 mixture median under the
     f=516 camera), projected per frame, tested against the filled-lumen mask
     and the any-tube mask - for the frames the robot actually VISITS, not the
     whole curve. Every off-lumen frame is classified: unfilled tube, or truly
     off structure (and by how many px).
  B. verification figure: native-resolution VIDEO crops at the worst stretches,
     contrast-enhanced so the pale fluid-free tubes are visible, with overlays.
  C. the results video re-rendered on REAL VIDEO FRAMES: the robot is visible,
     the reported dot is on it, and every vessel segment - filled or not - is
     tinted, so nothing can read as empty space.

Output video note (same as s3/v5): renders source-video frames, so it is a
local presentation asset and must not be committed.
"""
import os, shutil, tempfile
import numpy as np, cv2, matplotlib
matplotlib.use("Agg")
try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    HAVE_FFMPEG = True
except Exception:
    HAVE_FFMPEG = False
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from scipy.spatial.transform import Rotation as Rot
from config import DATA, OUT, ROOT, W, H, FPS, imread_u, imwrite_u
from a_centerline import load as load_cl
from e_hmm import ArcHMM

V_MAX, SIGMA_PX = 60.0, 9.0
s3c, C3, _, _ = load_cl(); L = s3c[-1]
DS = 0.25
SG = np.arange(0, L + 1e-9, DS)
CG = np.stack([np.interp(SG, s3c, C3[:, c]) for c in range(3)], 1)
fin = np.load(DATA / "i_final.npz"); UV, k_turn = fin["UV"], int(fin["k_turn"])
K = len(UV); dt = 1 / FPS

z10 = np.load(DATA / "c10_ba.npz")
R516 = Rot.from_rotvec(z10["rv"]).as_matrix(); t516 = z10["t"]; f516 = float(z10["f"])
c516 = np.array([float(z10["cx"]), float(z10["cy"])])
def proj516(X):
    Xc = X @ R516.T + t516
    return Xc[:, :2] / Xc[:, 2:3] * f516 + c516

v4 = np.load(DATA / "v4_mixture.npz")
X_mix, med_mix, sd_mix = v4["X_mix"], v4["med"], v4["sd"]
Q_mix = proj516(X_mix)
P_curve = proj516(CG)

# deliverable trajectory: the V8 tube-aware decode when present (the recommended
# configuration since Stage V8), else the V4 mixture median
if (DATA / "v8_tube.npz").exists():
    v8 = np.load(DATA / "v8_tube.npz")
    X_rep, Q_rep, s_rep, rp_rep = v8["X_tube"], v8["Q_tb"], v8["s_tb"], v8["rp_tb"]
    TAG = "pinhole f=516 - Viterbi - v60 - TUBE-aware (V8)"
    print(f"[V6] deliverable = V8 tube-aware decode")
else:
    X_rep, Q_rep, s_rep = X_mix, Q_mix, med_mix
    rp_rep = np.linalg.norm(Q_rep - UV, axis=1)
    TAG = "pinhole f=516 - Viterbi - v60"

# ---- the constraint-violation term the reported sigma was missing ----------
# sigma_mix covers detection noise + camera ambiguity UNDER the assumption the
# robot is on the centerline. V6's frame audit shows that assumption itself
# fails on a small set of frames (the wall-hugging return leg). The honest bar
# widens by the deliverable's own residual above its median floor; with the V8
# tube-aware decode the residual is already small there, so the bar narrows
# back - the widening now only fires on what the tube cannot explain.
_z = (CG @ R516.T + t516)[:, 2]
SC516 = f516 / np.median(_z)                       # px per mm, f=516 camera
r0 = float(np.median(rp_rep)) / SC516
excess = np.maximum(rp_rep / SC516 - r0, 0.0)
sigma_rep = np.sqrt(sd_mix ** 2 + excess ** 2)
viol = excess > 2.0
print(f"[V6] constraint-violation inflation (residual floor r0 = {r0:.2f} mm):")
print(f"     {int(viol.sum())} frames genuinely widened; sigma_rep median {np.median(sigma_rep):.2f} mm"
      f" (within-model {np.median(sd_mix):.2f}), p90 {np.percentile(sigma_rep,90):.2f}, max {sigma_rep.max():.2f} mm")

# ----------------------------------------------------------------- masks
bg_bgr = imread_u(OUT / "background_median.png")
_hsv = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2HSV)
_h, _s, _v = _hsv[..., 0], _hsv[..., 1], _hsv[..., 2]
gray = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY)
lumen = cv2.morphologyEx((((_h < 25) | (_h > 170)) & (_s > 40) & (_v > 40)).astype(np.uint8),
                         cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
flat = cv2.absdiff(gray, cv2.GaussianBlur(gray, (0, 0), 6)) < 4
anytube = (~((_v > 215) & (_s < 28) & flat)).astype(np.uint8)
anytube = cv2.morphologyEx(anytube, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
anytube = cv2.morphologyEx(anytube, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
dist_out = cv2.distanceTransform(1 - anytube, cv2.DIST_L2, 5)
unfilled = (anytube > 0) & ~(lumen > 0)

def audit(P):
    xi = np.clip(np.rint(P[:, 0]).astype(int), 0, W - 1)
    yi = np.clip(np.rint(P[:, 1]).astype(int), 0, H - 1)
    on_lum = lumen[yi, xi] > 0
    on_any = anytube[yi, xi] > 0
    d_off = dist_out[yi, xi]          # px to the nearest tube when off it
    return on_lum, on_any, d_off

# ============================================== PART A: per-frame on-vessel audit
print("=" * 78)
print("PART A - per-frame audit of the REPORTED trajectory (f=516 camera, V4 median)")
print("=" * 78)
on_lum_Q, on_any_Q, d_Q = audit(Q_rep)
on_lum_UV, on_any_UV, d_UV = audit(UV)
n_nan = np.isnan(UV[:, 0]).sum()
print(f"\n[A] the position the system REPORTS, frame by frame ({K} frames):")
print(f"    on a FLUID-FILLED lumen : {100*on_lum_Q.mean():5.1f}%")
print(f"    on ANY tube structure   : {100*on_any_Q.mean():5.1f}%   (unfilled tubes included)")
print(f"    off ALL tube structure  : {100*(~on_any_Q).mean():5.1f}%  "
      f"({int((~on_any_Q).sum())} frames, worst {d_Q.max():.1f} px = {d_Q.max()/6.43:.2f} mm)")
print(f"\n[A] the robot DETECTION itself (what the reviewer sees moving in the video):")
print(f"    on a fluid-filled lumen : {100*np.nanmean(on_lum_UV):5.1f}%   ({int(n_nan)} dropped frames)")
print(f"    on ANY tube structure   : {100*np.nanmean(on_any_UV):5.1f}%")

off_lum = ~on_lum_Q
if off_lum.any():
    # contiguous stretches of off-lumen frames, and whether each is an unfilled tube
    st = np.where(np.diff(np.concatenate([[0], off_lum.view(np.int8), [0]])) != 0)[0]
    print(f"\n[A] the off-filled-lumen stretches (what reads as 'outside the vessel'):")
    print(f"{'frames':>14}{'time (s)':>16}{'s range (mm)':>16}{'on unfilled tube':>18}")
    for a, b in zip(st[::2], st[1::2]):
        smed = med_mix[a:b]
        frac_unf = on_any_Q[a:b].mean()
        print(f"{a:6d}-{b-1:4d}{a/FPS:8.1f}-{(b-1)/FPS:5.1f}{smed.min():9.1f}-{smed.max():6.1f}"
              f"{100*frac_unf:15.0f}%")

# the frames where the CENTERLINE CONSTRAINT itself fails: detection far from
# BOTH surviving cameras' curves - the wall-hugging episode, localised
d516 = np.array([np.min(np.linalg.norm(P_curve - uv, axis=1)) for uv in UV])
pa = np.load(DATA / "l2_affine.npy"); Ra = Rot.from_rotvec(pa[:3]).as_matrix()
Xa = CG @ Ra.T; P_aff = Xa[:, :2] * np.exp(pa[3]) + pa[4:6]
daff = np.array([np.min(np.linalg.norm(P_aff - uv, axis=1)) for uv in UV])
both = (~np.isnan(UV[:, 0])) & (d516 > 30) & (daff > 30)
print(f"\n[A] frames where the detection is >30 px from BOTH cameras' curves")
print(f"    (the centerline constraint fails; camera choice does not help): "
      f"{int(both.sum())} frames")
if both.any():
    bs = np.where(np.diff(np.concatenate([[0], both.view(np.int8), [0]])) != 0)[0]
    for a, b in zip(bs[::2], bs[1::2]):
        print(f"     frames {a}-{b-1}  (t = {a/FPS:.1f}-{(b-1)/FPS:.1f} s, "
              f"s = {med_mix[a:b].min():.0f}-{med_mix[a:b].max():.0f} mm), worst "
              f"{d516[a:b].max():.1f} px = {d516[a:b].max()/SC516:.1f} mm")

# ============================================== PART B: video-frame verification crops
print(f"\n" + "=" * 78)
print("PART B - video-frame verification at the questioned stretches")
print("=" * 78)
tmp = os.path.join(tempfile.gettempdir(), "millirobot_video.mp4")
if not os.path.exists(tmp):
    shutil.copy(ROOT / "Video.mp4", tmp)
cap = cv2.VideoCapture(tmp)
def grab(k):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
    ok, fr = cap.read()
    if not ok: raise IOError(f"cannot read frame {k}")
    return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)

def enhance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(2.0, (8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

# pick the longest off-lumen stretch and the single worst off-tube frame
st = np.where(np.diff(np.concatenate([[0], off_lum.view(np.int8), [0]])) != 0)[0]
spans = sorted(zip(st[::2], st[1::2]), key=lambda ab: ab[1] - ab[0], reverse=True)
crops = []
for a, b in spans[:2]:
    k = (a + b) // 2
    crops.append((f"off-lumen stretch, frame {k} (t={k/FPS:.1f}s)", grab(k), Q_rep[k], UV[k]))
if (~on_any_Q).any():
    kw = int(np.argmax(np.where(~on_any_Q, d_Q, -1)))
    crops.append((f"worst off-tube frame {kw} (t={kw/FPS:.1f}s, {d_Q[kw]:.1f} px)", grab(kw), Q_rep[kw], UV[kw]))
# a return-leg wall-hugging moment: mid region, return leg, dot far from detection
d_det = np.linalg.norm(Q_rep - UV, axis=1)
RET = np.zeros(K, bool); RET[k_turn + 1:] = True
mid = (UV[:, 0] > 250) & (UV[:, 0] < 570) & (UV[:, 1] > 470) & (UV[:, 1] < 660) & RET
if mid.any():
    km = int(np.argmax(np.where(mid, d_det, -1)))
    crops.append((f"return-leg wall offset, frame {km} (t={km/FPS:.1f}s, {d_det[km]:.1f} px)", grab(km), Q_rep[km], UV[km]))

fig, axs = plt.subplots(1, len(crops), figsize=(6.0 * len(crops), 5.6))
if len(crops) == 1: axs = [axs]
for ax, (title, img, q, uv) in zip(axs, crops):
    img_e = enhance(img)
    img_e = (0.78 * img_e + 0.22 * np.array([120, 160, 255]) * (lumen[..., None] > 0)
             + 0.22 * np.array([128, 128, 128]) * (unfilled[..., None])).astype(np.uint8)
    x0 = max(0, int(q[0]) - 160); x1 = min(W, int(q[0]) + 160)
    y0 = max(0, int(q[1]) - 140); y1 = min(H, int(q[1]) + 140)
    ax.imshow(img_e[y0:y1, x0:x1])
    on_curve = (P_curve[:, 0] > x0) & (P_curve[:, 0] < x1) & (P_curve[:, 1] > y0) & (P_curve[:, 1] < y1)
    ax.plot(P_curve[on_curve, 0] - x0, P_curve[on_curve, 1] - y0, color="yellow", lw=1.2)
    ax.plot([q[0] - x0], [q[1] - y0], "o", ms=13, mec="w", mew=1.5, color="#b3452e")
    if np.isfinite(uv[0]):
        ax.plot([uv[0] - x0], [uv[1] - y0], "x", ms=11, mew=2, color="k")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)
plt.tight_layout()
import io
buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=110); buf.seek(0)
arr = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1].copy()
imwrite_u(OUT / "figs" / "v6_onvessel_verification.png", arr)
print(f"[V6] saved out/figs/v6_onvessel_verification.png  (red dot = reported, black x = detection)")

# ============================================== PART C: results video on REAL frames
print(f"\n" + "=" * 78)
print("PART C - re-rendering the results video on real video frames")
print("=" * 78)
bg_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
def render_frame(k):
    """Real video frame + honest tinting of the WHOLE vasculature + overlays."""
    fr = grab(k).astype(np.float32)
    # strong enough to read through the raw video's own colours: filled lumen
    # goes clearly blue; other tube-like structure (incl. fluid-free segments,
    # and permissively the phantom's supports - 62% of the frame, the Stage-T7
    # caveat) gets a light neutral tint so nothing pale reads as empty space
    fr = 0.55 * fr + 0.45 * np.array([70, 130, 255], np.float32) * (lumen[..., None] > 0)
    fr = 0.85 * fr + 0.15 * np.array([120, 120, 120], np.float32) * (unfilled[..., None])
    return fr.astype(np.uint8)

STEP = 4
frames = list(range(0, K, STEP))
print(f"[V6] {len(frames)} animation frames (stride {STEP}) from {K}")

fig = plt.figure(figsize=(15, 6.4))
axL = fig.add_subplot(1, 2, 1)
axR = fig.add_subplot(1, 2, 2, projection="3d")
imL = axL.imshow(render_frame(0), extent=[0, W, H, 0])
_xi = np.clip(np.rint(P_curve[:, 0]).astype(int), 0, W - 1)
_yi = np.clip(np.rint(P_curve[:, 1]).astype(int), 0, H - 1)
_on = lumen[_yi, _xi] > 0
_seg, _cur = [], [0]
for _i in range(1, len(P_curve)):
    if _on[_i] != _on[_i - 1]:
        _seg.append((_cur, _on[_i - 1])); _cur = [_i]
    else:
        _cur.append(_i)
_seg.append((_cur, _on[-1]))
for _ix, _isOn in _seg:
    if len(_ix) < 2: continue
    axL.plot(P_curve[_ix, 0], P_curve[_ix, 1], color="yellow", lw=1.2,
             ls="-" if _isOn else (0, (3, 2)))
axL.legend(handles=[Line2D([], [], color="yellow", lw=1.2, label="Path 2 projected (f=516)"),
                    Line2D([], [], color="yellow", lw=1.2, ls=(0, (3, 2)), label="...unfilled segment"),
                    Line2D([], [], color="#b3452e", marker="o", lw=0, ms=8, label="reported 3D, projected"),
                    Line2D([], [], color="k", marker="x", lw=0, ms=7, label="detection (raw)")],
           loc="lower left", fontsize=7, framealpha=.85)
trailL, = axL.plot([], [], "-", color="#4a7fb5", lw=1.8, alpha=.8)
circL = Circle((0, 0), 1.0, color="#b3452e", alpha=.22, ec="#b3452e", lw=1.8, ls=(0, (4, 2)))
axL.add_patch(circL)
dotL, = axL.plot([], [], "o", color="#b3452e", ms=11, mec="w", mew=1.5)
detL, = axL.plot([], [], "x", color="k", ms=8, mew=1.6)
axL.set_xlim(0, W); axL.set_ylim(H, 0); axL.set_xticks([]); axL.set_yticks([])
axL.set_title("Actual video frame  (blue = filled lumen, grey = unfilled vessel)", fontsize=11)

axR.plot(C3[:, 0], C3[:, 1], C3[:, 2], color="0.8", lw=1.5)
trailR, = axR.plot([], [], [], "-", color="#4a7fb5", lw=2)
dotR, = axR.plot([], [], [], "o", color="#b3452e", ms=9)
errR, = axR.plot([], [], [], "-", color="#b3452e", lw=4, alpha=.45)
axR.set_xlabel("x (mm)"); axR.set_ylabel("y (mm)"); axR.set_zlabel("z (mm)")
axR.set_xlim(C3[:, 0].min() - 3, C3[:, 0].max() + 3)
axR.set_ylim(C3[:, 1].min() - 3, C3[:, 1].max() + 3)
axR.set_zlim(C3[:, 2].min() - 3, C3[:, 2].max() + 3)
axR.set_title("Reported: 3D position + uncertainty", fontsize=12)
txt = axR.text2D(.02, .88, "", transform=axR.transAxes, fontsize=9.5, family="monospace",
                 bbox=dict(fc="w", alpha=.8, ec="0.7"))

T = np.gradient(C3, axis=0)
T /= np.clip(np.linalg.norm(T, axis=1, keepdims=True), 1e-9, None)
def nearest_tangent(x):
    return T[int(np.argmin(np.linalg.norm(C3 - x, axis=1)))]

def update(i):
    k = frames[i]
    imL.set_data(render_frame(k))
    lo = max(0, k - 120)
    trailL.set_data(Q_rep[lo:k + 1, 0], Q_rep[lo:k + 1, 1])
    circL.center = (Q_rep[k, 0], Q_rep[k, 1])
    circL.radius = sigma_rep[k] * SC516
    dotL.set_data([Q_rep[k, 0]], [Q_rep[k, 1]])
    if np.isfinite(UV[k, 0]):
        detL.set_data([UV[k, 0]], [UV[k, 1]])
    else:
        detL.set_data([], [])
    trailR.set_data(X_rep[lo:k + 1, 0], X_rep[lo:k + 1, 1])
    trailR.set_3d_properties(X_rep[lo:k + 1, 2])
    dotR.set_data([X_rep[k, 0]], [X_rep[k, 1]])
    dotR.set_3d_properties([X_rep[k, 2]])
    tg = nearest_tangent(X_rep[k]) * sigma_rep[k]
    e = np.stack([X_rep[k] - tg, X_rep[k] + tg])
    errR.set_data(e[:, 0], e[:, 1]); errR.set_3d_properties(e[:, 2])
    tag = "  <-- constraint widened" if viol[k] else ""
    txt.set_text(f"t = {k/FPS:5.1f} s   s = {s_rep[k]:6.1f} mm\n"
                 f"3D = ({X_rep[k,0]:5.1f},{X_rep[k,1]:5.1f},{X_rep[k,2]:5.1f}) mm\n"
                 f"+- {sigma_rep[k]:.2f} mm{tag}\n"
                 f"{TAG}")
    axR.view_init(elev=20, azim=-70 + 40 * np.sin(2 * np.pi * i / len(frames)))
    return trailL, circL, dotL, detL, trailR, dotR, errR, txt

anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 * STEP / FPS, blit=False)
if HAVE_FFMPEG:
    outp = OUT / "figs" / "V_results_animation.mp4"
    anim.save(str(outp), writer=FFMpegWriter(fps=int(FPS / STEP), bitrate=2600))
else:
    outp = OUT / "figs" / "V_results_animation.gif"
    anim.save(str(outp), writer=PillowWriter(fps=int(FPS / STEP)))
cap.release()
print(f"[V6] saved {outp}  (now on real video frames - the robot and the estimate are directly comparable)")
np.savez(DATA / "v6_onvessel.npz", on_lum_Q=on_lum_Q, on_any_Q=on_any_Q, d_Q=d_Q,
         on_lum_UV=on_lum_UV, on_any_UV=on_any_UV, sigma_rep=sigma_rep,
         rp_rep=rp_rep, both_fail=both)
print(f"[V6] saved data/v6_onvessel.npz")
