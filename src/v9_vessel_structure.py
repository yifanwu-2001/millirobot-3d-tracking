"""V9 - the vessel-visibility stage: what can and cannot be segmented, made visible.

The reviewer asks: "there is STILL a stretch outside the vessel - can't you
segment the vessels?" The honest answer, measured on this image:

  1. FLUID-FILLED lumen: yes - it is orange (hue 7-12), cleanly segmentable,
     30.7% of the frame (the Stage-D/T masks always did this).
  2. FLUID-FREE (transparent) tubes: their interior is 216-241 gray where the
     background is 228-235 - indistinguishable by appearance. Only the WALL
     LINES (60-100 gray-level dips) betray them, and by appearance alone they
     cannot be told from the phantom's clear acrylic supports. That is a
     physical limit of the imaging, not of the algorithm - no segmenter can
     separate two things that look the same. What CAN be done is detect and
     DISPLAY the wall lines, so a human can verify the route runs inside them.

So V9 does three things:

  A. measures the route against the segmentable structure (lumen + wall-bounded
     structure) and reports the honest numbers;
  B. renders the results video on CONTRAST-ENHANCED (CLAHE) frames with the
     wall lines drawn - the pale stretches become visible, so "outside the
     vessel" stops being an artefact of invisible walls;
  C. native-resolution verification crops at every remaining non-lumen stretch.

Video note (same as s3/v6): renders source-video frames - local asset, do not
commit.
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

# ------------------------------------------------------------------ masks
bg_bgr = imread_u(OUT / "background_median.png")
gray = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2HSV)
h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

lumen = (((h < 25) | (h > 170)) & (s > 40) & (v > 40)).astype(np.uint8)
lumen = cv2.morphologyEx(lumen, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

flat = cv2.absdiff(gray, cv2.GaussianBlur(gray, (0, 0), 6)) < 4
structure = (~((v > 215) & (s < 28) & flat)).astype(np.uint8)
structure = cv2.morphologyEx(structure, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
structure = cv2.morphologyEx(structure, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

gb = cv2.GaussianBlur(gray, (5, 5), 0)
walls = cv2.morphologyEx(gb, cv2.MORPH_BLACKHAT,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
wall_map = (walls > 25).astype(np.uint8)          # thin dark lines: tube walls
print(f"[V9] masks: filled lumen {100*lumen.mean():.1f}% | tube-like structure "
      f"{100*structure.mean():.1f}% | wall-line pixels {100*wall_map.mean():.1f}%")

dist_struct = cv2.distanceTransform(1 - structure, cv2.DIST_L2, 5)

# ------------------------------------------------ deliverable trajectory (V8)
s3c, C3, _, _ = load_cl(); L = s3c[-1]
SG = np.arange(0, L + 1e-9, 0.25)
CG = np.stack([np.interp(SG, s3c, C3[:, c]) for c in range(3)], 1)
z10 = np.load(DATA / "c10_ba.npz")
R516 = Rot.from_rotvec(z10["rv"]).as_matrix(); t516 = z10["t"]; f516 = float(z10["f"])
c516 = np.array([float(z10["cx"]), float(z10["cy"])])
def proj516(X):
    Xc = X @ R516.T + t516
    return Xc[:, :2] / Xc[:, 2:3] * f516 + c516
P_curve = proj516(CG)

fin = np.load(DATA / "i_final.npz"); UV, k_turn = fin["UV"], int(fin["k_turn"])
v4 = np.load(DATA / "v4_mixture.npz"); sd_mix = v4["sd"]
v8 = np.load(DATA / "v8_tube.npz")
X_rep, Q_rep, s_rep, rp_rep = v8["X_tube"], v8["Q_tb"], v8["s_tb"], v8["rp_tb"]
TAG = "pinhole f=516 - Viterbi - v60 - TUBE-aware (V8)"

# ------------------------------------------------- A: route vs structure
xi = np.clip(P_curve[:, 0].astype(int), 0, W - 1)
yi = np.clip(P_curve[:, 1].astype(int), 0, H - 1)
on_lum_c = lumen[yi, xi] > 0
on_str_c = structure[yi, xi] > 0
d_c = dist_struct[yi, xi]
xj = np.clip(Q_rep[:, 0].astype(int), 0, W - 1)
yj = np.clip(Q_rep[:, 1].astype(int), 0, H - 1)
on_lum_q = lumen[yj, xj] > 0
on_str_q = structure[yj, xj] > 0
d_q = dist_struct[yj, xj]
print(f"\n[V9] A. the projected ROUTE (Path 2 through the f=516 camera), per sample:")
print(f"    inside fluid-filled lumen : {100*on_lum_c.mean():5.1f}%")
print(f"    inside tube-like structure: {100*on_str_c.mean():5.1f}%   "
      f"(the rest is the transparent unfilled walls)")
print(f"    worst distance outside structure: {d_c.max():.1f} px = "
      f"{d_c.max()/5.8:.2f} mm; samples >2 px off: {int((d_c>2).sum())} of {len(d_c)}")
print(f"\n[V9] the REPORTED trajectory (V8 tube-aware), per frame:")
print(f"    inside fluid-filled lumen : {100*on_lum_q.mean():5.1f}%")
print(f"    inside tube-like structure: {100*on_str_q.mean():5.1f}%   worst off "
      f"{d_q.max():.1f} px = {d_q.max()/5.8:.2f} mm")

# ------------------------------------------------- B: video on CLAHE frames
tmp = os.path.join(tempfile.gettempdir(), "millirobot_video.mp4")
if not os.path.exists(tmp):
    shutil.copy(ROOT / "Video.mp4", tmp)
cap = cv2.VideoCapture(tmp)
def grab(k):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
    ok, fr = cap.read()
    if not ok: raise IOError(f"cannot read frame {k}")
    return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)

def enhance(img, clip=2.5):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clip, (8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

# static wall overlay: detected wall lines drawn as bright white lines on the
# natural-colour frame - walls are the visible evidence of "where vessels are"
_wl = cv2.morphologyEx(wall_map, cv2.MORPH_DILATE, np.ones((2, 2), np.uint8))
WALL_OV = (0.20 * _wl[..., None]).astype(np.float32)
WALL_COL = np.array([255, 255, 255], np.float32)

def render_frame(k):
    fr = enhance(grab(k), 1.6).astype(np.float32) * 0.92
    fr = fr * (1 - WALL_OV) + WALL_COL * WALL_OV
    return fr.astype(np.uint8)

# constraint-violation bar, as in V6, on the V8 residual
SC516 = f516 / float(np.median((CG @ R516.T + t516)[:, 2]))
r0 = float(np.median(rp_rep)) / SC516
excess = np.maximum(rp_rep / SC516 - r0, 0.0)
sigma_rep = np.sqrt(sd_mix ** 2 + excess ** 2)
viol = excess > 2.0
print(f"\n[V9] B. error bar: {int(viol.sum())} frames widened, max +-"
      f"{sigma_rep.max():.2f} mm")

STEP = 4
frames = list(range(0, len(UV), STEP))
fig = plt.figure(figsize=(15, 6.4))
axL = fig.add_subplot(1, 2, 1)
axR = fig.add_subplot(1, 2, 2, projection="3d")
imL = axL.imshow(render_frame(0), extent=[0, W, H, 0])
# wall lines: thin dark overlay so the transparent walls are visible
wl = np.zeros((H, W, 4), np.uint8)
wl[wall_map > 0] = (60, 60, 60, 255)
axL.imshow(wl, extent=[0, W, H, 0], alpha=0.35)
_xi = np.clip(P_curve[:, 0].astype(int), 0, W - 1)
_yi = np.clip(P_curve[:, 1].astype(int), 0, H - 1)
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
    axL.plot(P_curve[_ix, 0], P_curve[_ix, 1], color="0.15", lw=3.0,
             ls="-" if _isOn else (0, (3, 2)))
    axL.plot(P_curve[_ix, 0], P_curve[_ix, 1], color="yellow", lw=1.4,
             ls="-" if _isOn else (0, (3, 2)))
axL.legend(handles=[Line2D([], [], color="w", lw=2, label="detected tube walls (all tubes, filled or not)"),
                    Line2D([], [], color="yellow", lw=1.4, label="Path 2 projected - solid: inside FILLED lumen"),
                    Line2D([], [], color="yellow", lw=1.4, ls=(0, (3, 2)), label="Path 2 projected - dashed: inside UNFILLED segment"),
                    Line2D([], [], color="#b3452e", marker="o", lw=0, ms=8, label="reported position"),
                    Line2D([], [], color="k", marker="x", lw=0, ms=7, label="detection (raw)")],
           loc="lower left", fontsize=7, framealpha=.9)
trailL, = axL.plot([], [], "-", color="#4a7fb5", lw=1.8, alpha=.8)
circL = Circle((0, 0), 1.0, color="#b3452e", alpha=.22, ec="#b3452e", lw=1.8, ls=(0, (4, 2)))
axL.add_patch(circL)
dotL, = axL.plot([], [], "o", color="#b3452e", ms=11, mec="w", mew=1.5)
detL, = axL.plot([], [], "x", color="k", ms=8, mew=1.6)
axL.set_xlim(0, W); axL.set_ylim(H, 0); axL.set_xticks([]); axL.set_yticks([])
axL.set_title("Actual video frame, contrast-enhanced; bright lines = detected walls of ALL tubes", fontsize=10.5)

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
T3 = np.gradient(CG, axis=0)
T3 /= np.clip(np.linalg.norm(T3, axis=1, keepdims=True), 1e-9, None)
def nearest_tangent(x):
    return T3[int(np.argmin(np.linalg.norm(CG - x, axis=1)))]

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
    tag = "  <- constraint widened" if viol[k] else ""
    txt.set_text(f"t = {k/FPS:5.1f} s   s = {s_rep[k]:6.1f} mm\n"
                 f"3D = ({X_rep[k,0]:5.1f},{X_rep[k,1]:5.1f},{X_rep[k,2]:5.1f}) mm\n"
                 f"+- {sigma_rep[k]:.2f} mm{tag}\n{TAG}")
    axR.view_init(elev=20, azim=-70 + 40 * np.sin(2 * np.pi * i / len(frames)))
    return trailL, circL, dotL, detL, trailR, dotR, errR, txt

anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 * STEP / FPS, blit=False)
if HAVE_FFMPEG:
    outp = OUT / "figs" / "V_results_animation.mp4"
    anim.save(str(outp), writer=FFMpegWriter(fps=int(FPS / STEP), bitrate=2600))
else:
    outp = OUT / "figs" / "V_results_animation.gif"
    anim.save(str(outp), writer=PillowWriter(fps=int(FPS / STEP)))
print(f"[V9] saved {outp} (CLAHE-enhanced frames + wall lines)")

# ------------------------------------------------- C: verification crops
spans = []
off_lum = ~on_lum_q
st = np.where(np.diff(np.concatenate([[0], off_lum.view(np.int8), [0]])) != 0)[0]
for a, b in zip(st[::2], st[1::2]):
    if b - a >= 4:
        spans.append((a + b) // 2)
crops = [(k, f"frame {k} (t={k/FPS:.1f} s)") for k in spans[:3]]
fig, axs = plt.subplots(1, max(len(crops), 1), figsize=(6.4 * max(len(crops), 1), 5.8))
if len(crops) == 1: axs = [axs]
for ax, (k, title) in zip(axs, crops):
    img = enhance(grab(k))
    q = Q_rep[k]
    x0 = max(0, int(q[0]) - 170); x1 = min(W, int(q[0]) + 170)
    y0 = max(0, int(q[1]) - 150); y1 = min(H, int(q[1]) + 150)
    ax.imshow(img[y0:y1, x0:x1])
    m = (P_curve[:, 0] > x0) & (P_curve[:, 0] < x1) & (P_curve[:, 1] > y0) & (P_curve[:, 1] < y1)
    ax.plot(P_curve[m, 0] - x0, P_curve[m, 1] - y0, color="0.15", lw=3.0)
    ax.plot(P_curve[m, 0] - x0, P_curve[m, 1] - y0, color="yellow", lw=1.4)
    ww = (wall_map[y0:y1, x0:x1] > 0)
    ys, xs = np.where(ww)
    ax.scatter(xs, ys, s=0.3, c="k", alpha=.5)
    ax.plot([q[0] - x0], [q[1] - y0], "o", ms=13, mec="w", mew=1.5, color="#b3452e")
    if np.isfinite(UV[k, 0]):
        ax.plot([UV[k, 0] - x0], [UV[k, 1] - y0], "x", ms=11, mew=2, color="k")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title + " - CLAHE, walls in dark lines", fontsize=10)
plt.tight_layout()
import io
buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=110); buf.seek(0)
arr = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1].copy()
imwrite_u(OUT / "figs" / "v9_unfilled_verification.png", arr)
print(f"[V9] saved out/figs/v9_unfilled_verification.png")
np.savez(DATA / "v9_vessel_structure.npz", lumen=lumen, structure=structure,
         wall_map=wall_map, on_str_curve=on_str_c, d_curve=d_c,
         on_str_q=on_str_q, d_q=d_q)
print(f"[V9] saved data/v9_vessel_structure.npz")
cap.release()
