"""S3 - the talk's opening animation: 2D observation in, 3D position out.

Left panel is what the system actually sees (one 2D view, the detected robot).
Right panel is what it reports (a 3D position on the vessel centerline, with the
camera-family envelope drawn as a bar). Playing them synced is the clearest
statement of what the method does.

NOTE: the left panel renders frames of the source video, so the output of this
script must NOT be committed to a public repository - it would redistribute the
input. It is a local presentation asset.
"""
import numpy as np, cv2, matplotlib
matplotlib.use("Agg")
# use the ffmpeg binary that ships with imageio-ffmpeg; a GIF of this length is
# ~44 MB, an h264 mp4 of the same frames is ~1-2 MB and plays in any slide deck
try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    HAVE_FFMPEG = True
except Exception:
    HAVE_FFMPEG = False
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from config import DATA, OUT, W, H, FPS, imread_u
from a_centerline import load as load_cl

STEP = 4                      # frame stride
s3, C3, _, _ = load_cl()
u = np.load(DATA / "s1_uncertainty.npz")
X_ref, delta, tot = u["X_ref"], u["delta"], u["tot"]
fin = np.load(DATA / "i_final.npz"); UV = fin["UV"]
q = np.load(DATA / "q1_affine_downstream.npz"); PG = q["PG"]
bg_bgr = imread_u(OUT / "background_median.png")
bg = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)

# Stage T. Two honest-rendering problems, fixed together:
#   (a) tinting only the ORANGE lumen made the projected centerline look like it
#       ran through empty space where it was following an unfilled vessel;
#   (b) tinting every non-background pixel overcorrects - that mask is 62% of the
#       frame and includes the clear acrylic supports, implying all of it is
#       vasculature.
# So: tint the filled lumen (specific and correct), and draw the projected
# centerline dashed wherever it leaves it, with the reason stated in the legend.
_hsv = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2HSV)
_h, _s, _v = _hsv[..., 0], _hsv[..., 1], _hsv[..., 2]
_lumen = (((_h < 25) | (_h > 170)) & (_s > 40) & (_v > 40)).astype(np.uint8)
_lumen = cv2.morphologyEx(_lumen, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
bg = bg.copy()
bg[_lumen] = (0.82 * bg[_lumen] + 0.18 * np.array([70, 130, 255])).astype(np.uint8)

K = len(X_ref)
frames = list(range(0, K, STEP))
print(f"[S3] {len(frames)} animation frames (stride {STEP}) from {K}")

fig = plt.figure(figsize=(15, 6.4))
axL = fig.add_subplot(1, 2, 1)
axR = fig.add_subplot(1, 2, 2, projection="3d")

axL.imshow(bg)
_xi = np.clip(np.rint(PG[:, 0]).astype(int), 0, W - 1)
_yi = np.clip(np.rint(PG[:, 1]).astype(int), 0, H - 1)
_on = _lumen[_yi, _xi]
_seg, _cur = [], [0]
for _i in range(1, len(PG)):
    if _on[_i] != _on[_i - 1]:
        _seg.append((_cur, _on[_i - 1])); _cur = [_i]
    else:
        _cur.append(_i)
_seg.append((_cur, _on[-1]))
for _ix, _isOn in _seg:
    if len(_ix) < 2: continue
    axL.plot(PG[_ix, 0], PG[_ix, 1], color="0.35", lw=1.3,
             ls="-" if _isOn else (0, (3, 2)))
from matplotlib.lines import Line2D
axL.legend(handles=[Line2D([], [], color="0.35", lw=1.3, label="Path 2 projected"),
                    Line2D([], [], color="0.35", lw=1.3, ls=(0, (3, 2)),
                           label="...crossing an unfilled vessel segment")],
           loc="lower left", fontsize=7, framealpha=.85)
trailL, = axL.plot([], [], "-", color="#4a7fb5", lw=2, alpha=.8)
dotL, = axL.plot([], [], "o", color="#b3452e", ms=11, mec="w", mew=1.5)
axL.set_xlim(0, W); axL.set_ylim(H, 0); axL.set_xticks([]); axL.set_yticks([])
axL.set_title("What it sees: one 2D view  (blue tint = fluid-filled lumen)", fontsize=11)

axR.plot(C3[:, 0], C3[:, 1], C3[:, 2], color="0.8", lw=1.5)
trailR, = axR.plot([], [], [], "-", color="#4a7fb5", lw=2)
dotR, = axR.plot([], [], [], "o", color="#b3452e", ms=9)
errR, = axR.plot([], [], [], "-", color="#b3452e", lw=4, alpha=.45)
axR.set_xlabel("x (mm)"); axR.set_ylabel("y (mm)"); axR.set_zlabel("z (mm)")
axR.set_xlim(C3[:, 0].min() - 3, C3[:, 0].max() + 3)
axR.set_ylim(C3[:, 1].min() - 3, C3[:, 1].max() + 3)
axR.set_zlim(C3[:, 2].min() - 3, C3[:, 2].max() + 3)
axR.set_title("What it reports: 3D position + uncertainty", fontsize=12)
txt = axR.text2D(.02, .93, "", transform=axR.transAxes, fontsize=10,
                 family="monospace",
                 bbox=dict(fc="w", alpha=.75, ec="0.7"))

# local tangent, to draw the uncertainty bar along the vessel
T = np.gradient(C3, axis=0)
T /= np.clip(np.linalg.norm(T, axis=1, keepdims=True), 1e-9, None)


def nearest_tangent(x):
    return T[int(np.argmin(np.linalg.norm(C3 - x, axis=1)))]


def update(i):
    k = frames[i]
    lo = max(0, k - 120)
    trailL.set_data(UV[lo:k + 1, 0], UV[lo:k + 1, 1])
    dotL.set_data([UV[k, 0]], [UV[k, 1]])
    trailR.set_data(X_ref[lo:k + 1, 0], X_ref[lo:k + 1, 1])
    trailR.set_3d_properties(X_ref[lo:k + 1, 2])
    dotR.set_data([X_ref[k, 0]], [X_ref[k, 1]])
    dotR.set_3d_properties([X_ref[k, 2]])
    tg = nearest_tangent(X_ref[k]) * tot[k]
    e = np.stack([X_ref[k] - tg, X_ref[k] + tg])
    errR.set_data(e[:, 0], e[:, 1]); errR.set_3d_properties(e[:, 2])
    txt.set_text(f"t = {k/FPS:5.1f} s\n+- {tot[k]:.2f} mm")
    axR.view_init(elev=20, azim=-70 + 40 * np.sin(2 * np.pi * i / len(frames)))
    return trailL, dotL, trailR, dotR, errR, txt


anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 * STEP / FPS, blit=False)
if HAVE_FFMPEG:
    outp = OUT / "figs" / "SUMMARY_animation.mp4"
    anim.save(str(outp), writer=FFMpegWriter(fps=int(FPS / STEP), bitrate=2400))
else:
    outp = OUT / "figs" / "SUMMARY_animation.gif"
    anim.save(str(outp), writer=PillowWriter(fps=int(FPS / STEP)))
print(f"[S3] saved {outp}")
print("[S3] local presentation asset - do not commit (contains source video frames)")
