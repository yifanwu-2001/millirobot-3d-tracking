"""Z2 - the demo clip: actual video on the left, 3D position on the right.

Deliberately bare. Two small panel titles and nothing else - no residuals, no
status flags, no parameter strings. The trajectory shown is the deliverable
(V8 tube-aware decode, pinhole f=516, Viterbi, v_max=60, radius policy C).
"""
import numpy as np, cv2, io, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import DATA, OUT, VIDEO, FPS, imwrite_u
from a_centerline import load as load_cl

TRAIL = 90                      # frames of visible trail
s3, C3, _, _ = load_cl()
X = np.load(DATA / "v8_tube.npz")["X_tube"]
K = len(X)

cap = cv2.VideoCapture(str(VIDEO))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
PANEL = H                                    # right panel is square, video height
OUTW, OUTH = W + PANEL, H

fig = plt.figure(figsize=(PANEL / 100, OUTH / 100), dpi=100)
ax = fig.add_axes([.02, .02, .96, .96], projection="3d")
# The path is a long thin slab (190 x 28 x 20 mm). Rendered at true aspect it
# occupies a corner of a square panel, so the box aspect is stretched in x/z to
# fill the frame and the axes furniture is dropped entirely - the clip carries
# two titles and nothing else.
BOX = (2.3, 4.6, 2.0)
LIM = [(C3[:, c].min() - 3, C3[:, c].max() + 3) for c in range(3)]

# ---------------------------------------------------------------- fixed view
# One view, chosen by search rather than by eye, over elev 8-60 x azim 0-355 on
# three criteria:
#   mean |T . eye|  how much of the robot's travel is lost to foreshortening
#                   (0 = every segment moves across the screen, 1 = straight at
#                   the camera, where motion is invisible)
#   self-occlusion  projected points that land together while being far apart
#                   along the vessel - the path folding over itself
#   direction       on-screen travel must match the video (right-to-left);
#                   measured against the detected 2D track, constrained to >0.75
# Winner: elev 53, azim 175 -> |T.eye| 0.128, zero occlusion, cos +0.94.
# An earlier hand-picked elev 24 / azim -58 scored 0.506 / 0.453 / -0.98: half
# the motion foreshortened, the path folded over itself, and running backwards.
# A rotating view was tried first and dropped - tying the azimuth to the local
# tangent pinned it against its own clamp and left |T.eye| at 0.73, i.e. worse
# than this fixed view and busier to watch.
ELEV, AZIM = 53.0, 175.0

def render3d(k):
    ax.clear()
    ax.plot(C3[:, 0], C3[:, 1], C3[:, 2], color="#ccd6dc", lw=2.6, zorder=1)
    a = max(0, k - TRAIL)
    if k > a + 1:
        seg = X[a:k + 1]
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="#0b6fa4", lw=4.0,
                solid_capstyle="round", zorder=3)
    ax.scatter(*X[k], s=130, color="#c0392b", edgecolor="#7b241c", linewidth=1.0,
               depthshade=False, zorder=6)
    ax.set_xlim(*LIM[0]); ax.set_ylim(*LIM[1]); ax.set_zlim(*LIM[2])
    ax.set_box_aspect(BOX)
    # standard boxed 3D axes: panes, grid and mm ticks give the reference frame
    ax.set_xlabel("x (mm)", fontsize=11, labelpad=6)
    ax.set_ylabel("y (mm)", fontsize=11, labelpad=10)
    ax.set_zlabel("z (mm)", fontsize=11, labelpad=6)
    ax.tick_params(labelsize=9)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor("#fbfcfd"); a.pane.set_edgecolor("#d6dde1"); a.pane.set_alpha(1.0)
    ax.grid(True)
    ax.view_init(elev=ELEV, azim=AZIM)
    buf = io.BytesIO(); fig.savefig(buf, format="png", facecolor="w"); buf.seek(0)
    img = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), cv2.IMREAD_COLOR)
    return cv2.resize(img, (PANEL, OUTH))


def title(img, text, x):
    cv2.putText(img, text, (x, 34), cv2.FONT_HERSHEY_SIMPLEX, .78, (255, 255, 255), 5, cv2.LINE_AA)
    cv2.putText(img, text, (x, 34), cv2.FONT_HERSHEY_SIMPLEX, .78, (35, 35, 35), 2, cv2.LINE_AA)


RAW = OUT / "figs" / "_demo_raw.mp4"
four = cv2.VideoWriter(str(RAW), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (OUTW, OUTH))
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
for k in range(K):
    ok, frame = cap.read()
    if not ok:
        break
    right = render3d(k)
    canvas = np.hstack([frame, right])
    title(canvas, "Actual video", 24)
    title(canvas, "3D position", W + 24)
    four.write(canvas)
    if k % 150 == 0:
        print(f"  {k}/{K}")
four.release(); cap.release(); plt.close(fig)

# re-encode to h264 so it is small enough to attach to an email
import subprocess, imageio_ffmpeg
dst = OUT / "figs" / "DEMO_tracking.mp4"
r = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(RAW),
                    "-c:v", "libx264", "-crf", "28", "-preset", "slow",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst)],
                   capture_output=True, text=True)
if r.returncode:
    print(r.stderr[-1200:])
else:
    RAW.unlink(missing_ok=True)
    print(f"[Z2] wrote out/figs/DEMO_tracking.mp4  ({OUTW}x{OUTH}, {K} frames, "
          f"{dst.stat().st_size/1e6:.1f} MB)")
