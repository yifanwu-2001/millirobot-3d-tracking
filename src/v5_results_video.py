"""V5 - the improved configuration, measured end to end: 7-axis numbers + results video.

Stages V1-V4 changed the recommendation in two places: the uncertainty output is
now the V4 camera-family mixture (median + sigma_mix), and the working camera
now has four converging lines of evidence behind pinhole f=516 (R1 size cue,
R2a self-consistency tail, Q2's held-out columns, V4's held-out evidence).
This stage measures that configuration as a whole - the same columns the README's
"Current configuration" table and 7-axis evaluation quote - and renders the
results animation in the style of s3, with the V4 deliverable:

    left  : what the system sees - one 2D view, the detection, the projected
            centerline under the recommended camera (dashed where it crosses an
            unfilled vessel segment, the Stage-T honest-rendering rule)
    right : what it reports - the marginalised 3D position (mixture median) with
            its per-frame sigma_mix bar

Output video note (same as s3): renders the phantom background, so it is a
local presentation asset and must not be committed.
"""
import time
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
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from config import DATA, OUT, W, H, FPS, imread_u
from a_centerline import load as load_cl
from e_hmm import ArcHMM

V_MAX, SIGMA_PX = 60.0, 9.0
s3c, C3, _, _ = load_cl(); L = s3c[-1]
DS = 0.25
SG = np.arange(0, L + 1e-9, DS)
CG = np.stack([np.interp(SG, s3c, C3[:, c]) for c in range(3)], 1)
fin = np.load(DATA / "i_final.npz"); UV, k_turn = fin["UV"], int(fin["k_turn"])
K = len(UV); dt = 1 / FPS
OUT_LEG = np.zeros(K, bool); OUT_LEG[:k_turn + 1] = True
RET = ~OUT_LEG

z10 = np.load(DATA / "c10_ba.npz")
Xc = CG @ Rot.from_rotvec(z10["rv"]).as_matrix().T + z10["t"]
P_516 = Xc[:, :2] / Xc[:, 2:3] * float(z10["f"]) + np.array([float(z10["cx"]), float(z10["cy"])])
R516 = Rot.from_rotvec(z10["rv"]).as_matrix(); t516 = z10["t"]; f516 = float(z10["f"])
c516 = np.array([float(z10["cx"]), float(z10["cy"])])


def proj516(X):
    Xc = X @ R516.T + t516
    return Xc[:, :2] / Xc[:, 2:3] * f516 + c516


# ================================================================= PART A: metrics
print("=" * 78)
print("PART A - improved configuration: pinhole f=516 + Viterbi + v_max=60 + sigma=9")
print("=" * 78)
hmm = ArcHMM(SG, P_516, sigma_px=SIGMA_PX, v_max=V_MAX, n_v=41, accel_sigma=20., p_lost=0.05)
t0 = time.perf_counter(); s_hat_c, _, _ = hmm.forward(UV, dt); t_causal = time.perf_counter() - t0
t0 = time.perf_counter(); s_v = hmm.viterbi(UV, dt); t_vit = time.perf_counter() - t0
idx = np.clip(np.rint(s_v / DS).astype(int), 0, hmm.S - 1)
rp = np.linalg.norm(P_516[idx] - UV, axis=1)
speeds = np.abs(np.diff(s_v)) * FPS

print(f"\n[A] headline metrics (Viterbi, the recommended output)")
print(f"    reprojection median / p90      {np.median(rp):5.2f} / {np.percentile(rp,90):5.2f} px")
print(f"    failure rate (>30 px)          {100*np.mean(rp>30):5.1f}%")
print(f"    outbound (calibration set)     med {np.median(rp[OUT_LEG]):5.2f} px, {100*np.mean(rp[OUT_LEG]>30):4.1f}% fail")
print(f"    RETURN leg (held out)          med {np.median(rp[RET]):5.2f} px, {100*np.mean(rp[RET]>30):4.1f}% fail")
print(f"    speed violations > 60 mm/s     {100*np.mean(speeds>V_MAX):.1f}%   (max {speeds.max():.1f} mm/s - Viterbi is speed-bounded)")
A = np.arange(k_turn + 1); B = np.arange(k_turn + 1, K)
d_img, ia = cKDTree(UV[A]).query(UV[B]); sel = d_img < 8.0
dsr = np.abs(s_v[B][sel] - s_v[A][ia[sel]])
print(f"    out-and-back repeatability     median {np.median(dsr):.2f} mm, p90 {np.percentile(dsr,90):.2f} mm (match<8px, n={sel.sum()})")
print(f"    throughput                     causal {1000*t_causal/K:.2f} ms/frame ({int(K/t_causal)} fps, "
      f"{K/t_causal/30:.0f}x real time);  Viterbi {1000*t_vit/K:.2f} ms/frame (offline)")

# memoryless baseline, for the M2 reading of axis 2
d2all = ((P_516[:, None, :] - UV[None, :, :]) ** 2).sum(2)
am = np.argmin(d2all, axis=0); d_am = np.sqrt(d2all[am, np.arange(K)])
am_speed = np.abs(np.diff(SG[am])) * FPS
print(f"\n[A] memoryless argmax/frame baseline under the same camera: reproj med "
      f"{np.median(d_am):.2f} px, {100*np.mean(d_am>30):.1f}% fail, BUT "
      f"{100*np.mean(am_speed>V_MAX):.1f}% of frames exceed 60 mm/s (max {am_speed.max():.0f})")
print(f"    -> low reprojection alone is not evidence of tracking (M2); read with speed + held-out")

# ================================================================= PART B: axis-3 ablation
print(f"\n" + "=" * 78)
print("PART B - temporal robustness under the improved camera (Viterbi)")
print("=" * 78)
def run_sub(step, extra_noise=0.0, decode="causal"):
    """Stage N1's exact protocol: v_max stays FIXED (the motion model absorbs the
    larger gap through the advection, not by widening the velocity range), and
    injected noise widens sigma_px to the combined sqrt(9^2 + extra^2)."""
    ks = np.arange(0, K, step)
    uv = UV[ks]
    if extra_noise:
        uv = uv + np.random.default_rng(1).normal(0, extra_noise, (len(ks), 2))
    sig = float(np.hypot(SIGMA_PX, extra_noise))
    h = ArcHMM(SG, P_516, sigma_px=sig, v_max=V_MAX, n_v=41, accel_sigma=20., p_lost=0.05)
    sv = h.forward(uv, step / FPS)[0] if decode == "causal" else h.viterbi(uv, step / FPS)
    ix = np.clip(np.rint(sv / DS).astype(int), 0, hmm.S - 1)
    r = np.linalg.norm(P_516[ix] - uv, axis=1)
    return np.median(r), 100 * np.mean(r > 30)

print(f"{'fps':>6}{'reproj med':>12}{'fail%':>8}   causal filter, Stage N1's exact protocol (v_max fixed)")
for step in (1, 2, 5, 15):
    m, f = run_sub(step)
    print(f"{FPS/step:6.0f}{m:12.2f}{f:8.1f}")
print(f"\n{'extra noise px':>15}{'eff sigma':>10}{'reproj med':>12}{'fail%':>8}   (dose on per-pulse quality, causal)")
for en in (0, 10, 20):
    m, f = run_sub(1, en)
    print(f"{en:15.0f}{np.hypot(9.0,en):10.1f}{m:12.2f}{f:8.1f}")
print(f"\n     caveat - the OFFLINE Viterbi decoder quantises at low pulse rates")
print(f"     (41 velocity bins x a 0.5 s gap = 22.5 mm displacement granularity):")
for step in (2, 15):
    m, f = run_sub(step, decode="viterbi")
    print(f"       {FPS/step:4.0f} fps: med {m:6.2f} px, {f:4.1f}% fail")
print(f"     a scaled velocity grid would fix it, but the current O(V^2) transition")
print(f"     step makes that expensive; the real-time path is what the dose/rate")
print(f"     claim concerns, and Viterbi at 30 fps (the clinical rate) is unaffected.")

# ================================================================= PART C: anatomy
print(f"\n" + "=" * 78)
print("PART C - anatomical plausibility (the metric Stage T added)")
print("=" * 78)
bg_bgr = imread_u(OUT / "background_median.png")
_hsv = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2HSV)
_h, _s, _v = _hsv[..., 0], _hsv[..., 1], _hsv[..., 2]
gray = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY)
orange = (((_h < 25) | (_h > 170)) & (_s > 40) & (_v > 40)).astype(np.uint8)
flat = cv2.absdiff(gray, cv2.GaussianBlur(gray, (0, 0), 6)) < 4
anytube = (~((_v > 215) & (_s < 28) & flat)).astype(np.uint8)
anytube = cv2.morphologyEx(anytube, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
anytube = cv2.morphologyEx(anytube, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
dist_out = cv2.distanceTransform(1 - anytube, cv2.DIST_L2, 5)
lumen = cv2.morphologyEx(orange, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def mask_at(P, m):
    xi = np.clip(np.rint(P[:, 0]).astype(int), 0, W - 1)
    yi = np.clip(np.rint(P[:, 1]).astype(int), 0, H - 1)
    return m[yi, xi] > 0, dist_out[yi, xi]


on_lum516, _ = mask_at(P_516, lumen)
on516, d516 = mask_at(P_516, anytube)
on_det, _ = mask_at(UV, anytube)
print(f"    projected centerline (f=516):  on filled lumen {100*on_lum516.mean():.1f}%, "
      f"on any tube {100*on516.mean():.1f}%, worst deviation {d516.max():.1f} px")
print(f"    detected robot track:          on any tube {100*on_det.mean():.1f}%")
print(f"    random image point:            {100*anytube.mean():.1f}% (mask coverage; the metric to beat)")

# ================================================================= PART D: video
print(f"\n" + "=" * 78)
print("PART D - rendering the results video (improved configuration)")
print("=" * 78)
v4 = np.load(DATA / "v4_mixture.npz")
sd_mix, med_mix, X_mix = v4["sd"], v4["med"], v4["X_mix"]
Q_mix = proj516(X_mix)                     # reported 3D position, projected for the 2D panel

_lumen = cv2.morphologyEx(orange, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
bg = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
bg = bg.copy(); bg[_lumen] = (0.82 * bg[_lumen] + 0.18 * np.array([70, 130, 255])).astype(np.uint8)

STEP = 4
frames = list(range(0, K, STEP))
print(f"[V5] {len(frames)} animation frames (stride {STEP}) from {K}")

fig = plt.figure(figsize=(15, 6.4))
axL = fig.add_subplot(1, 2, 1)
axR = fig.add_subplot(1, 2, 2, projection="3d")
axL.imshow(bg)
_xi = np.clip(np.rint(P_516[:, 0]).astype(int), 0, W - 1)
_yi = np.clip(np.rint(P_516[:, 1]).astype(int), 0, H - 1)
_on = _lumen[_yi, _xi]
_seg, _cur = [], [0]
for _i in range(1, len(P_516)):
    if _on[_i] != _on[_i - 1]:
        _seg.append((_cur, _on[_i - 1])); _cur = [_i]
    else:
        _cur.append(_i)
_seg.append((_cur, _on[-1]))
for _ix, _isOn in _seg:
    if len(_ix) < 2: continue
    axL.plot(P_516[_ix, 0], P_516[_ix, 1], color="0.35", lw=1.3,
             ls="-" if _isOn else (0, (3, 2)))
axL.legend(handles=[Line2D([], [], color="0.35", lw=1.3, label="Path 2 projected (f=516 camera)"),
                    Line2D([], [], color="0.35", lw=1.3, ls=(0, (3, 2)),
                           label="...crossing an unfilled vessel segment")],
           loc="lower left", fontsize=7, framealpha=.85)
trailL, = axL.plot([], [], "-", color="#4a7fb5", lw=2, alpha=.8)
detL, = axL.plot([], [], "x", color="k", ms=7, mew=1.2)
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
axR.set_title("What it reports: marginalised 3D position + $\\sigma_{mix}$", fontsize=12)
txt = axR.text2D(.02, .93, "", transform=axR.transAxes, fontsize=9.5, family="monospace",
                 bbox=dict(fc="w", alpha=.75, ec="0.7"))

T = np.gradient(C3, axis=0)
T /= np.clip(np.linalg.norm(T, axis=1, keepdims=True), 1e-9, None)
def nearest_tangent(x):
    return T[int(np.argmin(np.linalg.norm(C3 - x, axis=1)))]

def update(i):
    k = frames[i]
    lo = max(0, k - 120)
    trailL.set_data(UV[lo:k + 1, 0], UV[lo:k + 1, 1])
    detL.set_data([UV[k, 0]], [UV[k, 1]])
    dotL.set_data([Q_mix[k, 0]], [Q_mix[k, 1]])
    trailR.set_data(X_mix[lo:k + 1, 0], X_mix[lo:k + 1, 1])
    trailR.set_3d_properties(X_mix[lo:k + 1, 2])
    dotR.set_data([X_mix[k, 0]], [X_mix[k, 1]])
    dotR.set_3d_properties([X_mix[k, 2]])
    tg = nearest_tangent(X_mix[k]) * sd_mix[k]
    e = np.stack([X_mix[k] - tg, X_mix[k] + tg])
    errR.set_data(e[:, 0], e[:, 1]); errR.set_3d_properties(e[:, 2])
    txt.set_text(f"t = {k/FPS:5.1f} s   s = {med_mix[k]:6.1f} mm\n"
                 f"3D = ({X_mix[k,0]:5.1f},{X_mix[k,1]:5.1f},{X_mix[k,2]:5.1f}) mm\n"
                 f"+- {sd_mix[k]:.2f} mm (V4 mixture)\n"
                 f"pinhole f=516 - Viterbi - v60")
    axR.view_init(elev=20, azim=-70 + 40 * np.sin(2 * np.pi * i / len(frames)))
    return trailL, detL, dotL, trailR, dotR, errR, txt

anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 * STEP / FPS, blit=False)
if HAVE_FFMPEG:
    outp = OUT / "figs" / "V_results_animation.mp4"
    anim.save(str(outp), writer=FFMpegWriter(fps=int(FPS / STEP), bitrate=2400))
else:
    outp = OUT / "figs" / "V_results_animation.gif"
    anim.save(str(outp), writer=PillowWriter(fps=int(FPS / STEP)))
print(f"[V5] saved {outp}")
print("[V5] local presentation asset - do not commit (contains source video frames)")

np.savez(DATA / "v5_final.npz", s_v=s_v, rp=rp, speeds=speeds, dsr=dsr,
         on516=on516, on_det=on_det, t_causal=t_causal, t_viterbi=t_vit)
print(f"[V5] saved data/v5_final.npz")
