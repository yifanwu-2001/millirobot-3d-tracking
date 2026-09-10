"""V8 - the vessel constraint upgraded from an AXIS to a TUBE. No new data.

The reviewer is right that "use the vessel as the constraint" should not mean
"collapse the vessel to its axis line". Path 2 gives the axis; the IMAGE gives
the tube width around it (Stage L1 already measured the width profile). The
axis-only estimator has no degree of freedom for where inside the lumen the
robot is, which is exactly why the return-leg wall-hugging (V6/V7: up to 8.7 mm
in-plane at the s=109-116 junction, against a ~1.6 mm outbound fit) shows up as
an unexplained 46 px residual instead of as an estimate.

V8 adds that degree of freedom as a per-frame latent state:

    d_k  = signed offset of the robot from the axis, along the projected
           curve's image normal, |d| <= R_tube(s)

  1. the s-decode runs on the TUBE-MARGINALISED likelihood -
     log p(uv | s) = logsumexp_d [ log N(uv; P(s) + d n(s), sigma)
                                   + log N(d; 0, sigma_d0) ] -
     so the observation can be explained by any point of the lumen, while the
     N(0, sigma_d0) prior keeps d near the axis wherever the axis fits;
  2. Viterbi decodes s exactly as before (same (s, v) machinery, speed-bounded);
  3. given the s-path, d is decoded by an exact 1-D chain DP with a smoothness
     kernel (the robot sweeps wall-to-wall in ~0.3 s, so sigma_dd = 1.5 mm/frame);
  4. the 3D position is C(s) + delta, where delta is the minimum-norm 3D offset
     whose f=516 projection equals the image offset d n(s) (2x3 Jacobian
     pseudo-inverse at the axis point), clamped to |delta| <= R_tube(s).

R_tube(s) is measured from the image itself on THIS camera's projection, where
both tube edges are visible; where they are not (the s=109-119 mm junction
among them) it is the conservative 3.0 mm floor, not an imputed median - V8b's
policy C, which reproduces the gain without inventing a radius. No new data.

Honest caveats stated up front: the minimum-norm delta is the SHORTEST 3D
offset consistent with the image offset (a depth component along the viewing
ray is invisible, so the true offset can be longer); and d is only observable
to the extent the camera model is right. Both are why the error bar keeps its
constraint-violation inflation.
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
SIGMA_D0, SIGMA_DD = 1.2, 1.5          # mm: offset prior width, per-frame smoothness
s3c, C3, _, _ = load_cl(); L = s3c[-1]
DS = 0.25
SG = np.arange(0, L + 1e-9, DS)
CG = np.stack([np.interp(SG, s3c, C3[:, c]) for c in range(3)], 1)
fin = np.load(DATA / "i_final.npz"); UV, k_turn = fin["UV"], int(fin["k_turn"])
K = len(UV); dt = 1 / FPS
OUT_LEG = np.zeros(K, bool); OUT_LEG[:k_turn + 1] = True
RET = ~OUT_LEG

z10 = np.load(DATA / "c10_ba.npz")
R516 = Rot.from_rotvec(z10["rv"]).as_matrix(); t516 = z10["t"]; f516 = float(z10["f"])
c516 = np.array([float(z10["cx"]), float(z10["cy"])])
def proj516(X):
    Xc = X @ R516.T + t516
    return Xc[:, :2] / Xc[:, 2:3] * f516 + c516

Xc = CG @ R516.T + t516
Zc = Xc[:, 2]
P_ax = Xc[:, :2] / Zc[:, None] * f516 + c516
sc = f516 / Zc                                    # local scale, px per mm

# image tangent and unit normal of the projected curve
T2 = np.gradient(P_ax, SG, axis=0)
T2 /= np.clip(np.linalg.norm(T2, axis=1, keepdims=True), 1e-9, None)
N2 = np.stack([-T2[:, 1], T2[:, 0]], 1)           # unit normal, consistent orientation

# tube radius, policy C (V8b): measured on THIS projection where both tube edges
# are visible, conservative 3.0 mm floor where they are not, nothing imputed.
# (The original version took L1's width profile - sampled along the old c8_11
# projection - and filled its 142 gaps with the global median; V13 showed the
# junction radius that produced was never measured, and V8b showed the 3 mm
# floor gives the same gain.)
from scipy.ndimage import map_coordinates
_bg = imread_u(OUT / "background_median.png")
_hsv = cv2.cvtColor(_bg, cv2.COLOR_BGR2HSV).astype(float)
_hue = np.minimum(_hsv[:, :, 0], 180 - _hsv[:, :, 0])
_orange = cv2.GaussianBlur(np.clip(1 - _hue / 25, 0, 1) * np.clip(_hsv[:, :, 1] / 60, 0, 1)
                           * np.clip(_hsv[:, :, 2] / 60, 0, 1), (3, 3), 0)
_offs = np.arange(-100, 100.01, .25)
wpx = np.full(len(SG), np.nan)
for _i in range(len(SG)):
    _p = P_ax[_i] + _offs[:, None] * N2[_i]
    _pr = map_coordinates(_orange, [_p[:, 1], _p[:, 0]], order=1, mode='constant')
    _c = len(_offs) // 2
    if _pr[_c] < .5: continue
    _a = _b = _c
    while _a > 0 and _pr[_a - 1] >= .5: _a -= 1
    while _b < len(_offs) - 1 and _pr[_b + 1] >= .5: _b += 1
    if _a == 0 or _b == len(_offs) - 1: continue
    _lo = _offs[_a - 1] + .25 * (.5 - _pr[_a - 1]) / (_pr[_a] - _pr[_a - 1])
    _hi = _offs[_b] + .25 * (.5 - _pr[_b]) / (_pr[_b + 1] - _pr[_b])
    if 5 <= _hi - _lo <= 100: wpx[_i] = _hi - _lo
R_meas = np.isfinite(wpx)
R_tube = np.where(R_meas, np.clip(wpx / 2 / sc, 3.0, 6.0), 3.0)
print(f"[V8] tube radius R_tube (policy C): measured on {R_meas.sum()}/{len(SG)} samples, median "
      f"{np.median(R_tube):.2f} mm, p10 {np.percentile(R_tube,10):.2f}, p90 {np.percentile(R_tube,90):.2f}; "
      f"junction s=109-119: {int((R_meas & (SG>109) & (SG<119)).sum())} measured -> floor "
      f"{np.median(R_tube[(SG>109)&(SG<119)]):.2f} mm (an assumption, not a measurement)")

# per-s offset grid, in mm and px
FR = np.array([-1.0, -2 / 3, -1 / 3, 0.0, 1 / 3, 2 / 3, 1.0])
D_mm = FR[None, :] * R_tube[:, None]              # (S, 7)
D_px = D_mm * sc[:, None]
W_D = np.exp(-0.5 * (D_mm / SIGMA_D0) ** 2)
W_D /= W_D.sum(1, keepdims=True)


class TubeArcHMM(ArcHMM):
    """ArcHMM with the observation likelihood marginalised over the tube offset."""

    def __init__(self, **kw):
        super().__init__(**kw)

    def loglik(self, uv):
        if uv is None or not np.isfinite(uv).all():
            return np.zeros(self.S)
        pos = P_ax[:, None, :] + D_px[:, :, None] * N2[:, None, :]     # (S, D, 2)
        d2 = ((pos - uv[None, None, :]) ** 2).sum(-1)                  # (S, D)
        g = np.exp(-0.5 * d2 / self.sigma ** 2) / (2 * np.pi * self.sigma ** 2)
        mix = (W_D * g).sum(1)
        return np.log((1 - self.p_lost) * mix + self.p_lost / (960.0 * 720.0))


# ---------------------------------------------------------------- decode both
print(f"\n[V8] decoding: axis-only (current default) vs tube-aware")
hmm_ax = ArcHMM(SG, P_ax, sigma_px=SIGMA_PX, v_max=V_MAX, n_v=41, accel_sigma=20., p_lost=0.05)
hmm_tb = TubeArcHMM(s_grid=SG, proj_xy=P_ax, sigma_px=SIGMA_PX, v_max=V_MAX,
                    n_v=41, accel_sigma=20., p_lost=0.05)
s_ax = hmm_ax.viterbi(UV, dt)
s_tb = hmm_tb.viterbi(UV, dt)
ix_ax = np.clip(np.rint(s_ax / DS).astype(int), 0, hmm_ax.S - 1)
ix_tb = np.clip(np.rint(s_tb / DS).astype(int), 0, hmm_tb.S - 1)

def decode_d(s_path, ix_path):
    """Exact 1-D Viterbi for d given the s-path (emission incl. the d prior)."""
    D = D_px.shape[1]
    em = np.zeros((K, D))
    for k in range(K):
        if not np.isfinite(UV[k, 0]):
            em[k] = 0.0; continue
        pos = P_ax[ix_path[k]][None, :] + D_px[ix_path[k]][:, None] * N2[ix_path[k]][None, :]
        d2 = ((pos - UV[k][None, :]) ** 2).sum(1)
        em[k] = -0.5 * d2 / SIGMA_PX ** 2 + np.log(np.maximum(W_D[ix_path[k]], 1e-300))
    diffs = D_mm[ix_path][1:][:, None, :] - D_mm[ix_path][:-1][:, :, None]   # (K-1, 7prev, 7cur)
    Td = np.log(np.maximum(np.exp(-0.5 * diffs ** 2 / SIGMA_DD ** 2), 1e-300))
    delta = em[0].copy(); back = np.zeros((K, D), int)
    for k in range(1, K):
        cand = delta[:, None] + Td[k - 1]          # (7prev, 7cur)
        back[k] = np.argmax(cand, axis=0)
        delta = cand[back[k], np.arange(D)] + em[k]
    d_idx = np.zeros(K, int)
    d_idx[-1] = int(np.argmax(delta))
    for k in range(K - 1, 0, -1):
        d_idx[k - 1] = back[k, d_idx[k]]
    return D_mm[ix_path, d_idx], D_px[ix_path, d_idx]

d_mm, d_px = decode_d(s_tb, ix_tb)

# 3D: minimum-norm offset whose f=516 projection equals the image offset, clamped
X_tube = CG[ix_tb].copy()
for k in range(K):
    if abs(d_px[k]) < 1e-6:
        continue
    i = ix_tb[k]
    X0 = CG[i]; z0 = Zc[i]; x0, y0 = Xc[i, 0], Xc[i, 1]
    A = np.array([[1 / z0, 0, -x0 / z0 ** 2],
                  [0, 1 / z0, -y0 / z0 ** 2]])
    J = f516 * A @ R516.T                        # 2x3 image Jacobian wrt 3D offset
    delta = np.linalg.pinv(J) @ (d_px[k] * N2[i])
    nrm = np.linalg.norm(delta)
    if nrm > R_tube[i]:
        delta *= R_tube[i] / nrm
    X_tube[k] = X0 + delta

Q_ax = P_ax[ix_ax]
Q_tb = proj516(X_tube)
rp_ax = np.linalg.norm(Q_ax - UV, axis=1)
rp_tb = np.linalg.norm(Q_tb - UV, axis=1)
sp_ax = np.abs(np.diff(s_ax)) * FPS
sp_tb = np.abs(np.diff(s_tb)) * FPS

def blk(name, rp, s_path):
    A = np.arange(k_turn + 1); B = np.arange(k_turn + 1, K)
    from scipy.spatial import cKDTree
    d_img, ia = cKDTree(UV[A]).query(UV[B]); sel = d_img < 8.0
    ob = np.abs(s_path[B][sel] - s_path[A][ia[sel]])
    ep = np.arange(885, 920)
    print(f"  {name:<26} med {np.median(rp):5.2f} | p90 {np.percentile(rp,90):5.2f} | "
          f">30px {100*np.mean(rp>30):4.1f}% | out {np.median(rp[OUT_LEG]):5.2f} | "
          f"RET {np.median(rp[RET]):5.2f} ({100*np.mean(rp[RET]>30):4.1f}%) | "
          f"ep885-920 med {np.median(rp[ep]):5.1f} max {np.max(rp[ep]):5.1f} | "
          f"out-back {np.median(ob):4.2f} mm | vmax {np.max(np.abs(np.diff(s_path))*FPS):4.0f} mm/s")

print(f"\n[V8] {'model':<26}{'med':>6} {'p90':>6} {'>30px':>6} | {'out':>5} | "
      f"{'RET':>12} | {'episode 885-920':>16} | {'out-back':>8} | {'speed':>5}")
blk("axis-only (default)", rp_ax, s_ax)
blk("TUBE-aware (V8)", rp_tb, s_tb)

print(f"\n[V8] decoded offset d: |d|>2 mm on {100*np.mean(np.abs(d_mm)>2):.0f}% of frames, "
      f"max {np.max(np.abs(d_mm)):.1f} mm; on the outbound leg median |d| = "
      f"{np.median(np.abs(d_mm[OUT_LEG])):.2f} mm, on the return leg {np.median(np.abs(d_mm[RET])):.2f} mm")
ep = np.arange(885, 920)
print(f"[V8] junction episode: d goes to {d_mm[ep][np.argmax(np.abs(d_mm[ep]))]:+.1f} mm "
      f"(R_tube there {np.median(R_tube[(SG>109)&(SG<116)]):.1f} mm) - the robot rides the wall "
      f"and the estimate follows")

# ------------------------------------------------------------------ figure
import io
fig, axs = plt.subplots(1, 3, figsize=(19, 5))
a = axs[0]
a.semilogy(np.maximum(rp_ax, .1), lw=.7, label="axis-only")
a.semilogy(np.maximum(rp_tb, .1), lw=.7, label="tube-aware (V8)")
a.axhline(30, color="r", ls="--"); a.legend(); a.set_xlabel("frame"); a.set_ylabel("reproj px")
a.set_title("residual over time")
a = axs[1]
a.plot(SG[ix_tb], d_mm, lw=.8)
a.plot(SG[ix_tb], R_tube[ix_tb], "r--", lw=.8, label="+-R_tube(s)")
a.plot(SG[ix_tb], -R_tube[ix_tb], "r--", lw=.8)
a.axhline(0, color="0.6", lw=.5); a.legend(); a.set_xlabel("arc length s (mm)")
a.set_ylabel("decoded wall offset d (mm)"); a.set_title("decoded offset along the path")
a = axs[2]
a.plot(s_ax, lw=.7, label="axis s")
a.plot(s_tb, lw=.7, label="tube s")
a.axvline(k_turn, color="r", ls="--"); a.legend(); a.set_xlabel("frame"); a.set_ylabel("s (mm)")
a.set_title("arc length: unchanged where the axis fits")
plt.tight_layout()
buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=105); buf.seek(0)
arr = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1].copy()
imwrite_u(OUT / "figs" / "v8_tube_tracker.png", arr)
print(f"\n[V8] saved out/figs/v8_tube_tracker.png")

np.savez(DATA / "v8_tube.npz", s_tb=s_tb, d_mm=d_mm, X_tube=X_tube, Q_tb=Q_tb,
         rp_tb=rp_tb, R_tube=R_tube, s_ax=s_ax, rp_ax=rp_ax)
print(f"[V8] saved data/v8_tube.npz")
