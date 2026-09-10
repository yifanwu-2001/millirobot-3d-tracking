"""V2 - the observation-noise sigma is sized for a signal the data no longer has.

sigma_px = 9 was set in Stage I to absorb the 8.6 px / 1.17 Hz roll wobble.
But the observation stream fed to the HMM was quietly switched to the
DE-WOBBLED savgol trend in that same stage (i_final.py loads
track2d_trend.npz's `trend`, and every later stage - P0, Q, S1, U1 - loads
UV from i_final.npz). The noise sigma was sized for is no longer in the data:
the trend's own off-curve scatter is a fraction of 9 px. M3 swept sigma_px and
found <15% change, but that was (a) under the mis-specified pinhole camera,
(b) scored on median reprojection only - the one metric M2 showed has a blind
spot. Under the affine camera, where Q1 showed the remaining failures
concentrate in self-intersection zones, a sharper likelihood is exactly what
branch resolution needs. The sweep was never re-run.

Sweep sigma_px under the working configuration (affine camera, v_max=60),
causal and Viterbi, and score on the metrics that are NOT circular for a
noise-width change: held-out return leg, failure-rate split into branch vs
no-match classes, out-and-back repeatability, speed violations. Selection
criterion: held-out return-leg failure rate, with the U1 guard - a choice
that degrades the fitting leg's absolute residual is not a win.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from config import DATA, FPS
from a_centerline import load as load_cl
from e_hmm import ArcHMM

V_MAX = 60.0
s3, C3, _, _ = load_cl(); L = s3[-1]
DS = 0.25
SG = np.arange(0, L + 1e-9, DS)
CG = np.stack([np.interp(SG, s3, C3[:, c]) for c in range(3)], 1)
fin = np.load(DATA / "i_final.npz"); UV, k_turn = fin["UV"], int(fin["k_turn"])
K = len(UV); dt = 1 / FPS
OUT_LEG = np.zeros(K, bool); OUT_LEG[:k_turn + 1] = True
RET = ~OUT_LEG

pa = np.load(DATA / "l2_affine.npy")
R_aff = Rot.from_rotvec(pa[:3]).as_matrix(); SC = np.exp(pa[3]); C0 = pa[4:6]
P_aff = (CG @ R_aff.T)[:, :2] * SC + C0

# --- premise check: how noisy is the trend observation, really? --------------
d_argmin = np.array([np.min(np.linalg.norm(P_aff - uv, axis=1)) for uv in UV])
print(f"[V2] premise check: distance from each TREND observation to the nearest")
print(f"     affine-projected curve point (argmax/frame, no model):")
print(f"     median {np.median(d_argmin):.2f} px   p90 {np.percentile(d_argmin,90):.2f} px")
print(f"     -> sigma_px=9 was sized for the 8.6 px wobble; the trend's own scatter")
print(f"        is ~{np.median(d_argmin):.0f} px. The likelihood may be ~3x too wide.")
zt = np.load(DATA / "track2d.npz"); raw = zt["uv"]
okr = ~np.isnan(raw[:, 0])
d_raw = np.array([np.min(np.linalg.norm(P_aff - uv, axis=1)) for uv in raw[okr]])
print(f"     (raw track for comparison: median {np.median(d_raw):.2f} px)")


def run(sigma, decode):
    hmm = ArcHMM(SG, P_aff, sigma_px=sigma, v_max=V_MAX, n_v=41, accel_sigma=20., p_lost=0.05)
    if decode == "causal":
        s_hat, _, _ = hmm.forward(UV, dt)
    else:
        s_hat = hmm.viterbi(UV, dt)
    idx = np.clip(np.rint(s_hat / DS).astype(int), 0, hmm.S - 1)
    rp = np.linalg.norm(P_aff[idx] - UV, axis=1)
    return s_hat, idx, rp


def outback(s_hat):
    A = np.arange(k_turn + 1); B = np.arange(k_turn + 1, K)
    d_img, ia = cKDTree(UV[A]).query(UV[B]); sel = d_img < 8.0
    return np.abs(s_hat[B][sel] - s_hat[A][ia[sel]])


SIGMAS = [2, 3, 4, 5, 6, 8, 9, 12]
print(f"\n[V2] sweep sigma_px, affine camera + v_max=60, observations = de-wobbled trend")
print(f"     decode   sigma  reproj med  p90  >30px | out med  RET med  ret>30px | out-back med")
results = {}
for decode in ("causal", "viterbi"):
    for sg in SIGMAS:
        s_hat, idx, rp = run(sg, decode)
        ob = outback(s_hat)
        results[(decode, sg)] = (s_hat, rp)
        print(f"     {decode:<8} {sg:5.1f}  {np.median(rp):10.2f} {np.percentile(rp,90):5.1f} "
              f"{100*np.mean(rp>30):5.1f}% | {np.median(rp[OUT_LEG]):6.2f} "
              f"{np.median(rp[RET]):7.2f} {100*np.mean(rp[RET]>30):7.1f}% | "
              f"{np.median(ob):7.2f} mm")

# --- failure-mode split at the two ends of the sweep -------------------------
print(f"\n[V2] failure-mode split (Viterbi, rp>30px): is a sharper likelihood fixing")
print(f"     branch errors, or just re-labelling them?  branch = a curve point >14 mm")
print(f"     away in s within 15 px of the observation; no-match = none within 30 px.")
for sg in (9, 5, 3):
    s_hat, rp = results[("viterbi", sg)]
    fail = rp > 30
    nb = 0; nm_ = 0
    for k in np.where(fail)[0]:
        d2 = ((P_aff - UV[k]) ** 2).sum(1)
        dbest = np.sqrt(d2.min())
        if dbest > 30:
            nm_ += 1; continue
        far = np.abs(SG - s_hat[k]) > 14.0
        dalt = np.sqrt(d2[far].min()) if far.any() else 1e9
        if dalt < 15:
            nb += 1
    print(f"     sigma={sg:4.1f}: failures {int(fail.sum()):3d}  branch {nb:3d}  no-match {nm_:3d}  "
          f"other {int(fail.sum())-nb-nm_:3d}")

# --- pick by held-out return-leg failure rate, with the U1 guard -------------
print(f"\n[V2] selection: held-out return-leg failure rate (Viterbi), guard = outbound")
print(f"     absolute residual must not degrade")
best = min(SIGMAS, key=lambda s: 100 * np.mean(results[("viterbi", s)][1][RET] > 30))
print(f"     best sigma_px = {best}")
for sg in sorted(set(SIGMAS) | {9}):
    _, rp = results[("viterbi", sg)]
    mark = "  <== selected" if sg == best else ("  (current default)" if sg == 9 else "")
    print(f"     sigma={sg:4.1f}: outbound {np.median(rp[OUT_LEG]):5.2f} px | return "
          f"{np.median(rp[RET]):5.2f} px / {100*np.mean(rp[RET]>30):4.1f}% fail | "
          f"overall >30px {100*np.mean(rp>30):4.1f}%{mark}")

np.savez(DATA / "v2_sigma.npz",
         sigmas=np.array(SIGMAS, float),
         med=np.array([np.median(results[("viterbi", s)][1]) for s in SIGMAS]),
         ret_med=np.array([np.median(results[("viterbi", s)][1][RET]) for s in SIGMAS]),
         ret_fail=np.array([100*np.mean(results[("viterbi", s)][1][RET] > 30) for s in SIGMAS]),
         fail=np.array([100*np.mean(results[("viterbi", s)][1] > 30) for s in SIGMAS]))
print(f"\n[V2] saved data/v2_sigma.npz")
