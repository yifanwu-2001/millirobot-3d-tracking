"""V11 - a genuinely online, tube-aware 2D -> 3D tracker.

The earlier best result (V8) is useful as an offline upper bound, but it has
two sources of future leakage: a centred 41-frame Savitzky-Golay filter and a
full-sequence Viterbi decode.  This module replaces both with components that
can run as frames arrive:

* a causal hard gate plus a narrow IIR notch at the measured roll frequency;
* a fixed-lag Viterbi decoder for (arc length, velocity), using only ``lag``
  future frames and constant memory;
* a fixed-lag decoder for signed position across the imaged lumen;
* minimum-norm lifting of that transverse image displacement back to 3D.

The camera/centreline registration and the tube-radius profile are setup data,
not future robot observations.  At the default lag of 12 frames the algorithm
has 400 ms algorithmic latency at 30 fps.  ``--lag 0`` is strictly causal,
but needs a known initial branch for this particular self-overlapping view.

Run from the repository root::

    python src/v11_realtime_tube.py
    python src/v11_realtime_tube.py --render

Outputs are ``data/v11_realtime.npz``, ``out/v11_realtime_tracking.csv``,
``out/figs/v11_realtime.png`` and, with ``--render``,
``out/figs/V11_realtime_tracking.mp4``.
"""
from __future__ import annotations

import argparse
from collections import deque
import csv
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import iirnotch, lfilter, lfilter_zi
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from scipy.special import logsumexp

from config import DATA, FPS, H, OUT, ROOT, VIDEO, W
from a_centerline import load as load_centerline


DS = 0.25
V_MAX = 60.0
N_V = 41
ACCEL_SIGMA = 20.0
P_LOST = 0.05
SIGMA_PX = 9.0
ROLL_HZ = 1.171875
NOTCH_Q = 3.0


def prepare_live_observations(gate_px: float = 80.0):
    """Causally reject detector glitches, bridge losses and remove roll wobble.

    No centred/windowed filter is used.  The hard gate operates only on the
    current detection and the previous accepted state.  The notch is a
    second-order causal IIR whose state can be retained between frames.
    """
    raw_file = np.load(DATA / "track2d.npz")
    raw = raw_file["uv"].copy()
    confidence = raw_file["conf"].copy()
    filled = np.empty_like(raw)
    accepted = np.zeros(len(raw), bool)
    velocity = np.zeros(2)
    last = None
    for k, uv in enumerate(raw):
        prediction = uv if last is None else last + velocity
        valid = np.isfinite(uv).all()
        if valid and last is not None:
            valid = np.linalg.norm(uv - prediction) <= gate_px
        if valid:
            filled[k] = uv
            accepted[k] = True
            if last is not None:
                velocity = 0.7 * velocity + 0.3 * (uv - last)
        else:
            filled[k] = prediction
            confidence[k] = 0.0
            velocity *= 0.8
        last = filled[k]

    b, a = iirnotch(ROLL_HZ, NOTCH_Q, FPS)
    filtered = np.empty_like(filled)
    for dim in range(2):
        zi = lfilter_zi(b, a) * filled[0, dim]
        filtered[:, dim], _ = lfilter(b, a, filled[:, dim], zi=zi)
    return filtered, raw, confidence, accepted


class TubeGeometry:
    """Registered centreline, projected lumen and 2D-to-3D lift."""

    def __init__(self, n_offset: int = 13, wall_mass: float = 0.35,
                 axis_sigma_mm: float = 1.5):
        s_raw, c_raw, _, _ = load_centerline()
        self.s = np.arange(0.0, s_raw[-1] + 1e-9, DS)
        self.C = np.stack([
            np.interp(self.s, s_raw, c_raw[:, dim]) for dim in range(3)
        ], axis=1)

        camera = np.load(DATA / "c10_ba.npz")
        self.R = Rot.from_rotvec(camera["rv"]).as_matrix()
        self.t = camera["t"]
        self.f = float(camera["f"])
        self.cxy = np.array([float(camera["cx"]), float(camera["cy"])])
        xc = self.C @ self.R.T + self.t
        self.xc = xc
        self.z = xc[:, 2]
        self.P = xc[:, :2] / self.z[:, None] * self.f + self.cxy
        self.scale = self.f / self.z

        tangent = np.gradient(self.P, self.s, axis=0)
        self.pixels_per_mm_s = np.linalg.norm(tangent, axis=1)
        tangent /= np.clip(self.pixels_per_mm_s[:, None], 1e-9, None)
        self.normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)

        # Tube radius, policy C (V8b): measured on THIS projection where both
        # tube edges are visible; conservative 3.0 mm floor where they are not
        # (the s=109-119 mm junction among them); nothing imputed from a median.
        # The original version took L1's width profile - sampled along the old
        # c8_11 projection - and median-filled its 142 gaps; V13 showed the
        # junction radius that produced was never measured (report.md, V8b).
        import cv2
        from scipy.ndimage import map_coordinates
        from config import OUT, imread_u
        bg = imread_u(OUT / "background_median.png")
        hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV).astype(float)
        hue = np.minimum(hsv[:, :, 0], 180 - hsv[:, :, 0])
        orange = cv2.GaussianBlur(np.clip(1 - hue / 25, 0, 1)
                                  * np.clip(hsv[:, :, 1] / 60, 0, 1)
                                  * np.clip(hsv[:, :, 2] / 60, 0, 1), (3, 3), 0)
        offs = np.arange(-100, 100.01, .25)
        width_px = np.full(len(self.s), np.nan)
        for i in range(len(self.s)):
            p = self.P[i] + offs[:, None] * self.normal[i]
            prof = map_coordinates(orange, [p[:, 1], p[:, 0]], order=1, mode="constant")
            c = len(offs) // 2
            if prof[c] < .5:
                continue
            a = b = c
            while a > 0 and prof[a - 1] >= .5:
                a -= 1
            while b < len(offs) - 1 and prof[b + 1] >= .5:
                b += 1
            if a == 0 or b == len(offs) - 1:
                continue
            lo = offs[a - 1] + .25 * (.5 - prof[a - 1]) / (prof[a] - prof[a - 1])
            hi = offs[b] + .25 * (.5 - prof[b]) / (prof[b + 1] - prof[b])
            if 5 <= hi - lo <= 100:
                width_px[i] = hi - lo
        self.radius_measured = np.isfinite(width_px)
        self.radius = np.where(self.radius_measured,
                               np.clip(width_px / (2.0 * self.scale), 3.0, 6.0), 3.0)
        self.fractions = np.linspace(-1.0, 1.0, n_offset)
        self.d_mm = self.radius[:, None] * self.fractions[None, :]
        self.d_px = self.d_mm * self.scale[:, None]
        self.offset_positions = (self.P[:, None, :] +
                                 self.d_px[:, :, None] *
                                 self.normal[:, None, :])

        # Most frames prefer the axis, but wall contact must retain real prior
        # mass.  This robust normal+uniform prior replaces V8's very narrow
        # Gaussian, which systematically stopped one bin short of the wall.
        normal = np.exp(-0.5 * (self.d_mm / axis_sigma_mm) ** 2)
        normal /= normal.sum(axis=1, keepdims=True)
        uniform = np.full_like(normal, 1.0 / n_offset)
        self.offset_prior = (1.0 - wall_mass) * normal + wall_mass * uniform
        self.log_offset_prior = np.log(np.maximum(self.offset_prior, 1e-300))

    def project(self, xyz):
        xc = np.asarray(xyz) @ self.R.T + self.t
        return xc[..., :2] / xc[..., 2:3] * self.f + self.cxy

    def tube_loglik(self, uv, sigma_px=SIGMA_PX):
        if uv is None or not np.isfinite(uv).all():
            return np.zeros(len(self.s))
        d2 = ((self.offset_positions - uv[None, None, :]) ** 2).sum(axis=2)
        log_g = (-0.5 * d2 / sigma_px ** 2 -
                 np.log(2.0 * np.pi * sigma_px ** 2))
        log_tube = logsumexp(self.log_offset_prior + log_g, axis=1)
        return np.logaddexp(np.log1p(-P_LOST) + log_tube,
                            np.log(P_LOST / (W * H)))

    def lift(self, s_index, d_mm):
        """Shortest 3D displacement that produces the decoded image offset."""
        i = int(s_index)
        d_px = float(d_mm) * self.scale[i]
        if abs(d_px) < 1e-12:
            return self.C[i].copy()
        z = self.z[i]
        x, y = self.xc[i, :2]
        perspective = np.array([[1.0 / z, 0.0, -x / z ** 2],
                                [0.0, 1.0 / z, -y / z ** 2]])
        jacobian = self.f * perspective @ self.R.T
        delta = np.linalg.pinv(jacobian) @ (d_px * self.normal[i])
        length = np.linalg.norm(delta)
        if length > self.radius[i]:
            delta *= self.radius[i] / length
        return self.C[i] + delta


class FixedLagTubeTracker:
    """Constant-memory max-product tracker with a bounded reporting delay."""

    def __init__(self, geometry: TubeGeometry, lag: int = 12,
                 sigma_px: float = SIGMA_PX, offset_step_sigma: float = 0.9):
        self.g = geometry
        self.lag = int(lag)
        self.sigma_px = float(sigma_px)
        self.offset_step_sigma = float(offset_step_sigma)
        self.velocity = np.linspace(-V_MAX, V_MAX, N_V)
        self.shift = np.rint(self.velocity / FPS / DS).astype(int)
        dv = self.velocity[:, None] - self.velocity[None, :]
        tv = np.exp(-0.5 * (dv / ACCEL_SIGMA) ** 2)
        tv /= tv.sum(axis=1, keepdims=True)
        self.log_tv = np.log(np.maximum(tv, 1e-300))
        self.S = len(self.g.s)
        self.V = len(self.velocity)
        self.delta = None
        self.back_ring = deque(maxlen=max(self.lag, 1))
        self.observations = []
        self.final_s_index = []
        self.final_d_mm = []
        self.previous_d = None

    def _advance(self):
        advected = np.full_like(self.delta, -np.inf)
        for j, shift in enumerate(self.shift):
            if shift == 0:
                advected[:, j] = self.delta[:, j]
            elif shift > 0:
                advected[shift:, j] = self.delta[:-shift, j]
            else:
                advected[:shift, j] = self.delta[-shift:, j]
        candidates = advected[:, :, None] + self.log_tv.T[None, :, :]
        back = np.argmax(candidates, axis=1).astype(np.uint8)
        best = np.take_along_axis(candidates, back[:, None, :], axis=1)[:, 0, :]
        return best, back

    def _current_path(self):
        """Return (s,v) indices from the oldest ring state to the current one."""
        si, vi = np.unravel_index(int(np.argmax(self.delta)), self.delta.shape)
        reverse_path = [(int(si), int(vi))]
        active_backs = list(self.back_ring)[-self.lag:] if self.lag else []
        for back in reversed(active_backs):
            v_previous = int(back[si, vi])
            # Advection happened with the PREVIOUS velocity bin.  The legacy
            # ArcHMM.viterbi subtracts shift[v_new] here, which is a subtle
            # backtracking bug: it can stitch together states that were never
            # connected by the forward recursion and therefore break v_max.
            si = int(np.clip(si - self.shift[v_previous], 0, self.S - 1))
            vi = v_previous
            reverse_path.append((si, vi))
        return list(reversed(reverse_path))

    def _offset_first(self, s_indices, observations):
        """Decode d on a short fixed-lag path and return its oldest value."""
        n = len(s_indices)
        d_count = len(self.g.fractions)
        emission = np.empty((n, d_count))
        for row, (si, uv) in enumerate(zip(s_indices, observations)):
            if uv is None or not np.isfinite(uv).all():
                emission[row] = self.g.log_offset_prior[si]
                continue
            d2 = ((self.g.offset_positions[si] - uv[None, :]) ** 2).sum(axis=1)
            emission[row] = (-0.5 * d2 / self.sigma_px ** 2 +
                             self.g.log_offset_prior[si])

        first_d = self.g.d_mm[s_indices[0]]
        if self.previous_d is None:
            score = emission[0] - 0.5 * (first_d / 1.5) ** 2
        else:
            score = (emission[0] - 0.5 *
                     ((first_d - self.previous_d) / self.offset_step_sigma) ** 2)
        backs = []
        for row in range(1, n):
            previous_d = self.g.d_mm[s_indices[row - 1]]
            current_d = self.g.d_mm[s_indices[row]]
            transition = -0.5 * ((current_d[None, :] - previous_d[:, None]) /
                                 self.offset_step_sigma) ** 2
            candidates = score[:, None] + transition
            back = np.argmax(candidates, axis=0)
            score = candidates[back, np.arange(d_count)] + emission[row]
            backs.append(back)
        state = int(np.argmax(score))
        for back in reversed(backs):
            state = int(back[state])
        value = float(first_d[state])
        self.previous_d = value
        return value

    def _finalise_oldest(self):
        path = self._current_path()
        # The ring path can begin before the target only during end flushing.
        target = len(self.final_s_index)
        current = len(self.observations) - 1
        path_start = current - len(path) + 1
        offset = max(0, target - path_start)
        path = path[offset:]
        observations = self.observations[target:current + 1]
        s_indices = [state[0] for state in path]
        if len(s_indices) != len(observations):
            raise RuntimeError("fixed-lag path/observation alignment failed")
        d_value = self._offset_first(s_indices, observations)
        self.final_s_index.append(s_indices[0])
        self.final_d_mm.append(d_value)
        return path[0]

    def _keep_committed_ancestor(self, committed_state):
        """Prevent later evidence from rewriting an already published state.

        Independent fixed-lag backtracks are individually feasible but their
        oldest states need not join into one feasible path.  Once a state is
        emitted, retain only current hypotheses whose ancestry passes through
        that exact state.  This is the online equivalent of committing the
        oldest node in a receding-horizon optimiser.
        """
        si = np.broadcast_to(np.arange(self.S)[:, None],
                             (self.S, self.V)).copy()
        vi = np.broadcast_to(np.arange(self.V)[None, :],
                             (self.S, self.V)).copy()
        active_backs = list(self.back_ring)[-self.lag:] if self.lag else []
        for back in reversed(active_backs):
            v_previous = back[si, vi].astype(int)
            si = np.clip(si - self.shift[v_previous], 0, self.S - 1)
            vi = v_previous
        # Only position is published.  Keeping every velocity hypothesis at
        # that position avoids artificial dead-ends at the centreline ends,
        # while the next committed position is still linked by one of the
        # model's bounded-velocity transitions.
        keep = si == committed_state[0]
        self.delta[~keep] = -np.inf
        if not np.any(keep):
            raise RuntimeError("no live hypothesis descends from committed state")
        self.delta -= np.max(self.delta)

    def update(self, uv):
        """Consume one frame and return the newly publishable state, if any.

        The returned dictionary is the actual streaming interface.  During
        warm-up (the first ``lag`` frames) it returns ``None``; afterwards it
        emits exactly one delayed estimate per input frame.
        """
        self.observations.append(np.asarray(uv, dtype=float))
        likelihood = self.g.tube_loglik(uv, self.sigma_px)
        if self.delta is None:
            self.delta = (np.full((self.S, self.V),
                                  -np.log(self.S * self.V)) + likelihood[:, None])
        else:
            self.delta, back = self._advance()
            self.back_ring.append(back)
            self.delta += likelihood[:, None]
        self.delta -= np.max(self.delta)
        if len(self.observations) - 1 >= self.lag:
            committed = self._finalise_oldest()
            self._keep_committed_ancestor(committed)
            frame = len(self.final_s_index) - 1
            si = self.final_s_index[-1]
            d_mm = self.final_d_mm[-1]
            xyz = self.g.lift(si, d_mm)
            return {"frame": frame, "s_mm": float(self.g.s[si]),
                    "offset_mm": float(d_mm), "xyz_mm": xyz,
                    "uv_px": self.g.project(xyz)}
        return None

    def finish(self):
        """Flush frames whose full lag was unavailable at stream end."""
        while len(self.final_s_index) < len(self.observations):
            self._finalise_oldest()
        s_index = np.asarray(self.final_s_index, dtype=int)
        d_mm = np.asarray(self.final_d_mm)
        xyz = np.stack([self.g.lift(si, d) for si, d in zip(s_index, d_mm)])
        projected = self.g.project(xyz)
        return s_index, d_mm, xyz, projected


def run_tracker(observations, geometry, lag):
    tracker = FixedLagTubeTracker(geometry, lag=lag)
    start = time.perf_counter()
    for uv in observations:
        tracker.update(uv)
    result = tracker.finish()
    elapsed_ms = 1000.0 * (time.perf_counter() - start) / len(observations)
    return result, elapsed_ms


def repeatability(s_path, observations, k_turn):
    outbound = np.arange(k_turn + 1)
    returning = np.arange(k_turn + 1, len(observations))
    image_distance, match = cKDTree(observations[outbound]).query(
        observations[returning])
    keep = image_distance < 8.0
    return np.abs(s_path[returning][keep] - s_path[outbound][match[keep]])


def save_csv(path, s_index, d_mm, xyz, projected, residual, radius,
             axis_sigma, unseen_radial_bound):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "time_s", "s_mm", "offset_mm", "x_mm",
                         "y_mm", "z_mm", "u_px", "v_px", "residual_px",
                         "axis_sigma_approx_mm", "unseen_radial_bound_mm",
                         "tube_radius_bound_mm"])
        for k in range(len(s_index)):
            writer.writerow([k, k / FPS, s_index[k] * DS, d_mm[k], *xyz[k],
                             *projected[k], residual[k], axis_sigma[k],
                             unseen_radial_bound[k], radius[s_index[k]]])


def make_figure(path, live_obs, oracle, s_path, d_mm, projected, residual,
                offline, lag, elapsed_ms, k_turn):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax = axes[0, 0]
    ax.plot(oracle[:, 0], oracle[:, 1], color="0.75", lw=2, label="offline trend")
    ax.plot(projected[:, 0], projected[:, 1], color="#1764ab", lw=1.2,
            label="V11 online estimate")
    ax.scatter(projected[::10, 0], projected[::10, 1], c=np.arange(len(projected))[::10],
               cmap="turbo", s=8)
    ax.invert_yaxis(); ax.set_aspect("equal"); ax.legend()
    ax.set_title("2D projection of the online 3D result")

    ax = axes[0, 1]
    ax.semilogy(np.maximum(residual, 0.1), lw=0.8, label="V11 vs offline trend")
    ax.semilogy(np.maximum(offline["rp_tb"], 0.1), lw=0.7, alpha=0.65,
                label="V8 full-video offline")
    ax.axhline(30, color="r", ls="--", lw=1)
    ax.axvline(k_turn, color="0.4", ls=":")
    ax.set_xlabel("frame"); ax.set_ylabel("pixel residual"); ax.legend()
    ax.set_title("Residual (trend is evaluation-only, never an online input)")

    ax = axes[1, 0]
    ax.plot(s_path, lw=1.0, label="arc length s")
    ax.axvline(k_turn, color="r", ls="--", label="turnaround")
    ax.set_xlabel("frame"); ax.set_ylabel("s (mm)"); ax.legend()
    ax.set_title(f"Speed-bounded fixed-lag path (lag={lag} frames)")

    ax = axes[1, 1]
    ax.plot(d_mm, lw=0.9, color="#a23b2a", label="lumen offset")
    ax.axhline(0, color="0.6", lw=0.7)
    ax.set_xlabel("frame"); ax.set_ylabel("signed offset (mm)"); ax.legend()
    ax.set_title("Online wall-following state")

    fig.suptitle(f"V11: real-time tube-aware 3D tracking | {elapsed_ms:.2f} ms/frame, "
                 f"{1000 / elapsed_ms:.0f} fps | latency {1000 * lag / FPS:.0f} ms",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def render_video(path, geometry, raw, accepted, projected, xyz, s_path,
                 d_mm, residual, lag):
    """Render an auditable side-by-side video without changing tracking."""
    cap = cv2.VideoCapture(str(VIDEO))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    canvas_w = W + 560
    writer = cv2.VideoWriter(str(path), fourcc, FPS, (canvas_w, H))

    # Fixed oblique display projection for the right-hand 3D panel.
    centre = geometry.C.mean(axis=0)
    view = Rot.from_euler("xyz", [25, 0, -55], degrees=True).as_matrix()
    vessel_2d = (geometry.C - centre) @ view.T
    scale = 4.2
    vessel_2d = vessel_2d[:, :2] * scale + np.array([280, H / 2])
    trail_3d = (xyz - centre) @ view.T
    trail_3d = trail_3d[:, :2] * scale + np.array([280, H / 2])

    route = np.rint(geometry.P).astype(np.int32)
    for k in range(len(projected)):
        ok, frame = cap.read()
        if not ok:
            break
        # A light linear contrast stretch keeps rendering comfortably faster
        # than acquisition; edge-preserving ``detailEnhance`` made video
        # export several times slower without changing any tracking result.
        frame = cv2.convertScaleAbs(frame, alpha=1.12, beta=-12)
        cv2.polylines(frame, [route], False, (35, 35, 35), 5, cv2.LINE_AA)
        cv2.polylines(frame, [route], False, (20, 220, 255), 2, cv2.LINE_AA)
        q = tuple(np.rint(projected[k]).astype(int))
        colour = (40, 190, 40) if residual[k] <= 30 else (20, 40, 230)
        cv2.circle(frame, q, 9, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(frame, q, 6, colour, -1, cv2.LINE_AA)
        if np.isfinite(raw[k]).all():
            r = tuple(np.rint(raw[k]).astype(int))
            cv2.drawMarker(frame, r, (255, 80, 20) if accepted[k] else (80, 80, 255),
                           cv2.MARKER_TILTED_CROSS, 14, 2, cv2.LINE_AA)
        cv2.putText(frame, f"LIVE frame {k:04d}  output delay {lag / FPS:.2f}s",
                    (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 3,
                    cv2.LINE_AA)
        cv2.putText(frame, f"LIVE frame {k:04d}  output delay {lag / FPS:.2f}s",
                    (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 1,
                    cv2.LINE_AA)

        panel = np.full((H, 560, 3), 248, np.uint8)
        cv2.polylines(panel, [np.rint(vessel_2d).astype(np.int32)], False,
                      (175, 175, 175), 2, cv2.LINE_AA)
        lo = max(0, k - 150)
        cv2.polylines(panel, [np.rint(trail_3d[lo:k + 1]).astype(np.int32)], False,
                      (175, 95, 35), 2, cv2.LINE_AA)
        point = tuple(np.rint(trail_3d[k]).astype(int))
        cv2.circle(panel, point, 8, (40, 70, 210), -1, cv2.LINE_AA)
        labels = ["REAL-TIME 3D ESTIMATE",
                  f"s = {s_path[k]:6.2f} mm",
                  f"offset = {d_mm[k]:+5.2f} mm",
                  f"x,y,z = ({xyz[k,0]:.1f}, {xyz[k,1]:.1f}, {xyz[k,2]:.1f}) mm",
                  f"residual = {residual[k]:.1f} px",
                  "circle: estimate   x: raw detection"]
        for row, label in enumerate(labels):
            cv2.putText(panel, label, (18, 35 + row * 31),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.57 if row else 0.68,
                        (35, 35, 35), 1, cv2.LINE_AA)
        writer.write(np.hstack([frame, panel]))
    cap.release(); writer.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lag", type=int, default=12,
                        help="future frames used by the online decoder (default: 12)")
    parser.add_argument("--render", action="store_true",
                        help="also render the auditable result video")
    args = parser.parse_args()

    live_obs, raw, confidence, accepted = prepare_live_observations()
    oracle_file = np.load(DATA / "track2d_trend.npz")
    oracle = oracle_file["trend"]
    k_turn = int(oracle_file["k_turn"])
    geometry = TubeGeometry()

    (s_index, d_mm, xyz, projected), elapsed_ms = run_tracker(
        live_obs, geometry, args.lag)
    s_path = geometry.s[s_index]
    residual = np.linalg.norm(projected - oracle, axis=1)
    residual_live = np.linalg.norm(projected - live_obs, axis=1)
    speed = np.abs(np.diff(s_path)) * FPS
    dsr = repeatability(s_path, oracle, k_turn)
    offline = np.load(DATA / "v8_tube.npz")
    xyz_delta = np.linalg.norm(xyz - offline["X_tube"], axis=1)
    axis_sigma = np.clip(SIGMA_PX / geometry.pixels_per_mm_s[s_index], 0.1, 10.0)
    transverse_norm = np.linalg.norm(xyz - geometry.C[s_index], axis=1)
    unseen_radial_bound = np.sqrt(np.maximum(
        geometry.radius[s_index] ** 2 - transverse_norm ** 2, 0.0))

    ret = np.arange(len(s_path)) > k_turn
    print("[V11] genuinely online input: raw detections -> hard gate -> causal IIR notch")
    print(f"[V11] accepted {accepted.sum()}/{len(accepted)} raw detections; "
          f"latency {args.lag} frames = {1000 * args.lag / FPS:.0f} ms")
    print(f"[V11] compute {elapsed_ms:.2f} ms/frame = {1000 / elapsed_ms:.0f} fps "
          f"({1000 / elapsed_ms / FPS:.1f}x acquisition rate)")
    print(f"[V11] vs offline trend (evaluation only): median {np.median(residual):.2f} px, "
          f"p90 {np.percentile(residual, 90):.2f} px, >30 px "
          f"{100 * np.mean(residual > 30):.2f}%")
    print(f"[V11] return leg: median {np.median(residual[ret]):.2f} px, >30 px "
          f"{100 * np.mean(residual[ret] > 30):.2f}%")
    print(f"[V11] physical checks: max axial speed {speed.max():.1f} mm/s, "
          f"violations {100 * np.mean(speed > V_MAX + 1e-6):.2f}%; "
          f"out/back median {np.median(dsr):.2f} mm")
    print(f"[V11] disagreement from full-video V8: median {np.median(xyz_delta):.2f} mm, "
          f"p90 {np.percentile(xyz_delta, 90):.2f} mm")
    print(f"[V11] honest single-view bound: unobserved radial component <= "
          f"{np.median(unseen_radial_bound):.2f} mm median / "
          f"{np.percentile(unseen_radial_bound, 90):.2f} mm p90")

    if speed.max() > V_MAX + 1e-6:
        raise AssertionError("fixed-lag path violated the configured speed bound")
    if not (np.isfinite(xyz).all() and np.isfinite(projected).all()):
        raise AssertionError("non-finite tracking output")
    if np.any(np.abs(d_mm) > geometry.radius[s_index] + 1e-9):
        raise AssertionError("decoded point left the measured lumen radius")

    np.savez(DATA / "v11_realtime.npz", s_index=s_index, s=s_path,
             d_mm=d_mm, X=xyz, Q=projected, residual=residual,
             residual_live=residual_live, speed=speed, repeatability=dsr,
             live_observation=live_obs, raw_observation=raw,
             accepted=accepted, lag=args.lag, elapsed_ms=elapsed_ms,
             xyz_delta_v8=xyz_delta, axis_sigma_approx=axis_sigma,
             unseen_radial_bound=unseen_radial_bound)
    save_csv(OUT / "v11_realtime_tracking.csv", s_index, d_mm, xyz,
             projected, residual, geometry.radius, axis_sigma,
             unseen_radial_bound)
    figure_path = OUT / "figs" / "v11_realtime.png"
    make_figure(figure_path, live_obs, oracle, s_path, d_mm, projected,
                residual, offline, args.lag, elapsed_ms, k_turn)
    print(f"[V11] saved {DATA / 'v11_realtime.npz'}")
    print(f"[V11] saved {OUT / 'v11_realtime_tracking.csv'}")
    print(f"[V11] saved {figure_path}")

    if args.render:
        video_path = OUT / "figs" / "V11_realtime_tracking.mp4"
        render_video(video_path, geometry, raw, accepted, projected, xyz,
                     s_path, d_mm, residual, args.lag)
        print(f"[V11] saved {video_path}")


if __name__ == "__main__":
    main()
