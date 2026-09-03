"""Stage E - map a 2D detection stream to arc length along the 3D centerline.

State is (s, v): arc-length position and arc-length velocity. Restricting the
robot to the known centerline collapses the ill-posed 2D->3D back-projection to
a 1-D estimation problem, and the velocity component is what resolves the
depth ambiguity at projected self-crossings: at a crossing the two candidate
arc lengths are equally likely from the current frame alone, but only one is
reachable given where the robot was and how fast it was going.

Provides:
  forward()  - causal filtering, this is what runs in real time
  viterbi()  - globally optimal path, for offline accuracy assessment / ground truth

Both are O(n_s * n_v * k) with a banded transition, i.e. real time at 30 fps.
"""
import numpy as np


class ArcHMM:
    def __init__(self, s_grid, proj_xy, sigma_px=6.0, v_max=40.0, n_v=41,
                 accel_sigma=25.0, p_lost=0.02):
        """
        s_grid  (S,)   arc-length grid, mm, uniform
        proj_xy (S,2)  image position of the centerline at each grid point, px
        sigma_px       detection noise std, px
        v_max          max |ds/dt|, mm/s
        accel_sigma    process noise on velocity, mm/s per second
        p_lost         per-frame probability the detection is an outlier
        """
        self.s = s_grid; self.P = proj_xy
        self.S = len(s_grid); self.ds = s_grid[1] - s_grid[0]
        self.v = np.linspace(-v_max, v_max, n_v); self.V = n_v
        self.sigma = sigma_px; self.accel = accel_sigma; self.p_lost = p_lost
        # velocity transition matrix (random walk on v), independent of s
        dv = self.v[:, None] - self.v[None, :]
        self.Tv = np.exp(-0.5 * (dv / accel_sigma) ** 2)
        self.Tv /= self.Tv.sum(1, keepdims=True)

    def loglik(self, uv):
        """(S,) log p(uv | s), with a uniform outlier component."""
        if uv is None or not np.isfinite(uv).all():
            return np.zeros(self.S)
        d2 = ((self.P - uv) ** 2).sum(1)
        l = np.exp(-0.5 * d2 / self.sigma ** 2)
        return np.log((1 - self.p_lost) * l / (2 * np.pi * self.sigma ** 2)
                      + self.p_lost / (960.0 * 720.0))

    def _advect(self, logp, dt):
        """Apply s <- s + v*dt for every velocity bin, then mix velocities."""
        shift = np.rint(self.v * dt / self.ds).astype(int)
        out = np.full_like(logp, -np.inf)
        for j, sh in enumerate(shift):
            if sh == 0:
                out[:, j] = logp[:, j]
            elif sh > 0:
                out[sh:, j] = logp[:-sh, j]
            else:
                out[:sh, j] = logp[-sh:, j]
        # mix over velocity (log-domain matmul)
        m = out.max()
        if not np.isfinite(m): m = 0.0
        p = np.exp(out - m) @ self.Tv.T
        return np.log(np.maximum(p, 1e-300)) + m

    def forward(self, UV, dt, s0=None):
        """Causal filter. Returns (s_hat, s_std, logZ) per frame."""
        K = len(UV)
        logp = np.full((self.S, self.V), -np.log(self.S * self.V))
        if s0 is not None:
            i0 = int(np.argmin(np.abs(self.s - s0)))
            logp[:] = -1e6; logp[max(0, i0-10):i0+10, :] = 0.0
        s_hat = np.zeros(K); s_std = np.zeros(K); v_hat = np.zeros(K)
        for k in range(K):
            if k > 0:
                logp = self._advect(logp, dt)
            logp = logp + self.loglik(UV[k])[:, None]
            logp -= logp.max()
            w = np.exp(logp); w /= w.sum()
            ps = w.sum(1)
            s_hat[k] = (ps * self.s).sum()
            s_std[k] = np.sqrt(np.maximum((ps * (self.s - s_hat[k]) ** 2).sum(), 0))
            v_hat[k] = (w.sum(0) * self.v).sum()
            logp = np.log(np.maximum(w, 1e-300))
        return s_hat, s_std, v_hat

    def viterbi(self, UV, dt):
        """Globally optimal (s,v) sequence. Offline reference."""
        K = len(UV)
        shift = np.rint(self.v * dt / self.ds).astype(int)
        logTv = np.log(np.maximum(self.Tv, 1e-300))
        delta = np.full((self.S, self.V), -np.log(self.S * self.V)) + self.loglik(UV[0])[:, None]
        back = np.zeros((K, self.S, self.V), np.int32)
        for k in range(1, K):
            # advect
            adv = np.full((self.S, self.V), -np.inf)
            for j, sh in enumerate(shift):
                if sh == 0: adv[:, j] = delta[:, j]
                elif sh > 0: adv[sh:, j] = delta[:-sh, j]
                else: adv[:sh, j] = delta[-sh:, j]
            # best previous velocity for each (s, v_new)
            cand = adv[:, :, None] + logTv.T[None, :, :]      # (S, v_prev, v_new)
            jbest = np.argmax(cand, axis=1)
            best = np.take_along_axis(cand, jbest[:, None, :], 1)[:, 0, :]
            back[k] = jbest
            delta = best + self.loglik(UV[k])[:, None]
            delta -= delta.max()
        # backtrack
        si, vi = np.unravel_index(int(np.argmax(delta)), delta.shape)
        path = np.zeros(K, int)
        for k in range(K - 1, -1, -1):
            path[k] = si
            if k == 0: break
            vprev = back[k, si, vi]
            si = int(np.clip(si - shift[vi], 0, self.S - 1)); vi = int(vprev)
        return self.s[path]


class MultiViewArcHMM:
    """Arc-length HMM that fuses N simultaneous views, and allows the projection
    to change every frame (a rotating C-arm).

    Same (s, sdot) state as ArcHMM. The only change is the measurement model:
    views are conditionally independent given s, so their log-likelihoods add.
    That is what lets a second view kill the self-crossing ambiguity - a pair of
    arc lengths that collide in one projection almost never collides in both.
    """

    def __init__(self, s_grid, sigma_px=6.0, v_max=25.0, n_v=41,
                 accel_sigma=18.0, p_lost=0.02, img_area=960.0 * 720.0):
        self.s = s_grid
        self.S = len(s_grid); self.ds = s_grid[1] - s_grid[0]
        self.v = np.linspace(-v_max, v_max, n_v); self.V = n_v
        self.sigma = sigma_px; self.p_lost = p_lost; self.img_area = img_area
        dv = self.v[:, None] - self.v[None, :]
        self.Tv = np.exp(-0.5 * (dv / accel_sigma) ** 2)
        self.Tv /= self.Tv.sum(1, keepdims=True)

    def _loglik(self, uvs, projs):
        """Sum over views. uvs[i] may be None (that view lost this frame)."""
        out = np.zeros(self.S)
        for uv, P in zip(uvs, projs):
            if uv is None or P is None or not np.isfinite(uv).all():
                continue
            d2 = ((P - uv) ** 2).sum(1)
            l = np.exp(-0.5 * d2 / self.sigma ** 2) / (2 * np.pi * self.sigma ** 2)
            out += np.log((1 - self.p_lost) * l + self.p_lost / self.img_area)
        return out

    def _advect(self, logp, dt):
        shift = np.rint(self.v * dt / self.ds).astype(int)
        out = np.full_like(logp, -np.inf)
        for j, sh in enumerate(shift):
            if sh == 0: out[:, j] = logp[:, j]
            elif sh > 0: out[sh:, j] = logp[:-sh, j]
            else: out[:sh, j] = logp[-sh:, j]
        m = out.max()
        if not np.isfinite(m): m = 0.0
        return np.log(np.maximum(np.exp(out - m) @ self.Tv.T, 1e-300)) + m

    def forward(self, obs, proj_fn, dt):
        """obs      : (K, n_views, 2) with nan for a lost view
           proj_fn  : k -> list of (S,2) projections of the centerline grid
        """
        K = len(obs)
        logp = np.full((self.S, self.V), -np.log(self.S * self.V))
        s_hat = np.zeros(K); s_std = np.zeros(K); logZ = 0.0
        for k in range(K):
            if k > 0: logp = self._advect(logp, dt)
            logp = logp + self._loglik(obs[k], proj_fn(k))[:, None]
            m = logp.max(); logp -= m
            w = np.exp(logp); tot = w.sum(); w /= tot
            logZ += m + np.log(tot)          # model evidence, for selecting (r, phi)
            ps = w.sum(1)
            s_hat[k] = (ps * self.s).sum()
            s_std[k] = np.sqrt(max((ps * (self.s - s_hat[k]) ** 2).sum(), 0.0))
            logp = np.log(np.maximum(w, 1e-300))
        return s_hat, s_std, logZ
