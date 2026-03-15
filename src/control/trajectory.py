"""
Reference Trajectory Generation
=================================
Generates smooth, physically feasible reference joint trajectories
for learning experiments. Supports multiple gait patterns and
interpolation methods.
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Optional


class GaitType(str, Enum):
    TROT   = "trot"
    WALK   = "walk"
    BOUND  = "bound"
    PACE   = "pace"
    STAND  = "stand"
    CUSTOM = "custom"


@dataclass
class TrajectoryConfig:
    n_joints: int = 12
    horizon: int = 500
    dt: float = 0.01
    gait: GaitType = GaitType.TROT
    stride_freq: float = 2.0   # Hz
    amplitude: float = 0.4     # rad, peak hip flexion
    smoothing: bool = True     # apply Gaussian smoothing


class TrajectoryGenerator:
    """
    Generates periodic gait reference trajectories for quadruped/humanoid.

    Joint ordering (Unitree Go2 / standard humanoid legs):
    [0-2]  FL: hip_ab, hip_fe, knee
    [3-5]  FR: hip_ab, hip_fe, knee
    [6-8]  RL: hip_ab, hip_fe, knee
    [9-11] RR: hip_ab, hip_fe, knee
    """

    def __init__(self, config: TrajectoryConfig):
        self.cfg = config

    def generate(self, gait: Optional[GaitType] = None) -> np.ndarray:
        """Returns reference trajectory of shape (horizon, n_joints)."""
        g = gait or self.cfg.gait
        t = np.linspace(0, self.cfg.horizon * self.cfg.dt, self.cfg.horizon)
        omega = 2 * np.pi * self.cfg.stride_freq

        generators = {
            GaitType.TROT:  self._trot,
            GaitType.WALK:  self._walk,
            GaitType.BOUND: self._bound,
            GaitType.PACE:  self._pace,
            GaitType.STAND: self._stand,
        }
        gen = generators.get(g, self._trot)
        ref = gen(t, omega)

        if self.cfg.smoothing:
            ref = self._smooth(ref)

        return ref

    # ── Gait patterns ──────────────────────────────────────────────────────────

    def _trot(self, t: np.ndarray, w: float) -> np.ndarray:
        """Diagonal pair synchronisation: (FL+RR) vs (FR+RL)."""
        n, H = self.cfg.n_joints, len(t)
        ref = np.zeros((H, n))
        # Phase offsets per leg (0=FL, 1=FR, 2=RL, 3=RR)
        leg_phase = [0.0, np.pi, np.pi, 0.0]
        a = self.cfg.amplitude
        for leg in range(4):
            ph = leg_phase[leg]
            j = leg * 3
            ref[:, j]     = 0.05 * np.sin(w * t + ph)          # hip abduction
            ref[:, j + 1] = a * np.sin(w * t + ph)             # hip flexion
            ref[:, j + 2] = -0.5 * a * np.abs(np.sin(w * t + ph))  # knee
        return ref

    def _walk(self, t: np.ndarray, w: float) -> np.ndarray:
        """Four-beat walk: FL → RR → FR → RL."""
        n, H = self.cfg.n_joints, len(t)
        ref = np.zeros((H, n))
        leg_phase = [0.0, np.pi/2, np.pi, 3*np.pi/2]
        a = self.cfg.amplitude * 0.6  # smaller amplitude for stability
        for leg in range(4):
            ph = leg_phase[leg]
            j = leg * 3
            ref[:, j + 1] = a * np.sin(w * t + ph)
            ref[:, j + 2] = -0.4 * a * np.abs(np.sin(w * t + ph))
        return ref

    def _bound(self, t: np.ndarray, w: float) -> np.ndarray:
        """Bounding: front pair / rear pair alternating."""
        n, H = self.cfg.n_joints, len(t)
        ref = np.zeros((H, n))
        a = self.cfg.amplitude * 1.2
        for leg in range(4):
            ph = 0.0 if leg < 2 else np.pi
            j = leg * 3
            ref[:, j + 1] = a * np.sin(2 * w * t + ph)
            ref[:, j + 2] = -0.6 * a * np.abs(np.sin(2 * w * t + ph))
        return ref

    def _pace(self, t: np.ndarray, w: float) -> np.ndarray:
        """Pace: ipsilateral pair synchronisation (FL+RL, FR+RR)."""
        n, H = self.cfg.n_joints, len(t)
        ref = np.zeros((H, n))
        # FL=0,FR=1,RL=2,RR=3 → FL+RL same phase, FR+RR same phase
        leg_phase = [0.0, np.pi, 0.0, np.pi]
        a = self.cfg.amplitude
        for leg in range(4):
            ph = leg_phase[leg]
            j = leg * 3
            ref[:, j + 1] = a * np.sin(w * t + ph)
            ref[:, j + 2] = -0.5 * a * np.abs(np.sin(w * t + ph))
        return ref

    def _stand(self, t: np.ndarray, w: float) -> np.ndarray:
        """Standing pose: zero trajectory (debugging / initial learning)."""
        return np.zeros((len(t), self.cfg.n_joints))

    # ── Utilities ──────────────────────────────────────────────────────────────

    def _smooth(self, ref: np.ndarray, sigma: int = 3) -> np.ndarray:
        """Apply Gaussian smoothing along time axis per joint."""
        from scipy.ndimage import gaussian_filter1d
        try:
            return gaussian_filter1d(ref, sigma=sigma, axis=0)
        except ImportError:
            # Fallback: simple moving average
            kernel = np.ones(sigma * 2 + 1) / (sigma * 2 + 1)
            return np.apply_along_axis(
                lambda x: np.convolve(x, kernel, mode='same'), 0, ref)

    def interpolate(self, ref: np.ndarray,
                    target_horizon: int) -> np.ndarray:
        """Resample trajectory to a different horizon length."""
        H_src = len(ref)
        t_src = np.linspace(0, 1, H_src)
        t_tgt = np.linspace(0, 1, target_horizon)
        result = np.zeros((target_horizon, ref.shape[1]))
        for j in range(ref.shape[1]):
            result[:, j] = np.interp(t_tgt, t_src, ref[:, j])
        return result
