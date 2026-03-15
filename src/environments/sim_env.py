"""
Lightweight Physics Simulation Environment
==========================================
Self-contained simulation (no Isaac Lab required) for rapid prototyping,
unit testing, and CI/CD. Models each joint as a second-order system with
stiffness, damping, and random perturbations (simulating model uncertainty).

Also acts as a drop-in replacement when Isaac Lab is unavailable,
ensuring the full pipeline can be tested on any machine.
"""

import numpy as np
from typing import Tuple
from .base_env import BaseRobotEnv, EnvConfig, RobotState


class JointModel:
    """
    Second-order joint dynamics: m*q'' + b*q' + k*q = k*u + d(t)
    Discretised via semi-implicit Euler.
    Models the uncertain, nonlinear dynamics mentioned in the thesis.
    """

    def __init__(self, mass: float = 0.5, damping: float = 2.0,
                 stiffness: float = 50.0, noise_std: float = 0.01,
                 dt: float = 0.01):
        self.m = mass
        self.b = damping
        self.k = stiffness
        self.noise_std = noise_std
        self.dt = dt
        self.q = 0.0
        self.dq = 0.0
        # Random perturbation parameters (simulate unmodeled dynamics)
        self._delta_k = np.random.uniform(0.85, 1.15)  # ±15% stiffness uncertainty
        self._delta_b = np.random.uniform(0.90, 1.10)

    def reset(self, q0: float = 0.0):
        self.q = q0
        self.dq = 0.0

    def step(self, u: float, disturbance: float = 0.0) -> Tuple[float, float]:
        """Advance by one timestep. Returns (q, dq)."""
        k_eff = self.k * self._delta_k
        b_eff = self.b * self._delta_b
        noise = np.random.normal(0, self.noise_std)

        acc = (k_eff * (u - self.q) - b_eff * self.dq + disturbance + noise) / self.m
        self.dq += acc * self.dt
        self.q  += self.dq * self.dt
        return self.q, self.dq


class LightweightSimEnv(BaseRobotEnv):
    """
    Lightweight simulation of a legged robot.
    Each joint is modelled independently as a second-order system.
    Base motion is approximated from joint kinematics.

    Use this environment for:
    - Fast algorithm prototyping on any machine
    - Unit testing and CI/CD
    - Benchmarking ILC convergence without Isaac Lab
    """

    def __init__(self, config: EnvConfig, gait: str = "trot",
                 perturbation_std: float = 0.05):
        super().__init__(config)
        self.gait = gait
        self.perturbation_std = perturbation_std
        self._joints = [
            JointModel(noise_std=0.01, dt=config.dt)
            for _ in range(config.n_joints)
        ]
        self._ref_traj = self._generate_gait_trajectory()
        self._t = 0

    # ── Reference trajectory generation ──────────────────────────────────────

    def _generate_gait_trajectory(self) -> np.ndarray:
        """
        Generate a periodic gait reference trajectory.
        Supports trot, walk, and bound patterns.
        """
        H = self.cfg.episode_length
        n = self.cfg.n_joints
        t = np.linspace(0, 2 * np.pi, H)
        ref = np.zeros((H, n))

        if self.gait == "trot":
            # Diagonal pair synchronisation (FL+RR, FR+RL)
            phases = [0, np.pi, np.pi, 0,      # hip abduction
                      0, np.pi, np.pi, 0,       # hip flexion
                      0, np.pi, np.pi, 0]       # knee
            for j in range(n):
                amp = 0.4 if j % 3 == 1 else 0.15   # larger hip flexion
                ref[:, j] = amp * np.sin(t + phases[j % len(phases)])

        elif self.gait == "walk":
            phases = [0, np.pi/2, np.pi, 3*np.pi/2] * (n // 4 + 1)
            for j in range(n):
                ref[:, j] = 0.3 * np.sin(t + phases[j % 4])

        elif self.gait == "bound":
            # Front pair / rear pair alternating
            for j in range(n):
                phase = 0 if j < n // 2 else np.pi
                ref[:, j] = 0.5 * np.sin(2 * t + phase)

        else:
            ref = np.zeros((H, n))

        return ref

    # ── BaseRobotEnv interface ────────────────────────────────────────────────

    def reset(self) -> RobotState:
        self._t = 0
        for jm in self._joints:
            jm.reset()
        return self._build_state()

    def step(self, action: np.ndarray) -> Tuple[RobotState, float, bool, dict]:
        assert len(action) == self.cfg.n_joints

        # External perturbation (terrain, wind, payload)
        disturbance = np.random.normal(0, self.perturbation_std, self.cfg.n_joints)

        for j, jm in enumerate(self._joints):
            jm.step(action[j], disturbance[j])

        self._t += 1
        state = self._build_state()
        ref = self._ref_traj[min(self._t, len(self._ref_traj) - 1)]
        reward = self.tracking_reward(state, ref)
        done = self._t >= self.cfg.episode_length

        info = {'t': self._t, 'disturbance_norm': float(np.linalg.norm(disturbance))}
        return state, reward, done, info

    def get_reference_trajectory(self) -> np.ndarray:
        return self._ref_traj.copy()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_state(self) -> RobotState:
        q   = np.array([jm.q  for jm in self._joints])
        dq  = np.array([jm.dq for jm in self._joints])
        t_idx = min(self._t, self.cfg.episode_length - 1)

        # Approximate base height from average leg extension
        base_z = 0.35 + 0.05 * np.mean(np.abs(q[:4]))
        contact = np.ones(self.cfg.n_feet) * max(0.0, base_z - 0.30)

        return RobotState(
            joint_pos=q,
            joint_vel=dq,
            base_pos=np.array([0.0, 0.0, base_z]),
            base_quat=np.array([0.0, 0.0, 0.0, 1.0]),
            base_lin_vel=np.zeros(3),
            base_ang_vel=np.zeros(3),
            contact_forces=contact,
            timestamp=t_idx * self.cfg.dt,
        )
