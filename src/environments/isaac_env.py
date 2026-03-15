"""
Isaac Lab Environment Wrapper
==============================
Wraps the Isaac Lab / IsaacGym-based Unitree Go2 (or humanoid) environment
to conform to our BaseRobotEnv interface.

Usage
-----
This module is imported ONLY when Isaac Lab is available.
All other code uses LightweightSimEnv as fallback.

Supported robots
----------------
- unitree_go2      : quadruped (12 DOF) — confirmed working
- unitree_h1       : humanoid  (19 DOF)
- unitree_g1       : humanoid  (23 DOF)
- tienkung         : TienKung from Open-X-Humanoid project

Isaac Lab installation
----------------------
Follow: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation
Then: pip install -e .  inside the IsaacLab repo.
"""

import numpy as np
from typing import Tuple, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .base_env import BaseRobotEnv, EnvConfig, RobotState


class IsaacLabEnv(BaseRobotEnv):
    """
    Isaac Lab wrapper for legged robot environments.

    Bridges the IsaacLab DirectRL API to our BaseRobotEnv interface,
    enabling the ILC algorithms to run directly in physics simulation
    with GPU-accelerated rigid body dynamics.

    Isaac Lab gives us:
    - Accurate contact dynamics (crucial for gait learning)
    - GPU parallelism (run 4096 envs simultaneously)
    - Sim-to-real transfer support
    - TienKung / Unitree robot models out of the box

    The ILC feedforward signal is added on top of a base PD feedback
    controller, forming a 'feedforward + feedback' architecture that
    mirrors what the thesis describes.
    """

    def __init__(self, config: EnvConfig,
                 num_envs: int = 1,
                 headless: bool = False,
                 device: str = "cuda:0"):
        super().__init__(config)
        self.num_envs = num_envs
        self.headless = headless
        self.device = device
        self._env = None
        self._ref_traj: Optional[np.ndarray] = None
        self._t = 0
        self._initialised = False

    def _lazy_init(self):
        """Deferred init — Isaac Lab takes ~30s to load."""
        if self._initialised:
            return
        try:
            # Isaac Lab imports (only available in Ubuntu 22.04 + Isaac install)
            import isaaclab  # noqa: F401
            from isaaclab_tasks.manager_based.locomotion.velocity.config import (
                unitree_go2 as go2_cfg
            )
            from isaaclab.envs import ManagerBasedRLEnv

            cfg = go2_cfg.UnitreeGo2RoughEnvCfg()
            cfg.scene.num_envs = self.num_envs
            self._env = ManagerBasedRLEnv(cfg=cfg)
            self._initialised = True
            print(f"[IsaacLabEnv] Loaded {self.cfg.robot_name} "
                  f"({self.num_envs} envs, device={self.device})")
        except ImportError as e:
            raise ImportError(
                f"Isaac Lab not found: {e}\n"
                "Install via: https://isaac-sim.github.io/IsaacLab\n"
                "Falling back to LightweightSimEnv is recommended."
            )

    def reset(self) -> RobotState:
        self._lazy_init()
        self._t = 0
        obs, _ = self._env.reset()
        return self._obs_to_state(obs)

    def step(self, action: np.ndarray) -> Tuple[RobotState, float, bool, dict]:
        self._lazy_init()
        assert self._env is not None

        if TORCH_AVAILABLE:
            import torch
            action_t = torch.tensor(action, dtype=torch.float32,
                                    device=self.device).unsqueeze(0)
            obs, reward, terminated, truncated, info = self._env.step(action_t)
            reward_val = float(reward[0].cpu())
            done = bool(terminated[0] or truncated[0])
        else:
            raise RuntimeError("PyTorch required for Isaac Lab environments.")

        self._t += 1
        state = self._obs_to_state(obs)
        return state, reward_val, done, info

    def get_reference_trajectory(self) -> np.ndarray:
        if self._ref_traj is None:
            # Default: sinusoidal trot gait
            H, n = self.cfg.episode_length, self.cfg.n_joints
            t = np.linspace(0, 2 * np.pi, H)
            ref = np.zeros((H, n))
            phases = [0, np.pi, np.pi, 0] * (n // 4 + 1)
            for j in range(n):
                amp = 0.4 if j % 3 == 1 else 0.15
                ref[:, j] = amp * np.sin(t + phases[j % 4])
            self._ref_traj = ref
        return self._ref_traj.copy()

    def _obs_to_state(self, obs) -> RobotState:
        """Extract structured state from Isaac Lab observation tensor."""
        if TORCH_AVAILABLE:
            import torch
            o = obs[0].cpu().numpy() if isinstance(obs, torch.Tensor) else obs
        else:
            o = np.array(obs)

        n = self.cfg.n_joints
        # Isaac Lab observation layout: [base_lin_vel(3), base_ang_vel(3),
        #   gravity(3), commands(3), joint_pos(n), joint_vel(n), actions(n), ...]
        base_lin_vel = o[0:3]
        base_ang_vel = o[3:6]
        joint_pos    = o[12:12 + n]
        joint_vel    = o[12 + n:12 + 2 * n]

        return RobotState(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            base_pos=np.array([0.0, 0.0, 0.35]),
            base_quat=np.array([0.0, 0.0, 0.0, 1.0]),
            base_lin_vel=base_lin_vel,
            base_ang_vel=base_ang_vel,
            contact_forces=np.ones(self.cfg.n_feet),
            timestamp=self._t * self.cfg.dt,
        )

    def close(self):
        if self._env is not None:
            self._env.close()


def make_env(backend: str = "sim", config: Optional[EnvConfig] = None,
             **kwargs) -> BaseRobotEnv:
    """
    Factory for environment selection.

    backend : 'isaac' | 'sim'
        'isaac' — requires Isaac Lab installation (Ubuntu 22.04 + GPU)
        'sim'   — lightweight built-in simulation (any machine)
    """
    if config is None:
        config = EnvConfig()

    if backend == "isaac":
        return IsaacLabEnv(config, **kwargs)
    elif backend == "sim":
        from .sim_env import LightweightSimEnv
        return LightweightSimEnv(config, **kwargs)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Use 'isaac' or 'sim'.")
