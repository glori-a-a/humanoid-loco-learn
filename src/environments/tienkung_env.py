"""
TienKung Humanoid Environment Wrapper
=======================================
Integrates the TienKung humanoid robot (Open-X-Humanoid/TienKung-Lab)
with the HumanoidLocoLearn ILC pipeline.

TienKung-Lab uses Isaac Lab 2.1.0 + Isaac Sim 4.5.0 and an AMP-based
(Adversarial Motion Prior) RL policy for base locomotion.

This wrapper supports two usage modes:

Mode A — ILC-only (no base policy):
    The ILC feedforward signal directly commands joint positions.
    Use this for learning from scratch on TienKung's dynamics.

Mode B — ILC residual correction (recommended for thesis):
    A pre-trained TienKung AMP policy provides base joint targets.
    The ILC learns a residual correction signal on top:
        u_total = u_amp(t) + u_ilc(t)
    This compensates for systematic errors the base policy cannot fix —
    the core contribution of "data-driven learning control".

Setup
-----
1. Install TienKung-Lab:
   git clone https://github.com/Open-X-Humanoid/TienKung-Lab
   cd TienKung-Lab && pip install -e .

2. Verify Isaac Lab 2.1.0 + Isaac Sim 4.5.0 is installed.

3. Run with:
   python scripts/run_ilc.py --backend tienkung --algo noilc --gait walk

TienKung Joint Layout (23 DOF — verify against URDF)
-----------------------------------------------------
Left leg  (0-5):  hip_roll, hip_yaw, hip_pitch, knee, ankle_pitch, ankle_roll
Right leg (6-11): hip_roll, hip_yaw, hip_pitch, knee, ankle_pitch, ankle_roll
Waist     (12):   waist_yaw
Left arm  (13-17): shoulder_pitch, shoulder_roll, elbow, wrist_yaw, wrist_pitch
Right arm (18-22): shoulder_pitch, shoulder_roll, elbow, wrist_yaw, wrist_pitch

For locomotion ILC, we typically control legs only (12 DOF).
Arms and waist use fixed nominal poses or a separate balance controller.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from .base_env import BaseRobotEnv, EnvConfig, RobotState


# ── TienKung joint definitions ─────────────────────────────────────────────────

TIENKUNG_JOINT_NAMES = [
    # Left leg
    "left_hip_roll_joint", "left_hip_yaw_joint", "left_hip_pitch_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    # Right leg
    "right_hip_roll_joint", "right_hip_yaw_joint", "right_hip_pitch_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    # Waist
    "waist_yaw_joint",
    # Left arm
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_elbow_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint",
    # Right arm
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint",
]

# Joint limits (rad) — approximate, verify against URDF
TIENKUNG_JOINT_LIMITS = {
    "hip_roll":       (-0.5,  0.5),
    "hip_yaw":        (-0.5,  0.5),
    "hip_pitch":      (-1.2,  0.6),
    "knee":           (-0.1,  2.0),
    "ankle_pitch":    (-0.8,  0.5),
    "ankle_roll":     (-0.4,  0.4),
    "waist_yaw":      (-0.8,  0.8),
    "shoulder_pitch": (-1.5,  1.5),
    "shoulder_roll":  (-1.0,  1.0),
    "elbow":          (-2.0,  0.0),
    "wrist_yaw":      (-1.0,  1.0),
    "wrist_pitch":    (-0.8,  0.8),
}

# Default standing pose (nominal joint angles for balance)
TIENKUNG_STAND_POSE = np.array([
    # Left leg
     0.0,  0.0, -0.3,  0.6, -0.3,  0.0,
    # Right leg
     0.0,  0.0, -0.3,  0.6, -0.3,  0.0,
    # Waist
     0.0,
    # Left arm (relaxed)
     0.0, -0.2, -0.5,  0.0,  0.0,
    # Right arm (relaxed)
     0.0,  0.2, -0.5,  0.0,  0.0,
], dtype=np.float32)

# Locomotion-only joints (legs only, indices 0-11)
LOCOMOTION_DOF = list(range(12))


@dataclass
class TienKungConfig(EnvConfig):
    n_joints: int = 23              # full body
    n_loco_joints: int = 12        # legs only for ILC
    n_feet: int = 2                 # bipedal
    dt: float = 0.02                # 50 Hz (TienKung-Lab default)
    episode_length: int = 400
    robot_name: str = "tienkung"
    use_base_policy: bool = False   # Mode A (False) or Mode B (True)
    policy_checkpoint: str = ""     # path to AMP policy checkpoint


class TienKungEnv(BaseRobotEnv):
    """
    TienKung humanoid environment for ILC experiments.

    In Mode B (use_base_policy=True), this implements the thesis core idea:
        u_total(t) = u_base_policy(t) + u_ilc(t)

    where u_ilc is learned by the ILC algorithm to compensate for
    the systematic residual errors of the AMP-trained base policy.

    This is the "data-driven learning control" paradigm:
    - Base policy: handles general locomotion (RL-learned)
    - ILC layer:   compensates for task-specific errors (data-driven)

    Research question (for your thesis):
        "Can ILC effectively compensate for residual errors in AMP-based
         humanoid locomotion, and how does convergence rate depend on
         the quality of the base policy?"
    """

    def __init__(self, config: TienKungConfig = None,
                 headless: bool = True):
        if config is None:
            config = TienKungConfig()
        super().__init__(config)
        self.tk_cfg = config
        self.headless = headless
        self._env = None
        self._base_policy = None
        self._initialised = False
        self._t = 0

        # For offline testing without Isaac Lab
        self._fallback_mode = False

    def _lazy_init(self):
        if self._initialised:
            return
        try:
            self._init_isaac()
        except ImportError:
            print("[TienKungEnv] Isaac Lab not found — using fallback simulation.")
            print("[TienKungEnv] Install TienKung-Lab for full functionality.")
            self._init_fallback()

    def _init_isaac(self):
        """Initialise TienKung via Isaac Lab."""
        import isaaclab  # noqa

        # Try TienKung-Lab direct import
        try:
            from tienkung_lab.envs import TienKungLocomotionEnv
            self._env = TienKungLocomotionEnv(
                num_envs=1,
                headless=self.headless
            )
            print("[TienKungEnv] TienKung-Lab environment loaded.")
        except ImportError:
            # Fall back to Isaac Lab generic humanoid
            from isaaclab_tasks.manager_based.locomotion.velocity.config import (
                unitree_h1 as h1_cfg
            )
            from isaaclab.envs import ManagerBasedRLEnv
            cfg = h1_cfg.H1RoughEnvCfg()
            cfg.scene.num_envs = 1
            self._env = ManagerBasedRLEnv(cfg=cfg)
            print("[TienKungEnv] Using Unitree H1 as TienKung proxy.")

        # Load base policy if in Mode B
        if self.tk_cfg.use_base_policy and self.tk_cfg.policy_checkpoint:
            self._load_base_policy(self.tk_cfg.policy_checkpoint)

        self._initialised = True

    def _init_fallback(self):
        """Fallback: use lightweight sim with humanoid-like dynamics."""
        from .sim_env import LightweightSimEnv
        fallback_cfg = EnvConfig(
            n_joints=self.tk_cfg.n_joints,
            episode_length=self.tk_cfg.episode_length,
            dt=self.tk_cfg.dt
        )
        self._fallback_env = LightweightSimEnv(fallback_cfg, gait='walk')
        self._fallback_mode = True
        self._initialised = True

    def _load_base_policy(self, checkpoint_path: str):
        """Load pre-trained AMP policy from TienKung-Lab checkpoint."""
        try:
            import torch
            # TienKung-Lab uses RSL-RL or similar checkpoint format
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            # Extract actor network weights
            if 'model_state_dict' in checkpoint:
                print(f"[TienKungEnv] Base policy loaded from {checkpoint_path}")
                self._base_policy = checkpoint
            else:
                print(f"[TienKungEnv] Warning: unexpected checkpoint format.")
        except Exception as e:
            print(f"[TienKungEnv] Could not load base policy: {e}")
            print("[TienKungEnv] Running in Mode A (ILC-only).")

    def reset(self) -> RobotState:
        self._lazy_init()
        self._t = 0
        if self._fallback_mode:
            self._fallback_env.reset()
            return self._fallback_env._build_state()

        obs, _ = self._env.reset()
        return self._obs_to_state(obs)

    def step(self, action: np.ndarray) -> Tuple[RobotState, float, bool, dict]:
        self._lazy_init()
        self._t += 1

        if self._fallback_mode:
            full_action = TIENKUNG_STAND_POSE.copy()
            full_action[:len(action)] = action
            state, reward, done, info = self._fallback_env.step(full_action)
            return state, reward, done, info

        # Mode B: add base policy output
        if self._base_policy is not None:
            u_base = self._get_base_policy_action()
            action = action + u_base[:len(action)]

        # Clip to joint limits
        action = np.clip(action, -3.14, 3.14)

        try:
            import torch
            action_t = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
            obs, reward, terminated, truncated, info = self._env.step(action_t)
            state = self._obs_to_state(obs)
            done = bool(terminated[0] or truncated[0])
            return state, float(reward[0]), done, info
        except Exception as e:
            # Graceful degradation
            print(f"[TienKungEnv] Step error: {e}")
            return self.reset(), 0.0, True, {}

    def get_reference_trajectory(self) -> np.ndarray:
        """
        Generate walking reference trajectory for TienKung legs.
        Leg joints only (12 DOF) — arms held at nominal pose.
        """
        H = self.tk_cfg.episode_length
        n_loco = self.tk_cfg.n_loco_joints
        t = np.linspace(0, 2 * np.pi * 2, H)   # 2 stride cycles

        ref = np.zeros((H, self.tk_cfg.n_joints))

        # Set standing pose as base
        ref[:] = TIENKUNG_STAND_POSE

        # Bipedal walking pattern (hip pitch + knee + ankle)
        stride_freq = 1.5   # Hz
        omega = 2 * np.pi * stride_freq
        t_s = np.linspace(0, H * self.tk_cfg.dt, H)

        # Left leg: phase 0, Right leg: phase π
        for side, start_idx, phase in [(0, 0, 0.0), (1, 6, np.pi)]:
            hip_pitch = start_idx + 2
            knee      = start_idx + 3
            ankle     = start_idx + 4

            ref[:, hip_pitch] += 0.3 * np.sin(omega * t_s + phase)
            ref[:, knee]      += 0.4 * np.abs(np.sin(omega * t_s + phase))
            ref[:, ankle]     -= 0.2 * np.sin(omega * t_s + phase)

        return ref

    def _obs_to_state(self, obs) -> RobotState:
        """Convert Isaac Lab observation tensor to RobotState."""
        try:
            import torch
            o = obs[0].cpu().numpy() if isinstance(obs, torch.Tensor) else np.array(obs[0])
        except Exception:
            o = np.zeros(100)

        n = self.tk_cfg.n_joints
        # Isaac Lab obs layout: [base_lin_vel(3), base_ang_vel(3), gravity(3),
        #                         commands(3), joint_pos(n), joint_vel(n), ...]
        offset = 12
        joint_pos = o[offset:offset + n] if len(o) > offset + n else np.zeros(n)
        joint_vel = o[offset + n:offset + 2*n] if len(o) > offset + 2*n else np.zeros(n)

        return RobotState(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            base_pos=np.array([0.0, 0.0, 0.95]),   # TienKung standing height ~0.95m
            base_quat=np.array([0.0, 0.0, 0.0, 1.0]),
            base_lin_vel=o[:3] if len(o) >= 3 else np.zeros(3),
            base_ang_vel=o[3:6] if len(o) >= 6 else np.zeros(3),
            contact_forces=np.ones(self.tk_cfg.n_feet),
            timestamp=self._t * self.tk_cfg.dt,
        )

    def _get_base_policy_action(self) -> np.ndarray:
        """Query the base AMP policy for nominal joint targets."""
        # Placeholder — implement with actual TienKung-Lab policy inference
        return TIENKUNG_STAND_POSE.copy()

    def close(self):
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass


def make_tienkung_env(use_base_policy: bool = False,
                      policy_checkpoint: str = "",
                      headless: bool = True) -> TienKungEnv:
    """
    Factory for TienKung environment.

    Parameters
    ----------
    use_base_policy : bool
        If True, runs in Mode B (ILC residual correction on AMP policy).
        This is the recommended mode for the thesis experiments.
    policy_checkpoint : str
        Path to TienKung-Lab AMP policy checkpoint (.pt file).
        Only used when use_base_policy=True.
    """
    cfg = TienKungConfig(
        use_base_policy=use_base_policy,
        policy_checkpoint=policy_checkpoint
    )
    return TienKungEnv(cfg, headless=headless)
