"""
Base Environment Interface
===========================
Abstract interface that both the Isaac Lab wrapper and the lightweight
simulation environment implement. This allows algorithm code to be
environment-agnostic.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple


@dataclass
class RobotState:
    """Unified robot state representation."""
    joint_pos: np.ndarray      # joint angles (n_joints,)
    joint_vel: np.ndarray      # joint velocities (n_joints,)
    base_pos: np.ndarray       # base position (3,)
    base_quat: np.ndarray      # base orientation quaternion (4,)
    base_lin_vel: np.ndarray   # linear velocity (3,)
    base_ang_vel: np.ndarray   # angular velocity (3,)
    contact_forces: np.ndarray # foot contact forces (n_feet,)
    timestamp: float = 0.0


@dataclass
class EnvConfig:
    n_joints: int = 12
    n_feet: int = 4
    dt: float = 0.01
    episode_length: int = 500
    robot_name: str = "unitree_go2"


class BaseRobotEnv(ABC):
    """
    Abstract robot environment.
    Subclassed by IsaacLabEnv and LightweightSimEnv.
    """

    def __init__(self, config: EnvConfig):
        self.cfg = config
        self._step_count = 0

    @abstractmethod
    def reset(self) -> RobotState:
        """Reset environment and return initial state."""

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[RobotState, float, bool, dict]:
        """
        Apply action and advance simulation.
        Returns (next_state, reward, done, info).
        action shape: (n_joints,) — joint position targets
        """

    @abstractmethod
    def get_reference_trajectory(self) -> np.ndarray:
        """
        Returns desired joint trajectory of shape (episode_length, n_joints).
        """

    def tracking_reward(self, state: RobotState, ref: np.ndarray) -> float:
        """Negative RMSE tracking error as reward."""
        e = ref - state.joint_pos
        return -float(np.sqrt(np.mean(e ** 2)))

    @property
    def n_joints(self) -> int:
        return self.cfg.n_joints

    @property
    def horizon(self) -> int:
        return self.cfg.episode_length
