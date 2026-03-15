"""
State Estimator
===============
Kalman filter-based state estimator for legged robot base state.
Fuses IMU (acceleration + gyro) with joint encoder data to estimate:
  - Base position and velocity
  - Contact state
  - Terrain slope

Used in the ILC loop to provide clean, low-latency state feedback
even under sensor noise and contact impacts.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class EstimatorConfig:
    dt: float = 0.01
    n_joints: int = 12
    # Process noise (how much we trust the model)
    q_pos: float = 0.001
    q_vel: float = 0.01
    # Measurement noise (how much we trust sensors)
    r_encoder: float = 0.001
    r_imu: float = 0.01


class KalmanStateEstimator:
    """
    Linear Kalman filter estimating [base_vel, base_pos, joint_pos, joint_vel].

    State vector x = [v_base(3), p_base(3), q(n), dq(n)]
    Observation  y = [q_enc(n), a_imu(3)]

    Provides clean estimates for ILC error computation,
    reducing noise-induced learning instability.
    """

    def __init__(self, config: EstimatorConfig):
        self.cfg = config
        n = config.n_joints
        self.n_state = 6 + 2 * n   # [v(3), p(3), q(n), dq(n)]
        self.n_obs   = n + 3        # [q_enc(n), a_imu(3)]

        # State transition matrix (constant velocity model)
        dt = config.dt
        self.F = np.eye(self.n_state)
        # p_{t+1} = p_t + v_t * dt
        self.F[3:6, 0:3] = dt * np.eye(3)
        # q_{t+1} = q_t + dq_t * dt
        self.F[6:6+n, 6+n:] = dt * np.eye(n)

        # Observation matrix: observe q_enc and a_imu
        self.H = np.zeros((self.n_obs, self.n_state))
        self.H[:n, 6:6+n] = np.eye(n)          # joint encoder → q
        self.H[n:, 0:3] = np.eye(3)             # imu accel → v_base (approx)

        # Noise covariances
        self.Q = np.eye(self.n_state)
        self.Q[:3, :3]   *= config.q_vel
        self.Q[3:6, 3:6] *= config.q_pos
        self.Q[6:, 6:]   *= config.q_encoder if hasattr(config, 'q_encoder') else 0.001

        self.R = np.eye(self.n_obs)
        self.R[:n, :n] *= config.r_encoder
        self.R[n:, n:] *= config.r_imu

        # Initial state and covariance
        self.x = np.zeros(self.n_state)
        self.P = np.eye(self.n_state) * 0.1

    @property
    def n_joints(self):
        return self.cfg.n_joints

    def reset(self, q0: np.ndarray = None):
        self.x = np.zeros(self.n_state)
        if q0 is not None:
            self.x[6:6 + self.n_joints] = q0
        self.P = np.eye(self.n_state) * 0.1

    def update(self, q_enc: np.ndarray, a_imu: np.ndarray) -> np.ndarray:
        """
        Kalman predict + update step.

        Parameters
        ----------
        q_enc : joint encoder readings (n_joints,)
        a_imu : IMU linear acceleration (3,)

        Returns
        -------
        x_hat : estimated state vector
        """
        # Predict
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Measurement
        y = np.concatenate([q_enc, a_imu])
        y_hat = self.H @ x_pred
        innov = y - y_hat

        # Kalman gain
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        # Update
        self.x = x_pred + K @ innov
        self.P = (np.eye(self.n_state) - K @ self.H) @ P_pred

        return self.x.copy()

    @property
    def joint_pos(self) -> np.ndarray:
        return self.x[6:6 + self.n_joints]

    @property
    def joint_vel(self) -> np.ndarray:
        return self.x[6 + self.n_joints:]

    @property
    def base_vel(self) -> np.ndarray:
        return self.x[:3]

    @property
    def base_pos(self) -> np.ndarray:
        return self.x[3:6]
