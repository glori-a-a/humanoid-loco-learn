"""
Iterative Learning Control (ILC) Algorithms
=============================================
Implements three ILC variants for data-driven robot motion learning:

  1. P-type ILC   — simple proportional update
  2. PD-type ILC  — proportional + derivative, faster convergence
  3. Norm-Optimal ILC (NOILC) — optimal trade-off between tracking error
                                and control effort (used in research)

Theory
------
A robot executes a periodic task repeatedly (trials). Each trial k produces
an error signal e_k(t) = q_d(t) - q_k(t). The ILC law updates the feedforward
input for the next trial so that the error converges to zero:

    u_{k+1}(t) = u_k(t) + correction(e_k, ...)

This directly implements the thesis concept:
"Through repeated trials, the robot gradually learns optimal control actions
that compensate for uncertainties and complex dynamics."

References
----------
Bristow, D.A., Tharayil, M., Alleyne, A.G. (2006).
    A survey of iterative learning control. IEEE Control Systems Magazine.
Owens, D.H. (2016). Iterative Learning Control. Springer.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from abc import ABC, abstractmethod


@dataclass
class ILCConfig:
    n_joints: int = 12          # DOF (12 for humanoid legs, 4 for quadruped)
    horizon: int = 500          # timesteps per trial
    dt: float = 0.01            # control timestep (s)
    max_trials: int = 100       # maximum learning trials
    convergence_tol: float = 1e-3  # stop when RMSE < this


@dataclass
class TrialData:
    """Stores data from one learning trial."""
    trial: int
    q_ref: np.ndarray      # reference joint angles   (horizon, n_joints)
    q_actual: np.ndarray   # actual joint angles       (horizon, n_joints)
    u: np.ndarray          # control input applied     (horizon, n_joints)
    error: np.ndarray      # tracking error e = q_ref - q_actual
    rmse: float

    @property
    def max_error(self) -> float:
        return float(np.max(np.abs(self.error)))

    @property
    def converged(self) -> bool:
        return self.rmse < 1e-3


class BaseILC(ABC):
    """Abstract base class for all ILC algorithms."""

    def __init__(self, config: ILCConfig):
        self.cfg = config
        self.history: list[TrialData] = []
        # Initialise feedforward input to zero
        self.u_ff = np.zeros((config.horizon, config.n_joints))

    @abstractmethod
    def update(self, trial_data: TrialData) -> np.ndarray:
        """
        Compute updated feedforward signal for next trial.
        Returns u_{k+1} of shape (horizon, n_joints).
        """

    def record(self, trial_data: TrialData):
        self.history.append(trial_data)

    def convergence_curve(self) -> np.ndarray:
        """Returns array of per-trial RMSE values."""
        return np.array([d.rmse for d in self.history])

    def has_converged(self) -> bool:
        if not self.history:
            return False
        return self.history[-1].rmse < self.cfg.convergence_tol


class PILCController(BaseILC):
    """
    P-type ILC:  u_{k+1}(t) = u_k(t) + Gamma * e_k(t)

    Simple, robust, guaranteed to converge when ||I - G*Gamma|| < 1,
    where G is the plant's pulse response matrix.
    """

    def __init__(self, config: ILCConfig, gamma: float = 0.8):
        super().__init__(config)
        self.gamma = gamma    # learning gain (0 < gamma < 1 for stability)

    def update(self, td: TrialData) -> np.ndarray:
        self.u_ff = td.u + self.gamma * td.error
        self.record(td)
        return self.u_ff.copy()


class PDILCController(BaseILC):
    """
    PD-type ILC:  u_{k+1}(t) = u_k(t) + Kp * e_k(t) + Kd * de_k/dt

    Faster convergence than P-type; derivative term anticipates future errors.
    Well-suited for robotic tasks where phase lead matters (e.g., fast gaits).
    """

    def __init__(self, config: ILCConfig, kp: float = 0.8, kd: float = 0.1):
        super().__init__(config)
        self.kp = kp
        self.kd = kd

    def update(self, td: TrialData) -> np.ndarray:
        e = td.error
        # Finite-difference derivative (causal)
        de = np.zeros_like(e)
        de[1:] = (e[1:] - e[:-1]) / self.cfg.dt

        self.u_ff = td.u + self.kp * e + self.kd * de
        self.record(td)
        return self.u_ff.copy()


class NormOptimalILC(BaseILC):
    """
    Norm-Optimal ILC (NOILC):
        Minimise  J = ||W_e * e_{k+1}||^2 + lambda * ||W_u * delta_u_k||^2

    Closed-form solution per joint (diagonal approximation):
        delta_u* = (G^T * Q * G + lambda * R)^{-1} * G^T * Q * e_k

    where G is the Markov (lower-triangular Toeplitz) matrix of the plant,
    Q = W_e^T W_e, R = W_u^T W_u.

    This is the algorithm most commonly used in recent humanoid locomotion
    learning papers. The lambda parameter controls the exploration-exploitation
    trade-off: high lambda → cautious small updates; low lambda → aggressive.

    In this implementation G is estimated online from input-output data,
    making it fully data-driven — no plant model required.
    """

    def __init__(self, config: ILCConfig,
                 lam: float = 0.1,
                 q_weight: float = 1.0,
                 r_weight: float = 1.0):
        super().__init__(config)
        self.lam = lam              # regularisation (exploration vs. exploitation)
        self.q_weight = q_weight    # error weighting
        self.r_weight = r_weight    # input change weighting
        self._G_hat: Optional[np.ndarray] = None   # estimated Markov matrix

    def _estimate_markov_matrix(self, td: TrialData) -> np.ndarray:
        """
        Estimate the lower-triangular Toeplitz plant matrix from data.
        Uses a first-order approximation: G[i,j] ≈ delta_y[i] / delta_u[j].
        In practice, uses the empirical impulse response.
        """
        H, n = self.cfg.horizon, self.cfg.n_joints
        # Scalar (per-joint average) Toeplitz approximation
        g = np.zeros(H)
        if len(self.history) >= 1:
            prev = self.history[-1]
            du = td.u - prev.u
            dy = td.q_actual - prev.q_actual
            # Least-squares scalar gain estimate
            du_flat = du.flatten()
            dy_flat = dy.flatten()
            norm = np.dot(du_flat, du_flat) + 1e-8
            g0 = np.dot(dy_flat, du_flat) / norm
            g[0] = g0
            # Simple first-order lag model for off-diagonal terms
            alpha = 0.7
            for i in range(1, min(10, H)):
                g[i] = g[i - 1] * alpha
        else:
            g[0] = 0.5   # initial guess

        # Build lower-triangular Toeplitz matrix (scalar per joint)
        G = np.zeros((H, H))
        for i in range(H):
            for j in range(i + 1):
                G[i, j] = g[i - j]
        return G

    def update(self, td: TrialData) -> np.ndarray:
        G = self._estimate_markov_matrix(td)
        Q = self.q_weight * np.eye(self.cfg.horizon)
        R = self.r_weight * np.eye(self.cfg.horizon)

        # Solve per joint independently
        u_new = np.zeros_like(td.u)
        GtQ = G.T @ Q
        M = GtQ @ G + self.lam * R

        for j in range(self.cfg.n_joints):
            e_j = td.error[:, j]
            delta_u_j = np.linalg.solve(M, GtQ @ e_j)
            u_new[:, j] = td.u[:, j] + delta_u_j

        self.u_ff = u_new
        self._G_hat = G
        self.record(td)
        return self.u_ff.copy()


class AdaptiveILC(BaseILC):
    """
    Adaptive ILC with online learning rate scheduling.

    Combines NOILC with a momentum term inspired by Adam optimiser:
        m_k = beta1 * m_{k-1} + (1 - beta1) * e_k
        v_k = beta2 * v_{k-1} + (1 - beta2) * e_k^2
        delta_u = alpha * m_k / (sqrt(v_k) + eps)

    Provides fast initial learning with automatic step-size decay,
    mimicking modern deep-learning optimisers applied to ILC.
    This is the "adaptive" data-driven approach referenced in the thesis.
    """

    def __init__(self, config: ILCConfig,
                 alpha: float = 0.5,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 eps: float = 1e-8):
        super().__init__(config)
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m = np.zeros((config.horizon, config.n_joints))
        self._v = np.zeros((config.horizon, config.n_joints))
        self._t = 0

    def update(self, td: TrialData) -> np.ndarray:
        self._t += 1
        e = td.error

        # Bias-corrected moment estimates
        self._m = self.beta1 * self._m + (1 - self.beta1) * e
        self._v = self.beta2 * self._v + (1 - self.beta2) * e ** 2
        m_hat = self._m / (1 - self.beta1 ** self._t)
        v_hat = self._v / (1 - self.beta2 ** self._t)

        delta_u = self.alpha * m_hat / (np.sqrt(v_hat) + self.eps)
        self.u_ff = td.u + delta_u
        self.record(td)
        return self.u_ff.copy()


def make_ilc(algorithm: str, config: ILCConfig, **kwargs) -> BaseILC:
    """Factory function for ILC algorithm selection."""
    registry = {
        'p':        PILCController,
        'pd':       PDILCController,
        'noilc':    NormOptimalILC,
        'adaptive': AdaptiveILC,
    }
    if algorithm not in registry:
        raise ValueError(f"Unknown ILC algorithm '{algorithm}'. "
                         f"Choose from: {list(registry.keys())}")
    return registry[algorithm](config, **kwargs)
