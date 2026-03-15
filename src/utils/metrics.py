"""
Learning Performance Metrics
==============================
Evaluation utilities for ILC convergence analysis.
"""

import numpy as np
from typing import List
from ..algorithms.ilc import TrialData


def compute_tracking_metrics(trials: List[TrialData]) -> dict:
    """Compute convergence and performance metrics across trials."""
    if not trials:
        return {}

    rmses     = [t.rmse for t in trials]
    max_errs  = [t.max_error for t in trials]

    # Convergence rate: exponential fit to RMSE curve
    n = len(rmses)
    if n > 2:
        x = np.arange(n, dtype=float)
        y = np.log(np.clip(rmses, 1e-8, None))
        coeffs = np.polyfit(x, y, 1)
        conv_rate = float(-coeffs[0])   # positive = converging
    else:
        conv_rate = 0.0

    # Improvement percentage from trial 1 to last
    improvement = (rmses[0] - rmses[-1]) / (rmses[0] + 1e-8) * 100

    # Trials to reach 50% / 90% improvement
    target_50 = rmses[0] * 0.5
    target_90 = rmses[0] * 0.1
    t50 = next((i for i, r in enumerate(rmses) if r <= target_50), n)
    t90 = next((i for i, r in enumerate(rmses) if r <= target_90), n)

    return {
        'n_trials':       n,
        'initial_rmse':   rmses[0],
        'final_rmse':     rmses[-1],
        'best_rmse':      min(rmses),
        'improvement_pct': improvement,
        'convergence_rate': conv_rate,
        'trials_to_50pct': t50,
        'trials_to_90pct': t90,
        'max_error_final': max_errs[-1],
        'converged':       rmses[-1] < 1e-3,
    }


def joint_error_breakdown(trial: TrialData) -> dict:
    """Per-joint error statistics for a single trial."""
    e = trial.error   # (horizon, n_joints)
    return {
        'per_joint_rmse':    np.sqrt(np.mean(e**2, axis=0)).tolist(),
        'per_joint_max':     np.max(np.abs(e), axis=0).tolist(),
        'worst_joint':       int(np.argmax(np.sqrt(np.mean(e**2, axis=0)))),
        'best_joint':        int(np.argmin(np.sqrt(np.mean(e**2, axis=0)))),
    }
