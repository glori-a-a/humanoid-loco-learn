"""
Experience Replay Buffer for Data-Driven Learning Control
==========================================================
Stores trial data and provides batched access for offline policy improvement.
Supports prioritised replay (recent + high-error trials sampled more often).
"""

import numpy as np
from collections import deque
from typing import Optional
from .ilc import TrialData


class ReplayBuffer:
    """
    Circular buffer storing trial histories for offline analysis and
    multi-trial policy gradient updates.

    Prioritised sampling: trials with higher RMSE are sampled more often,
    focusing learning on the most informative experiences.
    """

    def __init__(self, capacity: int = 200, priority_alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = priority_alpha        # 0 = uniform, 1 = fully prioritised
        self._buffer: deque[TrialData] = deque(maxlen=capacity)

    def push(self, trial: TrialData):
        self._buffer.append(trial)

    def sample(self, batch_size: int,
               prioritised: bool = True) -> list[TrialData]:
        """Sample batch_size trials, optionally with priority weighting."""
        buf = list(self._buffer)
        n = len(buf)
        if n == 0:
            return []
        batch_size = min(batch_size, n)

        if prioritised and n > 1:
            # Higher RMSE → higher priority
            rmses = np.array([t.rmse for t in buf]) + 1e-6
            probs = rmses ** self.alpha
            probs /= probs.sum()
            indices = np.random.choice(n, size=batch_size, replace=False, p=probs)
        else:
            indices = np.random.choice(n, size=batch_size, replace=False)

        return [buf[i] for i in indices]

    def best_trial(self) -> Optional[TrialData]:
        """Return trial with lowest RMSE."""
        if not self._buffer:
            return None
        return min(self._buffer, key=lambda t: t.rmse)

    def latest(self, n: int = 1) -> list[TrialData]:
        buf = list(self._buffer)
        return buf[-n:]

    def convergence_stats(self) -> dict:
        if not self._buffer:
            return {}
        rmses = [t.rmse for t in self._buffer]
        return {
            'n_trials': len(rmses),
            'initial_rmse': rmses[0],
            'final_rmse': rmses[-1],
            'best_rmse': min(rmses),
            'improvement_pct': (rmses[0] - rmses[-1]) / (rmses[0] + 1e-8) * 100,
        }

    def __len__(self) -> int:
        return len(self._buffer)
