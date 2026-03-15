"""Tests for ILC algorithms."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.algorithms.ilc import ILCConfig, TrialData, PILCController, PDILCController, NormOptimalILC, AdaptiveILC, make_ilc


def make_config(n=4, H=50):
    return ILCConfig(n_joints=n, horizon=H, dt=0.02)

def make_trial(cfg, rmse=0.1, trial=0):
    H, n = cfg.horizon, cfg.n_joints
    q_ref = np.random.randn(H, n) * 0.3
    error = np.random.randn(H, n) * rmse
    q_actual = q_ref - error
    u = np.random.randn(H, n) * 0.5
    return TrialData(trial=trial, q_ref=q_ref, q_actual=q_actual,
                     u=u, error=error, rmse=float(np.sqrt(np.mean(error**2))))


class TestILCBase:
    @pytest.mark.parametrize("algo", ["p", "pd", "noilc", "adaptive"])
    def test_update_returns_correct_shape(self, algo):
        cfg = make_config()
        ilc = make_ilc(algo, cfg)
        td  = make_trial(cfg)
        u_new = ilc.update(td)
        assert u_new.shape == (cfg.horizon, cfg.n_joints)

    @pytest.mark.parametrize("algo", ["p", "pd", "noilc", "adaptive"])
    def test_history_recorded(self, algo):
        cfg = make_config()
        ilc = make_ilc(algo, cfg)
        for i in range(5):
            ilc.update(make_trial(cfg, trial=i))
        assert len(ilc.history) == 5

    def test_convergence_curve_length(self):
        cfg = make_config()
        ilc = PILCController(cfg)
        for i in range(8):
            ilc.update(make_trial(cfg, trial=i))
        curve = ilc.convergence_curve()
        assert len(curve) == 8

    def test_factory_unknown_raises(self):
        with pytest.raises(ValueError):
            make_ilc("unknown", make_config())


class TestPILC:
    def test_p_ilc_reduces_error_on_simple_system(self):
        """P-ILC should reduce error over trials on a simple linear system."""
        cfg = ILCConfig(n_joints=1, horizon=100, dt=0.01)
        ilc = PILCController(cfg, gamma=0.5)
        # Simple first-order system: y_k = 0.8 * u_k
        G = 0.8
        q_ref = np.ones((100, 1)) * 0.5
        u = np.zeros((100, 1))
        rmses = []
        for _ in range(30):
            q_act = G * u + np.random.randn(100, 1) * 0.002
            error = q_ref - q_act
            rmse = float(np.sqrt(np.mean(error**2)))
            rmses.append(rmse)
            td = TrialData(0, q_ref, q_act, u, error, rmse)
            u = ilc.update(td)
        assert rmses[-1] < rmses[0] * 0.5, "P-ILC should reduce RMSE by 50%+"


class TestNormOptimalILC:
    def test_noilc_update_is_finite(self):
        cfg = make_config(n=2, H=30)
        ilc = NormOptimalILC(cfg, lam=0.1)
        for i in range(3):
            td = make_trial(cfg, trial=i)
            u_new = ilc.update(td)
            assert np.all(np.isfinite(u_new))

    def test_noilc_stores_markov_matrix(self):
        cfg = make_config(n=2, H=20)
        ilc = NormOptimalILC(cfg)
        for i in range(2):
            ilc.update(make_trial(cfg, trial=i))
        assert ilc._G_hat is not None
        assert ilc._G_hat.shape == (cfg.horizon, cfg.horizon)


class TestAdaptiveILC:
    def test_adaptive_step_counter(self):
        cfg = make_config()
        ilc = AdaptiveILC(cfg)
        for i in range(5):
            ilc.update(make_trial(cfg, trial=i))
        assert ilc._t == 5

    def test_adaptive_momentum_nonzero_after_trials(self):
        cfg = make_config()
        ilc = AdaptiveILC(cfg)
        ilc.update(make_trial(cfg, rmse=0.5))
        assert np.any(ilc._m != 0)
        assert np.any(ilc._v != 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
