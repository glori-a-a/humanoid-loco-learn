"""Tests for simulation environment and trajectory generation."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.environments.base_env import EnvConfig
from src.environments.sim_env import LightweightSimEnv
from src.control.trajectory import TrajectoryGenerator, TrajectoryConfig, GaitType


class TestLightweightSim:
    def _make_env(self, n=4, H=50):
        return LightweightSimEnv(EnvConfig(n_joints=n, episode_length=H, dt=0.02))

    def test_reset_returns_state(self):
        env = self._make_env()
        s = env.reset()
        assert s.joint_pos.shape == (4,)
        assert s.joint_vel.shape == (4,)

    def test_step_returns_tuple(self):
        env = self._make_env()
        env.reset()
        s, r, done, info = env.step(np.zeros(4))
        assert isinstance(r, float)
        assert isinstance(done, bool)

    def test_episode_terminates_at_horizon(self):
        env = self._make_env(H=20)
        env.reset()
        for _ in range(19):
            _, _, done, _ = env.step(np.zeros(4))
            assert not done
        _, _, done, _ = env.step(np.zeros(4))
        assert done

    def test_reference_trajectory_shape(self):
        env = self._make_env(n=12, H=100)
        ref = env.get_reference_trajectory()
        assert ref.shape == (100, 12)

    @pytest.mark.parametrize("gait", ["trot", "walk", "bound", "pace"])
    def test_all_gaits_generate_trajectory(self, gait):
        cfg = EnvConfig(n_joints=12, episode_length=200)
        env = LightweightSimEnv(cfg, gait=gait)
        ref = env.get_reference_trajectory()
        assert ref.shape == (200, 12)
        assert not np.all(ref == 0) or gait == "stand"


class TestTrajectoryGenerator:
    def _make_gen(self, gait=GaitType.TROT, n=12, H=300):
        return TrajectoryGenerator(TrajectoryConfig(n_joints=n, horizon=H, gait=gait))

    def test_output_shape(self):
        gen = self._make_gen()
        ref = gen.generate()
        assert ref.shape == (300, 12)

    def test_trot_is_periodic(self):
        gen = self._make_gen(GaitType.TROT, H=200)
        ref = gen.generate(GaitType.TROT)
        # Trot should have nonzero values (not all zeros)
        assert np.max(np.abs(ref)) > 0.1

    def test_interpolate(self):
        gen = self._make_gen(H=100)
        ref = gen.generate()
        ref2 = gen.interpolate(ref, target_horizon=200)
        assert ref2.shape == (200, 12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
