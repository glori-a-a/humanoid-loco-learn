"""
Run ILC learning experiment from command line.

Usage:
    python scripts/run_ilc.py --algo noilc --gait trot --trials 100
    python scripts/run_ilc.py --algo adaptive --gait bound --backend isaac
"""

import argparse
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.algorithms.ilc import ILCConfig, TrialData, make_ilc
from src.algorithms.replay_buffer import ReplayBuffer
from src.environments.isaac_env import make_env
from src.environments.base_env import EnvConfig
from src.control.trajectory import TrajectoryGenerator, TrajectoryConfig, GaitType
from src.utils.logger import ExperimentLogger
from src.utils.metrics import compute_tracking_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Run ILC learning experiment")
    p.add_argument('--algo',    default='noilc',
                   choices=['p','pd','noilc','adaptive'])
    p.add_argument('--gait',    default='trot',
                   choices=['trot','walk','bound','pace'])
    p.add_argument('--trials',  type=int, default=100)
    p.add_argument('--backend', default='sim', choices=['sim','isaac'])
    p.add_argument('--n-joints',type=int, default=12)
    p.add_argument('--horizon', type=int, default=300)
    p.add_argument('--dt',      type=float, default=0.02)
    p.add_argument('--kp',      type=float, default=40.0)
    p.add_argument('--kd',      type=float, default=2.0)
    p.add_argument('--plot',    action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    n, H, dt = args.n_joints, args.horizon, args.dt

    print(f"\n{'='*60}")
    print(f"  HumanoidLocoLearn — ILC Experiment")
    print(f"  Algorithm : {args.algo.upper()}")
    print(f"  Gait      : {args.gait}")
    print(f"  Backend   : {args.backend}")
    print(f"  Trials    : {args.trials}")
    print(f"{'='*60}\n")

    # Setup
    ilc_cfg  = ILCConfig(n_joints=n, horizon=H, dt=dt, max_trials=args.trials)
    env_cfg  = EnvConfig(n_joints=n, episode_length=H, dt=dt)
    traj_cfg = TrajectoryConfig(n_joints=n, horizon=H,
                                gait=GaitType(args.gait), dt=dt)

    ilc    = make_ilc(args.algo, ilc_cfg)
    env    = make_env(args.backend, env_cfg)
    ref    = TrajectoryGenerator(traj_cfg).generate()
    buffer = ReplayBuffer()
    logger = ExperimentLogger(run_name=f'{args.algo}_{args.gait}')

    # Learning loop
    for trial in range(args.trials):
        q_buf = np.zeros((H, n))
        u_buf = np.zeros((H, n))

        env.reset()
        for t in range(H):
            q_ref = ref[t]
            state = env._build_state() if hasattr(env, '_build_state') else env.reset()
            q  = state.joint_pos
            dq = state.joint_vel

            u_ff = ilc.u_ff[t]
            u_fb = args.kp * (q_ref - q) - args.kd * dq
            u    = np.clip(u_ff + u_fb, -5.0, 5.0)

            env.step(u)
            q_buf[t] = q
            u_buf[t] = u

        error = ref - q_buf
        rmse  = float(np.sqrt(np.mean(error**2)))
        td    = TrialData(trial=trial, q_ref=ref, q_actual=q_buf,
                          u=u_buf, error=error, rmse=rmse)
        buffer.push(td)
        ilc.update(td)
        logger.log(trial, rmse=rmse, max_error=td.max_error)

        print(f"  Trial {trial:03d} | RMSE={rmse:.5f} | "
              f"MaxErr={td.max_error:.4f} | "
              f"{'✓ CONVERGED' if td.converged else ''}")

        if td.converged:
            print(f"\n  Converged at trial {trial}!")
            break

    # Summary
    print(f"\n{'='*60}")
    metrics = compute_tracking_metrics(list(buffer._buffer))
    for k, v in metrics.items():
        print(f"  {k:25s}: {v}")

    log_path = logger.save()
    print(f"\n  Logs saved → {log_path}")
    print(f"{'='*60}\n")

    if args.plot:
        _plot(ilc.convergence_curve(), args)


def _plot(curve, args):
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,4))
        plt.plot(curve, 'steelblue', linewidth=2)
        plt.xlabel('Trial'); plt.ylabel('RMSE (rad)')
        plt.title(f'ILC Convergence — {args.algo.upper()} / {args.gait}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'ilc_{args.algo}_{args.gait}.png', dpi=150)
        print(f"  Plot saved → ilc_{args.algo}_{args.gait}.png")
    except ImportError:
        print("  (matplotlib not available, skipping plot)")


if __name__ == '__main__':
    main()
