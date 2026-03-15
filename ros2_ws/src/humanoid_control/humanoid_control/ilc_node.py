"""
ROS2 ILC Controller Node
=========================
Runs the Iterative Learning Control algorithm as a ROS2 node,
subscribing to joint states and publishing feedforward + feedback
control commands to the robot (real or simulated via Isaac Lab ROS2 bridge).

Topics
------
Subscribed:
  /joint_states          (sensor_msgs/JointState)  — encoder feedback
  /imu/data              (sensor_msgs/Imu)          — base IMU
  /ilc/reference         (std_msgs/Float64MultiArray) — override reference

Published:
  /joint_position_targets (std_msgs/Float64MultiArray) — ILC output
  /ilc/status             (std_msgs/String)             — JSON status
  /ilc/convergence        (std_msgs/Float64MultiArray)  — RMSE per trial

Services
--------
  /ilc/reset             — reset learning (start fresh)
  /ilc/set_algorithm     — switch P / PD / NOILC / Adaptive at runtime
  /ilc/set_gait          — switch gait pattern at runtime
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Float64MultiArray, String
from sensor_msgs.msg import JointState, Imu
from std_srvs.srv import Trigger, SetBool

import numpy as np
import json
import time
import sys
import os

# Allow importing src/ from the workspace root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../../..'))

from src.algorithms.ilc import ILCConfig, TrialData, make_ilc
from src.algorithms.replay_buffer import ReplayBuffer
from src.control.trajectory import TrajectoryGenerator, TrajectoryConfig, GaitType
from src.control.state_estimator import KalmanStateEstimator, EstimatorConfig
from src.utils.logger import ExperimentLogger
from src.utils.metrics import compute_tracking_metrics


class ILCNode(Node):
    """
    Main ROS2 node for iterative learning control.

    State machine:
        IDLE → TRIAL_RUNNING → TRIAL_DONE → UPDATE → IDLE (loop)

    On each trial:
        1. Reset environment / robot to initial pose
        2. Apply u_ff (feedforward) + u_fb (PD feedback) for horizon steps
        3. Collect (q_ref, q_actual, u, error) data
        4. ILC update: compute u_ff for next trial
        5. Publish convergence metrics and status
    """

    def __init__(self):
        super().__init__('ilc_controller')

        # ── Parameters (configurable via ros2 param set) ──────────────────────
        self.declare_parameter('algorithm',    'noilc')
        self.declare_parameter('n_joints',     12)
        self.declare_parameter('horizon',      500)
        self.declare_parameter('dt',           0.01)
        self.declare_parameter('gait',         'trot')
        self.declare_parameter('kp_fb',        50.0)
        self.declare_parameter('kd_fb',        2.0)
        self.declare_parameter('max_trials',   200)
        self.declare_parameter('auto_start',   True)

        algo    = self.get_parameter('algorithm').value
        n       = self.get_parameter('n_joints').value
        H       = self.get_parameter('horizon').value
        dt      = self.get_parameter('dt').value
        gait    = self.get_parameter('gait').value
        max_tr  = self.get_parameter('max_trials').value

        # ── ILC setup ─────────────────────────────────────────────────────────
        self.ilc_cfg = ILCConfig(n_joints=n, horizon=H, dt=dt,
                                 max_trials=max_tr)
        self.ilc = make_ilc(algo, self.ilc_cfg)
        self.buffer = ReplayBuffer(capacity=500)

        # ── Trajectory ────────────────────────────────────────────────────────
        traj_cfg = TrajectoryConfig(n_joints=n, horizon=H, dt=dt,
                                    gait=GaitType(gait))
        self.traj_gen = TrajectoryGenerator(traj_cfg)
        self.ref_traj = self.traj_gen.generate()

        # ── State estimator ───────────────────────────────────────────────────
        self.estimator = KalmanStateEstimator(EstimatorConfig(dt=dt, n_joints=n))

        # ── Logger ────────────────────────────────────────────────────────────
        self.logger = ExperimentLogger(log_dir='/tmp/ilc_logs',
                                       run_name=f'ilc_{algo}_{gait}')

        # ── Internal trial state ──────────────────────────────────────────────
        self.n  = n
        self.H  = H
        self.dt = dt
        self.kp = self.get_parameter('kp_fb').value
        self.kd = self.get_parameter('kd_fb').value

        self._step      = 0
        self._trial     = 0
        self._q_buf     = np.zeros((H, n))
        self._u_buf     = np.zeros((H, n))
        self._q_latest  = np.zeros(n)
        self._dq_latest = np.zeros(n)
        self._running   = False

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub_cmd  = self.create_publisher(
            Float64MultiArray, '/joint_position_targets', 10)
        self.pub_stat = self.create_publisher(
            String, '/ilc/status', 10)
        self.pub_conv = self.create_publisher(
            Float64MultiArray, '/ilc/convergence', 10)

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(JointState, '/joint_states',
                                 self._cb_joints, 10)
        self.create_subscription(Imu, '/imu/data', self._cb_imu, 10)
        self.create_subscription(Float64MultiArray, '/ilc/reference',
                                 self._cb_reference, 10)

        # ── Services ──────────────────────────────────────────────────────────
        self.create_service(Trigger,  '/ilc/reset',        self._srv_reset)
        self.create_service(Trigger,  '/ilc/start',        self._srv_start)

        # ── Control timer (runs at 1/dt Hz) ───────────────────────────────────
        self.ctrl_timer = self.create_timer(dt, self._control_loop)

        # ── Status timer (1 Hz) ───────────────────────────────────────────────
        self.stat_timer = self.create_timer(1.0, self._publish_status)

        self.get_logger().info(
            f'ILCNode ready | algo={algo} | gait={gait} | '
            f'n_joints={n} | horizon={H} | dt={dt}'
        )

        if self.get_parameter('auto_start').value:
            self._running = True
            self.get_logger().info('Auto-start enabled — learning begins now.')

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _cb_joints(self, msg: JointState):
        if len(msg.position) >= self.n:
            self._q_latest  = np.array(msg.position[:self.n])
            self._dq_latest = np.array(msg.velocity[:self.n]) \
                if len(msg.velocity) >= self.n else self._dq_latest

    def _cb_imu(self, msg: Imu):
        a = msg.linear_acceleration
        self.estimator.update(self._q_latest,
                               np.array([a.x, a.y, a.z]))

    def _cb_reference(self, msg: Float64MultiArray):
        data = np.array(msg.data).reshape(-1, self.n)
        if data.shape[0] == self.H:
            self.ref_traj = data
            self.get_logger().info('Reference trajectory updated from topic.')

    # ── Services ──────────────────────────────────────────────────────────────

    def _srv_reset(self, req, res):
        self._reset_trial()
        self.ilc = make_ilc(
            self.get_parameter('algorithm').value, self.ilc_cfg)
        self.buffer = ReplayBuffer()
        res.success = True
        res.message = 'ILC reset. Learning will restart from trial 0.'
        return res

    def _srv_start(self, req, res):
        self._running = True
        res.success = True
        res.message = f'ILC started. Algorithm: {self.get_parameter("algorithm").value}'
        return res

    # ── Control loop ──────────────────────────────────────────────────────────

    def _control_loop(self):
        if not self._running:
            return
        if self._trial >= self.ilc_cfg.max_trials:
            self._running = False
            self.get_logger().info('Max trials reached. Learning complete.')
            return

        t = self._step
        if t >= self.H:
            self._end_trial()
            return

        # Reference at this timestep
        q_ref = self.ref_traj[t]
        q     = self._q_latest
        dq    = self._dq_latest

        # Feedforward (ILC) + feedback (PD)
        u_ff = self.ilc.u_ff[t]
        e_fb = q_ref - q
        u_fb = self.kp * e_fb - self.kd * dq
        u    = np.clip(u_ff + u_fb, -5.0, 5.0)

        # Publish command
        cmd = Float64MultiArray()
        cmd.data = u.tolist()
        self.pub_cmd.publish(cmd)

        # Buffer recording
        self._q_buf[t] = q
        self._u_buf[t] = u
        self._step += 1

    def _end_trial(self):
        """Called when a trial is complete."""
        error = self.ref_traj - self._q_buf
        rmse  = float(np.sqrt(np.mean(error ** 2)))

        td = TrialData(
            trial=self._trial,
            q_ref=self.ref_traj.copy(),
            q_actual=self._q_buf.copy(),
            u=self._u_buf.copy(),
            error=error,
            rmse=rmse,
        )
        self.buffer.push(td)
        self.ilc.update(td)

        self.logger.log(self._trial, rmse=rmse,
                        max_error=td.max_error, trial=self._trial)

        # Publish convergence
        conv = Float64MultiArray()
        conv.data = self.ilc.convergence_curve().tolist()
        self.pub_conv.publish(conv)

        self.get_logger().info(
            f'Trial {self._trial:03d} | RMSE={rmse:.5f} | '
            f'MaxErr={td.max_error:.4f} | '
            f'{"CONVERGED ✓" if td.converged else "learning..."}'
        )

        self._trial += 1
        self._reset_trial()

    def _reset_trial(self):
        self._step  = 0
        self._q_buf = np.zeros((self.H, self.n))
        self._u_buf = np.zeros((self.H, self.n))

    # ── Status publisher ──────────────────────────────────────────────────────

    def _publish_status(self):
        stats = self.buffer.convergence_stats()
        stats['algorithm'] = self.get_parameter('algorithm').value
        stats['gait']      = self.get_parameter('gait').value
        stats['trial']     = self._trial
        stats['running']   = self._running
        stats['timestamp'] = time.time()

        msg = String()
        msg.data = json.dumps(stats)
        self.pub_stat.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ILCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.logger.save()
        node.get_logger().info('ILC node shut down. Logs saved.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
