"""
Real-Time ILC Monitoring Dashboard — FastAPI Backend
=====================================================
Serves:
  - REST API for experiment control and history
  - WebSocket stream for real-time learning metrics
  - Static frontend files

WebSocket message format (JSON, pushed every control step):
{
  "trial":        42,
  "step":         312,
  "rmse":         0.0341,
  "rmse_history": [0.21, 0.18, ...],
  "joint_errors": [0.01, 0.03, ...],    // per-joint, current step
  "u_ff_norm":    1.23,
  "converged":    false,
  "algorithm":    "noilc",
  "gait":         "trot",
  "timestamp":    1711234567.89
}
"""

import asyncio
import json
import time
import numpy as np
from pathlib import Path
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

import sys, os
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.algorithms.ilc import ILCConfig, make_ilc
from src.algorithms.replay_buffer import ReplayBuffer
from src.environments.sim_env import LightweightSimEnv
from src.environments.base_env import EnvConfig
from src.control.trajectory import TrajectoryGenerator, TrajectoryConfig, GaitType
from src.utils.metrics import compute_tracking_metrics
from src.utils.logger import ExperimentLogger


# ── App init ──────────────────────────────────────────────────────────────────
app = FastAPI(title="HumanoidLocoLearn Dashboard",
              description="Real-time ILC learning control monitor",
              version="1.0.0")

FRONTEND = Path(__file__).parent.parent / "frontend"
if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")

# ── Global experiment state ───────────────────────────────────────────────────
class ExperimentState:
    def __init__(self):
        self.reset()

    def reset(self, algorithm='noilc', gait='trot', n_joints=12):
        ilc_cfg = ILCConfig(n_joints=n_joints, horizon=300, dt=0.02)
        env_cfg = EnvConfig(n_joints=n_joints, episode_length=300, dt=0.02)
        traj_cfg = TrajectoryConfig(n_joints=n_joints, horizon=300,
                                    gait=GaitType(gait), dt=0.02)

        self.ilc = make_ilc(algorithm, ilc_cfg)
        self.env = LightweightSimEnv(env_cfg, gait=gait)
        self.traj = TrajectoryGenerator(traj_cfg).generate()
        self.buffer = ReplayBuffer()
        self.logger = ExperimentLogger('/tmp/dashboard_logs')

        self.algorithm  = algorithm
        self.gait       = gait
        self.n_joints   = n_joints
        self.trial      = 0
        self.running    = False
        self.max_trials = 100
        self.last_rmse  = 0.0
        self.rmse_history: list[float] = []
        self.current_joint_errors = np.zeros(n_joints).tolist()
        self.u_ff_norm = 0.0

exp = ExperimentState()
clients: Set[WebSocket] = set()


# ── Background learning loop ──────────────────────────────────────────────────

async def run_one_trial(state: ExperimentState) -> dict:
    """Run one ILC trial and return metrics dict."""
    H, n = state.ilc.cfg.horizon, state.ilc.cfg.n_joints
    q_buf = np.zeros((H, n))
    u_buf = np.zeros((H, n))

    state.env.reset()

    for t in range(H):
        q_ref = state.traj[t]
        # Get current joint state from env
        from src.environments.base_env import RobotState
        env_state = state.env._build_state()
        q = env_state.joint_pos
        dq = env_state.joint_vel

        # Feedforward + PD feedback
        u_ff = state.ilc.u_ff[t]
        u_fb = 40.0 * (q_ref - q) - 2.0 * dq
        u = np.clip(u_ff + u_fb, -5.0, 5.0)

        state.env.step(u)
        q_buf[t] = q
        u_buf[t] = u

        # Broadcast step update to websocket clients every 10 steps
        if t % 10 == 0 and clients:
            step_msg = {
                "type":          "step",
                "trial":         state.trial,
                "step":          t,
                "joint_errors":  (q_ref - q).tolist(),
                "u_ff_norm":     float(np.linalg.norm(u_ff)),
                "timestamp":     time.time(),
            }
            await broadcast(step_msg)

        await asyncio.sleep(0)   # yield to event loop

    error = state.traj - q_buf
    rmse  = float(np.sqrt(np.mean(error ** 2)))

    from src.algorithms.ilc import TrialData
    td = TrialData(trial=state.trial, q_ref=state.traj.copy(),
                   q_actual=q_buf, u=u_buf, error=error, rmse=rmse)
    state.buffer.push(td)
    state.ilc.update(td)
    state.logger.log(state.trial, rmse=rmse, max_error=td.max_error)

    state.last_rmse = rmse
    state.rmse_history.append(rmse)
    state.current_joint_errors = np.sqrt(np.mean(error**2, axis=0)).tolist()
    state.u_ff_norm = float(np.linalg.norm(state.ilc.u_ff))
    state.trial += 1

    return {
        "type":         "trial_end",
        "trial":        state.trial,
        "rmse":         rmse,
        "rmse_history": state.rmse_history[-50:],
        "joint_errors": state.current_joint_errors,
        "u_ff_norm":    state.u_ff_norm,
        "converged":    td.converged,
        "algorithm":    state.algorithm,
        "gait":         state.gait,
        "timestamp":    time.time(),
    }


async def learning_loop():
    """Background task: run ILC trials while exp.running == True."""
    while True:
        if exp.running and exp.trial < exp.max_trials:
            msg = await run_one_trial(exp)
            await broadcast(msg)
            if msg["converged"]:
                exp.running = False
                await broadcast({"type": "converged", "trial": exp.trial})
        else:
            await asyncio.sleep(0.1)


async def broadcast(msg: dict):
    dead = set()
    for ws in clients:
        try:
            await ws.send_json(msg)
        except Exception:
            dead.add(ws)
    clients.difference_update(dead)


@app.on_event("startup")
async def startup():
    asyncio.create_task(learning_loop())


# ── REST endpoints ────────────────────────────────────────────────────────────

class StartRequest(BaseModel):
    algorithm: str = "noilc"
    gait: str = "trot"
    max_trials: int = 100


@app.post("/api/start")
async def start(req: StartRequest):
    exp.reset(algorithm=req.algorithm, gait=req.gait)
    exp.max_trials = req.max_trials
    exp.running = True
    return {"status": "started", "algorithm": req.algorithm, "gait": req.gait}


@app.post("/api/stop")
async def stop():
    exp.running = False
    return {"status": "stopped", "trial": exp.trial}


@app.post("/api/reset")
async def reset():
    exp.reset()
    return {"status": "reset"}


@app.get("/api/status")
async def status():
    stats = exp.buffer.convergence_stats()
    return JSONResponse({
        "trial":        exp.trial,
        "running":      exp.running,
        "algorithm":    exp.algorithm,
        "gait":         exp.gait,
        "last_rmse":    exp.last_rmse,
        "rmse_history": exp.rmse_history[-100:],
        "converged":    exp.last_rmse < 1e-3 and exp.trial > 0,
        **stats,
    })


@app.get("/api/history")
async def history():
    return JSONResponse([
        {"trial": t.trial, "rmse": t.rmse, "max_error": t.max_error}
        for t in exp.buffer.latest(50)
    ])


@app.get("/api/metrics")
async def metrics():
    trials = list(exp.buffer._buffer)
    if not trials:
        return JSONResponse({"error": "No trials recorded yet."})
    return JSONResponse(compute_tracking_metrics(trials))


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        # Send current state immediately on connect
        await ws.send_json({
            "type":         "init",
            "trial":        exp.trial,
            "running":      exp.running,
            "algorithm":    exp.algorithm,
            "gait":         exp.gait,
            "rmse_history": exp.rmse_history[-50:],
        })
        while True:
            await ws.receive_text()   # keep-alive
    except WebSocketDisconnect:
        clients.discard(ws)


# ── Frontend ──────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    index = FRONTEND / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text())
    return HTMLResponse("<h1>Dashboard backend running. "
                        "Open /docs for API reference.</h1>")
