# HumanoidLocoLearn

**Data-Driven Learning Control of Humanoid & Legged Robots**

[![CI](https://github.com/glori-a-a/humanoid-loco-learn/actions/workflows/ci.yml/badge.svg)](https://github.com/glori-a-a/humanoid-loco-learn/actions)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11-blue)](https://python.org)
[![ROS2](https://img.shields.io/badge/ROS2-Humble-orange)](https://docs.ros.org/en/humble)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-compatible-green)](https://isaac-sim.github.io/IsaacLab)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)


> *"Data-Driven Learning Control of Humanoid Robots"*


---

## Overview

HumanoidLocoLearn implements a family of **Iterative Learning Control (ILC)** algorithms for legged robots. Through repeated trials, the robot learns a feedforward control signal that compensates for model uncertainty, unmodeled dynamics, and complex contact interactions — without requiring an accurate physics model.

```
Trial k:   robot executes motion  →  measure error e_k(t)
                                            ↓
                                    ILC update law
                                            ↓
Trial k+1: improved feedforward u_{k+1} = u_k + f(e_k)
```

This directly addresses the core challenge of humanoid locomotion: **the dynamics are too complex to model accurately, but data from repeated attempts can be used to learn optimal control actions.**

---

## Architecture

```
humanoid-loco-learn/
├── src/
│   ├── algorithms/
│   │   ├── ilc.py              ← P / PD / Norm-Optimal / Adaptive ILC
│   │   └── replay_buffer.py    ← Prioritised experience replay
│   ├── environments/
│   │   ├── base_env.py         ← Abstract environment interface
│   │   ├── sim_env.py          ← Lightweight built-in simulator
│   │   └── isaac_env.py        ← Isaac Lab / Unitree Go2 wrapper
│   ├── control/
│   │   ├── trajectory.py       ← Gait reference generation (trot/walk/bound)
│   │   └── state_estimator.py  ← Kalman filter for sensor fusion
│   └── utils/
│       ├── metrics.py          ← Convergence analysis
│       └── logger.py           ← JSON experiment logging
├── ros2_ws/src/humanoid_control/
│   └── ilc_node.py             ← Full ROS2 node (real robot deployment)
├── dashboard/
│   ├── backend/app.py          ← FastAPI + WebSocket server
│   └── frontend/index.html     ← Real-time monitoring dashboard
├── scripts/run_ilc.py          ← CLI experiment runner
├── tests/                      ← pytest test suite
├── .github/workflows/ci.yml    ← GitHub Actions CI
└── docker/                     ← Docker + Compose
```

---

## ILC Algorithms

| Algorithm | Update Law | Best For |
|---|---|---|
| **P-type** | `u_{k+1} = u_k + γ·e_k` | Simple, guaranteed convergence |
| **PD-type** | `u_{k+1} = u_k + Kp·e_k + Kd·ė_k` | Faster convergence, phase-sensitive tasks |
| **Norm-Optimal (NOILC)** | Minimises `‖e_{k+1}‖² + λ‖Δu_k‖²` | Research-grade, optimal trade-off |
| **Adaptive** | Adam-style moment estimation | Non-stationary dynamics, robust to noise |

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/glori-a-a/humanoid-loco-learn
cd humanoid-loco-learn
pip install -r requirements.txt
```

### 2. Run learning experiment (simulation)

```bash
python scripts/run_ilc.py --algo noilc --gait trot --trials 100 --plot
```

### 3. Launch real-time dashboard

```bash
uvicorn dashboard.backend.app:app --reload
# Open http://localhost:8000
```

### 4. Run with Isaac Lab (Ubuntu 22.04 + GPU)

```bash
python scripts/run_ilc.py --algo noilc --backend isaac --gait trot
```

### 5. ROS2 deployment (Unitree Go2 / humanoid)

```bash
cd ros2_ws
colcon build --packages-select humanoid_control
source install/setup.bash
ros2 run humanoid_control ilc_node \
  --ros-args -p algorithm:=noilc -p gait:=trot -p auto_start:=true
```

### 6. Docker

```bash
docker compose -f docker/docker-compose.yml up
```

---

## Real-Time Dashboard

Open `http://localhost:8000` after launching the backend.

Features:
- Live convergence curve (RMSE per trial)
- Per-joint tracking error bars
- Feedforward signal norm evolution
- Algorithm switching at runtime (P / PD / NOILC / Adaptive)
- Gait switching (trot / walk / bound / pace)
- WebSocket push — zero-latency updates

---

## Results (Simulation)

| Algorithm | Trials to 90% improvement | Final RMSE |
|---|---|---|
| P-type ILC | ~80 | 0.018 rad |
| PD-type ILC | ~45 | 0.012 rad |
| Norm-Optimal ILC | ~25 | 0.006 rad |
| Adaptive ILC | ~30 | 0.009 rad |

*Tested on Unitree Go2 trot gait, 12 DOF, LightweightSimEnv with ±15% stiffness uncertainty.*

---

## Tests & CI

```bash
pytest tests/ -v --cov=src
```

GitHub Actions runs tests on Python 3.10 and 3.11 on every push.

---

## Roadmap

- [ ] Sim-to-real transfer experiments on physical Unitree Go2
- [ ] TienKung humanoid integration ([Open-X-Humanoid](https://github.com/Open-X-Humanoid/TienKung-Lab))
- [ ] Multi-task ILC (generalisation across gait patterns)
- [ ] Model-free RL baseline comparison
- [ ] Hardware-in-the-loop experiments

---

## Author

**Xinyue Zhang**
MSc Advanced Control and Systems Engineering
University of Manchester
[github.com/glori-a-a](https://github.com/glori-a-a)
