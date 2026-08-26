# Heterogeneous GPU Cluster Scheduler with PPO & Discrete-Event Simulation

A discrete-event GPU cluster simulator and reinforcement learning environment (wrapped with OpenEnv) where a Proximal Policy Optimization (PPO) agent learns online workload placement across heterogeneous GPU nodes and is evaluated against classical scheduling baselines (FIFO, SJF, Priority, Best-Fit).

## Features
- **Discrete-Event Simulator**: Event-driven simulation (`JOB_ARRIVAL`, `JOB_COMPLETION`, `SCHEDULING_POINT`) for high-throughput trajectory rollouts without fixed-timestep overhead.
- **Heterogeneous GPU Topologies**: Models diverse GPU types (e.g., A100 80GB/40GB, H100 80GB, A10 24GB) with discrete GPU and VRAM capacity tracking.
- **Stochastic Workload Generator**: Six realistic AI workload scenarios (Balanced, Training-Heavy, Short-Job, Bursty, GPU Fragmentation, High-Load).
- **OpenEnv Integration**: OpenEnv server + client wrapping Gymnasium environment interface.
- **Classical Baselines**: Deterministic baselines (FIFO, Shortest Job First, Priority, Best-Fit) evaluated on identical seeds.
- **Custom PPO from Scratch**: Pure PyTorch actor-critic policy with Generalized Advantage Estimation (GAE), action masking, and GPU acceleration.
- **Parallel Environment Vectorization**: Synchronous multi-environment rollouts for rapid training.
- **Inference API**: FastAPI service for real-time scheduler inference and dashboard integration.
- **Notebooks**: Interactive exploration, training, and benchmarking notebooks via `notebooks/`.

## Directory Structure
```
GPU_scheduling_sim/
├── simulator/         # Core discrete-event simulator (Cluster, Node, GPU, Job, Queue)
├── workloads/         # Statistical distributions and scenario generators
├── env/               # OpenEnv models, server, and client
├── baselines/         # Classical heuristics (FIFO, SJF, SJF, BestFit)
├── rl/                # PPO implementation (ActorCritic, Buffer, Trainer)
├── evaluation/        # Unified metrics, scenario runner, report generation
├── api/               # FastAPI inference service for dashboard/UI
├── configs/           # YAML configurations for cluster, training, and rewards
├── notebooks/         # Jupyter notebooks for training, exploration, and evaluation
├── scripts/           # Standalone execution scripts
└── tests/             # Comprehensive pytest test suite
```

## Quick Start

### 1. Environment Setup
```bash
conda activate pygpu
pip install -e .
```

### 2. Run Tests
```bash
pytest tests/ -v
```

### 3. Run Baselines
```bash
python scripts/run_baselines.py
```

### 4. Train PPO
```bash
python scripts/train_ppo.py --config configs/training.yaml
```

### 5. Launch Inference API
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```
