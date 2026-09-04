# 🚀 Deep Reinforcement Learning GPU Cluster Scheduler (OpenEnv + PPO from Scratch)

An enterprise-grade, high-performance **GPU Cluster Scheduler** powered by **Proximal Policy Optimization (PPO)** written completely from scratch with PyTorch, CUDA hardware acceleration, OpenEnv protocol integration, and an interactive real-time Web Dashboard.

---

## 🌟 Highlights & Architecture

- **Discrete-Event Simulator Engine (`simulator/`)**: Event-driven queue simulation with sub-millisecond precision, tracking heterogeneous GPU nodes (A100 80GB, A100 40GB, H100 80GB, A10 24GB), granular VRAM allocations, multi-GPU topology constraints, and job life cycles.
- **Realistic Stochastic Workload Generator (`workloads/`)**: Models 6 real-world cluster scenarios (`balanced`, `training_heavy`, `short_job_heavy`, `bursty`, `gpu_fragmentation`, `high_load`) using log-normal runtime distributions and Poisson arrival processes.
- **Standardized OpenEnv Integration (`env/`)**: Fully compliant OpenEnv client/server architecture with Gymnasium discrete action spaces and logit-level action masking.
- **Classical Heuristic Baselines (`baselines/`)**: Battle-tested industry baselines (FIFO, Shortest Job First, Priority-First, Best-Fit Bin-Packing).
- **Custom PPO Engine with CUDA Acceleration (`rl/`)**:
  - Vectorized multi-environment parallel rollouts ($N=8$ CPU workers).
  - Generalized Advantage Estimation (GAE $\lambda=0.95, \gamma=0.99$).
  - Strict logit-level action masking ($-\infty$ masking on invalid placements).
  - Proactive future arrival lookahead features.
  - Multi-objective shaped reward formulation.
- **FastAPI REST API & Real-Time Web Dashboard (`api/`, `frontend/`)**:
  - Interactive live cluster topology visualizer with real-time VRAM progress bars and GPU core chip badges.
  - Interactive Custom Job Injector (submit live training/inference jobs on the fly).
  - Live policy switcher (PPO vs FIFO vs SJF vs Priority vs Best-Fit).
  - Live streaming decision feed.

---

## 📊 Final Benchmark Results (Held-Out Test Seeds)

| Scenario | Metric | FIFO | SJF | Priority | BestFit | **PPO (Ours)** | Winner |
|---|---|---|---|---|---|---|---|
| **Balanced** | **GPU Utilization (%)** | 36.87% | 30.22% | 31.80% | 33.26% | **45.84%** | **PPO (+8.9%)** |
| **Balanced** | **Throughput (jobs/hr)** | 49.33 | 40.67 | 41.33 | 43.00 | **54.67** | **PPO (+5.3)** |
| **Balanced** | **Deadline Miss (%)** | 29.60% | 25.45% | 27.99% | 34.26% | **23.45%** | **PPO (Lowest)** |
| **Short Job** | **Mean JCT (s)** | 68.46s | 47.79s | 55.49s | 66.58s | **46.81s** | **PPO (Fastest)** |
| **Short Job** | **P95 Tail JCT (s)** | 188.24s | 108.91s | 116.65s | 185.42s | **97.65s** | **PPO (Lowest)** |
| **Short Job** | **Mean Queue Wait (s)** | 40.22s | 19.82s | 27.22s | 38.43s | **18.73s** | **PPO (Lowest)** |
| **GPU Frag.** | **Deadline Miss (%)** | 22.12% | 21.85% | 23.96% | 22.24% | **20.03%** | **PPO (Lowest)** |
| **GPU Frag.** | **Cumulative Reward** | +23.49 | +25.04 | +24.27 | +24.98 | **+27.63** | **PPO (+2.6)** |

---

## 🚀 Quickstart

### 1. Run Complete 29-Test Verification Suite
```bash
python tests/run_tests.py
```

### 2. Launch FastAPI Server & Live Web Dashboard
```bash
python scripts/run_server.py --port 8000
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser to interact with the live cluster dashboard.

### 3. Train PPO from Scratch (CLI or Notebook)
```bash
python scripts/train_ppo.py --timesteps 500000 --num-envs 8
```
Or open [`notebooks/05_ppo_training.ipynb`](notebooks/05_ppo_training.ipynb).

### 4. Run Final Comprehensive Benchmark
```bash
python scripts/evaluate.py --checkpoint checkpoints/ppo_final.pt --test-seeds 501 602 703
```
Or open [`notebooks/06_final_evaluation.ipynb`](notebooks/06_final_evaluation.ipynb).

### Deploy To Vercel

The repository includes a Vercel ASGI entry point at `api/index.py` and a `vercel.json` rewrite so FastAPI serves both the dashboard and API routes. Install the runtime locally with `pip install -r requirements.txt`, then run the dashboard with `uvicorn api.app:app --host 0.0.0.0 --port 8000`.

Before a Git-based Vercel deployment, add the trained model artifact with `git add checkpoints/ppo_final.pt`; the `.gitignore` intentionally includes this final checkpoint while excluding intermediate training checkpoints.

The live simulator keeps mutable state in memory. That is reliable under local Uvicorn, but Vercel functions may run on different instances between requests. For a multi-user production deployment, persist the simulator/session state in an external store before relying on live run continuity.

---

## 📁 Repository Structure

```text
GPU_scheduling_sim/
├── api/
│   ├── app.py                     # FastAPI web application & static router
│   ├── inference.py               # Singleton cluster & multi-policy inference manager
│   └── schemas.py                 # Pydantic request/response schemas
├── baselines/
│   ├── base.py                    # BaseScheduler interface
│   ├── best_fit.py                # Best-Fit Bin-Packing heuristic
│   ├── fifo.py                    # First-In First-Out baseline
│   ├── priority.py                # Priority-First baseline
│   ├── runner.py                  # Single-episode baseline runner
│   └── sjf.py                     # Shortest Job First baseline
├── checkpoints/
│   └── ppo_final.pt               # Trained PyTorch ActorCritic weights
├── configs/
│   ├── cluster_medium.yaml        # 4-node 16-GPU cluster definition
│   ├── cluster_small.yaml         # 2-node 8-GPU cluster definition
│   ├── reward.yaml                # Multi-objective shaped reward config
│   └── training.yaml              # PPO hyperparameters & vectorization config
├── env/
│   ├── client.py                  # OpenEnv HTTP/Gymnasium client wrapper
│   ├── models.py                  # Observation, Action, Info Pydantic models
│   └── server/
│       ├── app.py                 # OpenEnv FastAPI environment server
│       └── gpu_scheduler_environment.py # Gymnasium environment with action masking
├── evaluation/
│   ├── evaluator.py               # Multi-scenario evaluation engine
│   ├── metrics.py                 # Core telemetry & SLA calculation
│   ├── ppo_policy.py              # PPO policy wrapper for BaseScheduler
│   └── reports.py                 # Tabular DataFrame & Markdown table formatters
├── frontend/
│   ├── app.js                     # Real-time dashboard controller (polling, step, auto-play)
│   ├── index.html                 # Dark-mode dashboard HTML single page
│   └── style.css                  # CSS design system (glassmorphism, glowing resource bars)
├── notebooks/
│   ├── 01_simulator_exploration.ipynb
│   ├── 02_workload_analysis.ipynb
│   ├── 03_openenv_env_demo.ipynb
│   ├── 04_baseline_comparison.ipynb
│   ├── 05_ppo_training.ipynb
│   └── 06_final_evaluation.ipynb
├── results/
│   ├── baselines_benchmark.json
│   ├── final_evaluation_report.md
│   └── final_evaluation_results.json
├── rl/
│   ├── actor_critic.py            # Neural network with logit-level action masking
│   ├── buffer.py                  # Rollout buffer with Generalized Advantage Estimation
│   ├── config.py                  # PPOConfig dataclass
│   ├── ppo.py                     # PPO algorithm (clipped surrogate loss, value MSE)
│   ├── trainer.py                 # Batched CUDA trainer loop with domain randomization
│   └── vector_env.py              # SyncVectorEnv parallel multi-scenario rollout runner
├── scripts/
│   ├── evaluate.py                # Full benchmark evaluation CLI
│   ├── run_baselines.py           # Baseline evaluation CLI
│   ├── run_env.py                 # OpenEnv server runner
│   ├── run_server.py              # Production API & dashboard server launcher
│   └── train_ppo.py               # PPO training CLI
├── simulator/
│   ├── cluster.py                 # Heterogeneous cluster manager
│   ├── events.py                  # Event queue & event types
│   ├── gpu.py                     # Physical GPU model with VRAM tracking
│   ├── job.py                     # Job definition, SLA, and lifecycle states
│   ├── node.py                    # Heterogeneous server node model
│   ├── queue.py                   # Cluster scheduling wait queue
│   ├── scheduler_state.py         # State snapshot with action masking logic
│   └── simulator.py               # Discrete-event simulation engine
├── tests/
│   ├── run_tests.py               # Automated test runner (29 tests)
│   ├── test_api.py                # FastAPI REST API unit tests
│   ├── test_baselines.py          # Heuristic logic tests
│   ├── test_cluster.py            # Node & Cluster invariant tests
│   ├── test_environment.py        # Gymnasium & OpenEnv tests
│   ├── test_evaluation.py         # PPO inference & evaluation tests
│   ├── test_gpu.py                # GPU lifecycle tests
│   ├── test_ppo.py                # PPO neural net & GAE tests
│   ├── test_queue.py              # Queue tests
│   ├── test_simulator.py          # Simulator deterministic stepping tests
│   └── test_workloads.py          # Stochastic workload scenario tests
├── openenv.yaml
└── pyproject.toml
```
