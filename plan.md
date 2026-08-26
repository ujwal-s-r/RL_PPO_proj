# GPU Cluster Scheduler — PPO-Based Resource Scheduling

## 1. Project Objective

Build a **production-grade simulated GPU cluster scheduler** where a PPO agent learns to make online workload-placement decisions.

The system models a heterogeneous AI compute cluster receiving jobs over time. At every scheduling opportunity, the scheduler observes the current cluster and queue state and decides which queued job should be placed on which available node.

The project must establish whether PPO can outperform strong deterministic scheduling policies under realistic and previously unseen workload patterns.

### V1 boundary

V1 is **simulation-only**.

```text
Workload Generator
       ↓
Discrete-Event GPU Cluster Simulator
       ↓
OpenEnv Environment
       ↓
PPO Scheduler
       ↓
Evaluation Framework
       ↓
PPO vs Classical Baselines
```

### V2 — explicitly out of scope for V1

```text
Kubernetes
vLLM
real GPUs
real cluster deployment
Kubernetes scheduler plugins/controllers
production inference workloads
```

V2 will replace parts of the simulated control plane with real Kubernetes/vLLM infrastructure.

---

# 2. Core Problem

Simulate an AI cluster containing heterogeneous GPU nodes.

Example:

```text
Node 0: 4 × A100 80GB
Node 1: 8 × A100 40GB
Node 2: 4 × H100 80GB
Node 3: 8 × A10 24GB
```

Jobs continuously arrive into a queue.

Each job has:

* GPU requirement
* VRAM requirement
* expected runtime
* priority
* deadline
* arrival time
* preemptibility
* workload class

The scheduler must decide how to allocate available resources.

---

# 3. Scheduling Objective

The scheduler should balance:

1. GPU utilization
2. Job completion time
3. Queue waiting time
4. Throughput
5. Deadline satisfaction
6. Fairness/starvation avoidance
7. Scheduling stability

The primary optimization objective is:

```text
maximize useful cluster throughput
while minimizing waiting time,
completion time,
GPU fragmentation,
and deadline violations.
```

---

# 4. Environment Architecture

Use **Hugging Face OpenEnv** as the environment interface.

OpenEnv provides Gymnasium-style `reset`, `step`, and state interaction while allowing the environment to run as an isolated service/container.

Architecture:

```text
                    ┌──────────────────────┐
                    │   PPO Training Loop  │
                    └──────────┬───────────┘
                               │
                         OpenEnv Client
                               │
                               ▼
                    ┌──────────────────────┐
                    │   OpenEnv Server     │
                    │      FastAPI         │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ GPU Cluster Env      │
                    │                      │
                    │ Discrete Event       │
                    │ Simulator            │
                    └──────────┬───────────┘
                               │
                 ┌─────────────┼─────────────┐
                 ▼             ▼             ▼
              Cluster       Queue        Workloads
                State        State        Generator
```

Use the current OpenEnv project structure rather than inventing a custom environment protocol. OpenEnv's current scaffolding separates models, client, server, environment implementation, manifest, and dependencies.

---

# 5. Repository Structure

Target structure:

```text
gpu-scheduler-rl/
│
├── README.md
├── pyproject.toml
├── openenv.yaml
├── .gitignore
│
├── env/
│   ├── __init__.py
│   ├── models.py
│   ├── client.py
│   │
│   └── server/
│       ├── app.py
│       └── gpu_scheduler_environment.py
│
├── simulator/
│   ├── __init__.py
│   ├── cluster.py
│   ├── node.py
│   ├── gpu.py
│   ├── job.py
│   ├── queue.py
│   ├── events.py
│   ├── scheduler_state.py
│   ├── simulator.py
│   └── workload.py
│
├── rl/
│   ├── __init__.py
│   ├── ppo.py
│   ├── actor_critic.py
│   ├── rollout.py
│   ├── buffer.py
│   ├── trainer.py
│   └── config.py
│
├── baselines/
│   ├── fifo.py
│   ├── sjf.py
│   ├── priority.py
│   ├── best_fit.py
│   └── runner.py
│
├── evaluation/
│   ├── evaluator.py
│   ├── metrics.py
│   ├── scenarios.py
│   └── reports.py
│
├── workloads/
│   ├── distributions.py
│   ├── generators.py
│   └── scenarios.py
│
├── configs/
│   ├── small.yaml
│   ├── medium.yaml
│   └── training.yaml
│
├── scripts/
│   ├── run_env.py
│   ├── train_ppo.py
│   ├── evaluate.py
│   └── run_baselines.py
│
└── tests/
    ├── test_cluster.py
    ├── test_jobs.py
    ├── test_queue.py
    ├── test_simulator.py
    ├── test_environment.py
    ├── test_baselines.py
    └── test_ppo.py
```

Keep simulator logic independent from OpenEnv. The simulator should be usable directly for deterministic testing and baseline evaluation.

---

# 6. Cluster Model

Represent a cluster as:

```python
Cluster
    nodes: list[Node]
    current_time: float
```

Each node contains:

```text
Node
├── node_id
├── gpu_type
├── gpu_count
├── GPUs
├── CPU capacity
├── RAM capacity
├── network/topology metadata
└── running_jobs
```

Each GPU contains:

```text
GPU
├── gpu_id
├── gpu_type
├── total_vram
├── allocated_vram
├── utilization
└── running_job
```

V1 should support heterogeneous GPU types.

Example configuration:

```yaml
nodes:
  - type: A100_80GB
    count: 4
    vram_gb: 80

  - type: A100_40GB
    count: 8
    vram_gb: 40

  - type: H100_80GB
    count: 4
    vram_gb: 80

  - type: A10_24GB
    count: 8
    vram_gb: 24
```

Do not model CUDA kernels or actual GPU execution in V1.

Runtime is represented by simulated job duration.

---

# 7. Job Model

Each job should contain:

```text
job_id
arrival_time
gpu_count
vram_per_gpu
estimated_runtime
priority
deadline
workload_type
preemptible
status
start_time
completion_time
allocated_node
allocated_gpus
```

Example:

```text
Job 1027

GPUs:             4
VRAM/GPU:         40 GB
Runtime:          900 sec
Priority:         7
Deadline:         1800 sec
Type:             training
Preemptible:      true
```

Supported workload classes:

```text
training
fine_tuning
evaluation
inference
experiment
```

The workload class should primarily influence generated characteristics rather than directly exposing a perfect scheduling answer to the agent.

---

# 8. Discrete-Event Simulation

Do **not** advance the simulator using tiny fixed time steps such as:

```text
t += 0.01
```

Use a discrete-event model.

Important events:

```text
JOB_ARRIVAL
JOB_COMPLETION
JOB_PREEMPTION
SCHEDULING_POINT
```

The simulator advances directly to the next relevant event.

Example:

```text
t=0       Job A arrives
t=10      Job B arrives
t=15      Scheduler dispatches A
t=400     Job A completes
t=400     Scheduler dispatches B
```

This keeps the simulation computationally efficient and makes the environment suitable for large-scale training.

---

# 9. Scheduling Decision

For V1, keep the action space manageable.

### Action

```text
(job_index, node_index)
```

The agent chooses:

```text
which queued job
+
which cluster node
```

The simulator determines whether the allocation is feasible.

Do not initially include:

```text
migration
preemption
GPU-level placement
MIG
```

These are later extensions.

However, the architecture should not prevent adding them later.

---

# 10. Invalid Actions

Invalid actions must be handled explicitly.

Examples:

```text
job does not exist
node does not have enough GPUs
node does not have enough VRAM
queue is empty
job already scheduled
```

Do not silently accept invalid actions.

Preferred behavior:

```text
invalid action
    ↓
negative reward
    ↓
state remains consistent
```

The environment must never enter an invalid cluster state.

---

# 11. Observation Space

The observation must contain enough information for the policy to make scheduling decisions.

Include:

### Cluster features

Per node:

```text
GPU count
available GPU count
GPU utilization
total VRAM
free VRAM
running job count
CPU utilization
RAM utilization
```

### Queue features

For each queue slot:

```text
job present
GPU requirement
VRAM requirement
estimated runtime
priority
time waiting
deadline remaining
preemptible
workload type
```

### Global features

```text
current simulation time
queue length
arrival rate
cluster utilization
```

Use a **fixed maximum queue size** for the initial PPO implementation.

Pad unused queue positions.

This keeps the neural-network input dimensionality fixed.

---

# 12. Action Masking

Implement action masking.

The agent should not waste probability mass on obviously impossible:

```text
(job, node)
```

pairs.

Create:

```text
valid_action_mask[job, node]
```

where:

```text
1 = feasible
0 = infeasible
```

The PPO policy should apply the mask to action logits before sampling.

This is important because the raw Cartesian action space grows with queue size × node count.

---

# 13. Reward Function

Use a shaped reward based on system outcomes.

Recommended formulation:

```text
R_t =
    + throughput_reward
    - waiting_cost
    - utilization_penalty
    - deadline_penalty
    - invalid_action_penalty
```

More concretely:

```text
waiting_cost
    ∝ number of queued jobs × elapsed simulation time

deadline_penalty
    ∝ jobs missing deadlines

utilization_penalty
    ∝ unused allocatable GPU capacity

throughput_reward
    ∝ completed jobs / useful work
```

Avoid making the reward excessively dominated by any single term.

All reward coefficients must live in configuration rather than being hard-coded.

Example:

```yaml
reward:
  completion: 1.0
  waiting: 0.01
  idle_gpu: 0.01
  deadline_miss: 2.0
  invalid_action: 1.0
```

The exact coefficients should be tuned empirically during development.

---

# 14. Episode Definition

An episode represents a simulated scheduling horizon.

Example:

```text
episode duration = 1 hour simulated time
```

At reset:

```text
seed workload generator
reset cluster
reset queue
reset metrics
set simulation time = 0
```

The episode ends when:

```text
simulation horizon reached
```

Optionally allow the simulator to drain remaining jobs for final accounting, but distinguish:

```text
jobs completed within horizon
jobs completed after horizon
unfinished jobs
```

---

# 15. Workload Generation

Do not train on one fixed workload.

Implement stochastic workload generation.

Parameters:

```text
arrival rate
runtime distribution
GPU requirement distribution
VRAM requirement distribution
priority distribution
deadline distribution
workload mix
burstiness
```

Scenarios:

### Balanced

```text
mixed workload
moderate arrival rate
```

### Training-heavy

```text
long-running
multi-GPU jobs
```

### Short-job-heavy

```text
many short jobs
```

### Bursty

```text
low traffic
→ sudden arrival spike
→ recovery
```

### GPU-fragmentation-heavy

Many jobs requiring different GPU counts and VRAM capacities.

### High-load

Arrival rate approaches/exceeds cluster service capacity.

---

# 16. Training/Test Separation

This is critical.

The PPO agent must not simply memorize workload seeds.

Training:

```text
random seeds
randomized workload parameters
```

Validation:

```text
different seeds
```

Test:

```text
completely unseen seeds
different workload distributions
```

Evaluation must report performance over multiple independent seeds.

---

# 17. Classical Baselines

Implement these before PPO.

## FIFO

Schedule the oldest feasible job first.

```text
queue order
→ first feasible job
→ first feasible node
```

## Shortest Job First

Prioritize the job with the smallest estimated runtime.

## Priority

Highest priority first.

Tie-break using waiting time.

## Best Fit

Choose the placement that minimizes resource waste.

For example:

```text
remaining GPU capacity
+
remaining VRAM capacity
```

after placement.

These baselines must use the **same simulator, workload generator, metrics, and evaluation seeds** as PPO.

This makes the comparison fair.

---

# 18. PPO Implementation

Implement PPO directly rather than hiding the algorithm behind a high-level RL library.

This is intentional.

The project should demonstrate understanding of:

```text
policy network
value network
rollout collection
GAE
advantages
returns
clipped surrogate objective
entropy bonus
value loss
minibatch updates
gradient clipping
```

Architecture:

```text
Observation
    ↓
MLP Encoder
    ↓
Policy Head
    ↓
Action logits

Observation
    ↓
MLP Encoder
    ↓
Value Head
    ↓
State value
```

The policy should support action masking.

---

# 19. PPO Configuration

Expose configuration for:

```text
learning_rate
gamma
gae_lambda
clip_ratio
entropy_coef
value_coef
max_grad_norm
rollout_length
minibatch_size
epochs_per_update
num_envs
total_timesteps
```

Use deterministic seeds.

Save:

```text
model checkpoint
optimizer state
training configuration
environment configuration
random seed
normalization statistics
```

---

# 20. Training Pipeline

The training pipeline should be:

```text
Load config
    ↓
Create OpenEnv environments
    ↓
Initialize PPO
    ↓
Collect rollout
    ↓
Compute rewards/returns/GAE
    ↓
PPO update
    ↓
Evaluate periodically
    ↓
Save checkpoint
```

Training logs should include:

```text
episode reward
mean JCT
queue wait
GPU utilization
throughput
deadline violations
policy entropy
value loss
policy loss
approx KL
clip fraction
```

Do not use reward as the only training signal.

---

# 21. Evaluation Framework

Create one unified evaluator.

Input:

```text
policy
workload scenario
random seeds
environment configuration
```

Output:

```text
mean
std
P50
P95
```

where applicable.

Metrics:

```text
mean job completion time
P95 job completion time
mean queue waiting time
GPU utilization
throughput
deadline violation rate
completed jobs
unfinished jobs
starvation rate
```

---

# 22. Main Comparison

The final V1 experiment should produce:

```text
FIFO
SJF
Priority
Best-Fit
PPO
```

evaluated on identical test workloads.

Example output:

```text
                    FIFO   SJF   Priority   BestFit   PPO
------------------------------------------------------------
Mean JCT
P95 JCT
Queue Wait
GPU Utilization
Throughput
Deadline Violations
Starvation
```

The exact numbers are generated by the experiments. Never fabricate expected performance.

The goal is not to guarantee that PPO wins every metric.

The goal is to determine **where PPO provides a useful scheduling policy and where classical heuristics remain superior**.

---

# 23. Reproducibility

Every experiment must record:

```text
git commit
config
environment version
PPO hyperparameters
workload parameters
random seeds
checkpoint
evaluation seeds
```

Use deterministic seed handling throughout:

```text
Python
NumPy
PyTorch
environment
workload generator
```

---

# 24. Testing Requirements

Before PPO training, the simulator must pass behavioral tests.

Test:

```text
job arrival
job completion
resource allocation
resource release
VRAM constraints
GPU constraints
queue ordering
deadline calculation
invalid actions
episode termination
random seed reproducibility
```

Baseline tests:

```text
FIFO correctness
SJF correctness
priority correctness
best-fit correctness
```

Environment tests:

```text
reset consistency
step consistency
observation shape
action validation
action mask correctness
reward calculation
state invariants
```

Critical invariant:

```text
allocated GPUs <= physical GPUs
allocated VRAM <= GPU VRAM
```

must always hold.

---

# 25. Development Phases

## Phase 0 — Project skeleton

Set up:

```text
OpenEnv
Python package
configuration
logging
testing
Docker
```

Deliverable:

```text
OpenEnv environment starts successfully.
```

---

## Phase 1 — Simulator

Implement:

```text
GPU
Node
Cluster
Job
Queue
Events
Simulator
```

Deliverable:

```text
Random jobs can execute correctly
through the simulated cluster.
```

No PPO yet.

---

## Phase 2 — Workload generator

Implement:

```text
multiple workload distributions
random seeds
scenario configurations
```

Deliverable:

```text
repeatable workload generation
with controllable difficulty.
```

---

## Phase 3 — OpenEnv integration

Implement:

```text
Action
Observation
State
Environment
Client
Server
```

Deliverable:

```text
reset()
step()
state()
```

work correctly through OpenEnv.

---

## Phase 4 — Baselines

Implement:

```text
FIFO
SJF
Priority
Best-Fit
```

Deliverable:

```text
all baselines run through the same environment
and evaluation framework.
```

---

## Phase 5 — PPO

Implement:

```text
actor
critic
rollout buffer
GAE
PPO update
action masking
checkpointing
training loop
```

Deliverable:

```text
PPO successfully learns a scheduling policy.
```

---

## Phase 6 — Evaluation

Run:

```text
PPO vs FIFO
PPO vs SJF
PPO vs Priority
PPO vs Best-Fit
```

across multiple unseen workload scenarios and seeds.

Deliverable:

```text
reproducible evaluation report.
```

---

## Phase 7 — Production-grade V1 packaging

Add:

```text
Docker
configuration management
structured logging
checkpoint management
experiment metadata
CLI scripts
tests
documentation
```

The final system should be runnable with a small number of commands and should not depend on notebooks.

---

# 26. Future V2 Boundary

Do not implement these now.

V2 will evolve the system toward:

```text
                    PPO Scheduler
                          │
                          ▼
                 Kubernetes Control Plane
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
          GPU Nodes               GPU Nodes
              │                       │
            vLLM                    vLLM
              │                       │
              └───────────┬───────────┘
                          ▼
                    Real Workloads
```

Potential V2 components:

```text
Kubernetes
NVIDIA device plugin
GPU metrics
real GPU allocation
Kubernetes scheduling/controller integration
vLLM
real inference workloads
Prometheus metrics
real-time scheduler decisions
```

V2 should reuse the **policy abstraction, metrics, experiment framework, and scheduling concepts** developed in V1.

---

# 27. Definition of Done

V1 is complete only when:

* [ ] OpenEnv environment runs locally
* [ ] Simulator is deterministic under fixed seeds
* [ ] Heterogeneous GPUs are supported
* [ ] Stochastic workload generation works
* [ ] FIFO works
* [ ] SJF works
* [ ] Priority works
* [ ] Best-Fit works
* [ ] PPO is implemented
* [ ] PPO supports action masking
* [ ] PPO successfully trains
* [ ] Checkpoints can be saved/loaded
* [ ] Evaluation uses unseen workloads
* [ ] PPO and baselines use identical evaluation conditions
* [ ] Metrics are automatically collected
* [ ] Tests cover simulator/environment invariants
* [ ] Dockerized environment runs
* [ ] Full experiment is reproducible from configuration
* [ ] README explains architecture, methodology, and results

---

# 28. Engineering Principles

The coding agent should follow these rules:

1. **Simulator first, PPO second.**
2. Keep simulator logic independent from OpenEnv.
3. Keep environment logic independent from training logic.
4. Baselines must use exactly the same simulator.
5. Never hard-code experiment parameters.
6. Never fabricate benchmark results.
7. Prefer deterministic tests with fixed seeds.
8. Use type hints throughout.
9. Fail loudly on invalid cluster states.
10. Keep V1 free of Kubernetes/vLLM code.
11. Avoid premature optimization.
12. Do not build the UI until the simulator, PPO, baselines, and evaluation pipeline are correct.
13. FastAPI is used by OpenEnv for the environment service; a separate project UI/API layer can be added later.
14. Every major component should have unit tests before it becomes a dependency of PPO training.

---

# 29. Final V1 Architecture

```text
                         ┌─────────────────┐
                         │ Workload Config │
                         └────────┬────────┘
                                  ↓
                         ┌─────────────────┐
                         │ Workload        │
                         │ Generator       │
                         └────────┬────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────┐
│                    OpenEnv Environment                       │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │ Job Queue    │    │ GPU Cluster  │    │ Event Engine │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│              └──────────────┬────────────────┘             │
│                             ↓                               │
│                     Scheduler State                         │
│                             ↓                               │
│                    Observation + Mask                       │
└─────────────────────────────┬───────────────────────────────┘
                              ↓
                 ┌────────────────────────┐
                 │ PPO / Baseline Policy  │
                 └────────────┬───────────┘
                              ↓
                       Scheduling Action
                              ↓
                         Environment
                              ↓
                           Reward
                              ↓
                     Metrics / Evaluation
                              ↓
                 ┌────────────────────────┐
                 │ FIFO / SJF / Priority  │
                 │ BestFit / PPO          │
                 └────────────────────────┘
```

## One-line project definition

**A discrete-event, OpenEnv-based heterogeneous GPU cluster simulator in which PPO learns online job-placement policies and is evaluated against FIFO, SJF, Priority, and Best-Fit schedulers under stochastic and unseen AI workload distributions.**
