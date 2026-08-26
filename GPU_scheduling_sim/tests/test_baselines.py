"""Unit tests for classical baseline schedulers."""

import numpy as np
from simulator.cluster import Cluster
from simulator.simulator import Simulator
from simulator.job import Job
from baselines.fifo import FIFOScheduler
from baselines.sjf import SJFScheduler
from baselines.priority import PriorityScheduler
from baselines.best_fit import BestFitScheduler
from baselines.runner import run_baseline_episode
from env.server.gpu_scheduler_environment import GPUSchedulerEnvironment


def test_fifo_logic():
    cluster = Cluster.from_yaml("configs/cluster_small.yaml")
    sim = Simulator(cluster=cluster, max_queue_size=8)
    sim.reset()

    j1 = Job(1, arrival_time=0.0, gpu_count=2, vram_per_gpu_gb=20.0, estimated_runtime=100.0, actual_runtime=100.0)
    j2 = Job(2, arrival_time=1.0, gpu_count=2, vram_per_gpu_gb=20.0, estimated_runtime=10.0, actual_runtime=10.0)
    sim.queue.push(j1)
    sim.queue.push(j2)

    fifo = FIFOScheduler()
    act = fifo.select_action(sim.get_state())
    assert act is not None
    assert act[0] == 0  # FIFO must pick index 0 (Job 1, oldest)


def test_sjf_logic():
    cluster = Cluster.from_yaml("configs/cluster_small.yaml")
    sim = Simulator(cluster=cluster, max_queue_size=8)
    sim.reset()

    j1 = Job(1, arrival_time=0.0, gpu_count=2, vram_per_gpu_gb=20.0, estimated_runtime=500.0, actual_runtime=500.0)
    j2 = Job(2, arrival_time=0.0, gpu_count=2, vram_per_gpu_gb=20.0, estimated_runtime=30.0, actual_runtime=30.0)
    sim.queue.push(j1)
    sim.queue.push(j2)

    sjf = SJFScheduler()
    act = sjf.select_action(sim.get_state())
    assert act is not None
    assert act[0] == 1  # SJF must pick index 1 (Job 2, shortest runtime 30s)


def test_priority_logic():
    cluster = Cluster.from_yaml("configs/cluster_small.yaml")
    sim = Simulator(cluster=cluster, max_queue_size=8)
    sim.reset()

    j1 = Job(1, arrival_time=0.0, gpu_count=2, vram_per_gpu_gb=20.0, estimated_runtime=100.0, actual_runtime=100.0, priority=3)
    j2 = Job(2, arrival_time=0.0, gpu_count=2, vram_per_gpu_gb=20.0, estimated_runtime=100.0, actual_runtime=100.0, priority=9)
    sim.queue.push(j1)
    sim.queue.push(j2)

    prio = PriorityScheduler()
    act = prio.select_action(sim.get_state())
    assert act is not None
    assert act[0] == 1  # Priority must pick index 1 (Job 2, priority 9 > 3)


def test_best_fit_logic():
    cluster = Cluster.from_yaml("configs/cluster_small.yaml")  # Node 0 has 4x40GB, Node 1 has 4x24GB
    sim = Simulator(cluster=cluster, max_queue_size=8)
    sim.reset()

    # Job requiring 2x24GB fits tightly on Node 1 (24GB VRAM) rather than Node 0 (40GB VRAM)
    job = Job(1, arrival_time=0.0, gpu_count=2, vram_per_gpu_gb=24.0, estimated_runtime=100.0, actual_runtime=100.0)
    sim.queue.push(job)

    best_fit = BestFitScheduler()
    act = best_fit.select_action(sim.get_state())
    assert act is not None
    assert act[1] == 1  # Must place on Node 1 (exact VRAM match, least slack waste)


def test_baseline_runner_episode():
    env = GPUSchedulerEnvironment(cluster_config_path="configs/cluster_small.yaml", horizon_seconds=600.0)
    fifo = FIFOScheduler()
    metrics = run_baseline_episode(fifo, env, seed=42, scenario="short_job_heavy")

    assert metrics.policy_name == "FIFO"
    assert metrics.completed_jobs > 0
    assert metrics.invalid_action_count == 0
    assert metrics.mean_jct > 0
