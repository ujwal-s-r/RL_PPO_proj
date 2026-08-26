"""Unit tests for PPOPolicyScheduler and final evaluation pipeline."""

import os
import torch
import numpy as np

from simulator.cluster import Cluster
from simulator.simulator import Simulator
from simulator.job import Job
from evaluation.ppo_policy import PPOPolicyScheduler
from evaluation.evaluator import Evaluator
from rl.config import PPOConfig
from rl.ppo import PPO


def test_ppo_policy_scheduler_inference():
    # Save a temporary checkpoint
    ckpt_path = "checkpoints/test_eval_ckpt.pt"
    cfg = PPOConfig(hidden_dim=64, device="cpu")
    ppo = PPO(cfg, obs_dim=164, action_dim=32)
    ppo.save_checkpoint(ckpt_path)

    # Initialize policy scheduler from checkpoint
    policy = PPOPolicyScheduler(checkpoint_path=ckpt_path, device="cpu")
    assert policy.name == "PPO"

    # Setup simulator with 2 jobs in queue
    cluster = Cluster.from_yaml("configs/cluster_small.yaml")
    sim = Simulator(cluster=cluster, max_queue_size=16)
    sim.reset()

    j1 = Job(1, arrival_time=0.0, gpu_count=2, vram_per_gpu_gb=20.0, estimated_runtime=50.0, actual_runtime=50.0)
    j2 = Job(2, arrival_time=5.0, gpu_count=4, vram_per_gpu_gb=20.0, estimated_runtime=80.0, actual_runtime=80.0)
    sim.queue.push(j1)
    sim.queue.push(j2)

    action = policy.select_action(sim.get_state())
    assert action is not None
    job_idx, node_idx = action
    assert 0 <= job_idx < len(sim.queue)
    assert 0 <= node_idx < cluster.num_nodes
    assert sim.get_state().is_action_valid(job_idx, node_idx)

    # Clean up
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)


def test_evaluator_with_ppo():
    evaluator = Evaluator(cluster_config_path="configs/cluster_small.yaml", horizon_seconds=300.0)
    policy = PPOPolicyScheduler(checkpoint_path="checkpoints/ppo_final.pt", device="cpu")
    results = evaluator.evaluate_scheduler(policy, scenarios=["balanced"], seeds=[999])

    assert "balanced" in results
    assert len(results["balanced"]["raw_episodes"]) == 1
    assert results["balanced"]["raw_episodes"][0].completed_jobs > 0
