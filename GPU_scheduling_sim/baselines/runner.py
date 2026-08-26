"""Runner for evaluating classical baseline schedulers against OpenEnv environments."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np

from baselines.base import BaseScheduler
from env.server.gpu_scheduler_environment import GPUSchedulerEnvironment
from evaluation.metrics import EpisodeMetrics


def run_baseline_episode(
    scheduler: BaseScheduler,
    env: GPUSchedulerEnvironment,
    seed: int = 42,
    scenario: str = "balanced",
) -> EpisodeMetrics:
    """
    Execute a full simulation episode using a classical heuristic scheduler.
    
    Args:
        scheduler: Baseline scheduler instance (FIFO, SJF, Priority, BestFit).
        env: OpenEnv environment instance.
        seed: Workload generation random seed.
        scenario: Scenario name from workloads.scenarios.
        
    Returns:
        EpisodeMetrics dataclass.
    """
    obs, info = env.reset(seed=seed, options={"scenario": scenario})
    
    done = False
    cumulative_reward = 0.0
    step_count = 0

    while not done:
        state = env.simulator.get_state()
        action_tuple = scheduler.select_action(state)

        if action_tuple is None:
            # No feasible placement right now; advance simulator
            done, _ = env.simulator.step_to_next_decision()
            if done:
                break
            continue

        job_idx, node_idx = action_tuple
        flat_action = env.encode_action(job_idx, node_idx)
        
        obs, reward, terminated, truncated, s_info = env.step(flat_action)
        cumulative_reward += reward
        done = terminated or truncated
        step_count += 1

    metrics_raw = env.simulator.get_metrics()

    return EpisodeMetrics(
        policy_name=scheduler.name,
        scenario_name=scenario,
        seed=seed,
        completed_jobs=metrics_raw["completed_jobs"],
        submitted_jobs=metrics_raw["submitted_jobs"],
        mean_jct=metrics_raw["mean_jct"],
        p95_jct=metrics_raw["p95_jct"],
        mean_wait_time=metrics_raw["mean_wait_time"],
        gpu_utilization=metrics_raw["gpu_utilization"],
        throughput_jobs_per_hour=metrics_raw["throughput_jobs_per_hour"],
        deadline_violation_rate=metrics_raw["deadline_violation_rate"],
        invalid_action_count=metrics_raw["invalid_action_count"],
        cumulative_reward=cumulative_reward,
        simulation_duration=metrics_raw["simulation_duration"],
    )
