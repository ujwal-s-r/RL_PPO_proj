"""Unified evaluation framework for both classical heuristics and RL policies."""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np

from baselines.base import BaseScheduler
from baselines.runner import run_baseline_episode
from env.server.gpu_scheduler_environment import GPUSchedulerEnvironment
from evaluation.metrics import EpisodeMetrics, aggregate_metrics
from workloads.scenarios import list_scenarios


class Evaluator:
    """
    Standardized evaluator ensuring identical test environments, workloads,
    and seeds across all baseline and PPO policies.
    """

    def __init__(
        self,
        cluster_config_path: str = "configs/cluster_small.yaml",
        reward_config_path: str = "configs/reward.yaml",
        max_queue_size: int = 16,
        horizon_seconds: float = 3600.0,
    ) -> None:
        self.cluster_config_path = cluster_config_path
        self.reward_config_path = reward_config_path
        self.max_queue_size = max_queue_size
        self.horizon_seconds = horizon_seconds

        self.env = GPUSchedulerEnvironment(
            cluster_config_path=self.cluster_config_path,
            reward_config_path=self.reward_config_path,
            max_queue_size=self.max_queue_size,
            horizon_seconds=self.horizon_seconds,
        )

    def evaluate_scheduler(
        self,
        scheduler: BaseScheduler,
        scenarios: Optional[List[str]] = None,
        seeds: Optional[List[int]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate a classical baseline scheduler across scenarios and seeds.
        
        Returns:
            Dict mapping scenario_name -> {
                "raw_episodes": List[EpisodeMetrics],
                "summary": Dict[str, Dict[str, float]]
            }
        """
        if scenarios is None:
            scenarios = list_scenarios()
        if seeds is None:
            seeds = [101, 202, 303, 404, 505]

        results: Dict[str, Dict[str, Any]] = {}

        for sc in scenarios:
            episodes: List[EpisodeMetrics] = []
            for s in seeds:
                ep_metric = run_baseline_episode(
                    scheduler=scheduler,
                    env=self.env,
                    seed=s,
                    scenario=sc,
                )
                episodes.append(ep_metric)

            results[sc] = {
                "raw_episodes": episodes,
                "summary": aggregate_metrics(episodes),
            }

        return results
