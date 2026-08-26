"""Standard evaluation metrics definitions and aggregation statistics."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List
import numpy as np


@dataclass
class EpisodeMetrics:
    """Detailed performance metrics for a single evaluation episode."""
    policy_name: str
    scenario_name: str
    seed: int
    completed_jobs: int
    submitted_jobs: int
    mean_jct: float
    p95_jct: float
    mean_wait_time: float
    gpu_utilization: float
    throughput_jobs_per_hour: float
    deadline_violation_rate: float
    invalid_action_count: int
    cumulative_reward: float
    simulation_duration: float


def aggregate_metrics(metrics_list: List[EpisodeMetrics]) -> Dict[str, Dict[str, float]]:
    """
    Compute statistical aggregate summary (mean, std, p50, p95) across multiple episode runs.
    """
    if not metrics_list:
        return {}

    keys = [
        "completed_jobs",
        "mean_jct",
        "p95_jct",
        "mean_wait_time",
        "gpu_utilization",
        "throughput_jobs_per_hour",
        "deadline_violation_rate",
        "cumulative_reward",
    ]

    aggregated: Dict[str, Dict[str, float]] = {}

    for key in keys:
        values = [getattr(m, key) for m in metrics_list]
        aggregated[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "p50": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    return aggregated
