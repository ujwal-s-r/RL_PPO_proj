"""Statistical distributions and sampling helpers for synthetic AI workloads."""

from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np


class WorkloadSampler:
    """Helper for sampling arrival times, runtimes, GPU counts, and deadlines."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def sample_inter_arrival_time(self, rate_per_sec: float, burstiness: float = 0.0) -> float:
        """
        Sample inter-arrival time in seconds.
        
        Args:
            rate_per_sec: Average arrival rate lambda (jobs/second).
            burstiness: Value in [0.0, 1.0]. If > 0, introduces bimodal burst behavior.
        """
        if rate_per_sec <= 0:
            return 3600.0

        if burstiness > 0 and self.rng.random() < burstiness:
            # Bursty mode: much shorter inter-arrival time (5x-10x denser)
            boosted_rate = rate_per_sec * (5.0 + 5.0 * self.rng.random())
            return float(self.rng.exponential(1.0 / boosted_rate))

        return float(self.rng.exponential(1.0 / rate_per_sec))

    def sample_runtime(
        self,
        lognorm_mean: float = 4.5,
        lognorm_sigma: float = 1.0,
        min_sec: float = 10.0,
        max_sec: float = 3600.0,
    ) -> Tuple[float, float]:
        """
        Sample actual and estimated runtime (seconds) using log-normal distribution.
        
        Returns:
            (actual_runtime, estimated_runtime)
        """
        raw = float(self.rng.lognormal(mean=lognorm_mean, sigma=lognorm_sigma))
        actual = float(np.clip(raw, min_sec, max_sec))
        
        # In reality, user estimation has ~10-20% estimation noise
        noise_factor = float(self.rng.uniform(0.85, 1.25))
        estimated = float(np.clip(actual * noise_factor, min_sec, max_sec * 1.5))
        
        return actual, estimated

    def sample_gpu_count(self, counts: List[int], probs: List[float]) -> int:
        """Sample requested GPU count (e.g. [1, 2, 4, 8]) from categorical distribution."""
        norm_probs = np.array(probs, dtype=np.float64) / sum(probs)
        return int(self.rng.choice(counts, p=norm_probs))

    def sample_vram(self, vram_choices_gb: List[float], weights: Optional[List[float]] = None) -> float:
        """Sample requested VRAM per GPU (GB)."""
        if weights is None:
            return float(self.rng.choice(vram_choices_gb))
        norm_weights = np.array(weights, dtype=np.float64) / sum(weights)
        return float(self.rng.choice(vram_choices_gb, p=norm_weights))

    def sample_priority(self, min_p: int = 1, max_p: int = 10) -> int:
        """Sample job priority from 1 (lowest) to 10 (highest)."""
        # Triangular/normal-like distribution centered around 5
        val = int(round(self.rng.triangular(min_p, 5, max_p)))
        return max(min_p, min(max_p, val))

    def sample_deadline(
        self,
        arrival_time: float,
        actual_runtime: float,
        slack_min: float = 1.3,
        slack_max: float = 3.5,
    ) -> float:
        """
        Sample deadline timestamp.
        
        Deadline = arrival_time + (slack * runtime)
        """
        slack = float(self.rng.uniform(slack_min, slack_max))
        return arrival_time + (slack * actual_runtime)
