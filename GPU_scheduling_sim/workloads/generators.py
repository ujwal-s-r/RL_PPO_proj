"""Workload generator creating realistic streams of heterogeneous AI jobs."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from simulator.job import Job, WorkloadType
from workloads.distributions import WorkloadSampler


@dataclass
class WorkloadConfig:
    """Configuration defining the parameters of a synthetic workload stream."""
    name: str = "custom"
    duration_seconds: float = 3600.0          # Time horizon for job arrivals
    arrival_rate_per_min: float = 6.0         # Average jobs arriving per minute
    burstiness: float = 0.0                   # 0.0 (smooth Poisson) to 1.0 (highly bursty)
    
    # GPU counts & probabilities (e.g. {1: 0.5, 2: 0.3, 4: 0.15, 8: 0.05})
    gpu_count_probs: Dict[int, float] = field(
        default_factory=lambda: {1: 0.50, 2: 0.25, 4: 0.15, 8: 0.10}
    )
    
    # VRAM choices in GB and corresponding sampling weights
    vram_choices_gb: List[float] = field(
        default_factory=lambda: [16.0, 24.0, 32.0, 40.0, 80.0]
    )
    vram_weights: Optional[List[float]] = None
    
    # Runtime distribution parameters (log-normal: exp(mean) is typical median in seconds)
    runtime_lognorm_mean: float = 4.8         # exp(4.8) ~ 121 seconds
    runtime_lognorm_sigma: float = 1.0
    runtime_min_sec: float = 10.0
    runtime_max_sec: float = 1800.0
    
    # Deadline slack factors (deadline = arrival + slack * runtime)
    deadline_slack_min: float = 1.5
    deadline_slack_max: float = 3.5
    
    # Workload mix breakdown
    workload_mix_probs: Dict[WorkloadType, float] = field(
        default_factory=lambda: {
            WorkloadType.TRAINING: 0.35,
            WorkloadType.FINE_TUNING: 0.25,
            WorkloadType.EVALUATION: 0.20,
            WorkloadType.INFERENCE: 0.10,
            WorkloadType.EXPERIMENT: 0.10,
        }
    )


class WorkloadGenerator:
    """Generates synthetic job sequences deterministically given a seed and config."""

    def __init__(self, config: WorkloadConfig) -> None:
        self.config = config

    def generate(self, seed: int = 42) -> List[Job]:
        """
        Generate a list of chronological jobs based on config.
        
        Args:
            seed: Random seed for 100% reproducible generation.
            
        Returns:
            List of Job objects sorted by arrival_time.
        """
        rng = np.random.default_rng(seed)
        sampler = WorkloadSampler(rng)
        
        jobs: List[Job] = []
        current_time = 0.0
        job_id_counter = 1
        
        arrival_rate_per_sec = self.config.arrival_rate_per_min / 60.0
        gpu_counts = list(self.config.gpu_count_probs.keys())
        gpu_probs = list(self.config.gpu_count_probs.values())
        
        workload_types = list(self.config.workload_mix_probs.keys())
        workload_probs = [float(p) for p in self.config.workload_mix_probs.values()]
        norm_workload_probs = np.array(workload_probs, dtype=np.float64) / sum(workload_probs)

        while current_time < self.config.duration_seconds:
            # Sample inter-arrival step
            delta_t = sampler.sample_inter_arrival_time(
                rate_per_sec=arrival_rate_per_sec,
                burstiness=self.config.burstiness,
            )
            current_time += delta_t
            if current_time >= self.config.duration_seconds:
                break

            # Sample job characteristics
            workload_type = rng.choice(workload_types, p=norm_workload_probs)
            gpu_count = sampler.sample_gpu_count(gpu_counts, gpu_probs)
            vram_gb = sampler.sample_vram(
                self.config.vram_choices_gb,
                weights=self.config.vram_weights,
            )
            actual_runtime, estimated_runtime = sampler.sample_runtime(
                lognorm_mean=self.config.runtime_lognorm_mean,
                lognorm_sigma=self.config.runtime_lognorm_sigma,
                min_sec=self.config.runtime_min_sec,
                max_sec=self.config.runtime_max_sec,
            )
            priority = sampler.sample_priority()
            deadline = sampler.sample_deadline(
                arrival_time=current_time,
                actual_runtime=actual_runtime,
                slack_min=self.config.deadline_slack_min,
                slack_max=self.config.deadline_slack_max,
            )

            job = Job(
                job_id=job_id_counter,
                arrival_time=round(current_time, 2),
                gpu_count=gpu_count,
                vram_per_gpu_gb=vram_gb,
                estimated_runtime=round(estimated_runtime, 2),
                actual_runtime=round(actual_runtime, 2),
                priority=priority,
                deadline=round(deadline, 2),
                workload_type=workload_type,
                preemptible=False,
            )
            jobs.append(job)
            job_id_counter += 1

        return jobs
