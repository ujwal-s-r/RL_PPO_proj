"""Scenario definitions and registry for reproducible benchmark evaluations."""

from __future__ import annotations
from typing import Dict, List
import numpy as np
from simulator.job import WorkloadType
from workloads.generators import WorkloadConfig, WorkloadGenerator


# 1. Balanced: Mixed workloads, moderate arrival rate (~5 jobs/min)
BALANCED = WorkloadConfig(
    name="balanced",
    duration_seconds=3600.0,
    arrival_rate_per_min=5.0,
    burstiness=0.0,
    gpu_count_probs={1: 0.40, 2: 0.30, 4: 0.20, 8: 0.10},
    vram_choices_gb=[16.0, 24.0, 32.0, 40.0, 80.0],
    runtime_lognorm_mean=4.5,   # median ~90s
    runtime_lognorm_sigma=0.8,
    runtime_min_sec=15.0,
    runtime_max_sec=1200.0,
)

# 2. Training Heavy: Multi-GPU jobs (4, 8 GPUs) with long runtimes
TRAINING_HEAVY = WorkloadConfig(
    name="training_heavy",
    duration_seconds=3600.0,
    arrival_rate_per_min=2.5,
    burstiness=0.0,
    gpu_count_probs={1: 0.10, 2: 0.20, 4: 0.40, 8: 0.30},
    vram_choices_gb=[40.0, 80.0],
    runtime_lognorm_mean=5.8,   # median ~330s
    runtime_lognorm_sigma=0.6,
    runtime_min_sec=60.0,
    runtime_max_sec=2400.0,
    workload_mix_probs={
        WorkloadType.TRAINING: 0.70,
        WorkloadType.FINE_TUNING: 0.20,
        WorkloadType.EVALUATION: 0.05,
        WorkloadType.INFERENCE: 0.05,
        WorkloadType.EXPERIMENT: 0.0,
    },
)

# 3. Short Job Heavy: Fast inference and evaluation workloads
SHORT_JOB_HEAVY = WorkloadConfig(
    name="short_job_heavy",
    duration_seconds=3600.0,
    arrival_rate_per_min=12.0,
    burstiness=0.0,
    gpu_count_probs={1: 0.70, 2: 0.25, 4: 0.05},
    vram_choices_gb=[16.0, 24.0],
    runtime_lognorm_mean=3.2,   # median ~24s
    runtime_lognorm_sigma=0.5,
    runtime_min_sec=5.0,
    runtime_max_sec=120.0,
    workload_mix_probs={
        WorkloadType.INFERENCE: 0.60,
        WorkloadType.EVALUATION: 0.30,
        WorkloadType.EXPERIMENT: 0.10,
        WorkloadType.TRAINING: 0.0,
        WorkloadType.FINE_TUNING: 0.0,
    },
)

# 4. Bursty: Idle stretches punctuated by sudden spikes of traffic
BURSTY = WorkloadConfig(
    name="bursty",
    duration_seconds=3600.0,
    arrival_rate_per_min=4.0,
    burstiness=0.45,            # Frequent intense traffic bursts
    gpu_count_probs={1: 0.45, 2: 0.35, 4: 0.15, 8: 0.05},
    vram_choices_gb=[16.0, 24.0, 40.0, 80.0],
    runtime_lognorm_mean=4.2,
    runtime_lognorm_sigma=0.9,
)

# 5. GPU Fragmentation Heavy: Diverse GPU counts and non-standard VRAM requirements
GPU_FRAGMENTATION = WorkloadConfig(
    name="gpu_fragmentation",
    duration_seconds=3600.0,
    arrival_rate_per_min=6.0,
    burstiness=0.1,
    gpu_count_probs={1: 0.25, 2: 0.30, 4: 0.25, 8: 0.20},
    vram_choices_gb=[24.0, 40.0, 80.0],
    runtime_lognorm_mean=4.6,
    runtime_lognorm_sigma=0.8,
)

# 6. High Load: Arrival rate approaches or exceeds cluster capacity
HIGH_LOAD = WorkloadConfig(
    name="high_load",
    duration_seconds=3600.0,
    arrival_rate_per_min=15.0,  # Saturated queue
    burstiness=0.1,
    gpu_count_probs={1: 0.40, 2: 0.30, 4: 0.20, 8: 0.10},
    vram_choices_gb=[16.0, 24.0, 40.0, 80.0],
    runtime_lognorm_mean=4.7,
    runtime_lognorm_sigma=0.7,
)

# Scenario registry mapping
SCENARIOS: Dict[str, WorkloadConfig] = {
    "balanced": BALANCED,
    "training_heavy": TRAINING_HEAVY,
    "short_job_heavy": SHORT_JOB_HEAVY,
    "bursty": BURSTY,
    "gpu_fragmentation": GPU_FRAGMENTATION,
    "high_load": HIGH_LOAD,
    "mixed": BALANCED,
    "all": BALANCED,
}


def get_scenario(name: str) -> WorkloadConfig:
    """Retrieve workload configuration by scenario name."""
    if name in ["mixed", "all", "random"]:
        # Default representative scenario for config inspection
        return SCENARIOS["balanced"]
    if name not in SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{name}'. Available scenarios: {list(SCENARIOS.keys())}"
        )
    return SCENARIOS[name]


def list_scenarios() -> List[str]:
    """List all available scenario names."""
    return list(SCENARIOS.keys())


def create_scenario_workload(name: str, seed: int = 42):
    """Generate jobs directly for a named scenario."""
    if name in ["mixed", "all", "random"]:
        rng = np.random.default_rng(seed)
        scenario_keys = list(SCENARIOS.keys())
        chosen_name = str(rng.choice(scenario_keys))
        cfg = SCENARIOS[chosen_name]
    else:
        cfg = get_scenario(name)
    gen = WorkloadGenerator(cfg)
    return gen.generate(seed=seed)
