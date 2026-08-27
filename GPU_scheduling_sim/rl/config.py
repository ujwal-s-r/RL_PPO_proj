"""PPO hyperparameters and training configuration dataclass."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import os
import yaml
import torch


@dataclass
class PPOConfig:
    """Hyperparameters and runtime settings for PPO training."""
    # Environment & Workload settings
    cluster_config: str = "configs/cluster_small.yaml"
    reward_config: str = "configs/reward.yaml"
    scenario: str = "mixed"
    max_nodes: int = 10             # Dynamic node capacity (1 to 10 nodes with slot padding)
    max_queue_size: int = 16
    sim_horizon_seconds: float = 3600.0
    seed: int = 42

    # PPO Hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Vectorized Rollout & Optimization Batching
    num_envs: int = 16              # 16 Parallel CPU simulator workers for 2x faster trajectory collection
    rollout_length: int = 256       # Steps collected per env before update (Total batch = 16 * 256 = 4096)
    minibatch_size: int = 256       # Scaled GPU minibatch tensor size for high RTX 3050 CUDA saturation
    epochs_per_update: int = 10     # Epochs per PPO optimization step
    total_timesteps: int = 500_000  # Total environment steps across all workers

    # Network Architecture
    hidden_dim: int = 256

    # Hardware & Logging
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = "checkpoints"
    save_freq_steps: int = 25_000
    eval_freq_steps: int = 10_000
    eval_episodes: int = 3

    @classmethod
    def from_yaml(cls, yaml_path: str) -> PPOConfig:
        """Load PPOConfig from YAML file."""
        if not os.path.exists(yaml_path):
            return cls()
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # Resolve device
        dev = data.get("device", "cuda")
        if dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"
        data["device"] = dev
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
