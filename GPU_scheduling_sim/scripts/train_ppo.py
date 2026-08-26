"""Standalone CLI script to execute PPO training."""

import argparse
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from rl.config import PPOConfig
from rl.trainer import PPOTrainer


def main():
    parser = argparse.ArgumentParser(description="Train PPO GPU Cluster Scheduler")
    parser.add_argument("--config", type=str, default="configs/training.yaml", help="Path to training.yaml")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total timesteps")
    parser.add_argument("--num-envs", type=int, default=None, help="Override number of parallel workers")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    cfg = PPOConfig.from_yaml(args.config)
    if args.timesteps:
        cfg.total_timesteps = args.timesteps
    if args.num_envs:
        cfg.num_envs = args.num_envs
    if args.device:
        cfg.device = args.device
    if args.seed:
        cfg.seed = args.seed

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    trainer = PPOTrainer(cfg)
    history = trainer.train()


if __name__ == "__main__":
    main()
