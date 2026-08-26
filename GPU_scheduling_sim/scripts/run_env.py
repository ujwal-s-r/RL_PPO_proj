"""Script to launch and interact with the OpenEnv cluster environment."""

import argparse
import sys
import os
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from env.client import OpenEnvClient


def main():
    parser = argparse.ArgumentParser(description="Test OpenEnv GPU Cluster Environment")
    parser.add_argument("--cluster-config", type=str, default="configs/cluster_small.yaml")
    parser.add_argument("--scenario", type=str, default="balanced")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Initializing OpenEnv Client with cluster '{args.cluster_config}', scenario '{args.scenario}'...")
    client = OpenEnvClient(cluster_config_path=args.cluster_config)
    obs, info = client.reset(seed=args.seed, scenario=args.scenario)

    print(f"Environment reset successful!")
    print(f"Observation shape: {obs.shape}")
    print(f"Total Action Space Dim: {client.action_dim}")

    total_reward = 0.0
    for step in range(args.steps):
        mask = client.get_action_mask()
        valid_indices = np.where(mask > 0)[0]
        if len(valid_indices) == 0:
            print(f"Step {step}: No feasible actions available.")
            break

        # Pick random valid action
        action = int(np.random.choice(valid_indices))
        obs, reward, term, trunc, s_info = client.step(action)
        total_reward += reward

        print(
            f"Step {step:02d} | Action {action:03d} | Reward {reward:+.3f} | "
            f"SimTime: {s_info['sim_time']:.1f}s | QueueLen: {len(client.env.simulator.queue)} | "
            f"Cluster GPU Util: {client.env.cluster.gpu_utilization*100:.1f}%"
        )

        if term or trunc:
            print("Episode reached terminal horizon.")
            break

    print(f"\nCompleted {args.steps} steps. Cumulative Reward: {total_reward:.3f}")


if __name__ == "__main__":
    main()
