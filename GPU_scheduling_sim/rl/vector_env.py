"""Vectorized synchronous multi-environment runner for parallel PPO rollouts."""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from env.server.gpu_scheduler_environment import GPUSchedulerEnvironment


class SyncVectorEnv:
    """
    Synchronous parallel environment vectorizer.
    
    Steps N environment instances together and stacks observations/masks into
    batched arrays for high-throughput CUDA tensor processing.
    """

    def __init__(
        self,
        num_envs: int = 8,
        cluster_config_path: str = "configs/cluster_small.yaml",
        reward_config_path: str = "configs/reward.yaml",
        scenario_name: str = "balanced",
        base_seed: int = 42,
    ) -> None:
        self.num_envs = num_envs
        self.base_seed = base_seed
        self.scenario_name = scenario_name

        # If scenario_name is "all" or "random", assign different benchmark scenarios across workers
        from workloads.scenarios import list_scenarios
        all_scenarios = list_scenarios()

        self.env_scenarios = []
        for i in range(num_envs):
            if scenario_name in ["all", "random", "mixed"]:
                sc = all_scenarios[i % len(all_scenarios)]
            else:
                sc = scenario_name
            self.env_scenarios.append(sc)

        self.envs = [
            GPUSchedulerEnvironment(
                cluster_config_path=cluster_config_path,
                reward_config_path=reward_config_path,
                scenario_name=self.env_scenarios[i],
            )
            for i in range(num_envs)
        ]

        self.obs_dim = self.envs[0].obs_dim
        self.action_dim = self.envs[0].action_dim
        self._current_seeds = [base_seed + (i * 1000) for i in range(num_envs)]

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reset all parallel environments.
        
        Returns:
            (stacked_obs, stacked_action_masks) of shapes (num_envs, obs_dim) and (num_envs, action_dim).
        """
        obs_list = []
        mask_list = []

        for i, env in enumerate(self.envs):
            obs, info = env.reset(seed=self._current_seeds[i], options={"scenario": self.env_scenarios[i]})
            obs_list.append(obs)
            mask_list.append(info["action_mask"])

        return np.array(obs_list, dtype=np.float32), np.array(mask_list, dtype=np.float32)

    def step(
        self,
        actions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Step all environments simultaneously with their respective actions.
        
        Args:
            actions: Array of shape (num_envs,) with discrete actions.
            
        Returns:
            (stacked_next_obs, stacked_rewards, stacked_dones, stacked_masks, infos)
        """
        next_obs_list = []
        rewards_list = []
        dones_list = []
        masks_list = []
        infos_list = []

        from workloads.scenarios import list_scenarios
        all_scenarios = list_scenarios()

        for i, env in enumerate(self.envs):
            action_i = int(actions[i])
            obs, reward, terminated, truncated, info = env.step(action_i)
            done = terminated or truncated

            if done:
                # Auto-reset with next seed; if mixed, rotate scenario
                self._current_seeds[i] += 1
                if self.scenario_name in ["all", "random", "mixed"]:
                    self.env_scenarios[i] = np.random.choice(all_scenarios)

                reset_obs, reset_info = env.reset(
                    seed=self._current_seeds[i],
                    options={"scenario": self.env_scenarios[i]}
                )
                next_obs_list.append(reset_obs)
                masks_list.append(reset_info["action_mask"])
                info["terminal_observation"] = obs
                info["episode_metrics"] = info.get("metrics", {})
            else:
                next_obs_list.append(obs)
                masks_list.append(info["action_mask"])

            rewards_list.append(reward)
            dones_list.append(float(done))
            infos_list.append(info)

        return (
            np.array(next_obs_list, dtype=np.float32),
            np.array(rewards_list, dtype=np.float32),
            np.array(dones_list, dtype=np.float32),
            np.array(masks_list, dtype=np.float32),
            infos_list,
        )
