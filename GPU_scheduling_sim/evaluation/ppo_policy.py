"""Policy wrapper to evaluate trained PPO checkpoints within the standard BaseScheduler interface."""

from __future__ import annotations
from typing import Optional, Tuple
import os
import torch
import numpy as np

from baselines.base import BaseScheduler
from simulator.scheduler_state import SchedulerState
from rl.actor_critic import ActorCritic
from rl.config import PPOConfig
from env.server.gpu_scheduler_environment import GPUSchedulerEnvironment


class PPOPolicyScheduler(BaseScheduler):
    """
    Wraps a trained PyTorch PPO checkpoint into a deterministic BaseScheduler policy.
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/ppo_final.pt",
        device: Optional[str] = None,
        cluster_config_path: str = "configs/cluster_small.yaml",
        deterministic: bool = True,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.deterministic = deterministic
        
        dev_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(dev_str)

        # Temporary env instance for state extraction logic
        self._temp_env = GPUSchedulerEnvironment(cluster_config_path=cluster_config_path)
        self.obs_dim = self._temp_env.obs_dim
        self.action_dim = self._temp_env.action_dim

        # Load network from checkpoint
        hidden_dim = 256
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            hidden_dim = checkpoint.get("config", {}).get("hidden_dim", 256)
            self.actor_critic = ActorCritic(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                hidden_dim=hidden_dim,
            ).to(self.device)
            self.actor_critic.load_state_dict(checkpoint["model_state_dict"])
            self.actor_critic.eval()
            print(f"Loaded PPO policy checkpoint from '{checkpoint_path}' (hidden_dim={hidden_dim}) on [{self.device.type.upper()}]")
        else:
            self.actor_critic = ActorCritic(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                hidden_dim=hidden_dim,
            ).to(self.device)
            print(f"Warning: Checkpoint '{checkpoint_path}' not found. Initialized with random weights.")

    @property
    def name(self) -> str:
        return "PPO"

    def select_action(self, state: SchedulerState) -> Optional[Tuple[int, int]]:
        """
        Choose (job_index, node_index) using trained neural network policy with action masking.
        """
        # Extract observation and mask from state
        obs_vec = self._temp_env._extract_observation(state)
        mask_2d = state.get_action_mask(max_nodes=self._temp_env.max_nodes)
        mask_vec = mask_2d.flatten()

        if not np.any(mask_vec > 0):
            return None

        # Convert to tensors
        obs_tensor = torch.tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_tensor = torch.tensor(mask_vec, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            action_tensor, _, _ = self.actor_critic.get_action(
                obs=obs_tensor,
                mask=mask_tensor,
                deterministic=self.deterministic,
            )

        flat_action = int(action_tensor.item())
        job_idx, node_idx = self._temp_env.decode_action(flat_action)

        # Validate that chosen action is feasible
        if state.is_action_valid(job_idx, node_idx):
            return job_idx, node_idx

        # Fallback to first valid action if any edge case occurs
        valid_indices = np.where(mask_vec > 0)[0]
        if len(valid_indices) > 0:
            return self._temp_env.decode_action(int(valid_indices[0]))

        return None
