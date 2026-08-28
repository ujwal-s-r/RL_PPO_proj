"""Rollout Buffer with Generalized Advantage Estimation (GAE) calculation."""

from __future__ import annotations
from typing import Generator, Tuple
import torch


class RolloutBuffer:
    """
    Stores experience collected across multiple parallel environments during rollouts.
    Computes GAE advantages and discounted returns.
    """

    def __init__(
        self,
        rollout_length: int,
        num_envs: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
    ) -> None:
        self.rollout_length = rollout_length
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        self.batch_size = rollout_length * num_envs

        # Allocate memory tensors directly on target device (CUDA / CPU)
        self.obs = torch.zeros((rollout_length, num_envs, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((rollout_length, num_envs), dtype=torch.long, device=device)
        self.rewards = torch.zeros((rollout_length, num_envs), dtype=torch.float32, device=device)
        self.dones = torch.zeros((rollout_length, num_envs), dtype=torch.float32, device=device)
        self.values = torch.zeros((rollout_length, num_envs), dtype=torch.float32, device=device)
        self.log_probs = torch.zeros((rollout_length, num_envs), dtype=torch.float32, device=device)
        self.masks = torch.zeros((rollout_length, num_envs, action_dim), dtype=torch.float32, device=device)

        self.advantages = torch.zeros((rollout_length, num_envs), dtype=torch.float32, device=device)
        self.returns = torch.zeros((rollout_length, num_envs), dtype=torch.float32, device=device)

        self.step_idx = 0

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        """Store one transition step across all parallel envs."""
        self.obs[self.step_idx] = obs
        self.actions[self.step_idx] = action
        self.rewards[self.step_idx] = reward
        self.dones[self.step_idx] = done
        self.values[self.step_idx] = value
        self.log_probs[self.step_idx] = log_prob
        self.masks[self.step_idx] = mask

        self.step_idx += 1

    def compute_gae(
        self,
        last_value: torch.Tensor,
        last_done: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """
        Compute Generalized Advantage Estimation (GAE) and target returns.
        
        Args:
            last_value: Value estimate V(s_{T}) for final observation, shape (num_envs,).
            last_done: Terminal flags for final observation, shape (num_envs,).
        """
        last_gae = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        for t in reversed(range(self.rollout_length)):
            if t == self.rollout_length - 1:
                next_values = last_value
            else:
                next_values = self.values[t + 1]

            next_non_terminal = 1.0 - self.dones[t]

            # TD error delta = r_t + gamma * V(s_{t+1}) * (1 - d_t) - V(s_t)
            delta = self.rewards[t] + (gamma * next_values * next_non_terminal) - self.values[t]
            
            # GAE: A_t = delta_t + gamma * lambda * (1 - d_t) * A_{t+1}
            last_gae = delta + (gamma * gae_lambda * next_non_terminal * last_gae)
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values
        self.step_idx = 0  # Reset for next rollout

    def get_minibatch_generator(
        self,
        minibatch_size: int,
    ) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        """
        Flatten rollout buffer and yield randomly shuffled minibatches.
        """
        # Flatten time and environment dimensions
        flat_obs = self.obs.reshape(-1, self.obs_dim)
        flat_actions = self.actions.reshape(-1)
        flat_log_probs = self.log_probs.reshape(-1)
        flat_returns = self.returns.reshape(-1)
        flat_advantages = self.advantages.reshape(-1)
        flat_masks = self.masks.reshape(-1, self.action_dim)

        # Normalize advantages across the full rollout batch for stable policy updates
        adv_mean = flat_advantages.mean()
        adv_std = flat_advantages.std() + 1e-8
        flat_advantages = (flat_advantages - adv_mean) / adv_std

        # Random permutation
        indices = torch.randperm(self.batch_size, device=self.device)

        for start_idx in range(0, self.batch_size, minibatch_size):
            end_idx = start_idx + minibatch_size
            mb_inds = indices[start_idx:end_idx]

            yield (
                flat_obs[mb_inds],
                flat_actions[mb_inds],
                flat_log_probs[mb_inds],
                flat_returns[mb_inds],
                flat_advantages[mb_inds],
                flat_masks[mb_inds],
            )
