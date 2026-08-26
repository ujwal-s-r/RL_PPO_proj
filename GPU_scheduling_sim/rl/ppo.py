"""Proximal Policy Optimization (PPO) core algorithm and update engine."""

from __future__ import annotations
from typing import Any, Dict, Optional
import os
import torch
import torch.nn as nn
import torch.optim as optim

from rl.actor_critic import ActorCritic
from rl.buffer import RolloutBuffer
from rl.config import PPOConfig


class PPO:
    """
    Proximal Policy Optimization (PPO-Clip) agent with action masking
    and GPU acceleration.
    """

    def __init__(self, config: PPOConfig, obs_dim: int, action_dim: int) -> None:
        self.config = config
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device(config.device)

        # Initialize network on device
        self.actor_critic = ActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=config.learning_rate,
            eps=1e-5,
        )

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """
        Execute PPO optimization epochs across all minibatches in the rollout buffer.
        
        Returns:
            Dictionary of training loss metrics and diagnostics.
        """
        pg_losses = []
        v_losses = []
        entropy_losses = []
        approx_kls = []
        clip_fractions = []

        for _ in range(self.config.epochs_per_update):
            for mb_obs, mb_actions, mb_old_log_probs, mb_returns, mb_advantages, mb_masks in buffer.get_minibatch_generator(self.config.minibatch_size):
                # Evaluate current policy on minibatch
                new_log_probs, new_values, entropy = self.actor_critic.evaluate_actions(
                    obs=mb_obs,
                    actions=mb_actions,
                    mask=mb_masks,
                )

                # Ratio: r_t(theta) = exp(log_pi(a|s) - log_pi_old(a|s))
                log_ratio = new_log_probs - mb_old_log_probs
                ratio = torch.exp(log_ratio)

                # Approximate KL divergence for monitoring
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    approx_kls.append(approx_kl.item())
                    clipped = ((ratio - 1.0).abs() > self.config.clip_ratio).float().mean()
                    clip_fractions.append(clipped.item())

                # 1. Clipped Surrogate Policy Loss
                surr1 = -mb_advantages * ratio
                surr2 = -mb_advantages * torch.clamp(
                    ratio,
                    1.0 - self.config.clip_ratio,
                    1.0 + self.config.clip_ratio,
                )
                policy_loss = torch.max(surr1, surr2).mean()

                # 2. Value Function Loss (MSE)
                value_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()

                # 3. Entropy Bonus
                entropy_loss = entropy.mean()

                # Total Loss
                total_loss = (
                    policy_loss
                    + (self.config.value_coef * value_loss)
                    - (self.config.entropy_coef * entropy_loss)
                )

                # Gradient Step
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                pg_losses.append(policy_loss.item())
                v_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

        return {
            "policy_loss": float(np.mean(pg_losses)) if "np" in globals() else float(sum(pg_losses)/len(pg_losses)),
            "value_loss": float(sum(v_losses) / len(v_losses)),
            "entropy": float(sum(entropy_losses) / len(entropy_losses)),
            "approx_kl": float(sum(approx_kls) / len(approx_kls)),
            "clip_fraction": float(sum(clip_fractions) / len(clip_fractions)),
        }

    def save_checkpoint(self, path: str, extra_metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save model weights, optimizer state, and training metadata."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        checkpoint = {
            "model_state_dict": self.actor_critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "config": self.config.__dict__,
            "metadata": extra_metadata or {},
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load model weights and optimizer from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint.get("metadata", {})
