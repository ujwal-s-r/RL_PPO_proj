"""PyTorch Actor-Critic neural network with action masking for PPO."""

from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def layer_init(layer: nn.Linear, std: float = np.sqrt(2) if "np" in globals() else 1.414, bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal parameter initialization for RL networks."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """
    Actor-Critic neural network for discrete action scheduling.
    
    Supports logit-level action masking to strictly prevent sampling of
    infeasible (job, node) placements.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Shared feature representation encoder
        self.encoder = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )

        # Policy Head (Actor): outputs unnormalized action logits
        self.actor = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)

        # Value Head (Critic): outputs scalar state-value estimate V(s)
        self.critic = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def _apply_mask(self, logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Apply action feasibility mask to raw policy logits.
        
        Where mask == 0.0, set logit to -1e9 so probability is exactly 0.0 after softmax.
        """
        if mask is None:
            return logits

        # Safety: if mask has an all-zero row, fall back to unmasked for that row to prevent NaN
        all_zeros = (mask.sum(dim=-1, keepdim=True) == 0)
        safe_mask = torch.where(all_zeros, torch.ones_like(mask), mask)

        masked_logits = logits + (safe_mask - 1.0) * 1e9
        return masked_logits

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Estimate state value V(s)."""
        features = self.encoder(obs)
        return self.critic(features).squeeze(-1)

    def get_action(
        self,
        obs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from masked policy distribution.
        
        Returns:
            (actions, log_probs, values)
        """
        features = self.encoder(obs)
        raw_logits = self.actor(features)
        masked_logits = self._apply_mask(raw_logits, mask)

        dist = Categorical(logits=masked_logits)
        
        if deterministic:
            action = torch.argmax(masked_logits, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        value = self.critic(features).squeeze(-1)

        return action, log_prob, value

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log-probabilities, state values, and distribution entropy for a batch.
        
        Returns:
            (log_probs, values, entropy)
        """
        features = self.encoder(obs)
        raw_logits = self.actor(features)
        masked_logits = self._apply_mask(raw_logits, mask)

        dist = Categorical(logits=masked_logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        value = self.critic(features).squeeze(-1)

        return log_prob, value, entropy
