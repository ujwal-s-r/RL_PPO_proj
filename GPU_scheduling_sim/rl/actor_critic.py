"""PyTorch Actor-Critic neural network with action masking for PPO."""

from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


import numpy as np

def layer_init(layer: nn.Linear, std: float = 1.414, bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal parameter initialization for RL networks."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """
    Structured Cross-Attention Actor-Critic network for GPU scheduling.
    
    Explicitly attends queue job requirements against node capabilities
    before producing masked discrete action distributions.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Dimensionality splits: 8 nodes * 8 feats = 64, 16 queue * 9 feats = 144, 7 globals (obs_dim = 215)
        self.num_nodes = 8
        self.node_feat_dim = 8
        self.max_queue = 16
        self.queue_feat_dim = 9
        self.global_dim = 7

        # Embedding projections
        self.node_proj = nn.Sequential(
            layer_init(nn.Linear(self.node_feat_dim, 64)),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        self.queue_proj = nn.Sequential(
            layer_init(nn.Linear(self.queue_feat_dim, 64)),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        self.global_proj = nn.Sequential(
            layer_init(nn.Linear(self.global_dim, 32)),
            nn.LayerNorm(32),
            nn.ReLU(),
        )

        # Cross-Attention: Queue queries attend to Node keys/values
        self.cross_attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.attn_norm = nn.LayerNorm(64)

        # Flat MLP backhaul for global synergy
        combined_dim = (self.max_queue * 64) + (self.num_nodes * 64) + 32
        self.encoder = nn.Sequential(
            layer_init(nn.Linear(combined_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )

        # Policy Head (Actor) & Value Head (Critic)
        self.actor = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        self.critic = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def _extract_features(self, obs: torch.Tensor) -> torch.Tensor:
        """Process structured observations through attention and projections."""
        dev = next(self.parameters()).device
        obs = obs.to(dev)
        batch_size = obs.shape[0]

        node_end = self.num_nodes * self.node_feat_dim  # 8 * 8 = 64
        q_end = node_end + (self.max_queue * self.queue_feat_dim)  # 64 + 144 = 208

        # 1. Slice node features (B, 8, 8)
        node_raw = obs[:, :node_end].view(batch_size, self.num_nodes, self.node_feat_dim)
        node_emb = self.node_proj(node_raw)  # (B, 8, 64)

        # 2. Slice queue features (B, 16, 9)
        q_raw = obs[:, node_end:q_end].view(batch_size, self.max_queue, self.queue_feat_dim)
        q_emb = self.queue_proj(q_raw)  # (B, 16, 64)

        # 3. Slice global features (B, 7)
        g_raw = obs[:, q_end:q_end + self.global_dim]
        g_emb = self.global_proj(g_raw)  # (B, 32)

        # 4. Cross Attention: Queue attends to Nodes
        attn_out, _ = self.cross_attn(query=q_emb, key=node_emb, value=node_emb)
        q_refined = self.attn_norm(q_emb + attn_out)  # Residual connection

        # 5. Flatten & fuse
        flat_q = q_refined.reshape(batch_size, -1)
        flat_node = node_emb.reshape(batch_size, -1)
        fused = torch.cat([flat_q, flat_node, g_emb], dim=-1)

        return self.encoder(fused)

    def _apply_mask(self, logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Apply action feasibility mask to raw policy logits.
        
        Where mask == 0.0, set logit to -1e9 so probability is exactly 0.0 after softmax.
        """
        if mask is None:
            return logits

        mask = mask.to(logits.device)
        # Safety: if mask has an all-zero row, fall back to unmasked for that row to prevent NaN
        all_zeros = (mask.sum(dim=-1, keepdim=True) == 0)
        safe_mask = torch.where(all_zeros, torch.ones_like(mask), mask)

        masked_logits = logits + (safe_mask - 1.0) * 1e9
        return masked_logits

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Estimate state value V(s)."""
        features = self._extract_features(obs)
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
        features = self._extract_features(obs)
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
        features = self._extract_features(obs)
        raw_logits = self.actor(features)
        masked_logits = self._apply_mask(raw_logits, mask)

        dist = Categorical(logits=masked_logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        value = self.critic(features).squeeze(-1)

        return log_prob, value, entropy
