"""PyTorch Structured Cross-Attention & Pairwise Placement Actor-Critic for GPU Scheduling."""

from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def layer_init(layer: nn.Linear, std: float = 1.414, bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal parameter initialization for RL networks."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """
    Structured Cross-Attention & Pairwise Placement Actor-Critic network.
    
    1. Encodes Node resources and Queue job tokens separately.
    2. Uses key_padding_mask so queue tokens only attend to REAL physical nodes (never padding).
    3. Directly scores all (Job_j, Node_n) candidate pairs via structured pairwise compatibility.
    4. Evaluates cluster value V(s) via entity-pooled attention representations.
    """

    def __init__(self, obs_dim: int = 215, action_dim: int = 128, hidden_dim: int = 256) -> None:
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

        # 1. Encoders
        self.node_proj = nn.Sequential(
            layer_init(nn.Linear(self.node_feat_dim, 64)),
            nn.LayerNorm(64),
            nn.GELU(),
            layer_init(nn.Linear(64, 64)),
        )
        self.queue_proj = nn.Sequential(
            layer_init(nn.Linear(self.queue_feat_dim, 64)),
            nn.LayerNorm(64),
            nn.GELU(),
            layer_init(nn.Linear(64, 64)),
        )
        self.global_proj = nn.Sequential(
            layer_init(nn.Linear(self.global_dim, 32)),
            nn.LayerNorm(32),
            nn.GELU(),
            layer_init(nn.Linear(32, 32)),
        )

        # 2. Cross-Attention: Queue queries attend to Node keys/values
        self.cross_attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.attn_norm = nn.LayerNorm(64)

        # 3. Pairwise (Job_j x Node_n) Placement Scorer
        # Inputs: q_refined (64) + node_emb (64) + global_emb (32) + pair_compat (4) = 164
        pair_in_dim = 64 + 64 + 32 + 4
        self.pair_scorer = nn.Sequential(
            layer_init(nn.Linear(pair_in_dim, 128)),
            nn.GELU(),
            layer_init(nn.Linear(128, 64)),
            nn.GELU(),
            layer_init(nn.Linear(64, 1), std=0.10),
        )

        # 4. Value Head (Critic) with entity pooling
        # Inputs: pooled_jobs (64) + pooled_nodes (64) + global_emb (32) = 160
        critic_in_dim = 64 + 64 + 32
        self.critic_head = nn.Sequential(
            layer_init(nn.Linear(critic_in_dim, 128)),
            nn.GELU(),
            layer_init(nn.Linear(128, 64)),
            nn.GELU(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def _extract_representations(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process structured observations into (placement_logits, state_value).
        
        Returns:
            placement_logits: Tensor of shape (B, 128)
            value: Tensor of shape (B,)
        """
        dev = next(self.parameters()).device
        obs = obs.to(dev)
        batch_size = obs.shape[0]

        node_end = self.num_nodes * self.node_feat_dim  # 8 * 8 = 64
        q_end = node_end + (self.max_queue * self.queue_feat_dim)  # 64 + 144 = 208

        # 1. Slice node features (B, 8, 8)
        node_raw = obs[:, :node_end].view(batch_size, self.num_nodes, self.node_feat_dim)
        # Node padding detection: real nodes have non-zero GPU count (node_raw[:, :, 0] > 0)
        node_is_padded = (node_raw[:, :, 0] <= 1e-5)  # (B, 8), True = padded/absent
        # Safety: if all nodes are flagged as padded (should not happen), unmask node 0
        all_nodes_padded = node_is_padded.all(dim=-1, keepdim=True)
        key_padding_mask = torch.where(all_nodes_padded, torch.zeros_like(node_is_padded), node_is_padded)

        node_emb = self.node_proj(node_raw)  # (B, 8, 64)

        # 2. Slice queue features (B, 16, 9)
        q_raw = obs[:, node_end:q_end].view(batch_size, self.max_queue, self.queue_feat_dim)
        job_is_present = (q_raw[:, :, 0] > 0.5)  # (B, 16), True = job exists
        q_emb = self.queue_proj(q_raw)  # (B, 16, 64)

        # 3. Slice global features (B, 7)
        g_raw = obs[:, q_end:q_end + self.global_dim]
        g_emb = self.global_proj(g_raw)  # (B, 32)

        # 4. Cross Attention with key_padding_mask
        attn_out, _ = self.cross_attn(
            query=q_emb,
            key=node_emb,
            value=node_emb,
            key_padding_mask=key_padding_mask,
        )
        q_refined = self.attn_norm(q_emb + attn_out)  # (B, 16, 64)

        # 5. Explicit Pairwise Compatibility Features (B, 16, 8, 4)
        # job_gpu_req: q_raw[:, :, 1], job_vram_req: q_raw[:, :, 2]
        # node_avail_gpu: node_raw[:, :, 1], node_vram: node_raw[:, :, 4]
        job_gpu_b = q_raw[:, :, 1].unsqueeze(-1).expand(-1, -1, self.num_nodes)
        job_vram_b = q_raw[:, :, 2].unsqueeze(-1).expand(-1, -1, self.num_nodes)
        node_gpu_b = node_raw[:, :, 1].unsqueeze(1).expand(-1, self.max_queue, -1)
        node_vram_b = node_raw[:, :, 4].unsqueeze(1).expand(-1, self.max_queue, -1)

        gpu_fit = (node_gpu_b - job_gpu_b).unsqueeze(-1)
        vram_fit = (node_vram_b - job_vram_b).unsqueeze(-1)
        j_present_b = job_is_present.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.num_nodes, 1).float()
        n_present_b = (~node_is_padded).unsqueeze(1).unsqueeze(-1).expand(-1, self.max_queue, -1, 1).float()

        pair_compat = torch.cat([gpu_fit, vram_fit, j_present_b, n_present_b], dim=-1)  # (B, 16, 8, 4)

        # 6. Form full Pair Tensor (B, 16, 8, 164)
        q_broadcast = q_refined.unsqueeze(2).expand(-1, -1, self.num_nodes, -1)  # (B, 16, 8, 64)
        n_broadcast = node_emb.unsqueeze(1).expand(-1, self.max_queue, -1, -1)   # (B, 16, 8, 64)
        g_broadcast = g_emb.unsqueeze(1).unsqueeze(2).expand(-1, self.max_queue, self.num_nodes, -1) # (B, 16, 8, 32)

        pair_tensor = torch.cat([q_broadcast, n_broadcast, g_broadcast, pair_compat], dim=-1) # (B, 16, 8, 164)

        # 7. Placement Logits (B, 16, 8) -> (B, 128)
        pair_scores = self.pair_scorer(pair_tensor).squeeze(-1)  # (B, 16, 8)
        placement_logits = pair_scores.view(batch_size, self.action_dim)  # (B, 128)

        # 8. Critic Value Estimation via Entity Pooling
        j_mask_f = job_is_present.unsqueeze(-1).float()  # (B, 16, 1)
        pooled_jobs = (q_refined * j_mask_f).sum(dim=1) / j_mask_f.sum(dim=1).clamp(min=1.0) # (B, 64)

        n_mask_f = (~node_is_padded).unsqueeze(-1).float()  # (B, 8, 1)
        pooled_nodes = (node_emb * n_mask_f).sum(dim=1) / n_mask_f.sum(dim=1).clamp(min=1.0) # (B, 64)

        critic_in = torch.cat([pooled_jobs, pooled_nodes, g_emb], dim=-1)  # (B, 160)
        value = self.critic_head(critic_in).squeeze(-1)  # (B,)

        return placement_logits, value

    def _apply_mask(self, logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Apply strict action feasibility mask to raw pairwise policy logits.
        
        Where mask == 0.0, set logit to -1e9 so probability is exactly 0.0 after softmax.
        """
        if mask is None:
            return logits

        mask = mask.to(logits.device)
        # Check if entire mask row is all zeros (e.g. cluster full / no action feasible anywhere)
        all_zeros = (mask.sum(dim=-1, keepdim=True) <= 1e-5)
        # For completely full states, provide a safe fallback so distribution does not NaN
        safe_mask = torch.where(all_zeros, torch.ones_like(mask), mask)

        masked_logits = logits + (safe_mask - 1.0) * 1e9
        return masked_logits

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Estimate state value V(s)."""
        _, value = self._extract_representations(obs)
        return value

    def get_action(
        self,
        obs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from masked pairwise policy distribution.
        
        Returns:
            (actions, log_probs, values)
        """
        raw_logits, value = self._extract_representations(obs)
        masked_logits = self._apply_mask(raw_logits, mask)

        dist = Categorical(logits=masked_logits)
        
        if deterministic:
            action = torch.argmax(masked_logits, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
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
        raw_logits, value = self._extract_representations(obs)
        masked_logits = self._apply_mask(raw_logits, mask)

        dist = Categorical(logits=masked_logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_prob, value, entropy
