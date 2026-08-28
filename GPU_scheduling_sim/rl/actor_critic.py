"""PyTorch Hierarchical Factorized (Job Selection -> Conditional Placement) Actor-Critic for GPU Scheduling."""

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
    Hierarchical Factorized Cross-Attention Actor-Critic network.
    
    1. Encodes Node resources and Queue job tokens separately.
    2. Cross-Attention: Queue tokens attend to physical cluster nodes.
    3. Head 1 (Job Selection): Computes P(job) over 16 queue slots, masked by cluster-wide job feasibility.
    4. Head 2 (Conditional Placement): Computes P(node | job) over 8 nodes conditioned on the chosen job.
    5. Autoregressive Action: log_prob = log P(job) + log P(node | job), Entropy = H(job) + H(node | job).
    6. Decoupled Critic: Evaluates cluster value V(s) via entity-pooled representations.
    """

    def __init__(self, obs_dim: int = 215, action_dim: int = 128, hidden_dim: int = 256) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Structural cluster dimensions (8 nodes, 16 queue slots)
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

        # 3. Head 1: Job Selection Head (16-way categorical)
        # Inputs: q_refined (64) + global_emb (32) = 96
        self.job_scorer = nn.Sequential(
            layer_init(nn.Linear(64 + 32, 64)),
            nn.GELU(),
            layer_init(nn.Linear(64, 1), std=0.10),
        )

        # 4. Head 2: Conditional Placement Head (8-way categorical conditioned on chosen job)
        # Inputs: q_chosen (64) + node_emb (64) + global_emb (32) + pair_compat (4) = 164
        cond_pair_in_dim = 64 + 64 + 32 + 4
        self.placement_scorer = nn.Sequential(
            layer_init(nn.Linear(cond_pair_in_dim, 128)),
            nn.GELU(),
            layer_init(nn.Linear(128, 64)),
            nn.GELU(),
            layer_init(nn.Linear(64, 1), std=0.10),
        )

        # 5. Value Head (Critic) with entity pooling
        # Inputs: pooled_jobs (64) + pooled_nodes (64) + global_emb (32) = 160
        critic_in_dim = 64 + 64 + 32
        self.critic_head = nn.Sequential(
            layer_init(nn.Linear(critic_in_dim, 128)),
            nn.GELU(),
            layer_init(nn.Linear(128, 64)),
            nn.GELU(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def _encode_representations(self, obs: torch.Tensor):
        """Extract structured entity tokens, cross-attention representations, and value."""
        dev = next(self.parameters()).device
        obs = obs.to(dev)
        batch_size = obs.shape[0]

        node_end = self.num_nodes * self.node_feat_dim  # 8 * 8 = 64
        q_end = node_end + (self.max_queue * self.queue_feat_dim)  # 64 + 144 = 208

        # 1. Node features (B, 8, 8)
        node_raw = obs[:, :node_end].view(batch_size, self.num_nodes, self.node_feat_dim)
        node_is_padded = (node_raw[:, :, 0] <= 1e-5)  # (B, 8), True = padded
        all_nodes_padded = node_is_padded.all(dim=-1, keepdim=True)
        key_padding_mask = torch.where(all_nodes_padded, torch.zeros_like(node_is_padded), node_is_padded)
        node_emb = self.node_proj(node_raw)  # (B, 8, 64)

        # 2. Queue features (B, 16, 9)
        q_raw = obs[:, node_end:q_end].view(batch_size, self.max_queue, self.queue_feat_dim)
        job_is_present = (q_raw[:, :, 0] > 0.5)  # (B, 16), True = job exists
        q_emb = self.queue_proj(q_raw)  # (B, 16, 64)

        # 3. Global features (B, 7)
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

        # 5. Critic Value Estimation via Entity Pooling
        j_mask_f = job_is_present.unsqueeze(-1).float()  # (B, 16, 1)
        pooled_jobs = (q_refined * j_mask_f).sum(dim=1) / j_mask_f.sum(dim=1).clamp(min=1.0)  # (B, 64)
        n_mask_f = (~node_is_padded).unsqueeze(-1).float()  # (B, 8, 1)
        pooled_nodes = (node_emb * n_mask_f).sum(dim=1) / n_mask_f.sum(dim=1).clamp(min=1.0)  # (B, 64)

        critic_in = torch.cat([pooled_jobs, pooled_nodes, g_emb], dim=-1)  # (B, 160)
        value = self.critic_head(critic_in).squeeze(-1)  # (B,)

        return {
            "batch_size": batch_size,
            "node_raw": node_raw,
            "q_raw": q_raw,
            "node_emb": node_emb,
            "q_refined": q_refined,
            "g_emb": g_emb,
            "job_is_present": job_is_present,
            "node_is_padded": node_is_padded,
            "value": value,
        }

    def _compute_job_logits(self, q_refined: torch.Tensor, g_emb: torch.Tensor) -> torch.Tensor:
        """Compute Head 1 raw logits over 16 job slots: (B, 16)."""
        batch_size = q_refined.shape[0]
        g_expanded = g_emb.unsqueeze(1).expand(batch_size, self.max_queue, -1)  # (B, 16, 32)
        job_features = torch.cat([q_refined, g_expanded], dim=-1)  # (B, 16, 96)
        job_logits = self.job_scorer(job_features).squeeze(-1)  # (B, 16)
        return job_logits

    def _compute_conditional_node_logits(
        self,
        chosen_job_idx: torch.Tensor,
        reps: dict,
    ) -> torch.Tensor:
        """Compute Head 2 raw logits over 8 nodes conditioned on chosen job: (B, 8)."""
        batch_size = reps["batch_size"]
        dev = chosen_job_idx.device

        # Extract chosen job representations: (B, 64)
        batch_idx = torch.arange(batch_size, device=dev)
        chosen_q_refined = reps["q_refined"][batch_idx, chosen_job_idx]  # (B, 64)
        chosen_q_raw = reps["q_raw"][batch_idx, chosen_job_idx]          # (B, 9)

        # Pairwise compatibility for this specific job across 8 nodes
        job_gpu_b = chosen_q_raw[:, 1].unsqueeze(-1).expand(-1, self.num_nodes)   # (B, 8)
        job_vram_b = chosen_q_raw[:, 2].unsqueeze(-1).expand(-1, self.num_nodes)  # (B, 8)
        node_gpu_b = reps["node_raw"][:, :, 1]                                     # (B, 8)
        node_vram_b = reps["node_raw"][:, :, 4]                                    # (B, 8)

        gpu_fit = (node_gpu_b - job_gpu_b).unsqueeze(-1)                          # (B, 8, 1)
        vram_fit = (node_vram_b - job_vram_b).unsqueeze(-1)                       # (B, 8, 1)
        j_present_b = reps["job_is_present"][batch_idx, chosen_job_idx].unsqueeze(-1).unsqueeze(-1).expand(-1, self.num_nodes, 1).float()
        n_present_b = (~reps["node_is_padded"]).unsqueeze(-1).float()             # (B, 8, 1)

        pair_compat_j = torch.cat([gpu_fit, vram_fit, j_present_b, n_present_b], dim=-1)  # (B, 8, 4)

        # Build conditional pair tensor: (B, 8, 164)
        q_exp = chosen_q_refined.unsqueeze(1).expand(-1, self.num_nodes, -1)       # (B, 8, 64)
        g_exp = reps["g_emb"].unsqueeze(1).expand(-1, self.num_nodes, -1)          # (B, 8, 32)
        cond_tensor = torch.cat([q_exp, reps["node_emb"], g_exp, pair_compat_j], dim=-1)  # (B, 8, 164)

        node_logits = self.placement_scorer(cond_tensor).squeeze(-1)               # (B, 8)
        return node_logits

    def _apply_mask_to_logits(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply strict feasibility mask to logits with safe fallback for all-zero rows."""
        mask = mask.to(logits.device).float()
        all_zeros = (mask.sum(dim=-1, keepdim=True) <= 1e-5)
        safe_mask = torch.where(all_zeros, torch.ones_like(mask), mask)
        return logits + (safe_mask - 1.0) * 1e9

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Estimate state value V(s)."""
        reps = self._encode_representations(obs)
        return reps["value"]

    def get_action(
        self,
        obs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Hierarchical 2-stage action sampling:
        1. Sample job_idx j ~ P(job | s)
        2. Sample node_idx n ~ P(node | job=j, s)
        
        Returns:
            (actions, log_probs, values) where actions are flat indices j * 8 + n.
        """
        reps = self._encode_representations(obs)
        batch_size = reps["batch_size"]
        dev = obs.device

        # Reshape 128-dim flat action mask into 2D (B, 16, 8)
        if mask is not None:
            mask_2d = mask.to(dev).view(batch_size, self.max_queue, self.num_nodes)
        else:
            mask_2d = torch.ones((batch_size, self.max_queue, self.num_nodes), device=dev)

        # ----------------------------------------------------
        # Stage 1: Job Selection P(job | s)
        # ----------------------------------------------------
        raw_job_logits = self._compute_job_logits(reps["q_refined"], reps["g_emb"])
        job_mask = (mask_2d.sum(dim=-1) > 0.5)  # (B, 16), True if job can fit on AT LEAST one node
        masked_job_logits = self._apply_mask_to_logits(raw_job_logits, job_mask)

        dist_job = Categorical(logits=masked_job_logits)
        if deterministic:
            chosen_job = torch.argmax(masked_job_logits, dim=-1)
        else:
            chosen_job = dist_job.sample()

        log_prob_job = dist_job.log_prob(chosen_job)

        # ----------------------------------------------------
        # Stage 2: Conditional Placement P(node | job, s)
        # ----------------------------------------------------
        raw_node_logits = self._compute_conditional_node_logits(chosen_job, reps)
        batch_idx = torch.arange(batch_size, device=dev)
        node_mask = mask_2d[batch_idx, chosen_job, :]  # (B, 8), True only for valid nodes for THIS job
        masked_node_logits = self._apply_mask_to_logits(raw_node_logits, node_mask)

        dist_node = Categorical(logits=masked_node_logits)
        if deterministic:
            chosen_node = torch.argmax(masked_node_logits, dim=-1)
        else:
            chosen_node = dist_node.sample()

        log_prob_node = dist_node.log_prob(chosen_node)

        # Combined flat action index and joint log-probability
        flat_action = (chosen_job * self.num_nodes) + chosen_node
        total_log_prob = log_prob_job + log_prob_node

        return flat_action, total_log_prob, reps["value"]

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Autoregressively evaluate log P(actions), state values, and joint entropy.
        
        log \pi(j, n) = log P(j) + log P(n | j)
        Entropy = H(j) + H(n | j)
        """
        reps = self._encode_representations(obs)
        batch_size = reps["batch_size"]
        dev = obs.device

        actions = actions.to(dev).long()
        chosen_job = actions // self.num_nodes  # (B,)
        chosen_node = actions % self.num_nodes  # (B,)

        if mask is not None:
            mask_2d = mask.to(dev).view(batch_size, self.max_queue, self.num_nodes)
        else:
            mask_2d = torch.ones((batch_size, self.max_queue, self.num_nodes), device=dev)

        # 1. Job Selection Evaluation
        raw_job_logits = self._compute_job_logits(reps["q_refined"], reps["g_emb"])
        job_mask = (mask_2d.sum(dim=-1) > 0.5)
        masked_job_logits = self._apply_mask_to_logits(raw_job_logits, job_mask)

        dist_job = Categorical(logits=masked_job_logits)
        log_prob_job = dist_job.log_prob(chosen_job)
        entropy_job = dist_job.entropy()

        # 2. Conditional Placement Evaluation
        raw_node_logits = self._compute_conditional_node_logits(chosen_job, reps)
        batch_idx = torch.arange(batch_size, device=dev)
        node_mask = mask_2d[batch_idx, chosen_job, :]
        masked_node_logits = self._apply_mask_to_logits(raw_node_logits, node_mask)

        dist_node = Categorical(logits=masked_node_logits)
        log_prob_node = dist_node.log_prob(chosen_node)
        entropy_node = dist_node.entropy()

        # Total Joint Metrics
        total_log_prob = log_prob_job + log_prob_node
        total_entropy = entropy_job + entropy_node

        return total_log_prob, reps["value"], total_entropy
