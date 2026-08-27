"""State representation and action mask extraction at scheduling points."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
from simulator.cluster import Cluster
from simulator.queue import JobQueue
from simulator.job import Job


@dataclass
class SchedulerState:
    """Snapshot of cluster and queue state passed to policies and RL models."""
    current_time: float
    cluster: Cluster
    queue: JobQueue
    max_queue_size: int
    completed_jobs_count: int
    total_jobs_arrived: int
    horizon_time: float

    @property
    def num_nodes(self) -> int:
        return self.cluster.num_nodes

    def is_action_valid(self, job_index: int, node_index: int) -> bool:
        """
        Check whether placing the job at job_index onto node_index is feasible.
        """
        job = self.queue.get_at(job_index)
        if job is None:
            return False

        node = self.cluster.get_node(node_index)
        if node is None:
            return False

        return node.can_schedule(job)

    def get_action_mask(self, max_nodes: Optional[int] = None) -> np.ndarray:
        """
        Construct boolean/float mask of shape (queue_slots, node_slots).
        
        If max_nodes is provided (e.g. for PPO env), padded unused slots receive 0.0 mask.
        1.0 indicates feasible action, 0.0 indicates infeasible action.
        """
        node_slots = max_nodes if max_nodes is not None else self.num_nodes
        q_len = len(self.queue)
        q_slots = self.max_queue_size if max_nodes is not None else max(self.max_queue_size, q_len)
        mask = np.zeros((q_slots, node_slots), dtype=np.float32)
        for j_idx in range(min(q_len, q_slots)):
            job = self.queue.get_at(j_idx)
            if job is None:
                continue
            for n_idx in range(min(self.num_nodes, node_slots)):
                node = self.cluster.get_node(n_idx)
                if node is not None and node.can_schedule(job):
                    mask[j_idx, n_idx] = 1.0
        return mask

    def has_any_valid_action(self) -> bool:
        """Whether at least one (job, node) placement is currently valid."""
        return bool(np.any(self.get_action_mask() > 0))
