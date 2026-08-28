"""Priority + Best-Fit Composite Baseline Scheduler."""

from __future__ import annotations
from typing import List, Optional, Tuple
from baselines.base import BaseScheduler
from simulator.scheduler_state import SchedulerState


class PriorityBestFitScheduler(BaseScheduler):
    """
    Composite Priority + Best-Fit policy.
    
    1. Selects the queued job with the highest user priority (breaks ties by queue wait age).
    2. Places that job onto the feasible node that minimizes stranded GPU and VRAM capacity.
    """

    @property
    def name(self) -> str:
        return "Priority_BestFit"

    def select_action(self, state: SchedulerState) -> Optional[Tuple[int, int]]:
        mask = state.get_action_mask()

        # Gather feasible candidate jobs: (j_idx, priority, wait_time)
        candidates: List[Tuple[int, int, float]] = []
        for j_idx in range(min(len(state.queue), mask.shape[0])):
            job = state.queue.get_at(j_idx)
            if job is not None and any(mask[j_idx, :] > 0):
                wait_time = max(0.0, state.current_time - job.arrival_time)
                candidates.append((j_idx, job.priority, wait_time))

        if not candidates:
            return None

        # Sort by priority descending (-priority), then wait_time descending (-wait_time)
        candidates.sort(key=lambda x: (-x[1], -x[2]))
        best_job_idx = candidates[0][0]
        job = state.queue.get_at(best_job_idx)
        if job is None:
            return None

        # Find the node that minimizes residual GPU core and VRAM fragmentation
        best_node_idx: Optional[int] = None
        best_waste = float("inf")

        for n_idx in range(state.num_nodes):
            if mask[best_job_idx, n_idx] > 0:
                node = state.cluster.get_node(n_idx)
                if node is None:
                    continue

                gpu_waste = node.available_gpu_count - job.gpu_count
                vram_waste = node.vram_per_gpu_gb - job.vram_per_gpu_gb
                waste_score = (gpu_waste * 100.0) + vram_waste

                if waste_score < best_waste:
                    best_waste = waste_score
                    best_node_idx = n_idx

        if best_node_idx is not None:
            return best_job_idx, best_node_idx

        return None
