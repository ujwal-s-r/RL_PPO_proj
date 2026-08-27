"""Best-Fit (Fragmentation Minimization) Baseline Scheduler."""

from __future__ import annotations
from typing import List, Optional, Tuple
from baselines.base import BaseScheduler
from simulator.scheduler_state import SchedulerState


class BestFitScheduler(BaseScheduler):
    """
    Best-Fit Bin-Packing policy.
    
    Evaluates all feasible (job, node) combinations and chooses the placement
    that leaves the least amount of unallocated GPU cores and VRAM slack on the target node.
    """

    @property
    def name(self) -> str:
        return "BestFit"

    def select_action(self, state: SchedulerState) -> Optional[Tuple[int, int]]:
        mask = state.get_action_mask()
        
        best_pair: Optional[Tuple[int, int]] = None
        best_score = float("inf")

        for j_idx in range(min(len(state.queue), mask.shape[0])):
            job = state.queue.get_at(j_idx)
            if job is None:
                continue

            for n_idx in range(state.num_nodes):
                if mask[j_idx, n_idx] > 0:
                    node = state.cluster.get_node(n_idx)
                    if node is None:
                        continue

                    # Residual waste score: lower is tighter fit
                    gpu_waste = node.available_gpu_count - job.gpu_count
                    vram_waste = node.vram_per_gpu_gb - job.vram_per_gpu_gb
                    score = (gpu_waste * 100.0) + vram_waste

                    if score < best_score:
                        best_score = score
                        best_pair = (j_idx, n_idx)

        return best_pair
