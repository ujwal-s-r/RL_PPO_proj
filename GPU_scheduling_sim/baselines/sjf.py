"""Shortest Job First (SJF) Baseline Scheduler."""

from __future__ import annotations
from typing import List, Optional, Tuple
from baselines.base import BaseScheduler
from simulator.scheduler_state import SchedulerState


class SJFScheduler(BaseScheduler):
    """
    Shortest Job First (SJF) policy.
    
    Prioritizes queued jobs with the smallest user-estimated runtimes to
    minimize average waiting time and turnaround latency.
    """

    @property
    def name(self) -> str:
        return "SJF"

    def select_action(self, state: SchedulerState) -> Optional[Tuple[int, int]]:
        mask = state.get_action_mask()
        
        # Collect candidate jobs with their queue indices and estimated runtimes
        candidates: List[Tuple[int, float]] = []
        for j_idx in range(len(state.queue)):
            job = state.queue.get_at(j_idx)
            if job is not None and any(mask[j_idx, :] > 0):
                candidates.append((j_idx, job.estimated_runtime))

        if not candidates:
            return None

        # Sort by estimated runtime ascending
        candidates.sort(key=lambda x: x[1])
        best_job_idx = candidates[0][0]

        # Find first feasible node for this best job
        for n_idx in range(state.num_nodes):
            if mask[best_job_idx, n_idx] > 0:
                return best_job_idx, n_idx

        return None
