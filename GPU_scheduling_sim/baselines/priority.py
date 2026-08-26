"""Priority-Based Baseline Scheduler."""

from __future__ import annotations
from typing import List, Optional, Tuple
from baselines.base import BaseScheduler
from simulator.scheduler_state import SchedulerState


class PriorityScheduler(BaseScheduler):
    """
    Priority-Aware policy.
    
    Selects the queued job with the highest user priority (10 is highest).
    Breaks ties by favoring the job that has waited the longest in the queue.
    """

    @property
    def name(self) -> str:
        return "Priority"

    def select_action(self, state: SchedulerState) -> Optional[Tuple[int, int]]:
        mask = state.get_action_mask()

        # Candidates: (j_idx, priority, wait_time)
        candidates: List[Tuple[int, int, float]] = []
        for j_idx in range(len(state.queue)):
            job = state.queue.get_at(j_idx)
            if job is not None and any(mask[j_idx, :] > 0):
                wait_time = max(0.0, state.current_time - job.arrival_time)
                candidates.append((j_idx, job.priority, wait_time))

        if not candidates:
            return None

        # Sort by priority descending (-priority), then wait_time descending (-wait_time)
        candidates.sort(key=lambda x: (-x[1], -x[2]))
        best_job_idx = candidates[0][0]

        # Find first feasible node for this job
        for n_idx in range(state.num_nodes):
            if mask[best_job_idx, n_idx] > 0:
                return best_job_idx, n_idx

        return None
