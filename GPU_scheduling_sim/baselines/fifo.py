"""First-In First-Out (FIFO) Baseline Scheduler."""

from __future__ import annotations
from typing import Optional, Tuple
from baselines.base import BaseScheduler
from simulator.scheduler_state import SchedulerState


class FIFOScheduler(BaseScheduler):
    """
    First-In First-Out (FIFO) policy.
    
    Examines queued jobs in chronological arrival order and assigns the oldest
    feasible job to the first available cluster node that meets its constraints.
    """

    @property
    def name(self) -> str:
        return "FIFO"

    def select_action(self, state: SchedulerState) -> Optional[Tuple[int, int]]:
        mask = state.get_action_mask()
        
        # Iterate queue in arrival order (0 is oldest)
        for j_idx in range(len(state.queue)):
            for n_idx in range(state.num_nodes):
                if mask[j_idx, n_idx] > 0:
                    return j_idx, n_idx
                    
        return None
