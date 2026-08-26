"""Abstract base class for classical and heuristic cluster schedulers."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from simulator.scheduler_state import SchedulerState


class BaseScheduler(ABC):
    """Abstract base class for all scheduling policies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the scheduling policy."""
        pass

    @abstractmethod
    def select_action(self, state: SchedulerState) -> Optional[Tuple[int, int]]:
        """
        Choose the next (job_index, node_index) placement decision.
        
        Args:
            state: Snapshot of current cluster and queue state.
            
        Returns:
            Tuple of (job_index, node_index), or None if no feasible placement exists.
        """
        pass

    def __repr__(self) -> str:
        return f"<Scheduler: {self.name}>"
