"""Job queue data structure with capacity limits and ordering helpers."""

from __future__ import annotations
from typing import List, Optional
from simulator.job import Job


class JobQueue:
    """Manages queued jobs awaiting cluster scheduling."""

    def __init__(self, max_size: int = 32) -> None:
        self.max_size = max_size
        self._jobs: List[Job] = []

    def __len__(self) -> int:
        return len(self._jobs)

    def __getitem__(self, idx: int) -> Job:
        return self._jobs[idx]

    @property
    def jobs(self) -> List[Job]:
        return list(self._jobs)

    @property
    def is_empty(self) -> bool:
        return len(self._jobs) == 0

    @property
    def is_full(self) -> bool:
        return len(self._jobs) >= self.max_size

    def push(self, job: Job) -> bool:
        """
        Add a job to the queue.
        
        Returns:
            True if job was added, False if queue was full.
        """
        if self.is_full:
            return False
        self._jobs.append(job)
        return True

    def pop_at(self, index: int) -> Optional[Job]:
        """Remove and return job at specified slot index."""
        if 0 <= index < len(self._jobs):
            return self._jobs.pop(index)
        return None

    def remove_by_id(self, job_id: int) -> Optional[Job]:
        """Remove and return job by its unique ID."""
        for i, job in enumerate(self._jobs):
            if job.job_id == job_id:
                return self._jobs.pop(i)
        return None

    def get_at(self, index: int) -> Optional[Job]:
        """Inspect job at slot index without removing it."""
        if 0 <= index < len(self._jobs):
            return self._jobs[index]
        return None

    def total_waiting_time(self, current_time: float) -> float:
        """Sum of elapsed wait times for all queued jobs."""
        return sum(max(0.0, current_time - j.arrival_time) for j in self._jobs)

    def clear(self) -> None:
        """Clear all queued jobs."""
        self._jobs.clear()
