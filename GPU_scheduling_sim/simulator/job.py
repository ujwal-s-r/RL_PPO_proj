"""Job model and status definitions for GPU cluster workloads."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class WorkloadType(str, Enum):
    TRAINING = "training"
    FINE_TUNING = "fine_tuning"
    EVALUATION = "evaluation"
    INFERENCE = "inference"
    EXPERIMENT = "experiment"


@dataclass
class Job:
    """Represents a compute job requesting GPU cluster resources."""
    job_id: int
    arrival_time: float
    gpu_count: int
    vram_per_gpu_gb: float
    estimated_runtime: float
    actual_runtime: float
    priority: int = 5                  # 1 (lowest) to 10 (highest)
    deadline: float = float("inf")     # Absolute timestamp for deadline
    workload_type: WorkloadType = WorkloadType.TRAINING
    preemptible: bool = False
    
    # Dynamic lifecycle tracking
    status: JobStatus = JobStatus.QUEUED
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    allocated_node_id: Optional[int] = None
    allocated_gpu_ids: List[int] = field(default_factory=list)

    @property
    def waiting_time(self) -> float:
        """Time spent waiting in queue before scheduling."""
        if self.start_time is None:
            return 0.0
        return max(0.0, self.start_time - self.arrival_time)

    @property
    def turnaround_time(self) -> Optional[float]:
        """Total turnaround time (completion_time - arrival_time)."""
        if self.completion_time is None:
            return None
        return max(0.0, self.completion_time - self.arrival_time)

    @property
    def is_completed(self) -> bool:
        return self.status == JobStatus.COMPLETED

    def is_deadline_missed(self, current_time: Optional[float] = None) -> bool:
        """Check if job has missed or will miss its deadline."""
        end_time = self.completion_time if self.completion_time is not None else current_time
        if end_time is None:
            return False
        return end_time > self.deadline

    def mark_started(self, current_time: float, node_id: int, gpu_ids: List[int]) -> None:
        """Transition job from QUEUED to RUNNING."""
        self.status = JobStatus.RUNNING
        self.start_time = current_time
        self.allocated_node_id = node_id
        self.allocated_gpu_ids = list(gpu_ids)

    def mark_completed(self, current_time: float) -> None:
        """Transition job from RUNNING to COMPLETED."""
        self.status = JobStatus.COMPLETED
        self.completion_time = current_time
