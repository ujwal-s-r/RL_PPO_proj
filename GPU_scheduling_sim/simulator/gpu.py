"""GPU model representing a single physical or virtual GPU."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from simulator.job import Job


@dataclass
class GPU:
    """Represents a single GPU device within a node."""
    gpu_id: int
    gpu_type: str
    total_vram_gb: float
    allocated_vram_gb: float = 0.0
    running_job_id: Optional[int] = None

    @property
    def free_vram_gb(self) -> float:
        """Remaining allocatable VRAM in GB."""
        return max(0.0, self.total_vram_gb - self.allocated_vram_gb)

    @property
    def is_free(self) -> bool:
        """Whether this GPU has no job currently executing on it."""
        return self.running_job_id is None

    @property
    def utilization(self) -> float:
        """Memory utilization fraction (0.0 to 1.0)."""
        if self.total_vram_gb <= 0:
            return 0.0
        return min(1.0, self.allocated_vram_gb / self.total_vram_gb)

    def can_fit(self, vram_required_gb: float) -> bool:
        """Check if this GPU is free and has enough VRAM capacity."""
        return self.is_free and (vram_required_gb <= self.total_vram_gb)

    def allocate(self, job_id: int, vram_gb: float) -> bool:
        """
        Allocate this GPU to a job.
        
        Returns:
            True if allocation succeeded, False otherwise.
        """
        if not self.is_free or vram_gb > self.total_vram_gb:
            return False
        self.running_job_id = job_id
        self.allocated_vram_gb = vram_gb
        return True

    def release(self) -> None:
        """Release the GPU, resetting allocation state."""
        self.running_job_id = None
        self.allocated_vram_gb = 0.0
