"""Node model representing a physical server with GPUs, CPU, and RAM."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from simulator.gpu import GPU
from simulator.job import Job


@dataclass
class Node:
    """Represents a heterogeneous cluster node with mixed or uniform GPU chips."""
    node_id: int
    gpu_type: str = "A100-SXM4-80GB"
    gpu_count: int = 4
    vram_per_gpu_gb: float = 80.0
    cpu_cores: int = 32
    ram_gb: float = 128.0
    gpu_specs: Optional[List[Dict[str, Any]]] = None
    
    gpus: List[GPU] = field(init=False)
    running_jobs: Dict[int, Job] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.gpu_specs and len(self.gpu_specs) > 0:
            self.gpus = [
                GPU(
                    gpu_id=i,
                    gpu_type=str(s.get("type", s.get("gpu_type", self.gpu_type))),
                    total_vram_gb=float(s.get("vram", s.get("vram_gb", s.get("total_vram_gb", self.vram_per_gpu_gb))))
                )
                for i, s in enumerate(self.gpu_specs)
            ]
            self.gpu_count = len(self.gpus)
            self.vram_per_gpu_gb = max((g.total_vram_gb for g in self.gpus), default=80.0)
            self.gpu_type = self.gpus[0].gpu_type if self.gpus else "A100-SXM4-80GB"
        else:
            self.gpus = [
                GPU(
                    gpu_id=i,
                    gpu_type=self.gpu_type,
                    total_vram_gb=self.vram_per_gpu_gb
                )
                for i in range(self.gpu_count)
            ]
        self.running_jobs = {}

    @property
    def available_gpus(self) -> List[GPU]:
        """List of currently free GPUs on this node."""
        return [gpu for gpu in self.gpus if gpu.is_free]

    @property
    def available_gpu_count(self) -> int:
        """Number of free GPUs."""
        return len(self.available_gpus)

    @property
    def total_vram_gb(self) -> float:
        """Total VRAM capacity across all GPUs in this node."""
        return sum(gpu.total_vram_gb for gpu in self.gpus)

    @property
    def allocated_vram_gb(self) -> float:
        """Total allocated VRAM across all GPUs in this node."""
        return sum(gpu.allocated_vram_gb for gpu in self.gpus)

    @property
    def free_vram_gb(self) -> float:
        """Total unallocated VRAM across all GPUs in this node."""
        return self.total_vram_gb - self.allocated_vram_gb

    @property
    def gpu_utilization(self) -> float:
        """Fraction of GPU units currently occupied (0.0 to 1.0)."""
        if self.gpu_count == 0:
            return 0.0
        busy_count = self.gpu_count - self.available_gpu_count
        return busy_count / self.gpu_count

    @property
    def vram_utilization(self) -> float:
        """Fraction of total VRAM capacity currently allocated (0.0 to 1.0)."""
        if self.total_vram_gb <= 0:
            return 0.0
        return self.allocated_vram_gb / self.total_vram_gb

    def can_schedule(self, job: Job) -> bool:
        """
        Check if the job can fit onto this node:
        1. Node must have at least job.gpu_count free GPUs.
        2. Each allocated GPU must have total VRAM >= job's vram_per_gpu_gb requirement.
        """
        matching_free = [g for g in self.available_gpus if g.total_vram_gb >= job.vram_per_gpu_gb]
        return len(matching_free) >= job.gpu_count

    def schedule(self, job: Job, current_time: float) -> List[int]:
        """
        Allocate matching GPUs on this node for the job.
        Sorts free GPUs by VRAM ascending to pick the tightest fit (saving high-VRAM GPUs for bigger tasks).
        """
        matching_free = [g for g in self.available_gpus if g.total_vram_gb >= job.vram_per_gpu_gb]
        if len(matching_free) < job.gpu_count:
            raise ValueError(
                f"Node {self.node_id} cannot fit Job {job.job_id}: "
                f"req={job.gpu_count}x{job.vram_per_gpu_gb}GB, "
                f"matching_free={len(matching_free)}/{self.gpu_count} GPUs"
            )

        # Best-fit sort: pick smallest sufficient VRAM first
        matching_free.sort(key=lambda g: g.total_vram_gb)
        selected_gpus = matching_free[:job.gpu_count]
        assigned_gpu_ids: List[int] = []

        for gpu in selected_gpus:
            success = gpu.allocate(job.job_id, job.vram_per_gpu_gb)
            if not success:
                # Rollback
                for g_id in assigned_gpu_ids:
                    self.gpus[g_id].release()
                raise RuntimeError(f"Failed to allocate GPU {gpu.gpu_id} on Node {self.node_id}")
            assigned_gpu_ids.append(gpu.gpu_id)

        job.mark_started(current_time, self.node_id, assigned_gpu_ids)
        self.running_jobs[job.job_id] = job
        return assigned_gpu_ids

    def complete(self, job_id: int, current_time: float) -> Optional[Job]:
        """
        Complete a running job and release its allocated GPUs.
        
        Returns:
            The completed Job object, or None if job_id was not running on this node.
        """
        job = self.running_jobs.pop(job_id, None)
        if job is None:
            return None

        for gpu_id in job.allocated_gpu_ids:
            if 0 <= gpu_id < len(self.gpus):
                self.gpus[gpu_id].release()

        job.mark_completed(current_time)
        return job

    def reset(self) -> None:
        """Reset all GPUs and clear running jobs."""
        for gpu in self.gpus:
            gpu.release()
        self.running_jobs.clear()
