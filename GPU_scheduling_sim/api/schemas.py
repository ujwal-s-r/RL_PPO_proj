"""Pydantic schemas for the FastAPI inference and cluster management API."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class JobSubmissionRequest(BaseModel):
    """Payload for submitting a custom user job into the live cluster queue."""
    gpu_count: int = Field(default=2, ge=1, le=8, description="Number of GPUs requested (1, 2, 4, 8)")
    vram_per_gpu_gb: float = Field(default=24.0, ge=8.0, le=80.0, description="VRAM required per GPU in GB")
    estimated_runtime: float = Field(default=120.0, ge=5.0, le=3600.0, description="Estimated duration in seconds")
    priority: int = Field(default=5, ge=1, le=10, description="Job priority (1=lowest, 10=highest)")
    workload_type: str = Field(default="training", description="training, fine_tuning, evaluation, inference, experiment")
    deadline_slack: float = Field(default=2.0, ge=1.1, le=10.0, description="Deadline multiplier (arrival + slack * runtime)")


class ScheduleStepRequest(BaseModel):
    """Payload for executing a scheduling step."""
    policy: str = Field(default="PPO", description="Scheduling policy: PPO, FIFO, SJF, Priority, BestFit")
    auto_advance: bool = Field(default=True, description="Whether to advance simulation time to the next decision point")


class ClusterResetRequest(BaseModel):
    """Payload for resetting the cluster simulation."""
    cluster_config: str = Field(default="configs/cluster_small.yaml", description="Path to cluster YAML")
    scenario: str = Field(default="balanced", description="Workload scenario name")
    seed: int = Field(default=42, description="Workload generation seed")


class CustomClusterSetupRequest(BaseModel):
    """Payload for setting up custom 1-10 node cluster topology."""
    nodes: List[Dict[str, Any]] = Field(..., description="List of node specifications")
    scenario: str = Field(default="balanced", description="Workload scenario name")
    seed: int = Field(default=42, description="Workload seed")


class GPUSlotInfo(BaseModel):
    gpu_id: int
    gpu_type: str
    total_vram_gb: float
    allocated_vram_gb: float
    is_free: bool
    running_job_id: Optional[int] = None


class NodeStatusInfo(BaseModel):
    node_id: int
    gpu_type: str
    total_gpus: int
    available_gpus: int
    gpu_utilization_pct: float
    total_vram_gb: float
    allocated_vram_gb: float
    free_vram_gb: float
    vram_utilization_pct: float
    gpus: List[GPUSlotInfo]
    running_jobs: List[Dict[str, Any]]


class QueuedJobInfo(BaseModel):
    job_id: int
    slot_index: int
    gpu_count: int
    vram_per_gpu_gb: float
    estimated_runtime: float
    priority: int
    workload_type: str
    arrival_time: float
    waiting_time: float
    deadline: float
    is_urgent: bool


class ClusterStatusResponse(BaseModel):
    """Comprehensive snapshot of live cluster state, nodes, queue, and metrics."""
    simulation_time: float
    is_running: bool
    active_policy: str
    scenario: str
    nodes: List[NodeStatusInfo]
    queue: List[QueuedJobInfo]
    total_gpus: int
    available_gpus: int
    cluster_gpu_utilization_pct: float
    cluster_vram_utilization_pct: float
    metrics: Dict[str, Any]
    action_mask: List[List[float]]
    recent_logs: List[Dict[str, Any]]
