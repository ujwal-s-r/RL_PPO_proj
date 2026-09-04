"""Inference service manager hosting live simulator state and policy models."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import os

from simulator.cluster import Cluster
from simulator.simulator import Simulator
from simulator.job import Job, WorkloadType
from evaluation.onnx_policy import ONNXPolicyScheduler
from workloads.scenarios import create_scenario_workload, list_scenarios
from api.schemas import (
    ClusterStatusResponse,
    NodeStatusInfo,
    GPUSlotInfo,
    QueuedJobInfo,
    JobSubmissionRequest,
)


class ClusterServiceManager:
    """Singleton service managing live cluster simulation and multi-policy inference."""

    def __init__(
        self,
        cluster_config: str = "configs/cluster_small.yaml",
        checkpoint_path: str = "checkpoints/ppo_final.onnx",
        default_scenario: str = "balanced",
    ) -> None:
        self.cluster_config = cluster_config
        self.checkpoint_path = checkpoint_path
        self.current_scenario = default_scenario
        self.active_policy_name = "PPO"

        # Load cluster and simulator
        self.cluster = Cluster.from_yaml(self.cluster_config)
        self.simulator = Simulator(cluster=self.cluster, max_queue_size=16, horizon_seconds=3600.0)

        # Live simulation is PPO-only. Baselines remain available in offline evaluation.
        self.policies = {
            "PPO": ONNXPolicyScheduler(
                checkpoint_path=checkpoint_path,
                cluster_config_path=cluster_config,
            ),
        }

        self.recent_logs: List[Dict[str, Any]] = []
        self.custom_job_counter = 9000

        # Initial reset with default scenario
        self.reset(self.cluster_config, self.current_scenario, seed=42)

    def log_event(self, message: str, level: str = "info", details: Optional[Dict[str, Any]] = None) -> None:
        """Append log message to recent decision buffer (keeps last 30 entries)."""
        entry = {
            "timestamp": round(self.simulator.current_time, 1),
            "message": message,
            "level": level,
            "details": details or {},
        }
        self.recent_logs.append(entry)
        if len(self.recent_logs) > 30:
            self.recent_logs.pop(0)

    def reset_dynamic(self, node_specs: List[Dict[str, Any]], scenario: str = "balanced", seed: int = 42) -> None:
        """Reset simulator with custom user-built node specifications (1 to 8 nodes)."""
        self.current_scenario = scenario
        self.current_node_specs = node_specs
        self.current_seed = seed
        self.cluster = Cluster.create_dynamic(node_specs)
        # Horizon is None so all tasks in workload execute to 100% completion
        self.simulator = Simulator(cluster=self.cluster, max_queue_size=16, horizon_seconds=None)
        self.simulator.reset()

        # Adapt job requirements to cluster max node capacities so every job is physically schedulable
        max_cluster_gpus_per_node = max((n.gpu_count for n in self.cluster.nodes), default=8)
        max_cluster_vram = max((n.vram_per_gpu_gb for n in self.cluster.nodes), default=80.0)

        raw_jobs = create_scenario_workload(scenario, seed=seed)
        jobs = []
        for j in raw_jobs:
            j.gpu_count = max(1, min(j.gpu_count, max_cluster_gpus_per_node))
            j.vram_per_gpu_gb = min(j.vram_per_gpu_gb, max_cluster_vram)
            jobs.append(j)

        self.total_scenario_jobs = len(jobs)
        self.simulator.load_workload(jobs)

        # Advance to first scheduling decision point
        done, _ = self.simulator.step_to_next_decision()

        self.recent_logs.clear()
        self.completed_history: List[Dict[str, Any]] = []
        self.log_event(f"Custom cluster ({len(node_specs)} Nodes) initialized with '{scenario}' scenario ({self.total_scenario_jobs} jobs)", level="system")

    def reset(self, cluster_config: str, scenario: str, seed: int = 42) -> None:
        """Reset cluster and populate initial scenario workload."""
        self.cluster_config = cluster_config
        self.current_scenario = scenario
        self.current_seed = seed
        self.cluster = Cluster.from_yaml(self.cluster_config)
        self.simulator = Simulator(cluster=self.cluster, max_queue_size=16, horizon_seconds=100000.0)
        self.simulator.reset()

        # Adapt job requirements to cluster max node capacities so every job is physically schedulable
        max_cluster_gpus_per_node = max((n.gpu_count for n in self.cluster.nodes), default=8)
        max_cluster_vram = max((n.vram_per_gpu_gb for n in self.cluster.nodes), default=80.0)

        raw_jobs = create_scenario_workload(scenario, seed=seed)
        jobs = []
        for j in raw_jobs:
            j.gpu_count = max(1, min(j.gpu_count, max_cluster_gpus_per_node))
            j.vram_per_gpu_gb = min(j.vram_per_gpu_gb, max_cluster_vram)
            jobs.append(j)

        self.total_scenario_jobs = len(jobs)
        self.simulator.load_workload(jobs)

        # Advance to first scheduling decision
        done, _ = self.simulator.step_to_next_decision()

        self.recent_logs.clear()
        self.completed_history = []
        self.log_event(f"Cluster reset to '{scenario}' scenario ({self.total_scenario_jobs} jobs)", level="system")

    def submit_custom_job(self, req: JobSubmissionRequest) -> Dict[str, Any]:
        """Inject a custom user job directly into the live cluster queue."""
        self.custom_job_counter += 1
        w_type = WorkloadType.TRAINING
        try:
            w_type = WorkloadType(req.workload_type)
        except ValueError:
            pass

        now = self.simulator.current_time
        deadline = now + (req.deadline_slack * req.estimated_runtime)

        custom_job = Job(
            job_id=self.custom_job_counter,
            arrival_time=round(now, 1),
            gpu_count=req.gpu_count,
            vram_per_gpu_gb=req.vram_per_gpu_gb,
            estimated_runtime=req.estimated_runtime,
            actual_runtime=req.estimated_runtime,
            priority=req.priority,
            deadline=round(deadline, 1),
            workload_type=w_type,
        )

        pushed = self.simulator.queue.push(custom_job)
        if pushed:
            self.log_event(
                f"Custom Job #{custom_job.job_id} submitted: {req.gpu_count}x GPUs ({req.vram_per_gpu_gb}GB), Prio={req.priority}",
                level="success",
            )
            return {"status": "queued", "job_id": custom_job.job_id}
        else:
            self.log_event(f"Queue full! Could not accept Job #{custom_job.job_id}", level="warning")
            return {"status": "queue_full", "job_id": custom_job.job_id}

    def step_ppo(self) -> Dict[str, Any]:
        """
        Execute one PPO scheduling decision epoch on the live simulator.
        """
        ppo_policy = self.policies["PPO"]
        
        # 1. Multi-Action Placement Loop: place qualifying jobs iteratively at the same timestamp t
        action_taken = False
        placed_job_id = None
        target_node_id = None

        for _ in range(16): # Up to max queue size placements per decision point
            state = self.simulator.get_state()
            if not state.has_any_valid_action():
                break
            action = ppo_policy.select_action(state)
            if action is None:
                break
            job_idx, node_idx = action
            job = state.queue.get_at(job_idx)
            if job and self.simulator.apply_action(job_idx, node_idx):
                action_taken = True
                if placed_job_id is None:
                    placed_job_id = job.job_id
                    target_node_id = node_idx
            else:
                break

        # Advance PPO simulator to next event
        done, completed = self.simulator.step_to_next_decision()
        for c_job in completed:
            self.completed_history.insert(0, {
                "job_id": c_job.job_id,
                "gpu_count": c_job.gpu_count,
                "vram_gb": c_job.vram_per_gpu_gb,
                "turnaround_time": round(c_job.turnaround_time, 1) if c_job.turnaround_time is not None else 0.0,
                "waiting_time": round(c_job.waiting_time, 1),
                "workload_type": c_job.workload_type.value if hasattr(c_job.workload_type, "value") else str(c_job.workload_type),
                "finish_time": round(c_job.completion_time if c_job.completion_time is not None else self.simulator.current_time, 1),
                "node_id": c_job.allocated_node_id if c_job.allocated_node_id is not None else 0,
            })
            if len(self.completed_history) > 50:
                self.completed_history.pop()

        total_jobs = getattr(self, "total_scenario_jobs", len(self.simulator.submitted_jobs))
        completed_cnt = len(self.simulator.completed_jobs)
        running_cnt = sum(len(n.running_jobs) for n in self.simulator.cluster.nodes)
        queue_cnt = len(self.simulator.queue)
        remaining_cnt = max(0, total_jobs - completed_cnt)

        return {
            "policy": "PPO",
            "action_taken": action_taken,
            "job_id": placed_job_id,
            "node_id": target_node_id,
            "sim_time": self.simulator.current_time,
            "is_done": done,
            "completed_history": self.completed_history,
            "total_scenario_jobs": total_jobs,
            "completed_jobs_count": completed_cnt,
            "running_jobs_count": running_cnt,
            "queue_jobs_count": queue_cnt,
            "remaining_jobs_count": remaining_cnt,
        }

    def get_status(self) -> ClusterStatusResponse:
        """Construct full diagnostic state payload."""
        state = self.simulator.get_state()
        now = self.simulator.current_time

        # 1. Format Nodes
        nodes_info: List[NodeStatusInfo] = []
        for node in self.cluster.nodes:
            gpus_info = [
                GPUSlotInfo(
                    gpu_id=gpu.gpu_id,
                    gpu_type=gpu.gpu_type,
                    total_vram_gb=gpu.total_vram_gb,
                    allocated_vram_gb=gpu.allocated_vram_gb,
                    is_free=gpu.is_free,
                    running_job_id=gpu.running_job_id,
                )
                for gpu in node.gpus
            ]

            running_jobs_list = [
                {
                    "job_id": j.job_id,
                    "gpu_count": j.gpu_count,
                    "vram_gb": j.vram_per_gpu_gb,
                    "workload_type": j.workload_type.value if hasattr(j.workload_type, "value") else str(j.workload_type),
                    "start_time": j.start_time,
                    "elapsed_sec": round(now - (j.start_time or now), 1),
                    "total_runtime_sec": j.actual_runtime,
                    "progress_pct": min(100.0, max(0.0, (now - (j.start_time or now)) / max(1.0, j.actual_runtime) * 100.0)),
                }
                for j in node.running_jobs.values()
            ]

            nodes_info.append(
                NodeStatusInfo(
                    node_id=node.node_id,
                    gpu_type=node.gpu_type,
                    total_gpus=node.gpu_count,
                    available_gpus=node.available_gpu_count,
                    gpu_utilization_pct=round(node.gpu_utilization * 100.0, 1),
                    total_vram_gb=node.total_vram_gb,
                    allocated_vram_gb=node.allocated_vram_gb,
                    free_vram_gb=node.free_vram_gb,
                    vram_utilization_pct=round(node.vram_utilization * 100.0, 1),
                    gpus=gpus_info,
                    running_jobs=running_jobs_list,
                )
            )

        # 2. Format Queue
        queue_info: List[QueuedJobInfo] = []
        for slot_idx, job in enumerate(self.simulator.queue.jobs):
            wait_time = max(0.0, now - job.arrival_time)
            remaining_dl = max(0.0, job.deadline - now)
            is_urgent = remaining_dl < (job.estimated_runtime * 1.5)
            wtype_val = job.workload_type.value if hasattr(job.workload_type, "value") else str(job.workload_type)

            queue_info.append(
                QueuedJobInfo(
                    job_id=job.job_id,
                    slot_index=slot_idx,
                    gpu_count=job.gpu_count,
                    vram_per_gpu_gb=job.vram_per_gpu_gb,
                    estimated_runtime=job.estimated_runtime,
                    priority=job.priority,
                    workload_type=wtype_val,
                    arrival_time=job.arrival_time,
                    waiting_time=round(wait_time, 1),
                    deadline=job.deadline,
                    is_urgent=is_urgent,
                )
            )

        # 3. Aggregate Metrics
        metrics = self.simulator.get_metrics()

        return ClusterStatusResponse(
            simulation_time=round(now, 1),
            is_running=not self.simulator.event_queue.is_empty,
            active_policy=self.active_policy_name,
            scenario=self.current_scenario,
            nodes=nodes_info,
            queue=queue_info,
            total_gpus=self.cluster.total_gpus,
            available_gpus=self.cluster.available_gpus,
            cluster_gpu_utilization_pct=round(self.cluster.gpu_utilization * 100.0, 1),
            cluster_vram_utilization_pct=round(self.cluster.vram_utilization * 100.0, 1),
            metrics=metrics,
            action_mask=state.get_action_mask().tolist(),
            recent_logs=list(reversed(self.recent_logs)),
        )
