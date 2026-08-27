"""Inference service manager hosting live simulator state and policy models."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import os
import torch
import numpy as np

from simulator.cluster import Cluster
from simulator.simulator import Simulator
from simulator.job import Job, WorkloadType
from baselines.fifo import FIFOScheduler
from baselines.sjf import SJFScheduler
from baselines.priority import PriorityScheduler
from baselines.best_fit import BestFitScheduler
from evaluation.ppo_policy import PPOPolicyScheduler
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
        checkpoint_path: str = "checkpoints/ppo_final.pt",
        default_scenario: str = "balanced",
    ) -> None:
        self.cluster_config = cluster_config
        self.checkpoint_path = checkpoint_path
        self.current_scenario = default_scenario
        self.active_policy_name = "PPO"

        # Load cluster and simulator
        self.cluster = Cluster.from_yaml(self.cluster_config)
        self.simulator = Simulator(cluster=self.cluster, max_queue_size=16, horizon_seconds=3600.0)

        # Initialize all 5 policies
        self.policies = {
            "FIFO": FIFOScheduler(),
            "SJF": SJFScheduler(),
            "Priority": PriorityScheduler(),
            "BestFit": BestFitScheduler(),
            "PPO": PPOPolicyScheduler(
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
        """Reset simulator with custom user-built node specifications (1 to 10 nodes)."""
        self.current_scenario = scenario
        self.current_node_specs = node_specs
        self.current_seed = seed
        self.cluster = Cluster.create_dynamic(node_specs)
        self.simulator = Simulator(cluster=self.cluster, max_queue_size=16, horizon_seconds=3600.0)
        self.simulator.reset()

        # Update PPO internal temp environment for custom cluster
        if "PPO" in self.policies:
            self.policies["PPO"] = PPOPolicyScheduler(
                checkpoint_path=self.checkpoint_path,
                cluster_config_path=self.cluster_config,
            )

        # Load scenario jobs
        jobs = create_scenario_workload(scenario, seed=seed)
        self.simulator.load_workload(jobs)

        # Advance to first scheduling decision point
        done, _ = self.simulator.step_to_next_decision()

        # Also initialize simultaneous baseline simulators on identical workload clone
        self.baseline_sims: Dict[str, Simulator] = {}
        for b_name in ["FIFO", "SJF", "Priority", "BestFit"]:
            b_cluster = Cluster.create_dynamic(node_specs)
            b_sim = Simulator(cluster=b_cluster, max_queue_size=16, horizon_seconds=3600.0)
            b_sim.reset()
            b_jobs = create_scenario_workload(scenario, seed=seed)
            b_sim.load_workload(b_jobs)
            b_sim.step_to_next_decision()
            self.baseline_sims[b_name] = b_sim

        self.recent_logs.clear()
        self.completed_history: List[Dict[str, Any]] = []
        self.log_event(f"Custom cluster ({len(node_specs)} Nodes) initialized with '{scenario}' scenario", level="system")

    def reset(self, cluster_config: str, scenario: str, seed: int = 42) -> None:
        """Reset cluster and populate initial scenario workload."""
        self.cluster_config = cluster_config
        self.current_scenario = scenario
        self.current_seed = seed
        self.cluster = Cluster.from_yaml(self.cluster_config)
        self.simulator = Simulator(cluster=self.cluster, max_queue_size=16, horizon_seconds=3600.0)
        self.simulator.reset()

        # Load scenario jobs
        jobs = create_scenario_workload(scenario, seed=seed)
        self.simulator.load_workload(jobs)

        # Advance to first scheduling decision
        done, _ = self.simulator.step_to_next_decision()

        # Initialize simultaneous baseline simulators
        self.baseline_sims = {}
        for b_name in ["FIFO", "SJF", "Priority", "BestFit"]:
            b_cluster = Cluster.from_yaml(self.cluster_config)
            b_sim = Simulator(cluster=b_cluster, max_queue_size=16, horizon_seconds=3600.0)
            b_sim.reset()
            b_jobs = create_scenario_workload(scenario, seed=seed)
            b_sim.load_workload(b_jobs)
            b_sim.step_to_next_decision()
            self.baseline_sims[b_name] = b_sim

        self.recent_logs.clear()
        self.completed_history = []
        self.log_event(f"Cluster initialized with '{scenario}' scenario (seed={seed})", level="system")

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

    def step_simultaneous(self) -> Dict[str, Any]:
        """
        Step PPO on the primary interactive simulator and advance all 4 baselines simultaneously.
        """
        ppo_policy = self.policies["PPO"]
        state = self.simulator.get_state()
        action = ppo_policy.select_action(state)

        action_taken = False
        placed_job_id = None
        target_node_id = None

        if action is not None:
            job_idx, node_idx = action
            job = state.queue.get_at(job_idx)
            if job and self.simulator.apply_action(job_idx, node_idx):
                action_taken = True
                placed_job_id = job.job_id
                target_node_id = node_idx

        # Advance PPO simulator
        done, completed = self.simulator.step_to_next_decision()
        for c_job in completed:
            self.completed_history.insert(0, {
                "job_id": c_job.job_id,
                "gpu_count": c_job.gpu_count,
                "vram_gb": c_job.vram_per_gpu_gb,
                "turnaround_time": round(c_job.turnaround_time, 1),
                "waiting_time": round(c_job.waiting_time, 1),
                "workload_type": c_job.workload_type.value if hasattr(c_job.workload_type, "value") else str(c_job.workload_type),
                "finish_time": round(self.simulator.current_time, 1),
                "node_id": getattr(c_job, "node_id", target_node_id if target_node_id is not None else 0),
            })
            if len(self.completed_history) > 40:
                self.completed_history.pop()

        # Step all 4 baselines simultaneously up to current simulation timestamp
        current_t = self.simulator.current_time
        for b_name, b_sim in self.baseline_sims.items():
            b_policy = self.policies[b_name]
            # Step until caught up or finished
            for _ in range(20):
                if b_sim.current_time >= current_t:
                    break
                b_state = b_sim.get_state()
                b_act = b_policy.select_action(b_state)
                if b_act is not None:
                    b_sim.apply_action(*b_act)
                b_done, _ = b_sim.step_to_next_decision()
                if b_done:
                    break

        # Compute live comparative metrics
        benchmark_summary = self._compute_benchmark_summary()

        return {
            "action_taken": action_taken,
            "job_id": placed_job_id,
            "node_id": target_node_id,
            "sim_time": self.simulator.current_time,
            "is_done": done,
            "benchmark_summary": benchmark_summary,
            "completed_history": self.completed_history,
        }

    def _compute_benchmark_summary(self) -> List[Dict[str, Any]]:
        """Calculate metrics across PPO and all 4 baselines."""
        all_metrics = []

        # 1. PPO Metrics
        ppo_m = self.simulator.metrics.compute_summary()
        all_metrics.append({
            "policy": "PPO (Reinforcement Learning)",
            "short_name": "PPO",
            "is_rl": True,
            "completed": ppo_m.get("completed_jobs_count", 0),
            "mean_turnaround": round(ppo_m.get("mean_turnaround_time_sec", 0.0), 1),
            "mean_waiting": round(ppo_m.get("mean_waiting_time_sec", 0.0), 1),
            "gpu_util_pct": round(ppo_m.get("cluster_gpu_utilization_pct", 0.0), 1),
            "deadline_miss_pct": round(ppo_m.get("deadline_miss_rate_pct", 0.0), 1),
            "makespan": round(ppo_m.get("makespan_sec", self.simulator.current_time), 1),
        })

        # 2. Baseline Metrics
        for b_name in ["FIFO", "SJF", "Priority", "BestFit"]:
            if b_name in self.baseline_sims:
                b_m = self.baseline_sims[b_name].metrics.compute_summary()
                all_metrics.append({
                    "policy": b_name,
                    "short_name": b_name,
                    "is_rl": False,
                    "completed": b_m.get("completed_jobs_count", 0),
                    "mean_turnaround": round(b_m.get("mean_turnaround_time_sec", 0.0), 1),
                    "mean_waiting": round(b_m.get("mean_waiting_time_sec", 0.0), 1),
                    "gpu_util_pct": round(b_m.get("cluster_gpu_utilization_pct", 0.0), 1),
                    "deadline_miss_pct": round(b_m.get("deadline_miss_rate_pct", 0.0), 1),
                    "makespan": round(b_m.get("makespan_sec", self.baseline_sims[b_name].current_time), 1),
                })

        return all_metrics

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
