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

        # Also initialize simultaneous baseline simulators on identical workload clone
        self.baseline_sims: Dict[str, Simulator] = {}
        for b_name in ["FIFO", "SJF", "Priority", "BestFit"]:
            b_cluster = Cluster.create_dynamic(node_specs)
            b_sim = Simulator(cluster=b_cluster, max_queue_size=16, horizon_seconds=None)
            b_sim.reset()
            b_jobs = [
                Job(
                    job_id=j.job_id,
                    workload_type=j.workload_type,
                    gpu_count=j.gpu_count,
                    vram_per_gpu_gb=j.vram_per_gpu_gb,
                    actual_runtime=j.actual_runtime,
                    estimated_runtime=j.estimated_runtime,
                    priority=j.priority,
                    arrival_time=j.arrival_time,
                    deadline=j.deadline,
                ) for j in jobs
            ]
            b_sim.load_workload(b_jobs)
            b_sim.step_to_next_decision()
            self.baseline_sims[b_name] = b_sim

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

        # Load scenario jobs
        jobs = create_scenario_workload(scenario, seed=seed)
        self.total_scenario_jobs = len(jobs)
        self.simulator.load_workload(jobs)

        # Advance to first scheduling decision
        done, _ = self.simulator.step_to_next_decision()

        # Initialize simultaneous baseline simulators
        self.baseline_sims = {}
        for b_name in ["FIFO", "SJF", "Priority", "BestFit"]:
            b_cluster = Cluster.from_yaml(self.cluster_config)
            b_sim = Simulator(cluster=b_cluster, max_queue_size=16, horizon_seconds=100000.0)
            b_sim.reset()
            b_jobs = create_scenario_workload(scenario, seed=seed)
            b_sim.load_workload(b_jobs)
            b_sim.step_to_next_decision()
            self.baseline_sims[b_name] = b_sim

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

    def step_simultaneous(self) -> Dict[str, Any]:
        """
        Step PPO on the primary interactive simulator and advance all 4 baselines simultaneously.
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

        # Step all 4 baselines simultaneously up to current simulation timestamp
        current_t = self.simulator.current_time
        for b_name, b_sim in self.baseline_sims.items():
            b_policy = self.policies[b_name]
            for _ in range(25):
                if b_sim.current_time >= current_t:
                    break
                b_state = b_sim.get_state()
                b_act = b_policy.select_action(b_state)
                if b_act is not None:
                    b_sim.apply_action(*b_act)
                b_done, _ = b_sim.step_to_next_decision()
                if b_done:
                    break

        # If PPO simulation finished all jobs, let baselines finish all remaining jobs too
        if done:
            for b_name, b_sim in self.baseline_sims.items():
                b_policy = self.policies[b_name]
                for _ in range(400):
                    b_state = b_sim.get_state()
                    b_act = b_policy.select_action(b_state)
                    if b_act is not None:
                        b_sim.apply_action(*b_act)
                    b_done, _ = b_sim.step_to_next_decision()
                    if b_done:
                        break

        # Compute live comparative metrics
        benchmark_summary = self._compute_benchmark_summary()

        total_jobs = getattr(self, "total_scenario_jobs", len(self.simulator.submitted_jobs))
        completed_cnt = len(self.simulator.completed_jobs)
        running_cnt = sum(len(n.running_jobs) for n in self.simulator.cluster.nodes)
        queue_cnt = len(self.simulator.queue)
        remaining_cnt = max(0, total_jobs - completed_cnt)

        return {
            "action_taken": action_taken,
            "job_id": placed_job_id,
            "node_id": target_node_id,
            "sim_time": self.simulator.current_time,
            "is_done": done,
            "benchmark_summary": benchmark_summary,
            "completed_history": self.completed_history,
            "total_scenario_jobs": total_jobs,
            "completed_jobs_count": completed_cnt,
            "running_jobs_count": running_cnt,
            "queue_jobs_count": queue_cnt,
            "remaining_jobs_count": remaining_cnt,
        }

    def _extract_sim_metrics(self, sim: Simulator) -> Dict[str, Any]:
        """Extract academic and systems benchmark metrics (JCT, Slowdown, Makespan, Fairness, Util)."""
        c_jobs = sim.completed_jobs
        c_count = len(c_jobs)
        if c_count > 0:
            turnarounds = [j.turnaround_time for j in c_jobs if j.turnaround_time is not None]
            mean_jct = float(np.mean(turnarounds)) if turnarounds else 0.0
            
            # Bounded Slowdown: JCT / max(Runtime, 10.0s) (Decima / DeepRM standard)
            slowdowns = [
                j.turnaround_time / max(10.0, j.actual_runtime)
                for j in c_jobs if j.turnaround_time is not None
            ]
            mean_slowdown = float(np.mean(slowdowns)) if slowdowns else 1.0
            p95_slowdown = float(np.percentile(slowdowns, 95)) if slowdowns else 1.0

            # Jain's Fairness Index on Inverse Slowdown: (sum x)^2 / (n * sum x^2)
            inv_slowdowns = [1.0 / s for s in slowdowns if s > 0]
            if inv_slowdowns:
                sum_x = sum(inv_slowdowns)
                sum_sq_x = sum(x * x for x in inv_slowdowns)
                jains_fairness = float((sum_x * sum_x) / (len(inv_slowdowns) * sum_sq_x)) if sum_sq_x > 0 else 1.0
            else:
                jains_fairness = 1.0

            mean_waiting = float(np.mean([j.waiting_time for j in c_jobs]))
            miss_count = sum(1 for j in c_jobs if j.is_deadline_missed())
            deadline_miss_pct = (miss_count / c_count) * 100.0
            last_completion = max([j.completion_time for j in c_jobs if j.completion_time is not None] or [sim.current_time])
            total_duration = last_completion
            high_prio = [j.waiting_time for j in c_jobs if j.priority >= 7]
            low_prio = [j.waiting_time for j in c_jobs if j.priority < 7]
            mean_high_prio_wait = float(np.mean(high_prio)) if high_prio else 0.0
            mean_low_prio_wait = float(np.mean(low_prio)) if low_prio else 0.0
        else:
            mean_jct = 0.0
            mean_slowdown = 1.0
            p95_slowdown = 1.0
            jains_fairness = 1.0
            mean_waiting = 0.0
            mean_high_prio_wait = 0.0
            mean_low_prio_wait = 0.0
            deadline_miss_pct = 0.0
            total_duration = sim.current_time

        duration = max(1.0, total_duration)
        total_cluster_gpus = sim.cluster.total_gpus
        if total_cluster_gpus > 0:
            avg_gpu_util = (sim.integrated_busy_gpu_seconds / (total_cluster_gpus * duration)) * 100.0
            avg_gpu_util = min(100.0, max(0.0, avg_gpu_util))
        else:
            avg_gpu_util = 0.0

        return {
            "completed": c_count,
            "mean_jct": round(mean_jct, 1),
            "mean_turnaround": round(mean_jct, 1),
            "mean_slowdown": round(mean_slowdown, 2),
            "p95_slowdown": round(p95_slowdown, 2),
            "jains_fairness": round(jains_fairness, 3),
            "mean_waiting": round(mean_waiting, 1),
            "high_prio_wait": round(mean_high_prio_wait, 1),
            "low_prio_wait": round(mean_low_prio_wait, 1),
            "gpu_util_pct": round(avg_gpu_util, 1),
            "deadline_miss_pct": round(deadline_miss_pct, 1),
            "sla_compliance_pct": round(100.0 - deadline_miss_pct, 1),
            "makespan": round(total_duration, 1),
        }

    def _compute_benchmark_summary(self) -> List[Dict[str, Any]]:
        """Calculate academic metrics across PPO and all 4 baselines."""
        all_metrics = []

        # 1. PPO Metrics
        ppo_m = self._extract_sim_metrics(self.simulator)
        all_metrics.append({
            "policy": "PPO (Reinforcement Learning)",
            "short_name": "PPO",
            "is_rl": True,
            **ppo_m,
        })

        # 2. Baseline Metrics
        for b_name in ["FIFO", "SJF", "Priority", "BestFit"]:
            if b_name in self.baseline_sims:
                b_m = self._extract_sim_metrics(self.baseline_sims[b_name])
                all_metrics.append({
                    "policy": b_name,
                    "short_name": b_name,
                    "is_rl": False,
                    **b_m,
                })

        # Compute relative Pareto Efficiency Score (0 - 100)
        # Weights: 35% SLA Compliance + 30% Effective GPU Util + 20% Low Slowdown + 15% Makespan
        min_makespan = min((m["makespan"] for m in all_metrics if m["makespan"] > 0), default=1.0) or 1.0
        for m in all_metrics:
            sla_score = m["sla_compliance_pct"]
            util_score = m["gpu_util_pct"]
            slowdown_score = max(0.0, 100.0 * (1.0 - ((m["mean_slowdown"] - 1.0) / 4.0)))
            makespan_score = min(100.0, (min_makespan / max(1.0, m["makespan"])) * 100.0)
            composite = (sla_score * 0.35) + (util_score * 0.30) + (slowdown_score * 0.20) + (makespan_score * 0.15)
            m["efficiency_score"] = round(composite, 1)

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
