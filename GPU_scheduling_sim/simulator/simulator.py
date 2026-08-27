"""Discrete-event GPU Cluster Simulator Engine."""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from simulator.cluster import Cluster
from simulator.node import Node
from simulator.job import Job, JobStatus
from simulator.queue import JobQueue
from simulator.events import Event, EventPriority, EventQueue, EventType
from simulator.scheduler_state import SchedulerState


class Simulator:
    """
    High-performance discrete-event simulator for GPU cluster scheduling.
    
    Time advances jump directly to the next scheduled event rather than
    fixed small delta increments.
    """

    def __init__(
        self,
        cluster: Cluster,
        max_queue_size: int = 16,
        horizon_seconds: float = 3600.0,
    ) -> None:
        self.cluster = cluster
        self.max_queue_size = max_queue_size
        self.horizon_seconds = horizon_seconds

        self.queue = JobQueue(max_size=max_queue_size)
        self.event_queue = EventQueue()
        self.current_time: float = 0.0
        self.last_time: float = 0.0

        # Lifecycle tracking
        self.submitted_jobs: List[Job] = []
        self.completed_jobs: List[Job] = []
        self.invalid_action_count: int = 0

        # Time-integrated metrics (for continuous reward accounting)
        self.integrated_queue_wait_seconds: float = 0.0
        self.integrated_idle_gpu_seconds: float = 0.0
        self.integrated_busy_gpu_seconds: float = 0.0

    def reset(self) -> SchedulerState:
        """Reset the simulator to initial empty cluster state at t=0."""
        self.cluster.reset()
        self.queue.clear()
        self.event_queue.clear()
        self.current_time = 0.0
        self.last_time = 0.0

        self.submitted_jobs.clear()
        self.completed_jobs.clear()
        self.invalid_action_count = 0

        self.integrated_queue_wait_seconds = 0.0
        self.integrated_idle_gpu_seconds = 0.0
        self.integrated_busy_gpu_seconds = 0.0

        # Schedule end of horizon event only if explicitly configured
        if self.horizon_seconds is not None and self.horizon_seconds > 0:
            self.event_queue.push(
                Event(
                    timestamp=self.horizon_seconds,
                    priority=EventPriority.HORIZON,
                    event_type=EventType.HORIZON_REACHED,
                )
            )
        return self.get_state()

    def submit_job(self, job: Job) -> None:
        """Register a future job arrival event in the timeline."""
        self.submitted_jobs.append(job)
        self.event_queue.push(
            Event(
                timestamp=job.arrival_time,
                priority=EventPriority.ARRIVAL,
                event_type=EventType.JOB_ARRIVAL,
                job=job,
            )
        )

    def load_workload(self, jobs: List[Job]) -> None:
        """Bulk load a sequence of workload jobs."""
        for job in jobs:
            self.submit_job(job)

    def _update_integrals(self, new_time: float) -> None:
        """Accumulate area under curve for queue length and GPU idle/busy states."""
        dt = max(0.0, new_time - self.current_time)
        if dt > 0:
            self.integrated_queue_wait_seconds += len(self.queue) * dt
            idle_gpus = self.cluster.available_gpus
            busy_gpus = self.cluster.total_gpus - idle_gpus
            self.integrated_idle_gpu_seconds += idle_gpus * dt
            self.integrated_busy_gpu_seconds += busy_gpus * dt
            self.current_time = new_time

    def apply_action(self, job_index: int, node_index: int) -> bool:
        """
        Attempt to place job at job_index onto node_index.
        
        Returns:
            True if placement succeeded, False if action was invalid.
        """
        # Validate action
        state = self.get_state()
        if not state.is_action_valid(job_index, node_index):
            self.invalid_action_count += 1
            return False

        job = self.queue.pop_at(job_index)
        if job is None:
            self.invalid_action_count += 1
            return False

        node = self.cluster.get_node(node_index)
        if node is None:
            self.invalid_action_count += 1
            return False

        # Schedule job on node
        node.schedule(job, current_time=self.current_time)

        # Register future completion event
        completion_time = self.current_time + job.actual_runtime
        self.event_queue.push(
            Event(
                timestamp=completion_time,
                priority=EventPriority.COMPLETION,
                event_type=EventType.JOB_COMPLETION,
                job=job,
                node_id=node.node_id,
            )
        )

        # Invariant check
        self.cluster.validate_invariants()
        return True

    def step_to_next_decision(self) -> Tuple[bool, List[Job]]:
        """
        Process non-decision events (completions, arrivals, horizon) until:
        1. A valid scheduling action becomes possible, OR
        2. Episode terminates (horizon reached or all jobs completed).
        
        Returns:
            (is_done, list_of_jobs_completed_in_this_step)
        """
        step_completed_jobs: List[Job] = []

        while not self.event_queue.is_empty:
            event = self.event_queue.pop()
            self._update_integrals(event.timestamp)

            if event.event_type == EventType.HORIZON_REACHED:
                return True, step_completed_jobs

            elif event.event_type == EventType.JOB_COMPLETION:
                if event.job is not None and event.node_id is not None:
                    node = self.cluster.get_node(event.node_id)
                    if node is not None:
                        completed = node.complete(event.job.job_id, self.current_time)
                        if completed is not None:
                            self.completed_jobs.append(completed)
                            step_completed_jobs.append(completed)

            elif event.event_type == EventType.JOB_ARRIVAL:
                if event.job is not None:
                    # Queue is unlimited so push always succeeds — zero drops
                    self.queue.push(event.job)

            # Recheck invariants after every event processing
            self.cluster.validate_invariants()

            # If we now have an actionable state, stop and present to scheduler
            if self.get_state().has_any_valid_action():
                return False, step_completed_jobs

        # No more events remaining: done only if ALL submitted jobs completed and nothing running
        running_count = sum(len(n.running_jobs) for n in self.cluster.nodes)
        is_done = (
            len(self.completed_jobs) >= len(self.submitted_jobs)
            and len(self.queue) == 0
            and running_count == 0
        )
        return is_done, step_completed_jobs

    def peek_next_arrival(self) -> Tuple[float, float, float]:
        """
        Inspect the event queue for the next upcoming job arrival.
        
        Returns:
            (time_until_arrival_sec, gpu_count, vram_per_gpu_gb).
            If no upcoming arrival exists, returns (3600.0, 0.0, 0.0).
        """
        for event in list(self.event_queue._heap):
            if event.event_type == EventType.JOB_ARRIVAL and event.job is not None:
                dt = max(0.0, event.timestamp - self.current_time)
                return dt, float(event.job.gpu_count), float(event.job.vram_per_gpu_gb)
        return 3600.0, 0.0, 0.0

    def get_state(self) -> SchedulerState:
        """Generate snapshot of current scheduler state."""
        return SchedulerState(
            current_time=self.current_time,
            cluster=self.cluster,
            queue=self.queue,
            max_queue_size=self.max_queue_size,
            completed_jobs_count=len(self.completed_jobs),
            total_jobs_arrived=len(self.submitted_jobs),
            horizon_time=self.horizon_seconds,
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Compute end-of-episode or checkpoint evaluation metrics."""
        duration = max(1.0, self.current_time)
        completed = self.completed_jobs
        
        jcts = [j.turnaround_time for j in completed if j.turnaround_time is not None]
        waits = [j.waiting_time for j in completed]
        deadline_misses = [j for j in completed if j.is_deadline_missed()]

        mean_jct = float(np.mean(jcts)) if jcts else 0.0
        p95_jct = float(np.percentile(jcts, 95)) if jcts else 0.0
        mean_wait = float(np.mean(waits)) if waits else 0.0
        
        total_cluster_gpus = self.cluster.total_gpus
        avg_gpu_utilization = (
            self.integrated_busy_gpu_seconds / (total_cluster_gpus * duration)
            if total_cluster_gpus > 0 else 0.0
        )

        return {
            "completed_jobs": len(completed),
            "submitted_jobs": len(self.submitted_jobs),
            "mean_jct": mean_jct,
            "p95_jct": p95_jct,
            "mean_wait_time": mean_wait,
            "gpu_utilization": avg_gpu_utilization,
            "throughput_jobs_per_hour": (len(completed) / duration) * 3600.0,
            "deadline_violation_rate": (len(deadline_misses) / len(completed)) if completed else 0.0,
            "invalid_action_count": self.invalid_action_count,
            "simulation_duration": duration,
        }
