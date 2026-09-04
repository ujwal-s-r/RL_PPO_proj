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
        # Keep arrivals beyond the policy-visible queue instead of dropping them.
        # The observation/action space stays bounded; this backlog preserves workload correctness.
        self.overflow_jobs: List[Job] = []
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
        self.integrated_weighted_queue_seconds: float = 0.0
        self.integrated_weighted_running_seconds: float = 0.0

    def reset(self) -> SchedulerState:
        """Reset the simulator to initial empty cluster state at t=0."""
        self.cluster.reset()
        self.queue.clear()
        self.overflow_jobs.clear()
        self.event_queue.clear()
        self.current_time = 0.0
        self.last_time = 0.0

        self.submitted_jobs.clear()
        self.completed_jobs.clear()
        self.invalid_action_count = 0

        self.integrated_queue_wait_seconds = 0.0
        self.integrated_idle_gpu_seconds = 0.0
        self.integrated_busy_gpu_seconds = 0.0
        self.integrated_weighted_queue_seconds = 0.0
        self.integrated_weighted_running_seconds = 0.0

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

    def _promote_overflow_jobs(self) -> None:
        """Move arrived jobs into the visible queue whenever capacity is available."""
        while self.overflow_jobs and not self.queue.is_full:
            self.queue.push(self.overflow_jobs.pop(0))

    def _enqueue_arrival(self, job: Job) -> None:
        """Enqueue an arrival without losing it when the visible queue is full."""
        if not self.queue.push(job):
            self.overflow_jobs.append(job)

    def _update_integrals(self, new_time: float) -> None:
        """Accumulate area under curve for queue length, GPU idle/busy states, and priority flow-time."""
        dt = max(0.0, new_time - self.current_time)
        if dt > 0:
            self.integrated_queue_wait_seconds += len(self.queue) * dt
            idle_gpus = self.cluster.available_gpus
            busy_gpus = self.cluster.total_gpus - idle_gpus
            self.integrated_idle_gpu_seconds += idle_gpus * dt
            self.integrated_busy_gpu_seconds += busy_gpus * dt

            # Priority-weighted flow-time: sum(priority / 5.0 * dt) across queue and running jobs
            pending_jobs = list(self.queue.jobs) + list(self.overflow_jobs)
            self.integrated_queue_wait_seconds += len(self.overflow_jobs) * dt
            q_weight = sum(j.priority / 5.0 for j in pending_jobs)
            r_weight = sum(j.priority / 5.0 for n in self.cluster.nodes for j in n.running_jobs.values())
            self.integrated_weighted_queue_seconds += q_weight * dt
            self.integrated_weighted_running_seconds += r_weight * dt

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

        self._promote_overflow_jobs()

        # Invariant check
        self.cluster.validate_invariants()
        return True

    def _process_event(self, event: Event, step_completed_jobs: List[Job]) -> bool:
        """Process a single event. Returns True if HORIZON_REACHED."""
        if event.event_type == EventType.HORIZON_REACHED:
            return True
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
                self._enqueue_arrival(event.job)
        return False

    def step_to_next_decision(self) -> Tuple[bool, List[Job]]:
        """
        Process non-decision events until:
        1. An actionable scheduling state exists at current_time (allowing multi-placement at same t), OR
        2. Time advances to the next timestamp where an actionable scheduling state exists, OR
        3. Episode terminates (horizon reached or all jobs completed).
        
        Returns:
            (is_done, list_of_jobs_completed_in_this_step)
        """
        step_completed_jobs: List[Job] = []

        self._promote_overflow_jobs()

        # 1. Process all pending events at current_time (arrivals / completions right now)
        while not self.event_queue.is_empty and self.event_queue.peek().timestamp <= self.current_time + 1e-9:
            event = self.event_queue.pop()
            is_horizon = self._process_event(event, step_completed_jobs)
            if is_horizon:
                return True, step_completed_jobs

        self.cluster.validate_invariants()

        # 2. If valid placements exist RIGHT NOW at current_time, return immediately so policy can place next job at same t!
        if self.get_state().has_any_valid_action():
            return False, step_completed_jobs

        # 3. Only when NO valid action exists at current_time, advance time to future events
        while not self.event_queue.is_empty:
            next_event = self.event_queue.peek()
            next_time = next_event.timestamp
            self._update_integrals(next_time)

            # Process all events occurring at this new timestamp
            while not self.event_queue.is_empty and self.event_queue.peek().timestamp <= self.current_time + 1e-9:
                event = self.event_queue.pop()
                is_horizon = self._process_event(event, step_completed_jobs)
                if is_horizon:
                    return True, step_completed_jobs

            self.cluster.validate_invariants()

            # Check if all submitted jobs are now completed (done immediately without jumping to horizon)
            running_count = sum(len(n.running_jobs) for n in self.cluster.nodes)
            if (
                len(self.completed_jobs) >= len(self.submitted_jobs)
                and len(self.queue) == 0
                and len(self.overflow_jobs) == 0
                and running_count == 0
            ):
                return True, step_completed_jobs

            # If an actionable state now exists at this new timestamp, pause and let policy decide
            if self.get_state().has_any_valid_action():
                return False, step_completed_jobs

        # No more events remaining: check if simulation is fully complete
        running_count = sum(len(n.running_jobs) for n in self.cluster.nodes)
        is_done = (
            len(self.completed_jobs) >= len(self.submitted_jobs)
            and len(self.queue) == 0
            and len(self.overflow_jobs) == 0
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
            horizon_time=self.horizon_seconds if self.horizon_seconds is not None else 100000.0,
            next_arrival=self.peek_next_arrival(),
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Compute end-of-episode or checkpoint evaluation metrics (uncensored)."""
        duration = max(1.0, self.current_time)
        completed = self.completed_jobs
        total_submitted = max(1, len(self.submitted_jobs))
        
        jcts = [j.turnaround_time for j in completed if j.turnaround_time is not None]
        waits = [j.waiting_time for j in completed]

        # Calculate deadline violations across ALL submitted jobs (uncensored)
        # 1. Completed jobs that finished after deadline
        completed_misses = [j for j in completed if j.is_deadline_missed()]
        # 2. Unfinished jobs (in queue or still running) whose deadline has already elapsed
        running_jobs = [job for node in self.cluster.nodes for job in node.running_jobs.values()]
        queued_jobs = list(self.queue.jobs) + list(self.overflow_jobs)
        unfinished_misses = [j for j in (running_jobs + queued_jobs) if self.current_time > j.deadline]
        
        total_misses_count = len(completed_misses) + len(unfinished_misses)
        deadline_violation_rate = total_misses_count / total_submitted
        sla_compliance_rate = max(0.0, 1.0 - deadline_violation_rate)

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
            "completion_rate": len(completed) / total_submitted,
            "mean_jct": mean_jct,
            "p95_jct": p95_jct,
            "mean_wait_time": mean_wait,
            "gpu_utilization": avg_gpu_utilization,
            "throughput_jobs_per_hour": (len(completed) / duration) * 3600.0,
            "deadline_violation_rate": deadline_violation_rate,
            "sla_compliance_rate": sla_compliance_rate,
            "sla_compliance_pct": sla_compliance_rate * 100.0,
            "invalid_action_count": self.invalid_action_count,
            "simulation_duration": duration,
            "integrated_busy_gpu_seconds": self.integrated_busy_gpu_seconds,
            "integrated_idle_gpu_seconds": self.integrated_idle_gpu_seconds,
            "integrated_queue_wait_seconds": self.integrated_queue_wait_seconds,
        }
