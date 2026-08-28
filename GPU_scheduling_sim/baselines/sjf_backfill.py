"""Shortest Job First with Conservative Backfilling (Slurm/Volcano style)."""

from __future__ import annotations
from typing import List, Optional, Tuple
from baselines.base import BaseScheduler
from simulator.scheduler_state import SchedulerState


class SJFBackfillScheduler(BaseScheduler):
    """
    SJF + Conservative Backfilling Scheduler.
    
    1. Tries to schedule the shortest job first.
    2. If the head/shortest job cannot fit right now, computes its earliest availability
       reservation and backfills subsequent smaller jobs that fit without delaying it.
    """

    @property
    def name(self) -> str:
        return "SJF_Backfill"

    def select_action(self, state: SchedulerState) -> Optional[Tuple[int, int]]:
        mask = state.get_action_mask()
        queue_len = min(len(state.queue), mask.shape[0])
        if queue_len == 0:
            return None

        # Collect all jobs in the visible queue
        all_jobs: List[Tuple[int, float, int, float]] = [] # (j_idx, est_runtime, gpu_req, vram_req)
        for j_idx in range(queue_len):
            job = state.queue.get_at(j_idx)
            if job is not None:
                all_jobs.append((j_idx, job.estimated_runtime, job.gpu_count, job.vram_per_gpu_gb))

        if not all_jobs:
            return None

        # Sort all jobs by SJF order
        sjf_sorted = sorted(all_jobs, key=lambda x: x[1])
        head_job = sjf_sorted[0]
        head_idx = head_job[0]

        # 1. If head job can be scheduled immediately on any node -> schedule it on best-fit node
        feasible_nodes_head = [n for n in range(state.num_nodes) if mask[head_idx, n] > 0]
        if feasible_nodes_head:
            # Pick node with least free GPUs remaining after placement (best fit)
            best_node = min(
                feasible_nodes_head,
                key=lambda n: state.cluster.nodes[n].available_gpu_count - head_job[2]
            )
            return head_idx, best_node

        # 2. Head job cannot fit right now -> find earliest completion time among running jobs
        # to establish reservation window T_avail
        running_jobs = [j for n in state.cluster.nodes for j in n.running_jobs.values()]
        if not running_jobs:
            return None

        # Estimate time until head job can run
        min_remaining_time = min(
            max(5.0, (j.start_time or state.current_time) + j.actual_runtime - state.current_time)
            for j in running_jobs
        )

        # 3. Backfill: search subsequent jobs that fit right now AND complete within min_remaining_time
        for candidate in sjf_sorted[1:]:
            c_idx, c_runtime, c_gpus, _ = candidate
            feasible_nodes = [n for n in range(state.num_nodes) if mask[c_idx, n] > 0]
            if feasible_nodes and c_runtime <= min_remaining_time:
                best_node = min(
                    feasible_nodes,
                    key=lambda n: state.cluster.nodes[n].available_gpu_count - c_gpus
                )
                return c_idx, best_node

        # 4. If no backfill job meets the strict window, schedule any feasible candidate
        for candidate in sjf_sorted[1:]:
            c_idx, _, c_gpus, _ = candidate
            feasible_nodes = [n for n in range(state.num_nodes) if mask[c_idx, n] > 0]
            if feasible_nodes:
                best_node = min(
                    feasible_nodes,
                    key=lambda n: state.cluster.nodes[n].available_gpu_count - c_gpus
                )
                return c_idx, best_node

        return None
