"""Unit tests for JobQueue operations."""

from simulator.queue import JobQueue
from simulator.job import Job


def test_queue_push_pop():
    q = JobQueue(max_size=3)
    assert q.is_empty
    assert not q.is_full

    j1 = Job(1, arrival_time=0.0, gpu_count=1, vram_per_gpu_gb=10.0, estimated_runtime=10.0, actual_runtime=10.0)
    j2 = Job(2, arrival_time=5.0, gpu_count=2, vram_per_gpu_gb=20.0, estimated_runtime=20.0, actual_runtime=20.0)
    j3 = Job(3, arrival_time=10.0, gpu_count=1, vram_per_gpu_gb=10.0, estimated_runtime=30.0, actual_runtime=30.0)
    j4 = Job(4, arrival_time=15.0, gpu_count=1, vram_per_gpu_gb=10.0, estimated_runtime=40.0, actual_runtime=40.0)

    assert q.push(j1)
    assert q.push(j2)
    assert q.push(j3)
    assert q.is_full
    assert not q.push(j4)  # capacity limit hit

    assert len(q) == 3
    popped = q.pop_at(1)
    assert popped is not None
    assert popped.job_id == 2
    assert len(q) == 2
    assert not q.is_full

    removed = q.remove_by_id(1)
    assert removed is not None
    assert removed.job_id == 1
    assert len(q) == 1
