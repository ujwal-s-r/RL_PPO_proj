"""Unit tests for GPU and Node models."""

from simulator.gpu import GPU
from simulator.node import Node
from simulator.job import Job, WorkloadType


def test_gpu_lifecycle():
    gpu = GPU(gpu_id=0, gpu_type="A100_80GB", total_vram_gb=80.0)
    assert gpu.is_free
    assert gpu.free_vram_gb == 80.0
    assert gpu.utilization == 0.0

    # Allocate
    assert gpu.can_fit(40.0)
    assert not gpu.can_fit(90.0)
    assert gpu.allocate(job_id=101, vram_gb=40.0)
    assert not gpu.is_free
    assert gpu.allocated_vram_gb == 40.0
    assert gpu.utilization == 0.5

    # Cannot double-allocate
    assert not gpu.allocate(job_id=102, vram_gb=20.0)

    # Release
    gpu.release()
    assert gpu.is_free
    assert gpu.allocated_vram_gb == 0.0


def test_node_scheduling():
    node = Node(
        node_id=0,
        gpu_type="A100_40GB",
        gpu_count=4,
        vram_per_gpu_gb=40.0,
        cpu_cores=32,
        ram_gb=128.0,
    )
    assert node.available_gpu_count == 4
    assert node.free_vram_gb == 160.0

    # Job requiring 2 GPUs with 32GB VRAM each
    job = Job(
        job_id=1,
        arrival_time=0.0,
        gpu_count=2,
        vram_per_gpu_gb=32.0,
        estimated_runtime=100.0,
        actual_runtime=100.0,
    )

    assert node.can_schedule(job)
    assigned_gpus = node.schedule(job, current_time=10.0)
    assert len(assigned_gpus) == 2
    assert node.available_gpu_count == 2
    assert node.gpu_utilization == 0.5
    assert job.status.value == "RUNNING"
    assert job.start_time == 10.0

    # Oversized VRAM requirement
    big_vram_job = Job(
        job_id=2,
        arrival_time=10.0,
        gpu_count=1,
        vram_per_gpu_gb=80.0,
        estimated_runtime=50.0,
        actual_runtime=50.0,
    )
    assert not node.can_schedule(big_vram_job)

    # Complete job
    completed_job = node.complete(job.job_id, current_time=110.0)
    assert completed_job is not None
    assert completed_job.is_completed
    assert node.available_gpu_count == 4
    assert node.gpu_utilization == 0.0
