"""Unit tests for Cluster and safety invariants."""

from simulator.cluster import Cluster
from simulator.node import Node
from simulator.job import Job


def test_cluster_initialization():
    cluster = Cluster.from_yaml("configs/cluster_small.yaml")
    assert cluster.num_nodes == 2
    assert cluster.total_gpus == 8
    assert cluster.available_gpus == 8
    assert cluster.gpu_utilization == 0.0

    cluster.validate_invariants()


def test_cluster_invariants_during_execution():
    cluster = Cluster.from_yaml("configs/cluster_small.yaml")
    node0 = cluster.get_node(0)
    assert node0 is not None

    job = Job(
        job_id=42,
        arrival_time=0.0,
        gpu_count=4,
        vram_per_gpu_gb=32.0,
        estimated_runtime=200.0,
        actual_runtime=200.0,
    )

    node0.schedule(job, current_time=0.0)
    cluster.validate_invariants()

    assert cluster.available_gpus == 4
    assert cluster.gpu_utilization == 0.5

    node0.complete(job.job_id, current_time=200.0)
    cluster.validate_invariants()
    assert cluster.available_gpus == 8
    assert cluster.gpu_utilization == 0.0
