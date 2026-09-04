"""Unit tests for discrete-event simulator execution."""

from simulator.cluster import Cluster
from simulator.simulator import Simulator
from simulator.job import Job


def test_simulator_basic_flow():
    cluster = Cluster.from_yaml("configs/cluster_small.yaml")
    sim = Simulator(cluster=cluster, max_queue_size=8, horizon_seconds=1000.0)
    sim.reset()

    # Create 3 jobs arriving at different times
    jobs = [
        Job(job_id=1, arrival_time=0.0, gpu_count=2, vram_per_gpu_gb=20.0, estimated_runtime=50.0, actual_runtime=50.0),
        Job(job_id=2, arrival_time=10.0, gpu_count=4, vram_per_gpu_gb=20.0, estimated_runtime=100.0, actual_runtime=100.0),
        Job(job_id=3, arrival_time=20.0, gpu_count=2, vram_per_gpu_gb=20.0, estimated_runtime=30.0, actual_runtime=30.0),
    ]
    sim.load_workload(jobs)

    # Step to first decision
    done, _ = sim.step_to_next_decision()
    assert not done
    state = sim.get_state()
    assert len(state.queue) >= 1
    assert state.has_any_valid_action()

    # Schedule Job 1 on Node 0
    assert sim.apply_action(job_index=0, node_index=0)

    # Run remaining events with simple greedy FIFO policy
    while not done:
        state = sim.get_state()
        action_taken = False
        mask = state.get_action_mask()

        # Find first feasible action
        for j_idx in range(len(state.queue)):
            for n_idx in range(state.num_nodes):
                if mask[j_idx, n_idx] > 0:
                    sim.apply_action(j_idx, n_idx)
                    action_taken = True
                    break
            if action_taken:
                break

        done, _ = sim.step_to_next_decision()

    metrics = sim.get_metrics()
    assert metrics["completed_jobs"] == 3
    assert metrics["invalid_action_count"] == 0
    assert metrics["mean_jct"] > 0


def test_full_visible_queue_preserves_arrivals():
    """Arrivals beyond the policy-visible queue must still complete."""
    cluster = Cluster.from_yaml("configs/cluster_small.yaml")
    sim = Simulator(cluster=cluster, max_queue_size=1, horizon_seconds=None)
    sim.reset()

    jobs = [
        Job(job_id=i, arrival_time=0.0, gpu_count=1, vram_per_gpu_gb=20.0,
            estimated_runtime=1.0, actual_runtime=1.0)
        for i in range(1, 4)
    ]
    sim.load_workload(jobs)

    done, _ = sim.step_to_next_decision()
    while not done:
        state = sim.get_state()
        for job_idx in range(len(state.queue)):
            for node_idx in range(state.num_nodes):
                if state.is_action_valid(job_idx, node_idx):
                    assert sim.apply_action(job_idx, node_idx)
                    break
            else:
                continue
            break
        done, _ = sim.step_to_next_decision()

    assert len(sim.completed_jobs) == len(jobs)
    assert sim.overflow_jobs == []
