"""Unit tests for workload generation and scenario distributions."""

from workloads.generators import WorkloadConfig, WorkloadGenerator
from workloads.scenarios import SCENARIOS, get_scenario, create_scenario_workload, list_scenarios
from simulator.job import Job


def test_seed_determinism():
    cfg = get_scenario("balanced")
    gen = WorkloadGenerator(cfg)

    jobs1 = gen.generate(seed=123)
    jobs2 = gen.generate(seed=123)
    jobs3 = gen.generate(seed=456)

    assert len(jobs1) > 0
    assert len(jobs1) == len(jobs2)
    assert len(jobs1) != len(jobs3) or jobs1[0].actual_runtime != jobs3[0].actual_runtime

    # Verify exact match on attributes
    for j1, j2 in zip(jobs1, jobs2):
        assert j1.job_id == j2.job_id
        assert j1.arrival_time == j2.arrival_time
        assert j1.gpu_count == j2.gpu_count
        assert j1.vram_per_gpu_gb == j2.vram_per_gpu_gb
        assert j1.actual_runtime == j2.actual_runtime
        assert j1.deadline == j2.deadline


def test_all_scenarios_generation():
    scenario_names = list_scenarios()
    assert len(scenario_names) == 6

    for name in scenario_names:
        jobs = create_scenario_workload(name, seed=42)
        assert len(jobs) > 0, f"Scenario {name} produced 0 jobs"
        
        # Verify chronological order and sanity bounds
        last_arrival = 0.0
        for job in jobs:
            assert job.arrival_time >= last_arrival, f"Out of order in {name}"
            last_arrival = job.arrival_time
            assert job.gpu_count in [1, 2, 4, 8]
            assert job.vram_per_gpu_gb > 0
            assert job.actual_runtime > 0
            assert job.deadline >= job.arrival_time + job.actual_runtime
            assert 1 <= job.priority <= 10
