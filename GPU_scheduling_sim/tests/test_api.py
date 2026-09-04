"""Unit tests for FastAPI Inference & Dashboard REST API."""

from fastapi.testclient import TestClient
from api.app import app


def test_api_health(client=None):
    c = client or TestClient(app)
    res = c.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "active_policy" in data


def test_api_policies_and_scenarios(client=None):
    c = client or TestClient(app)
    res_pol = c.get("/api/policies")
    assert res_pol.status_code == 200
    assert res_pol.json()["policies"] == ["PPO"]

    res_sc = c.get("/api/scenarios")
    assert res_sc.status_code == 200
    assert "balanced" in res_sc.json()["scenarios"]
    assert "bursty" in res_sc.json()["scenarios"]


def test_api_cluster_status(client=None):
    c = client or TestClient(app)
    res = c.get("/api/cluster/status")
    assert res.status_code == 200
    data = res.json()
    assert "simulation_time" in data
    assert "nodes" in data
    assert len(data["nodes"]) > 0
    assert "queue" in data
    assert "metrics" in data


def test_api_job_submission(client=None):
    c = client or TestClient(app)
    payload = {
        "gpu_count": 2,
        "vram_per_gpu_gb": 24.0,
        "estimated_runtime": 60.0,
        "priority": 8,
        "workload_type": "training",
        "deadline_slack": 2.0,
    }
    res = c.post("/api/job/submit", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ["queued", "queue_full"]
    assert "job_id" in data


def test_api_schedule_step_ppo(client=None):
    c = client or TestClient(app)
    step_payload = {
        "policy": "PPO",
        "auto_advance": True,
    }
    res = c.post("/api/simulation/step", json=step_payload)
    assert res.status_code == 200
    data = res.json()
    assert "policy" in data
    assert data["policy"] == "PPO"
    assert "sim_time" in data


def test_api_benchmarks(client=None):
    c = client or TestClient(app)
    res = c.get("/api/benchmarks")
    assert res.status_code == 200
    data = res.json()
    assert data["evaluation_seeds"] == [501, 602, 703]
    assert "balanced" in data["descriptions"]
    assert "PPO" in data["results"]
    assert "SJF_Backfill" in data["results"]


def test_api_cluster_reset(client=None):
    c = client or TestClient(app)
    reset_payload = {
        "cluster_config": "configs/cluster_small.yaml",
        "scenario": "bursty",
        "seed": 999,
    }
    res = c.post("/api/cluster/reset", json=reset_payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "reset_successful"
    assert data["scenario"] == "bursty"
