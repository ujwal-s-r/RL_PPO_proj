"""FastAPI Application serving cluster scheduling REST APIs and Web Dashboard."""

from __future__ import annotations
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.schemas import (
    ClusterStatusResponse,
    JobSubmissionRequest,
    ScheduleStepRequest,
    ClusterResetRequest,
    CustomClusterSetupRequest,
)
from workloads.scenarios import list_scenarios

if TYPE_CHECKING:
    from api.inference import ClusterServiceManager


SCENARIO_DESCRIPTIONS = {
    "balanced": "A representative production mix of training, tuning, evaluation, and inference work.",
    "training_heavy": "Long-running, multi-GPU training jobs competing for large VRAM allocations.",
    "short_job_heavy": "High-volume inference and evaluation tasks with short execution times.",
    "bursty": "Idle periods interrupted by concentrated arrival spikes that stress queue decisions.",
    "gpu_fragmentation": "Mixed GPU and VRAM requests designed to expose inefficient resource packing.",
    "high_load": "Sustained demand near cluster capacity, emphasizing SLA protection under pressure.",
}

app = FastAPI(
    title="GPU Cluster Scheduler Real-Time API",
    description="Inference and cluster control service for PPO scheduling.",
    version="0.1.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@lru_cache(maxsize=1)
def get_service() -> "ClusterServiceManager":
    """Create the expensive PPO runtime only when a live API endpoint needs it."""
    from api.inference import ClusterServiceManager

    return ClusterServiceManager(
        cluster_config="configs/cluster_small.yaml",
        checkpoint_path="checkpoints/ppo_final.onnx",
        default_scenario="balanced",
    )


@app.api_route("/health", methods=["GET", "HEAD"])
def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "service": "gpu-scheduler-api", "active_policy": "PPO"}


@app.get("/api/policies")
def get_policies() -> Dict[str, List[str]]:
    """Return available scheduling policies."""
    return {"policies": ["PPO"]}


@app.get("/api/scenarios")
def get_scenarios() -> Dict[str, List[str]]:
    """Return available workload scenarios."""
    return {"scenarios": list_scenarios()}


@app.get("/api/benchmarks")
def get_benchmarks() -> Dict[str, Any]:
    """Return immutable held-out benchmark results for the comparison dashboard."""
    result_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evaluation", "final_benchmark_results.md"))
    with open(result_path, "r", encoding="utf-8") as result_file:
        report_lines = result_file.readlines()

    results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    current_scenario: Optional[str] = None
    policies: List[str] = []
    metric_map = {
        "Mean JCT (s)": "mean_jct",
        "Deadline Violation (%)": "deadline_violation_rate",
    }

    for line in report_lines:
        stripped = line.strip()
        if stripped.startswith("### Scenario: "):
            current_scenario = stripped.split(": ", 1)[1].lower()
            results[current_scenario] = {}
            policies = []
            continue
        if current_scenario is None or not stripped.startswith("|"):
            continue

        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if cells and cells[0] == "Metric":
            policies = cells[1:]
            continue
        if not policies or not cells or cells[0] not in metric_map or set(cells[0]) == {"-"}:
            continue

        metric_key = metric_map[cells[0]]
        for policy, raw_value in zip(policies, cells[1:]):
            value = float(raw_value.rstrip("%").strip())
            if metric_key == "deadline_violation_rate":
                value /= 100.0
            results[current_scenario].setdefault(policy, {})[metric_key] = {"mean": value}

    # The report is the source of truth because it includes SJF Backfill and Priority Best-Fit.
    results_by_policy: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for scenario, policy_data in results.items():
        for policy, metrics in policy_data.items():
            results_by_policy.setdefault(policy, {})[scenario] = metrics

    return {
        "evaluation_seeds": [501, 602, 703],
        "descriptions": SCENARIO_DESCRIPTIONS,
        "results": results_by_policy,
    }


@app.get("/api/cluster/status", response_model=ClusterStatusResponse)
def get_cluster_status() -> ClusterStatusResponse:
    """Retrieve full live snapshot of cluster nodes, GPUs, queued jobs, and metrics."""
    return get_service().get_status()


@app.post("/api/job/submit")
def submit_job(req: JobSubmissionRequest) -> Dict[str, Any]:
    """Submit an interactive custom job into the live cluster queue."""
    return get_service().submit_custom_job(req)


@app.post("/api/cluster/reset")
def reset_cluster(req: ClusterResetRequest) -> Dict[str, Any]:
    """Reset the cluster simulation with a specific scenario and seed."""
    get_service().reset(req.cluster_config, req.scenario, seed=req.seed)
    return {"status": "reset_successful", "scenario": req.scenario, "seed": req.seed}


@app.post("/api/cluster/setup_custom")
def setup_custom_cluster(req: CustomClusterSetupRequest) -> Dict[str, Any]:
    """Setup custom 1-8 node cluster topology built interactively by user."""
    get_service().reset_dynamic(req.nodes, scenario=req.scenario, seed=req.seed)
    return {"status": "custom_setup_successful", "nodes_count": len(req.nodes), "scenario": req.scenario}


@app.post("/api/simulation/step")
def simulation_step() -> Dict[str, Any]:
    """Execute one PPO scheduling step in the live cluster."""
    service = get_service()
    step_res = service.step_ppo()
    cluster_status = service.get_status()
    return {
        **step_res,
        "cluster_status": cluster_status.dict(),
    }


@app.post("/api/simulation/fast_forward")
def simulation_fast_forward() -> Dict[str, Any]:
    """Fast forward simulation to complete all workload jobs immediately."""
    service = get_service()
    step_res = {}
    for _ in range(500):
        step_res = service.step_ppo()
        if step_res.get("is_done", False):
            break
    cluster_status = service.get_status()
    return {
        **step_res,
        "cluster_status": cluster_status.dict(),
    }


@app.get("/api/simulation/completed")
def get_completed_tasks() -> Dict[str, Any]:
    """Retrieve history of finished tasks."""
    return {"completed_tasks": get_service().completed_history}


# Mount Frontend static assets
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.api_route("/", methods=["GET", "HEAD"])
    def serve_frontend():
        return FileResponse(os.path.join(frontend_dir, "index.html"))
