"""FastAPI Application serving cluster scheduling REST APIs and Web Dashboard."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
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
from api.inference import ClusterServiceManager
from workloads.scenarios import list_scenarios

app = FastAPI(
    title="GPU Cluster Scheduler Real-Time API",
    description="Inference and cluster control service for PPO & Classical Scheduling Policies",
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

# Global Cluster Service Manager
service = ClusterServiceManager(
    cluster_config="configs/cluster_small.yaml",
    checkpoint_path="checkpoints/ppo_final.pt",
    default_scenario="balanced",
)


@app.api_route("/health", methods=["GET", "HEAD"])
def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "service": "gpu-scheduler-api", "active_policy": service.active_policy_name}


@app.get("/api/policies")
def get_policies() -> Dict[str, List[str]]:
    """Return available scheduling policies."""
    return {"policies": list(service.policies.keys())}


@app.get("/api/scenarios")
def get_scenarios() -> Dict[str, List[str]]:
    """Return available workload scenarios."""
    return {"scenarios": list_scenarios()}


@app.get("/api/cluster/status", response_model=ClusterStatusResponse)
def get_cluster_status() -> ClusterStatusResponse:
    """Retrieve full live snapshot of cluster nodes, GPUs, queued jobs, and metrics."""
    return service.get_status()


@app.post("/api/job/submit")
def submit_job(req: JobSubmissionRequest) -> Dict[str, Any]:
    """Submit an interactive custom job into the live cluster queue."""
    return service.submit_custom_job(req)


@app.post("/api/cluster/reset")
def reset_cluster(req: ClusterResetRequest) -> Dict[str, Any]:
    """Reset the cluster simulation with a specific scenario and seed."""
    service.reset(req.cluster_config, req.scenario, seed=req.seed)
    return {"status": "reset_successful", "scenario": req.scenario, "seed": req.seed}


@app.post("/api/cluster/setup_custom")
def setup_custom_cluster(req: CustomClusterSetupRequest) -> Dict[str, Any]:
    """Setup custom 1-8 node cluster topology built interactively by user."""
    service.reset_dynamic(req.nodes, scenario=req.scenario, seed=req.seed)
    return {"status": "custom_setup_successful", "nodes_count": len(req.nodes), "scenario": req.scenario}


@app.post("/api/simulation/benchmark_step")
def simulation_benchmark_step() -> Dict[str, Any]:
    """Execute simultaneous multi-policy simulation step (PPO + FIFO + SJF + Priority + BestFit)."""
    step_res = service.step_simultaneous()
    cluster_status = service.get_status()
    return {
        **step_res,
        "cluster_status": cluster_status.dict(),
    }


@app.get("/api/simulation/completed")
def get_completed_tasks() -> Dict[str, Any]:
    """Retrieve history of finished tasks."""
    return {"completed_tasks": service.completed_history}


# Mount Frontend static assets
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.api_route("/", methods=["GET", "HEAD"])
    def serve_frontend():
        return FileResponse(os.path.join(frontend_dir, "index.html"))
