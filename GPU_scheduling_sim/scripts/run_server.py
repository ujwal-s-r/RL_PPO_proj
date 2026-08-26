"""Launcher script for FastAPI GPU Cluster Scheduler Service and Real-Time Web Dashboard."""

import argparse
import os
import sys
import uvicorn

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))


def main():
    parser = argparse.ArgumentParser(description="Start GPU Cluster Scheduler Dashboard & API")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"[STARTING] GPU CLUSTER SCHEDULER SERVER")
    print(f"[WEB UI]   Dashboard : http://localhost:{args.port}")
    print(f"[API DOCS] Swagger UI: http://localhost:{args.port}/docs")
    print(f"[DEVICE]   Target    : CUDA / RTX 3050")
    print(f"{'='*70}\n")

    uvicorn.run("api.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
