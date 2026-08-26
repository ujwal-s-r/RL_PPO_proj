"""Script to evaluate all classical baselines across standard benchmark scenarios."""

import argparse
import json
import os
import sys
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from baselines.fifo import FIFOScheduler
from baselines.sjf import SJFScheduler
from baselines.priority import PriorityScheduler
from baselines.best_fit import BestFitScheduler
from evaluation.evaluator import Evaluator
from evaluation.reports import generate_comparison_dataframe, format_markdown_table
from workloads.scenarios import list_scenarios


def main():
    parser = argparse.ArgumentParser(description="Run Classical Baselines Evaluation Benchmark")
    parser.add_argument("--cluster-config", type=str, default="configs/cluster_small.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--output-json", type=str, default="results/baselines_benchmark.json")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    schedulers = [
        FIFOScheduler(),
        SJFScheduler(),
        PriorityScheduler(),
        BestFitScheduler(),
    ]

    scenarios = list_scenarios()
    evaluator = Evaluator(cluster_config_path=args.cluster_config)

    print(f"Evaluating {len(schedulers)} baselines across {len(scenarios)} scenarios with seeds {args.seeds}...")
    
    all_results = {}
    for sched in schedulers:
        print(f"-> Running {sched.name}...")
        res = evaluator.evaluate_scheduler(
            scheduler=sched,
            scenarios=scenarios,
            seeds=args.seeds,
        )
        all_results[sched.name] = res

    # Print comparative tables for each scenario
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON BENCHMARK (Mean Values across seeds)")
    print("=" * 70)

    for sc in scenarios:
        df_sc = generate_comparison_dataframe(all_results, scenario_name=sc)
        print(f"\nScenario: {sc.upper()}")
        print(df_sc.to_string(index=False))

    # Save summary JSON for persistence & notebooks
    serializable = {}
    for pol, sc_map in all_results.items():
        serializable[pol] = {}
        for sc_name, sc_data in sc_map.items():
            serializable[pol][sc_name] = sc_data["summary"]

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    print(f"\nSaved benchmark results to {args.output_json}")


if __name__ == "__main__":
    main()
