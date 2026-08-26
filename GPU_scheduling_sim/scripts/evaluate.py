"""Final comprehensive benchmark evaluation script (PPO vs FIFO, SJF, Priority, Best-Fit)."""

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
from evaluation.ppo_policy import PPOPolicyScheduler
from evaluation.evaluator import Evaluator
from evaluation.reports import generate_comparison_dataframe, format_markdown_table
from workloads.scenarios import list_scenarios


def main():
    parser = argparse.ArgumentParser(description="Final Comprehensive Benchmark Evaluation")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ppo_final.pt")
    parser.add_argument("--cluster-config", type=str, default="configs/cluster_small.yaml")
    parser.add_argument("--test-seeds", type=int, nargs="+", default=[501, 602, 703])
    parser.add_argument("--output-json", type=str, default="results/final_evaluation_results.json")
    parser.add_argument("--horizon", type=float, default=3600.0)
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    # Initialize all 5 policies (4 classical heuristics + trained PPO)
    policies = [
        FIFOScheduler(),
        SJFScheduler(),
        PriorityScheduler(),
        BestFitScheduler(),
        PPOPolicyScheduler(checkpoint_path=args.checkpoint, cluster_config_path=args.cluster_config),
    ]

    scenarios = ["balanced", "training_heavy", "short_job_heavy", "bursty", "gpu_fragmentation", "high_load"]
    evaluator = Evaluator(cluster_config_path=args.cluster_config, horizon_seconds=args.horizon)

    print(f"\n{'='*75}")
    print(f"RUNNING FINAL HEAD-TO-HEAD BENCHMARK EVALUATION")
    print(f"Policies: {[p.name for p in policies]}")
    print(f"Scenarios: {scenarios}")
    print(f"Unseen Test Seeds: {args.test_seeds}")
    print(f"{'='*75}\n")

    all_results = {}
    for pol in policies:
        print(f"-> Evaluating [{pol.name}] across all {len(scenarios)} scenarios...")
        res = evaluator.evaluate_scheduler(
            scheduler=pol,
            scenarios=scenarios,
            seeds=args.test_seeds,
        )
        all_results[pol.name] = res

    # Print comparative tables for each scenario
    print("\n" + "=" * 75)
    print("FINAL HEAD-TO-HEAD BENCHMARK RESULTS (Mean Values Across Test Seeds)")
    print("=" * 75)

    markdown_reports = []
    for sc in scenarios:
        df_sc = generate_comparison_dataframe(all_results, scenario_name=sc)
        print(f"\nScenario: {sc.upper()}")
        print(df_sc.to_string(index=False))
        markdown_reports.append(format_markdown_table(df_sc, title=f"Scenario: {sc.capitalize()}"))

    # Save summary JSON for persistence & notebooks
    serializable = {}
    for pol_name, sc_map in all_results.items():
        serializable[pol_name] = {}
        for sc_name, sc_data in sc_map.items():
            serializable[pol_name][sc_name] = sc_data["summary"]

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    # Save markdown summary report
    with open("results/final_evaluation_report.md", "w", encoding="utf-8") as f:
        f.write("# Final Evaluation Benchmark: PPO vs Classical Schedulers\n\n")
        f.write("\n\n".join(markdown_reports))

    print(f"\n[SUCCESS] Saved final evaluation results to '{args.output_json}'")
    print(f"[SUCCESS] Saved markdown report to 'results/final_evaluation_report.md'")


if __name__ == "__main__":
    main()
