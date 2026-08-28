import os
import sys
import pandas as pd
import numpy as np

# Ensure project root is in python path
sys.path.insert(0, os.getcwd())

from evaluation.evaluator import Evaluator
from evaluation.ppo_policy import PPOPolicyScheduler
from baselines.sjf_backfill import SJFBackfillScheduler
from baselines.sjf import SJFScheduler
from baselines.fifo import FIFOScheduler
from baselines.priority import PriorityScheduler
from baselines.priority_best_fit import PriorityBestFitScheduler
from baselines.best_fit import BestFitScheduler
from evaluation.reports import generate_comparison_dataframe, format_markdown_table

evaluator = Evaluator(cluster_config_path="configs/cluster_small.yaml", horizon_seconds=3600.0)

ppo_final = PPOPolicyScheduler(checkpoint_path="checkpoints/ppo_final.pt", cluster_config_path="configs/cluster_small.yaml")
sjf_bf = SJFBackfillScheduler()
p_bf = PriorityBestFitScheduler()
sjf = SJFScheduler()
priority = PriorityScheduler()
best_fit = BestFitScheduler()
fifo = FIFOScheduler()

policies = [ppo_final, sjf_bf, p_bf, sjf, priority, best_fit, fifo]
scenarios = ["balanced", "training_heavy", "short_job_heavy", "bursty", "gpu_fragmentation", "high_load"]
test_seeds = [501, 602, 703]

all_results = {}
for p in policies:
    all_results[p.name] = evaluator.evaluate_scheduler(p, scenarios=scenarios, seeds=test_seeds)

all_tables = []
for sc in scenarios:
    title = f"Scenario: {sc.upper()}"
    df = generate_comparison_dataframe(all_results, scenario_name=sc)
    md = format_markdown_table(df, title=title)
    all_tables.append(md)

# Save complete evaluation report
report_path = "evaluation/final_benchmark_results.md"
os.makedirs("evaluation", exist_ok=True)
with open(report_path, "w", encoding="utf-8") as f:
    f.write("# Final PPO Benchmark Evaluation Report\n\n")
    f.write(f"Evaluated on {len(scenarios)} workloads across unseen test seeds {test_seeds}.\n\n")
    f.write("\n\n---\n\n".join(all_tables))

print(f"Saved complete markdown benchmark report to '{report_path}'")
