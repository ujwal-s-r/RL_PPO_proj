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

print("================================================================================")
print(f"[BENCHMARK] STARTING OFFICIAL BENCHMARK EVALUATION (SEEDS: {test_seeds})")
print("================================================================================\n")

all_results = {}
for p in policies:
    print(f"-> Evaluating [{p.name}] on all {len(scenarios)} scenarios...")
    all_results[p.name] = evaluator.evaluate_scheduler(p, scenarios=scenarios, seeds=test_seeds)

all_tables = []
for sc in scenarios:
    title = f"Scenario: {sc.upper()}"
    df = generate_comparison_dataframe(all_results, scenario_name=sc)
    md = format_markdown_table(df, title=title)
    all_tables.append(md)
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(df.to_string(index=False))

# Compute Paired Win Rates for PPO vs each baseline across all runs
print("\n" + "=" * 80)
print("[PAIRED] HEAD-TO-HEAD STATISTICAL SUMMARY ACROSS ALL 18 TEST TRACES")
print("=" * 80)

baselines_list = ["SJF_Backfill", "Priority_BestFit", "SJF", "Priority", "BestFit", "FIFO"]
paired_summary = []

for b_name in baselines_list:
    jct_wins = 0
    sla_wins = 0
    p95_wins = 0
    total_runs = 0
    jct_diffs = []
    sla_diffs = []
    p95_diffs = []

    for sc in scenarios:
        for seed_idx, seed in enumerate(test_seeds):
            total_runs += 1
            ppo_run = all_results["PPO"][sc]["raw_episodes"][seed_idx]
            base_run = all_results[b_name][sc]["raw_episodes"][seed_idx]

            # JCT comparison (lower is better)
            if ppo_run.mean_jct <= base_run.mean_jct:
                jct_wins += 1
            jct_diffs.append(base_run.mean_jct - ppo_run.mean_jct)

            # P95 JCT comparison (lower is better)
            if ppo_run.p95_jct <= base_run.p95_jct:
                p95_wins += 1
            p95_diffs.append(base_run.p95_jct - ppo_run.p95_jct)

            # SLA deadline violation rate comparison (lower is better)
            if ppo_run.deadline_violation_rate <= base_run.deadline_violation_rate:
                sla_wins += 1
            sla_diffs.append((base_run.deadline_violation_rate - ppo_run.deadline_violation_rate) * 100.0)

    paired_summary.append({
        "Baseline": b_name,
        "JCT Win Rate (%)": f"{(jct_wins / total_runs) * 100.0:.1f}%",
        "Mean JCT Reduction (s)": f"{np.mean(jct_diffs):+.1f}s",
        "P95 Win Rate (%)": f"{(p95_wins / total_runs) * 100.0:.1f}%",
        "Mean P95 Reduction (s)": f"{np.mean(p95_diffs):+.1f}s",
        "SLA Win Rate (%)": f"{(sla_wins / total_runs) * 100.0:.1f}%",
        "Mean SLA Miss Reduction (%)": f"{np.mean(sla_diffs):+.2f}%",
    })

df_paired = pd.DataFrame(paired_summary)
print(df_paired.to_string(index=False))

# Save complete evaluation report
report_path = "evaluation/final_benchmark_results.md"
os.makedirs("evaluation", exist_ok=True)
with open(report_path, "w", encoding="utf-8") as f:
    f.write("# Final PPO Benchmark Evaluation Report\n\n")
    f.write(f"Evaluated on {len(scenarios)} workloads across unseen test seeds {test_seeds}.\n\n")
    f.write("## Paired Head-to-Head Win Rate Summary\n\n")
    f.write(format_markdown_table(df_paired) + "\n\n---\n\n")
    f.write("\n\n---\n\n".join(all_tables))

print(f"\nSaved complete markdown benchmark report to '{report_path}'")
print("\n[SUCCESS] ALL OFFICIAL BENCHMARK TESTS COMPLETED SUCCESSFULLY!")
