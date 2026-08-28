import os
import sys

# Ensure project root is in python path
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "OneDrive", "Documents", "Github", "RL_PPO_proj", "GPU_scheduling_sim"))
if os.path.exists(proj_root):
    sys.path.insert(0, proj_root)
else:
    sys.path.insert(0, os.getcwd())

from evaluation.evaluator import Evaluator
from evaluation.ppo_policy import PPOPolicyScheduler
from baselines.sjf_backfill import SJFBackfillScheduler
from baselines.sjf import SJFScheduler
from baselines.fifo import FIFOScheduler
from baselines.priority import PriorityScheduler

evaluator = Evaluator(cluster_config_path="configs/cluster_small.yaml", horizon_seconds=3600.0)
ppo = PPOPolicyScheduler(checkpoint_path="checkpoints/ppo_best.pt", cluster_config_path="configs/cluster_small.yaml")
sjf_bf = SJFBackfillScheduler()
sjf = SJFScheduler()
fifo = FIFOScheduler()
priority = PriorityScheduler()

policies = [ppo, sjf_bf, sjf, priority, fifo]
scenarios = ["balanced", "training_heavy", "bursty", "gpu_fragmentation"]
test_seeds = [501, 602]

from evaluation.reports import generate_comparison_dataframe

all_results = {}
for p in policies:
    print(f"Evaluating [{p.name}]...")
    all_results[p.name] = evaluator.evaluate_scheduler(p, scenarios=scenarios, seeds=test_seeds)

for sc in scenarios:
    print("\n" + "=" * 65)
    print(f"Scenario: {sc.upper()}")
    print("=" * 65)
    df = generate_comparison_dataframe(all_results, scenario_name=sc)
    print(df.to_string(index=False))

print("\n" + "=" * 65)
print("EVALUATION CHECK COMPLETE!")
