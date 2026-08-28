import json

nb_path = "notebooks/06_final_evaluation.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update imports in cell 1
nb["cells"][1]["source"] = [
    "%load_ext autoreload\n",
    "%autoreload 2\n",
    "\n",
    "import sys\n",
    "import os\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Ensure project root is in python path\n",
    "sys.path.insert(0, os.path.abspath('..'))\n",
    "\n",
    "from baselines.fifo import FIFOScheduler\n",
    "from baselines.sjf import SJFScheduler\n",
    "from baselines.sjf_backfill import SJFBackfillScheduler\n",
    "from baselines.priority import PriorityScheduler\n",
    "from baselines.priority_best_fit import PriorityBestFitScheduler\n",
    "from baselines.best_fit import BestFitScheduler\n",
    "from evaluation.ppo_policy import PPOPolicyScheduler\n",
    "from evaluation.evaluator import Evaluator\n",
    "from evaluation.reports import generate_comparison_dataframe, format_markdown_table\n",
    "from workloads.scenarios import list_scenarios\n",
    "\n",
    "print(\"All evaluation modules imported successfully!\")\n"
]

# Update benchmark execution in cell 3
nb["cells"][3]["source"] = [
    "test_seeds = [501, 602, 703]\n",
    "scenarios = [\"balanced\", \"training_heavy\", \"short_job_heavy\", \"bursty\", \"gpu_fragmentation\", \"high_load\"]\n",
    "\n",
    "policies = [\n",
    "    FIFOScheduler(),\n",
    "    SJFScheduler(),\n",
    "    SJFBackfillScheduler(),\n",
    "    PriorityScheduler(),\n",
    "    PriorityBestFitScheduler(),\n",
    "    BestFitScheduler(),\n",
    "    PPOPolicyScheduler(checkpoint_path=\"../checkpoints/ppo_final.pt\", cluster_config_path=\"../configs/cluster_small.yaml\"),\n",
    "]\n",
    "\n",
    "evaluator = Evaluator(cluster_config_path=\"../configs/cluster_small.yaml\", horizon_seconds=3600.0)\n",
    "\n",
    "all_results = {}\n",
    "for pol in policies:\n",
    "    print(f\"Evaluating [{pol.name}] on all {len(scenarios)} scenarios...\")\n",
    "    all_results[pol.name] = evaluator.evaluate_scheduler(pol, scenarios=scenarios, seeds=test_seeds)\n",
    "\n",
    "print(\"\\nHead-to-head benchmark complete!\")\n"
]

# Update comparative benchmark tables in cell 5
nb["cells"][5]["source"] = [
    "# 1. Display Scenario Breakdown Tables\n",
    "for sc in scenarios:\n",
    "    df_sc = generate_comparison_dataframe(all_results, scenario_name=sc)\n",
    "    print(f\"\\n{'='*75}\")\n",
    "    print(f\"Scenario: {sc.upper()}\")\n",
    "    print(f\"{'='*75}\")\n",
    "    print(df_sc.to_string(index=False))\n",
    "\n",
    "# 2. Compute and Display Paired Head-to-Head Statistical Summary Matrix\n",
    "print(f\"\\n\\n{'='*95}\")\n",
    "print(\"🏆 PAIRED HEAD-TO-HEAD WIN RATE & IMPACT MATRIX ACROSS ALL 18 TEST TRACES\")\n",
    "print(f\"{'='*95}\")\n",
    "\n",
    "baselines_list = [\"SJF_Backfill\", \"Priority_BestFit\", \"SJF\", \"Priority\", \"BestFit\", \"FIFO\"]\n",
    "paired_summary = []\n",
    "\n",
    "for b_name in baselines_list:\n",
    "    jct_wins = 0\n",
    "    p95_wins = 0\n",
    "    sla_wins = 0\n",
    "    total_runs = 0\n",
    "    jct_diffs = []\n",
    "    p95_diffs = []\n",
    "    sla_diffs = []\n",
    "\n",
    "    for sc in scenarios:\n",
    "        for seed_idx, seed in enumerate(test_seeds):\n",
    "            total_runs += 1\n",
    "            ppo_run = all_results[\"PPO\"][sc][\"raw_episodes\"][seed_idx]\n",
    "            base_run = all_results[b_name][sc][\"raw_episodes\"][seed_idx]\n",
    "\n",
    "            # JCT comparison (lower is better)\n",
    "            if ppo_run.mean_jct <= base_run.mean_jct:\n",
    "                jct_wins += 1\n",
    "            jct_diffs.append(base_run.mean_jct - ppo_run.mean_jct)\n",
    "\n",
    "            # P95 JCT comparison (lower is better)\n",
    "            if ppo_run.p95_jct <= base_run.p95_jct:\n",
    "                p95_wins += 1\n",
    "            p95_diffs.append(base_run.p95_jct - ppo_run.p95_jct)\n",
    "\n",
    "            # SLA violation rate comparison (lower is better)\n",
    "            if ppo_run.deadline_violation_rate <= base_run.deadline_violation_rate:\n",
    "                sla_wins += 1\n",
    "            sla_diffs.append((base_run.deadline_violation_rate - ppo_run.deadline_violation_rate) * 100.0)\n",
    "\n",
    "    paired_summary.append({\n",
    "        \"Baseline\": b_name,\n",
    "        \"JCT Win Rate (%)\": f\"{(jct_wins / total_runs) * 100.0:.1f}%\",\n",
    "        \"Mean JCT Reduction (s)\": f\"{np.mean(jct_diffs):+.1f}s\",\n",
    "        \"P95 Win Rate (%)\": f\"{(p95_wins / total_runs) * 100.0:.1f}%\",\n",
    "        \"Mean P95 Reduction (s)\": f\"{np.mean(p95_diffs):+.1f}s\",\n",
    "        \"SLA Win Rate (%)\": f\"{(sla_wins / total_runs) * 100.0:.1f}%\",\n",
    "        \"Mean SLA Miss Reduction (%)\": f\"{np.mean(sla_diffs):+.2f}%\",\n",
    "    })\n",
    "\n",
    "df_paired = pd.DataFrame(paired_summary)\n",
    "print(df_paired.to_string(index=False))\n"
]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Updated 06_final_evaluation.ipynb successfully!")
