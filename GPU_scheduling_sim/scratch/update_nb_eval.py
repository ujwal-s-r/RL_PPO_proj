import json
import os

nb_path = "notebooks/06_final_evaluation.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

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
    "from baselines.best_fit import BestFitScheduler\n",
    "from evaluation.ppo_policy import PPOPolicyScheduler\n",
    "from evaluation.evaluator import Evaluator\n",
    "from evaluation.reports import generate_comparison_dataframe\n",
    "from workloads.scenarios import list_scenarios\n",
    "\n",
    "print(\"All evaluation modules imported successfully!\")\n"
]

nb["cells"][3]["source"] = [
    "test_seeds = [501, 602, 703]\n",
    "scenarios = [\"balanced\", \"training_heavy\", \"short_job_heavy\", \"bursty\", \"gpu_fragmentation\", \"high_load\"]\n",
    "\n",
    "policies = [\n",
    "    FIFOScheduler(),\n",
    "    SJFScheduler(),\n",
    "    SJFBackfillScheduler(),\n",
    "    PriorityScheduler(),\n",
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

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Updated 06_final_evaluation.ipynb successfully!")
