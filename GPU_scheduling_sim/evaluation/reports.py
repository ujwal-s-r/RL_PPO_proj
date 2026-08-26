"""Report generator producing clean markdown and tabular comparison summaries."""

from __future__ import annotations
from typing import Any, Dict, List
import pandas as pd


def generate_comparison_dataframe(
    all_results: Dict[str, Dict[str, Dict[str, Any]]],
    scenario_name: str = "balanced",
    metric_key: str = "mean",
) -> pd.DataFrame:
    """
    Generate comparative pandas DataFrame for a single scenario across all policies.
    
    Args:
        all_results: Dict mapping policy_name -> {scenario_name -> {"summary": ...}}
        scenario_name: Target scenario to inspect.
        metric_key: Statistical reduction ('mean', 'p50', 'p95', 'std').
    """
    metrics_to_show = [
        ("mean_jct", "Mean JCT (s)"),
        ("p95_jct", "P95 JCT (s)"),
        ("mean_wait_time", "Mean Queue Wait (s)"),
        ("gpu_utilization", "GPU Utilization (%)"),
        ("throughput_jobs_per_hour", "Throughput (jobs/hr)"),
        ("deadline_violation_rate", "Deadline Violation (%)"),
        ("completed_jobs", "Completed Jobs"),
        ("cumulative_reward", "Cumulative Reward"),
    ]

    policies = list(all_results.keys())
    data: Dict[str, List[Any]] = {"Metric": [label for _, label in metrics_to_show]}

    for pol in policies:
        col_vals = []
        sc_res = all_results.get(pol, {}).get(scenario_name, {}).get("summary", {})
        for metric_id, _ in metrics_to_show:
            val = sc_res.get(metric_id, {}).get(metric_key, 0.0)
            if metric_id in ["gpu_utilization", "deadline_violation_rate"]:
                col_vals.append(f"{val * 100.0:.2f}%")
            elif metric_id == "completed_jobs":
                col_vals.append(f"{int(val)}")
            else:
                col_vals.append(f"{val:.2f}")
        data[pol] = col_vals

    return pd.DataFrame(data)


def format_markdown_table(df: pd.DataFrame, title: str = "") -> str:
    """Format DataFrame as GitHub Markdown table."""
    md_lines = []
    if title:
        md_lines.append(f"### {title}\n")
    md_lines.append(df.to_markdown(index=False))
    return "\n".join(md_lines)
