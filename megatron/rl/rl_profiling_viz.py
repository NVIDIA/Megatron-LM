# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
RL Profiling Visualization

Generates charts and HTML reports from profiling data.

Usage:
    # Generate HTML report for a single run (outputs to folder)
    python -m megatron.rl.rl_profiling_viz report /path/to/profiles/ --output my_report

    # Compare multiple runs with charts
    python -m megatron.rl.rl_profiling_viz compare run1/profiles run2/profiles --names baseline optimized

    # Generate charts only (PNG files)
    python -m megatron.rl.rl_profiling_viz charts /path/to/profiles/ --output ./charts/
"""

import argparse
import json
import os
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Try to import visualization libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# Phase definitions for high-level breakdown
# Note: These are mutually exclusive leaf timers to avoid double-counting
# Phases marked with "(R0)" are rank-0 only operations - other ranks are idle/waiting
PHASE_DEFINITIONS = {
    "Rollout Gen (R0)": ["rl/collect-rollouts"],       # Rank-0 only: vLLM inference
    "Engine Suspend (R0)": ["rl/suspend-engine"],      # Rank-0 only: shutting down vLLM engine
    "Optimizer Offload": ["rl/offload-optimizer-before-inference"],
    "Optimizer Onload": ["rl/onload-optimizer-after-inference"],
    "Old Logprobs": ["rl/compute-old-logprobs"],
    "Ref Logprobs": ["rl/compute-ref-logprobs"],
    "Training": ["forward-backward"],
    "Sync Rollouts": ["rl/sync-rollouts"],             # Broadcast rollouts from rank-0 to all ranks
    "Data Prep": ["rl/prepare-trajectories", "rl/compute-group-stats", "rl/get-ltor-masks"],
}

# Colors for phases (using a colorblind-friendly palette)
PHASE_COLORS = {
    "Rollout Gen (R0)": "#2ecc71",    # Green
    "Engine Suspend (R0)": "#27ae60", # Dark Green (rank-0 related)
    "Optimizer Offload": "#e74c3c",   # Red
    "Optimizer Onload": "#c0392b",    # Dark Red
    "Old Logprobs": "#3498db",        # Blue
    "Ref Logprobs": "#2980b9",        # Dark Blue
    "Training": "#9b59b6",            # Purple
    "Sync Rollouts": "#f39c12",       # Orange
    "Data Prep": "#1abc9c",           # Teal
    "Other": "#95a5a6",               # Gray
}

# Container timers that should be excluded from Timer Statistics table
# because they contain nested timers and would double-count time
CONTAINER_TIMERS = [
    "rl/rollout-collection",      # Contains: collect-rollouts, suspend-engine, offload/onload, etc.
    "rl/prepare-data-for-update", # Contains: compute-group-stats, prepare-trajectories, logprobs, etc.
    "rl/compute-logprobs",        # Contains: compute-old-logprobs, compute-ref-logprobs
    "rl/get-logprobs",            # Contains: forward-pass, log-softmax
    "rl/inference-setup",         # Setup container
]


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load profile data from JSONL file."""
    profiles = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                profiles.append(json.loads(line))
    return profiles


def load_csv_summary(path: str) -> Dict[str, Dict[str, float]]:
    """Load summary CSV into dict."""
    import csv
    stats = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timer_name = row["timer_name"]
            stats[timer_name] = {k: float(v) if k != "timer_name" else v 
                                 for k, v in row.items() if k != "timer_name"}
    return stats


def find_profile_files(directory: str) -> Tuple[Optional[str], Optional[str]]:
    """Find JSONL and CSV files in a profile directory."""
    directory = Path(directory)
    
    jsonl_files = list(directory.glob("profile_*.jsonl"))
    csv_files = list(directory.glob("summary_*.csv"))
    
    jsonl_path = str(sorted(jsonl_files)[-1]) if jsonl_files else None
    csv_path = str(sorted(csv_files)[-1]) if csv_files else None
    
    return jsonl_path, csv_path


def filter_outliers(profiles: List[Dict[str, Any]], warmup_iterations: int = 3) -> List[Dict[str, Any]]:
    """Filter out warm-up iterations which typically have different performance patterns."""
    if len(profiles) <= warmup_iterations:
        return profiles
    return profiles[warmup_iterations:]


def compute_phase_times(profile: Dict[str, Any]) -> Dict[str, float]:
    """Extract phase times from a profile record."""
    phases = {}
    
    for phase_name, timer_names in PHASE_DEFINITIONS.items():
        total = 0.0
        for timer in timer_names:
            key = f"timer_{timer.replace('/', '_').replace('-', '_')}_max_ms"
            total += profile.get(key, 0.0)
        phases[phase_name] = total
    
    known_total = sum(phases.values())
    iteration_time = profile.get("elapsed_time_ms", 0)
    phases["Other"] = max(0, iteration_time - known_total)
    
    return phases


def extract_timer_stats_from_profiles(profiles: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Compute timer statistics directly from JSONL profiles."""
    timer_values = defaultdict(list)
    
    for profile in profiles:
        for key, value in profile.items():
            if key.startswith("timer_") and key.endswith("_max_ms"):
                timer_name = key.replace("timer_", "").replace("_max_ms", "")
                timer_name = timer_name.replace("_", "-")
                if timer_name.startswith("rl-"):
                    timer_name = "rl/" + timer_name[3:]
                timer_values[timer_name].append(value)
    
    stats = {}
    for timer_name, values in timer_values.items():
        if not values or all(v == 0 for v in values):
            continue
        stats[timer_name] = {
            "mean_ms": statistics.mean(values),
            "min_ms": min(values),
            "max_ms": max(values),
        }
    
    return stats


def generate_iteration_timeline_chart(
    profiles: List[Dict[str, Any]], 
    output_path: str,
    title: str = "Iteration Timeline"
):
    """Generate a stacked area chart showing phase breakdown over iterations."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping chart generation")
        return
    
    iterations = [p["iteration"] for p in profiles]
    phase_data = defaultdict(list)
    
    for p in profiles:
        phases = compute_phase_times(p)
        for phase, time in phases.items():
            phase_data[phase].append(time)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    bottom = [0] * len(iterations)
    for phase in PHASE_DEFINITIONS.keys():
        if phase in phase_data:
            values = phase_data[phase]
            color = PHASE_COLORS.get(phase, "#95a5a6")
            ax.bar(iterations, values, bottom=bottom, label=phase, color=color, width=0.8)
            bottom = [b + v for b, v in zip(bottom, values)]
    
    if "Other" in phase_data:
        ax.bar(iterations, phase_data["Other"], bottom=bottom, 
               label="Other", color=PHASE_COLORS["Other"], width=0.8)
    
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def generate_phase_pie_chart(
    profiles: List[Dict[str, Any]], 
    output_path: str,
    title: str = "Average Time Breakdown",
    exclude_warmup: bool = True
):
    """Generate a pie chart showing average phase distribution."""
    if not HAS_MATPLOTLIB:
        return
    
    filtered_profiles = filter_outliers(profiles) if exclude_warmup else profiles
    
    phase_totals = defaultdict(float)
    for p in filtered_profiles:
        phases = compute_phase_times(p)
        for phase, time in phases.items():
            phase_totals[phase] += time
    
    n = len(filtered_profiles)
    phase_avgs = {k: v / n for k, v in phase_totals.items() if v > 0}
    
    sorted_phases = sorted(phase_avgs.items(), key=lambda x: x[1], reverse=True)
    labels = [p[0] for p in sorted_phases]
    sizes = [p[1] for p in sorted_phases]
    colors = [PHASE_COLORS.get(l, "#95a5a6") for l in labels]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct=lambda pct: f'{pct:.1f}%' if pct > 3 else '',
        startangle=90, counterclock=False
    )
    
    legend_labels = [f"{l}: {s/1000:.1f}s" for l, s in zip(labels, sizes)]
    ax.legend(wedges, legend_labels, loc='upper left', bbox_to_anchor=(1.05, 1))
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def generate_timer_comparison_chart(
    summaries: List[Dict[str, Dict[str, float]]],
    run_names: List[str],
    output_path: str,
    top_n: int = 12
):
    """Generate a grouped bar chart comparing timer means across runs."""
    if not HAS_MATPLOTLIB:
        return
    
    all_timers = set()
    for summary in summaries:
        all_timers.update(summary.keys())
    
    timer_max_means = {}
    for timer in all_timers:
        max_mean = max(s.get(timer, {}).get("mean_ms", 0) for s in summaries)
        timer_max_means[timer] = max_mean
    
    top_timers = sorted(timer_max_means.items(), key=lambda x: x[1], reverse=True)[:top_n]
    timer_names = [t[0] for t in top_timers]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = range(len(timer_names))
    width = 0.8 / len(summaries)
    
    colors = plt.cm.Set2(range(len(summaries)))
    
    for i, (summary, name) in enumerate(zip(summaries, run_names)):
        means = [summary.get(t, {}).get("mean_ms", 0) for t in timer_names]
        offset = (i - len(summaries) / 2 + 0.5) * width
        bars = ax.bar([xi + offset for xi in x], means, width, label=name, color=colors[i])
    
    ax.set_xlabel("Timer", fontsize=12)
    ax.set_ylabel("Mean Time (ms)", fontsize=12)
    ax.set_title("Timer Comparison Across Runs", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("rl/", "") for t in timer_names], rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def generate_load_imbalance_chart(
    profiles: List[Dict[str, Any]],
    output_path: str,
    title: str = "Load Imbalance Over Iterations"
):
    """Generate chart showing load imbalance (max/min ratio) over time."""
    if not HAS_MATPLOTLIB:
        return
    
    # Exclude rank0-only operations like rl/collect-rollouts
    # Timers for load imbalance analysis - only include distributed operations
    # Excluded:
    #   - rl/collect-rollouts: rank-0 only (vLLM inference)
    #   - rl/suspend-engine: rank-0 only (engine management)
    #   - rl/sync-rollouts: synchronization barrier (imbalance is expected)
    key_timers = [
        "forward-backward",
        "optimizer",
        "rl/compute-old-logprobs",
        "rl/compute-ref-logprobs",
        "rl/train-forward",
    ]
    
    iterations = [p["iteration"] for p in profiles]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.tab10(range(len(key_timers)))
    
    for timer, color in zip(key_timers, colors):
        imbalance_key = f"imbalance_{timer.replace('/', '_').replace('-', '_')}"
        values = [p.get(imbalance_key, 1.0) for p in profiles]
        if any(v > 1.0 for v in values):
            ax.plot(iterations, values, label=timer, color=color, linewidth=2, marker='o', markersize=4)
    
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect balance')
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Imbalance Ratio (max/min)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_ylim(bottom=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def generate_html_report(
    profile_dir: str,
    output_name: str,
    run_name: Optional[str] = None,
    warmup_iterations: int = 3,
):
    """Generate an interactive HTML report for a single run."""
    
    jsonl_path, csv_path = find_profile_files(profile_dir)
    
    if not jsonl_path:
        print(f"No JSONL profile found in {profile_dir}")
        return
    
    profiles = load_jsonl(jsonl_path)
    
    # Always compute stats from JSONL with warmup filtering for consistency
    filtered_for_stats = filter_outliers(profiles, warmup_iterations)
    summary = extract_timer_stats_from_profiles(filtered_for_stats)
    
    run_name = run_name or Path(profile_dir).parent.name
    
    output_dir = Path(output_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_dir = output_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating report in: {output_dir}/")
    
    timeline_chart = chart_dir / "timeline.png"
    pie_chart = chart_dir / "breakdown.png"
    imbalance_chart = chart_dir / "imbalance.png"
    
    if HAS_MATPLOTLIB:
        generate_iteration_timeline_chart(profiles, str(timeline_chart), f"{run_name} - Iteration Timeline")
        generate_phase_pie_chart(profiles, str(pie_chart), f"{run_name} - Average Time Breakdown (excl. warmup)")
        generate_load_imbalance_chart(profiles, str(imbalance_chart), f"{run_name} - Load Imbalance")
    
    filtered_profiles = filter_outliers(profiles, warmup_iterations)
    total_iterations = len(profiles)
    analyzed_iterations = len(filtered_profiles)
    
    avg_iteration_time = sum(p["elapsed_time_ms"] for p in filtered_profiles) / analyzed_iterations if filtered_profiles else 0
    min_iteration_time = min(p["elapsed_time_ms"] for p in filtered_profiles) if filtered_profiles else 0
    max_iteration_time = max(p["elapsed_time_ms"] for p in filtered_profiles) if filtered_profiles else 0
    
    throughputs = [p.get("throughput_tflops", 0) for p in filtered_profiles if p.get("throughput_tflops")]
    avg_throughput = statistics.mean(throughputs) if throughputs else 0
    
    tokens_per_sec = [p.get("tokens_per_sec", 0) for p in filtered_profiles if p.get("tokens_per_sec")]
    avg_tokens_per_sec = statistics.mean(tokens_per_sec) if tokens_per_sec else 0
    
    tokens_per_sec_per_gpu = [p.get("tokens_per_sec_per_gpu", 0) for p in filtered_profiles if p.get("tokens_per_sec_per_gpu")]
    avg_tokens_per_sec_per_gpu = statistics.mean(tokens_per_sec_per_gpu) if tokens_per_sec_per_gpu else 0
    
    # Get GPU count from profiles if available
    global_batch_sizes = [p.get("global_batch_size", 0) for p in filtered_profiles if p.get("global_batch_size")]
    avg_global_batch_size = statistics.mean(global_batch_sizes) if global_batch_sizes else 0
    
    phase_totals = defaultdict(float)
    for p in filtered_profiles:
        phases = compute_phase_times(p)
        for phase, time in phases.items():
            phase_totals[phase] += time
    phase_avgs = {k: v / analyzed_iterations for k, v in phase_totals.items()}
    
    # Simplified HTML with cleaner colors
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RL Profiling Report - {run_name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.5;
            padding: 2rem;
        }}
        
        .container {{ max-width: 1400px; margin: 0 auto; }}
        
        header {{
            margin-bottom: 1.5rem;
            padding: 1.5rem;
            background: #fff;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        
        h1 {{
            font-size: 1.75rem;
            color: #222;
            margin-bottom: 0.25rem;
        }}
        
        .subtitle {{ color: #666; font-size: 0.9rem; }}
        
        .section {{
            background: #fff;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        
        .section h2 {{
            color: #333;
            margin-bottom: 1rem;
            font-size: 1.2rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #eee;
        }}
        
        .chart-container {{
            text-align: center;
            margin: 1rem 0;
        }}
        
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        
        .charts-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 1.5rem;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        th, td {{
            padding: 0.5rem 0.75rem;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        
        th {{
            background: #fafafa;
            color: #555;
            font-weight: 600;
            font-size: 0.85rem;
        }}
        
        td.numeric {{ text-align: right; font-family: 'Consolas', 'Monaco', monospace; }}
        
        tr:hover {{ background: #fafafa; }}
        
        .summary-table {{ width: auto; }}
        .summary-table td {{ padding: 0.3rem 1.5rem 0.3rem 0; border: none; }}
        .summary-table .label {{ color: #666; }}
        .summary-table .value {{ color: #222; font-weight: 600; font-family: 'Consolas', monospace; }}
        
        .phase-bar {{
            display: flex;
            height: 28px;
            border-radius: 4px;
            overflow: hidden;
            margin: 1rem 0;
        }}
        
        .phase-segment {{
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 0.7rem;
            font-weight: 600;
            min-width: 25px;
        }}
        
        .legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
            margin-top: 0.75rem;
            font-size: 0.85rem;
        }}
        
        .legend-item {{ display: flex; align-items: center; gap: 0.3rem; }}
        .legend-color {{ width: 12px; height: 12px; border-radius: 2px; }}
        
        footer {{
            text-align: center;
            color: #999;
            padding: 1.5rem;
            font-size: 0.8rem;
        }}
        
        .note {{ font-size: 0.8rem; color: #888; margin-top: 0.5rem; }}
        
        .view-toggle {{
            margin-bottom: 1rem;
            display: flex;
            gap: 1.5rem;
        }}
        
        .view-toggle label {{
            display: flex;
            align-items: center;
            gap: 0.4rem;
            cursor: pointer;
            font-size: 0.9rem;
            color: #555;
        }}
        
        .view-toggle input[type="radio"] {{
            cursor: pointer;
        }}
        
        .container-timer {{
            background: #f8f8f8;
            font-weight: 600;
        }}
        
        .container-timer td:first-child {{
            color: #333;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>RL Profiling Report</h1>
            <p class="subtitle">{run_name} &bull; {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
        </header>
        
        <div class="section">
            <h2>Summary</h2>
            <table class="summary-table">
                <tr><td class="label">Total Iterations:</td><td class="value">{total_iterations}</td>
                    <td class="label">Analyzed:</td><td class="value">{analyzed_iterations}</td></tr>
                <tr><td class="label">Avg Iteration Time:</td><td class="value">{avg_iteration_time/1000:.2f}s</td>
                    <td class="label">Min / Max:</td><td class="value">{min_iteration_time/1000:.2f}s / {max_iteration_time/1000:.2f}s</td></tr>
                <tr><td class="label">Avg Training:</td><td class="value">{phase_avgs.get('Training', 0)/1000:.2f}s</td>
                    <td class="label">Avg Rollout (R0):</td><td class="value">{phase_avgs.get('Rollout Gen (R0)', 0)/1000:.2f}s</td></tr>
            </table>
            
            <h3 style="margin-top: 1.5rem; margin-bottom: 0.75rem; font-size: 1rem; color: #444;">Throughput Metrics</h3>
            <table class="summary-table">
                <tr><td class="label">System TFLOPS:</td><td class="value">{avg_throughput:.2f}</td>
                    <td class="label" style="padding-left: 2rem;">Global Batch Size:</td><td class="value">{avg_global_batch_size:.0f}</td></tr>
                <tr><td class="label">Tokens/sec (system):</td><td class="value">{avg_tokens_per_sec:.1f}</td>
                    <td class="label" style="padding-left: 2rem;">Tokens/sec (per GPU):</td><td class="value">{avg_tokens_per_sec_per_gpu:.1f}</td></tr>
            </table>
            <p class="note">Statistics exclude first {warmup_iterations} warm-up iterations. System metrics are aggregated across all GPUs.</p>
        </div>
        
        <div class="section">
            <h2>Phase Breakdown</h2>
            <div class="phase-bar">
"""
    
    total_phase_time = sum(phase_avgs.values())
    for phase, time in sorted(phase_avgs.items(), key=lambda x: x[1], reverse=True):
        if time > 0:
            pct = (time / total_phase_time) * 100
            color = PHASE_COLORS.get(phase, "#95a5a6")
            label = f"{pct:.0f}%" if pct > 5 else ""
            html += f'                <div class="phase-segment" style="width: {pct}%; background: {color};" title="{phase}: {time/1000:.1f}s">{label}</div>\n'
    
    html += """            </div>
            <div class="legend">
"""
    
    for phase, time in sorted(phase_avgs.items(), key=lambda x: x[1], reverse=True):
        if time > 0:
            color = PHASE_COLORS.get(phase, "#95a5a6")
            pct = (time / total_phase_time) * 100
            html += f'                <div class="legend-item"><div class="legend-color" style="background: {color};"></div>{phase}: {time/1000:.1f}s ({pct:.1f}%)</div>\n'
    
    html += """            </div>
            <p class="note">(R0) = Rank-0 only operation. Other ranks are idle/waiting during this time.</p>
        </div>
        
        <div class="section">
            <h2>Timeline</h2>
            <div class="charts-row">
"""
    
    if HAS_MATPLOTLIB:
        html += """                <div class="chart-container">
                    <img src="charts/timeline.png" alt="Iteration Timeline">
                </div>
                <div class="chart-container">
                    <img src="charts/breakdown.png" alt="Time Breakdown">
                </div>
"""
    else:
        html += """                <p>Charts not available (matplotlib not installed)</p>
"""
    
    html += """            </div>
        </div>
        
        <div class="section">
            <h2>Load Imbalance</h2>
            <p class="note">Max/min ratio across ranks (1.0 = perfect). Excludes rank-0 only operations.</p>
"""
    
    if HAS_MATPLOTLIB:
        html += """            <div class="chart-container">
                <img src="charts/imbalance.png" alt="Load Imbalance">
            </div>
"""
    
    html += """        </div>
        
        <div class="section">
            <h2>Timer Statistics</h2>
            <div class="view-toggle">
                <label>
                    <input type="radio" name="timer-view" value="flat" checked onchange="toggleTimerView('flat')">
                    Flat (no containers)
                </label>
                <label>
                    <input type="radio" name="timer-view" value="hierarchy" onchange="toggleTimerView('hierarchy')">
                    Hierarchical
                </label>
            </div>
            
            <table id="timer-table-flat">
                <thead>
                    <tr>
                        <th>Timer</th>
                        <th style="text-align:right">Avg (ms)</th>
                        <th style="text-align:right">Min (ms)</th>
                        <th style="text-align:right">Max (ms)</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    # Flat view: Filter out container timers to avoid double-counting
    filtered_summary = {k: v for k, v in summary.items() if k not in CONTAINER_TIMERS}
    sorted_summary = sorted(filtered_summary.items(), key=lambda x: x[1].get("mean_ms", 0), reverse=True)
    for timer, stats in sorted_summary[:25]:
        html += f"""                    <tr>
                        <td>{timer}</td>
                        <td class="numeric">{stats.get('mean_ms', 0):.1f}</td>
                        <td class="numeric">{stats.get('min_ms', 0):.1f}</td>
                        <td class="numeric">{stats.get('max_ms', 0):.1f}</td>
                    </tr>
"""
    
    html += """                </tbody>
            </table>
            
            <table id="timer-table-hierarchy" style="display: none;">
                <thead>
                    <tr>
                        <th>Timer</th>
                        <th style="text-align:right">Avg (ms)</th>
                        <th style="text-align:right">Min (ms)</th>
                        <th style="text-align:right">Max (ms)</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    # Hierarchical view: show all timers with indentation
    # Define the hierarchy
    timer_hierarchy = {
        "forward-backward": [],
        "optimizer": ["optimizer-copy-to-main-grad", "optimizer-inner-step", "optimizer-copy-main-to-model-params"],
        "rl/rollout-collection": [
            "rl/inference-setup",
            "rl/collect-rollouts",
            "rl/compute-ref-logprobs",
        "rl/train-forward",
            "forward-backward",
            "rl/offload-optimizer-before-inference",
            "rl/onload-optimizer-after-inference",
            "rl/offload-kv-cache-after-inference",
            "rl/onload-kv-cache-before-inference",
        ],
        "rl/prepare-data-for-update": [
            "rl/compute-group-stats",
            "rl/prepare-advantages",
            "rl/prepare-trajectories",
            "rl/get-ltor-masks",
            "rl/create-dataloader",
            "rl/align-inference-logprobs",
            "rl/compute-logprobs",
        ],
        "rl/compute-logprobs": [
            "rl/compute-old-logprobs",
            "rl/compute-ref-logprobs",
        ],
    }
    
    # Track which timers we've already shown
    shown_timers = set()
    
    def add_timer_row(timer_name, indent=0):
        nonlocal html, shown_timers
        if timer_name in shown_timers:
            return
        if timer_name not in summary:
            return
        shown_timers.add(timer_name)
        stats = summary[timer_name]
        indent_style = f"padding-left: {20 + indent * 20}px;" if indent > 0 else ""
        is_container = timer_name in CONTAINER_TIMERS
        container_class = ' class="container-timer"' if is_container else ''
        prefix = "├─ " if indent > 0 else ""
        html += f"""                    <tr{container_class}>
                        <td style="{indent_style}">{prefix}{timer_name}</td>
                        <td class="numeric">{stats.get('mean_ms', 0):.1f}</td>
                        <td class="numeric">{stats.get('min_ms', 0):.1f}</td>
                        <td class="numeric">{stats.get('max_ms', 0):.1f}</td>
                    </tr>
"""
        # Add children
        if timer_name in timer_hierarchy:
            for child in timer_hierarchy[timer_name]:
                add_timer_row(child, indent + 1)
    
    # Add top-level timers in order
    top_level_order = ["forward-backward", "optimizer", "rl/rollout-collection", "rl/prepare-data-for-update"]
    for timer in top_level_order:
        add_timer_row(timer, 0)
    
    # Add any remaining timers not in hierarchy
    remaining = sorted(
        [(k, v) for k, v in summary.items() if k not in shown_timers],
        key=lambda x: x[1].get("mean_ms", 0),
        reverse=True
    )
    for timer, stats in remaining[:15]:
        html += f"""                    <tr>
                        <td>{timer}</td>
                        <td class="numeric">{stats.get('mean_ms', 0):.1f}</td>
                        <td class="numeric">{stats.get('min_ms', 0):.1f}</td>
                        <td class="numeric">{stats.get('max_ms', 0):.1f}</td>
                    </tr>
"""
    
    html += """                </tbody>
            </table>
        </div>
        
        <script>
        function toggleTimerView(view) {
            document.getElementById('timer-table-flat').style.display = view === 'flat' ? '' : 'none';
            document.getElementById('timer-table-hierarchy').style.display = view === 'hierarchy' ? '' : 'none';
        }
        </script>
        
        <footer>
            Generated by megatron.rl.rl_profiling_viz
        </footer>
    </div>
</body>
</html>
"""
    
    html_path = output_dir / "index.html"
    with open(html_path, "w") as f:
        f.write(html)
    
    print(f"Generated report: {html_path}")
    print(f"  Charts: {chart_dir}/")


def generate_comparison_report(
    profile_dirs: List[str],
    run_names: List[str],
    output_name: str
):
    """Generate a comparison report for multiple runs."""
    
    summaries = []
    all_profiles = []
    
    for profile_dir in profile_dirs:
        jsonl_path, csv_path = find_profile_files(profile_dir)
        profiles = load_jsonl(jsonl_path) if jsonl_path else []
        
        if csv_path:
            summary = load_csv_summary(csv_path)
        elif profiles:
            filtered = filter_outliers(profiles)
            summary = extract_timer_stats_from_profiles(filtered)
        else:
            summary = {}
            
        if summary:
            summaries.append(summary)
        if profiles:
            all_profiles.append(profiles)
    
    if not summaries:
        print("No summary files found")
        return
    
    output_dir = Path(output_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_dir = output_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating comparison report in: {output_dir}/")
    
    if HAS_MATPLOTLIB:
        generate_timer_comparison_chart(
            summaries, run_names, 
            str(chart_dir / "comparison.png")
        )
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RL Profiling Comparison</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            color: #333;
            padding: 2rem;
        }}
        
        .container {{ max-width: 1400px; margin: 0 auto; }}
        
        header {{
            margin-bottom: 1.5rem;
            padding: 1.5rem;
            background: #fff;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        
        h1 {{ font-size: 1.75rem; color: #222; }}
        .subtitle {{ color: #666; font-size: 0.9rem; margin-top: 0.25rem; }}
        
        .section {{
            background: #fff;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        
        .section h2 {{
            color: #333;
            margin-bottom: 1rem;
            font-size: 1.2rem;
            border-bottom: 2px solid #eee;
            padding-bottom: 0.5rem;
        }}
        
        .chart-container {{ text-align: center; margin: 1rem 0; }}
        .chart-container img {{ max-width: 100%; border-radius: 4px; }}
        
        table {{ width: 100%; border-collapse: collapse; }}
        
        th, td {{
            padding: 0.5rem 0.75rem;
            text-align: right;
            border-bottom: 1px solid #eee;
        }}
        
        th:first-child, td:first-child {{ text-align: left; }}
        th {{ background: #fafafa; color: #555; font-weight: 600; }}
        tr:hover {{ background: #fafafa; }}
        
        .better {{ color: #27ae60; }}
        .worse {{ color: #e67e22; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>RL Profiling Comparison</h1>
            <p class="subtitle">Comparing: {', '.join(run_names)}</p>
        </header>
        
        <div class="section">
            <h2>Timer Comparison</h2>
            <div class="chart-container">
                <img src="charts/comparison.png" alt="Timer Comparison">
            </div>
        </div>
        
        <div class="section">
            <h2>Detailed Comparison</h2>
            <table>
                <thead>
                    <tr>
                        <th>Timer</th>
"""
    
    for name in run_names:
        html += f"                        <th>{name} (ms)</th>\n"
    
    if len(run_names) == 2:
        html += "                        <th>Δ (%)</th>\n"
    
    html += """                    </tr>
                </thead>
                <tbody>
"""
    
    all_timers = set()
    for s in summaries:
        all_timers.update(s.keys())
    
    sorted_timers = sorted(all_timers, 
                           key=lambda t: summaries[0].get(t, {}).get("mean_ms", 0), 
                           reverse=True)
    
    for timer in sorted_timers[:25]:
        html += f"                    <tr>\n                        <td>{timer}</td>\n"
        
        values = []
        for s in summaries:
            val = s.get(timer, {}).get("mean_ms", 0)
            values.append(val)
            html += f"                        <td>{val:.1f}</td>\n"
        
        if len(values) == 2 and values[0] > 0:
            delta = ((values[1] - values[0]) / values[0]) * 100
            css_class = "better" if delta < -5 else ("worse" if delta > 5 else "")
            sign = "+" if delta > 0 else ""
            html += f'                        <td class="{css_class}">{sign}{delta:.1f}%</td>\n'
        
        html += "                    </tr>\n"
    
    html += """                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""
    
    html_path = output_dir / "index.html"
    with open(html_path, "w") as f:
        f.write(html)
    
    print(f"Generated comparison report: {html_path}")


def main():
    parser = argparse.ArgumentParser(description="RL Profiling Visualization")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    report_parser = subparsers.add_parser("report", help="Generate HTML report for a single run")
    report_parser.add_argument("profile_dir", help="Directory containing profile files")
    report_parser.add_argument("--output", "-o", default="profile_report", 
                               help="Output folder name")
    report_parser.add_argument("--name", help="Run name for the report")
    report_parser.add_argument("--warmup", type=int, default=3,
                               help="Warm-up iterations to exclude (default: 3)")
    
    compare_parser = subparsers.add_parser("compare", help="Compare multiple runs")
    compare_parser.add_argument("profile_dirs", nargs="+", help="Directories containing profile files")
    compare_parser.add_argument("--names", nargs="+", help="Names for each run")
    compare_parser.add_argument("--output", "-o", default="comparison_report")
    
    charts_parser = subparsers.add_parser("charts", help="Generate PNG charts only")
    charts_parser.add_argument("profile_dir", help="Directory containing profile files")
    charts_parser.add_argument("--output", "-o", default="./charts")
    
    args = parser.parse_args()
    
    if args.command == "report":
        generate_html_report(args.profile_dir, args.output, args.name, args.warmup)
    
    elif args.command == "compare":
        names = args.names or [Path(d).parent.name for d in args.profile_dirs]
        generate_comparison_report(args.profile_dirs, names, args.output)
    
    elif args.command == "charts":
        jsonl_path, _ = find_profile_files(args.profile_dir)
        if jsonl_path:
            profiles = load_jsonl(jsonl_path)
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            generate_iteration_timeline_chart(profiles, str(output_dir / "timeline.png"))
            generate_phase_pie_chart(profiles, str(output_dir / "breakdown.png"))
            generate_load_imbalance_chart(profiles, str(output_dir / "imbalance.png"))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
