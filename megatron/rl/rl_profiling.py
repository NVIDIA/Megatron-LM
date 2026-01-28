# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
RL Profiling Infrastructure

Provides structured logging of timer data for analysis across runs and iterations.
Outputs:
  - JSONL file: One JSON object per iteration with full timer breakdown
  - CSV summary: Aggregated stats (mean, min, max, p50, p95, p99) at run end
  - WandB/TensorBoard: Timer metrics logged as scalars for visualization

Usage:
    from megatron.rl.rl_profiling import get_rl_profiler, log_iteration_profile

    # At each iteration (after timers.log)
    log_iteration_profile(iteration, timers)

    # At run end
    get_rl_profiler().export_summary()
"""

import csv
import json
import logging
import os
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


# Timer names we care about for RL profiling (in hierarchical order)
RL_TIMER_NAMES = [
    # Top-level phases
    "forward-backward",
    "optimizer",
    "rl/rollout-collection",
    "rl/prepare-data-for-update",
    
    # Rollout collection breakdown
    "rl/inference-setup",
    "rl/collect-rollouts",
    "rl/sync-rollouts",
    "rl/suspend-engine",
    
    # Optimizer offload/onload
    "rl/offload-optimizer-before-inference",
    "rl/onload-optimizer-after-inference",
    "rl/offload-kv-cache-after-inference",
    "rl/onload-kv-cache-before-inference",
    
    # Weight prefetching
    "rl/prefetch-weights-to-gpu",
    "rl/prefetch-weights-to-cpu",
    
    # Data preparation breakdown
    "rl/compute-group-stats",
    "rl/prepare-advantages",
    "rl/prepare-trajectories",
    "rl/get-ltor-masks",
    "rl/create-dataloader",
    "rl/sequence-packing",
    "rl/pack-logprobs",
    "rl/align-inference-logprobs",
    "rl/log-wandb-tb",
    
    # Logprobs computation
    "rl/compute-logprobs",
    "rl/compute-old-logprobs",
    "rl/compute-ref-logprobs",
    "rl/get-logprobs",
    "rl/forward-pass",
    "rl/log-softmax",
    
    # Training
    "rl/train/forward",
    "rl/train/grpo-loss",
    
    # Gradient sync
    "embedding-grads-all-reduce",
    "all-grads-sync",
    "params-all-gather",
    "optimizer-copy-to-main-grad",
    "optimizer-inner-step",
    "optimizer-copy-main-to-model-params",
]

# Define timer hierarchy for de-duplication analysis
TIMER_HIERARCHY = {
    "rl/rollout-collection": [
        "rl/inference-setup",
        "rl/collect-rollouts", 
        "rl/sync-rollouts",
        "rl/suspend-engine",
        "rl/offload-optimizer-before-inference",
        "rl/onload-optimizer-after-inference",
        "rl/prefetch-weights-to-gpu",
        "rl/prefetch-weights-to-cpu",
        "rl/onload-kv-cache-before-inference",
        "rl/offload-kv-cache-after-inference",
    ],
    "rl/prepare-data-for-update": [
        "rl/compute-group-stats",
        "rl/prepare-advantages",
        "rl/prepare-trajectories",
        "rl/get-ltor-masks",
        "rl/create-dataloader",
        "rl/sequence-packing",
        "rl/pack-logprobs",
        "rl/align-inference-logprobs",
        "rl/log-wandb-tb",
        "rl/compute-logprobs",
    ],
    "rl/compute-logprobs": [
        "rl/compute-old-logprobs",
        "rl/compute-ref-logprobs",
    ],
    "rl/get-logprobs": [
        "rl/forward-pass",
        "rl/log-softmax",
    ],
    "optimizer": [
        "optimizer-copy-to-main-grad",
        "optimizer-inner-step",
        "optimizer-copy-main-to-model-params",
    ],
}


@dataclass
class IterationProfile:
    """Profile data for a single iteration."""
    iteration: int
    timestamp: str
    elapsed_time_ms: float  # Total iteration time
    timers: Dict[str, Tuple[float, float]]  # timer_name -> (min_ms, max_ms)
    throughput_tflops: Optional[float] = None
    global_batch_size: Optional[int] = None
    tokens_per_sec: Optional[float] = None
    tokens_per_sec_per_gpu: Optional[float] = None
    
    # Computed metrics
    load_imbalance: Dict[str, float] = field(default_factory=dict)  # timer -> max/min ratio
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "iteration": self.iteration,
            "timestamp": self.timestamp,
            "elapsed_time_ms": self.elapsed_time_ms,
            "throughput_tflops": self.throughput_tflops,
            "global_batch_size": self.global_batch_size,
            "tokens_per_sec": self.tokens_per_sec,
            "tokens_per_sec_per_gpu": self.tokens_per_sec_per_gpu,
        }
        
        # Flatten timer data
        for name, (min_t, max_t) in self.timers.items():
            safe_name = name.replace("/", "_").replace("-", "_")
            result[f"timer_{safe_name}_min_ms"] = min_t
            result[f"timer_{safe_name}_max_ms"] = max_t
        
        # Add load imbalance metrics
        for name, ratio in self.load_imbalance.items():
            safe_name = name.replace("/", "_").replace("-", "_")
            result[f"imbalance_{safe_name}"] = ratio
            
        return result


@dataclass 
class RunSummary:
    """Aggregated statistics across all iterations in a run."""
    run_id: str
    start_time: str
    end_time: str
    num_iterations: int
    world_size: int
    
    # Per-timer aggregated stats
    timer_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Format: timer_name -> {"mean", "min", "max", "std", "p50", "p95", "p99"}


class RLProfiler:
    """
    Profiling infrastructure for RL training.
    
    Collects timer data at each iteration and provides:
    - Per-iteration JSONL logging
    - End-of-run CSV summary
    - Integration with WandB/TensorBoard
    """
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        run_id: Optional[str] = None,
        enabled: bool = True,
        log_to_wandb: bool = True,
        log_to_tensorboard: bool = True,
        timer_names: Optional[List[str]] = None,
    ):
        """
        Initialize the RL Profiler.
        
        Args:
            output_dir: Directory to write profiling data. If None, uses LANGRL_LOG_DIR or ./profiles
            run_id: Unique identifier for this run. If None, generates from timestamp
            enabled: Whether profiling is enabled
            log_to_wandb: Whether to log timer metrics to WandB
            log_to_tensorboard: Whether to log timer metrics to TensorBoard
            timer_names: List of timer names to track. If None, uses RL_TIMER_NAMES
        """
        self.enabled = enabled
        self.log_to_wandb = log_to_wandb
        self.log_to_tensorboard = log_to_tensorboard
        self.timer_names = timer_names or RL_TIMER_NAMES
        
        # Determine output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            env_dir = os.environ.get("LANGRL_LOG_DIR")
            if env_dir:
                self.output_dir = Path(env_dir) / "profiles"
            else:
                self.output_dir = Path("./profiles")
        
        # Generate run ID
        if run_id:
            self.run_id = run_id
        else:
            self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        self.start_time = datetime.now().isoformat()
        
        # Storage for iteration data
        self.iteration_profiles: List[IterationProfile] = []
        self._timer_history: Dict[str, List[float]] = defaultdict(list)  # For stats
        
        # File handles (lazy init)
        self._jsonl_file = None
        self._initialized = False
        
    def _ensure_initialized(self):
        """Lazy initialization of output files (only on rank 0)."""
        if self._initialized:
            return
            
        # Only rank 0 writes files
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank != 0:
            self._initialized = True
            return
            
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Open JSONL file for streaming writes
        jsonl_path = self.output_dir / f"profile_{self.run_id}.jsonl"
        self._jsonl_file = open(jsonl_path, "w")
        
        logger.info(f"[RLProfiler] Writing profiles to {jsonl_path}")
        self._initialized = True
        
    def log_iteration(
        self,
        iteration: int,
        timers,  # Timers object from megatron
        elapsed_time_ms: float,
        throughput_tflops: Optional[float] = None,
        global_batch_size: Optional[int] = None,
        tokens_per_sec: Optional[float] = None,
        tokens_per_sec_per_gpu: Optional[float] = None,
        extra_metrics: Optional[Dict[str, float]] = None,
        wandb_writer=None,
        tb_writer=None,
    ):
        """
        Log profiling data for a single iteration.
        
        Args:
            iteration: Current training iteration
            timers: Megatron Timers object
            elapsed_time_ms: Total iteration time in milliseconds
            throughput_tflops: TFLOPS throughput
            global_batch_size: Global batch size
            tokens_per_sec: Token throughput
            tokens_per_sec_per_gpu: Token throughput per GPU
            extra_metrics: Additional metrics to log
            wandb_writer: WandB writer for metric logging
            tb_writer: TensorBoard writer for metric logging
        """
        if not self.enabled:
            return
            
        self._ensure_initialized()
        
        # Collect timer data (min, max across ranks)
        timer_data = self._collect_timer_data(timers)
        
        # Warn if no timer data was collected (might indicate timing issue)
        if not timer_data:
            logger.warning(f"[RLProfiler] No timer data collected for iteration {iteration}. "
                          "Timers may have been reset before profiling.")
        
        # Compute load imbalance metrics
        load_imbalance = {}
        for name, (min_t, max_t) in timer_data.items():
            if min_t > 0:
                load_imbalance[name] = max_t / min_t
        
        # Create profile
        profile = IterationProfile(
            iteration=iteration,
            timestamp=datetime.now().isoformat(),
            elapsed_time_ms=elapsed_time_ms,
            timers=timer_data,
            throughput_tflops=throughput_tflops,
            global_batch_size=global_batch_size,
            tokens_per_sec=tokens_per_sec,
            tokens_per_sec_per_gpu=tokens_per_sec_per_gpu,
            load_imbalance=load_imbalance,
        )
        
        self.iteration_profiles.append(profile)
        
        # Track history for summary stats
        for name, (_, max_t) in timer_data.items():
            self._timer_history[name].append(max_t)
        
        # Write to JSONL (rank 0 only)
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0 and self._jsonl_file:
            self._jsonl_file.write(json.dumps(profile.to_dict()) + "\n")
            self._jsonl_file.flush()
        
        # Log to WandB/TensorBoard
        if self.log_to_wandb and wandb_writer:
            self._log_to_wandb(profile, wandb_writer, iteration, extra_metrics)
        if self.log_to_tensorboard and tb_writer:
            self._log_to_tensorboard(profile, tb_writer, iteration, extra_metrics)
            
    def _collect_timer_data(self, timers) -> Dict[str, Tuple[float, float]]:
        """Collect min/max timer data across ranks."""
        # Use timers' internal method to get min/max times
        # This does an all_gather internally
        name_to_min_max = timers._get_global_min_max_time(
            names=self.timer_names,
            reset=False,  # Don't reset - let the main logging code handle that
            barrier=False,
            normalizer=1.0 / 1000.0,  # Convert to milliseconds
        )
        return name_to_min_max or {}
        
    def _log_to_wandb(
        self, 
        profile: IterationProfile, 
        wandb_writer, 
        iteration: int,
        extra_metrics: Optional[Dict[str, float]] = None,
    ):
        """Log timer metrics to WandB."""
        metrics = {}
        
        # Log max times for each timer (most relevant for optimization)
        for name, (min_t, max_t) in profile.timers.items():
            metrics[f"profile/{name}_max_ms"] = max_t
            metrics[f"profile/{name}_min_ms"] = min_t
            
            # Log imbalance for timers with significant spread
            if name in profile.load_imbalance and profile.load_imbalance[name] > 1.1:
                metrics[f"profile/{name}_imbalance"] = profile.load_imbalance[name]
        
        # Log aggregated phase times
        phase_times = self._compute_phase_times(profile)
        for phase, time_ms in phase_times.items():
            metrics[f"profile/phase_{phase}_ms"] = time_ms
            
        if extra_metrics:
            metrics.update(extra_metrics)
            
        wandb_writer.log(metrics, step=iteration)
        
    def _log_to_tensorboard(
        self, 
        profile: IterationProfile, 
        tb_writer, 
        iteration: int,
        extra_metrics: Optional[Dict[str, float]] = None,
    ):
        """Log timer metrics to TensorBoard."""
        # Log max times for each timer
        for name, (min_t, max_t) in profile.timers.items():
            tb_writer.add_scalar(f"profile/{name}_max_ms", max_t, iteration)
            
        # Log phase times
        phase_times = self._compute_phase_times(profile)
        for phase, time_ms in phase_times.items():
            tb_writer.add_scalar(f"profile/phase_{phase}_ms", time_ms, iteration)
            
        if extra_metrics:
            for key, value in extra_metrics.items():
                tb_writer.add_scalar(key, value, iteration)
                
    def _compute_phase_times(self, profile: IterationProfile) -> Dict[str, float]:
        """Compute high-level phase times from detailed timers."""
        timers = profile.timers
        phases = {}
        
        # Helper to get max time safely
        def get_max(name: str) -> float:
            return timers.get(name, (0, 0))[1]
        
        # Rollout generation (actual generation, not container)
        phases["rollout_generation"] = get_max("rl/collect-rollouts")
        
        # Optimizer memory management
        phases["optimizer_offload"] = get_max("rl/offload-optimizer-before-inference")
        phases["optimizer_onload"] = get_max("rl/onload-optimizer-after-inference")
        phases["optimizer_memory_mgmt"] = phases["optimizer_offload"] + phases["optimizer_onload"]
        
        # Logprobs computation
        phases["logprobs_old"] = get_max("rl/compute-old-logprobs")
        phases["logprobs_ref"] = get_max("rl/compute-ref-logprobs")
        phases["logprobs_total"] = phases["logprobs_old"] + phases["logprobs_ref"]
        
        # Training
        phases["training"] = get_max("forward-backward")
        
        # Wait/sync time (proxy for load imbalance)
        phases["sync_wait"] = get_max("rl/suspend-engine") + get_max("rl/sync-rollouts")
        
        return phases
        
    def export_summary(self, output_path: Optional[str] = None) -> Optional[str]:
        """
        Export aggregated summary statistics to CSV.
        
        Args:
            output_path: Path for CSV output. If None, uses default in output_dir.
            
        Returns:
            Path to the exported CSV file, or None if no data.
        """
        if not self.enabled or not self._timer_history:
            return None
            
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank != 0:
            return None
            
        self._ensure_initialized()
        
        # Compute statistics for each timer
        stats = {}
        for name, values in self._timer_history.items():
            if not values:
                continue
            sorted_values = sorted(values)
            n = len(sorted_values)
            stats[name] = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if n > 1 else 0,
                "min": min(values),
                "max": max(values),
                "p50": sorted_values[int(n * 0.50)],
                "p95": sorted_values[int(n * 0.95)] if n >= 20 else sorted_values[-1],
                "p99": sorted_values[int(n * 0.99)] if n >= 100 else sorted_values[-1],
                "count": n,
            }
        
        # Write CSV
        if output_path:
            csv_path = Path(output_path)
        else:
            csv_path = self.output_dir / f"summary_{self.run_id}.csv"
            
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timer_name", "mean_ms", "std_ms", "min_ms", "max_ms", 
                "p50_ms", "p95_ms", "p99_ms", "count"
            ])
            for name in self.timer_names:
                if name in stats:
                    s = stats[name]
                    writer.writerow([
                        name, 
                        f"{s['mean']:.2f}",
                        f"{s['std']:.2f}",
                        f"{s['min']:.2f}",
                        f"{s['max']:.2f}",
                        f"{s['p50']:.2f}",
                        f"{s['p95']:.2f}",
                        f"{s['p99']:.2f}",
                        s['count'],
                    ])
                    
        logger.info(f"[RLProfiler] Exported summary to {csv_path}")
        return str(csv_path)
        
    def print_summary(self):
        """Print a summary of timer statistics to stdout."""
        if not self.enabled or not self._timer_history:
            return
            
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank != 0:
            return
            
        print("\n" + "=" * 80)
        print("RL PROFILING SUMMARY")
        print("=" * 80)
        print(f"Run ID: {self.run_id}")
        print(f"Iterations: {len(self.iteration_profiles)}")
        print("-" * 80)
        print(f"{'Timer Name':<50} {'Mean':>8} {'P95':>8} {'Max':>8}")
        print("-" * 80)
        
        for name in self.timer_names:
            if name not in self._timer_history or not self._timer_history[name]:
                continue
            values = self._timer_history[name]
            sorted_values = sorted(values)
            n = len(sorted_values)
            mean = statistics.mean(values)
            p95 = sorted_values[int(n * 0.95)] if n >= 20 else sorted_values[-1]
            max_v = max(values)
            print(f"{name:<50} {mean:>7.1f}ms {p95:>7.1f}ms {max_v:>7.1f}ms")
            
        print("=" * 80 + "\n")
        
    def close(self):
        """Close file handles and export final summary."""
        if self._jsonl_file:
            self._jsonl_file.close()
            self._jsonl_file = None
        self.export_summary()
        self.print_summary()


# Global profiler instance
_RL_PROFILER: Optional[RLProfiler] = None


def get_rl_profiler() -> Optional[RLProfiler]:
    """Get the global RL profiler instance."""
    return _RL_PROFILER


def initialize_rl_profiler(
    output_dir: Optional[str] = None,
    run_id: Optional[str] = None,
    enabled: bool = True,
    **kwargs,
) -> RLProfiler:
    """
    Initialize the global RL profiler.
    
    Should be called once at training start.
    """
    global _RL_PROFILER
    _RL_PROFILER = RLProfiler(
        output_dir=output_dir,
        run_id=run_id,
        enabled=enabled,
        **kwargs,
    )
    return _RL_PROFILER


def log_iteration_profile(
    iteration: int,
    timers,
    elapsed_time_ms: float,
    throughput_tflops: Optional[float] = None,
    global_batch_size: Optional[int] = None,
    tokens_per_sec: Optional[float] = None,
    tokens_per_sec_per_gpu: Optional[float] = None,
    extra_metrics: Optional[Dict[str, float]] = None,
    wandb_writer=None,
    tb_writer=None,
):
    """
    Convenience function to log iteration profile using global profiler.
    
    This is the main entry point for logging timer data.
    """
    profiler = get_rl_profiler()
    if profiler:
        profiler.log_iteration(
            iteration=iteration,
            timers=timers,
            elapsed_time_ms=elapsed_time_ms,
            throughput_tflops=throughput_tflops,
            global_batch_size=global_batch_size,
            tokens_per_sec=tokens_per_sec,
            tokens_per_sec_per_gpu=tokens_per_sec_per_gpu,
            extra_metrics=extra_metrics,
            wandb_writer=wandb_writer,
            tb_writer=tb_writer,
        )


def shutdown_rl_profiler():
    """Shutdown the global RL profiler and export final data."""
    global _RL_PROFILER
    if _RL_PROFILER:
        _RL_PROFILER.close()
        _RL_PROFILER = None


# ============================================================================
# Analysis utilities for cross-run comparison
# ============================================================================

def load_profile_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load profile data from a JSONL file."""
    profiles = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                profiles.append(json.loads(line))
    return profiles


def load_summary_csv(path: str) -> Dict[str, Dict[str, float]]:
    """Load summary statistics from a CSV file."""
    stats = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timer_name = row["timer_name"]
            stats[timer_name] = {
                "mean_ms": float(row["mean_ms"]),
                "std_ms": float(row["std_ms"]),
                "min_ms": float(row["min_ms"]),
                "max_ms": float(row["max_ms"]),
                "p50_ms": float(row["p50_ms"]),
                "p95_ms": float(row["p95_ms"]),
                "p99_ms": float(row["p99_ms"]),
                "count": int(row["count"]),
            }
    return stats


def compare_runs(
    run_paths: List[str],
    run_names: Optional[List[str]] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    Compare profiling data across multiple runs.
    
    Args:
        run_paths: List of paths to summary CSV files
        run_names: Optional names for each run (defaults to filenames)
        output_path: Optional path to write comparison CSV
        
    Returns:
        Formatted comparison string
    """
    if run_names is None:
        run_names = [Path(p).stem for p in run_paths]
    
    # Load all summaries
    summaries = []
    for path in run_paths:
        summaries.append(load_summary_csv(path))
    
    # Get all timer names across all runs
    all_timers = set()
    for summary in summaries:
        all_timers.update(summary.keys())
    
    # Build comparison table
    lines = []
    header = f"{'Timer':<50}"
    for name in run_names:
        header += f" {name:>12}"
    lines.append(header)
    lines.append("-" * len(header))
    
    for timer in sorted(all_timers):
        line = f"{timer:<50}"
        for summary in summaries:
            if timer in summary:
                line += f" {summary[timer]['mean_ms']:>11.1f}ms"
            else:
                line += f" {'N/A':>12}"
        lines.append(line)
    
    result = "\n".join(lines)
    
    if output_path:
        with open(output_path, "w") as f:
            f.write(result)
    
    return result


def analyze_bottlenecks(profile_path: str, top_n: int = 10) -> str:
    """
    Analyze a profile to identify top bottlenecks.
    
    Args:
        profile_path: Path to summary CSV file
        top_n: Number of top timers to report
        
    Returns:
        Formatted analysis string
    """
    stats = load_summary_csv(profile_path)
    
    # Sort by mean time (descending)
    sorted_timers = sorted(
        stats.items(), 
        key=lambda x: x[1]["mean_ms"], 
        reverse=True
    )
    
    lines = []
    lines.append("=" * 70)
    lines.append("BOTTLENECK ANALYSIS")
    lines.append("=" * 70)
    lines.append(f"{'Rank':<6} {'Timer':<45} {'Mean':>8} {'P95':>8}")
    lines.append("-" * 70)
    
    for i, (timer, s) in enumerate(sorted_timers[:top_n], 1):
        lines.append(f"{i:<6} {timer:<45} {s['mean_ms']:>7.1f}ms {s['p95_ms']:>7.1f}ms")
    
    # Compute phase breakdown
    lines.append("")
    lines.append("PHASE BREAKDOWN:")
    lines.append("-" * 70)
    
    phases = {
        "Rollout Generation": ["rl/collect-rollouts"],
        "Optimizer Memory Mgmt": [
            "rl/offload-optimizer-before-inference",
            "rl/onload-optimizer-after-inference",
        ],
        "Logprobs Computation": [
            "rl/compute-old-logprobs",
            "rl/compute-ref-logprobs",
        ],
        "Training": ["forward-backward"],
        "Sync/Wait": ["rl/suspend-engine", "rl/sync-rollouts"],
    }
    
    for phase_name, timers in phases.items():
        total = sum(stats.get(t, {"mean_ms": 0})["mean_ms"] for t in timers)
        if total > 0:
            lines.append(f"  {phase_name:<40} {total:>7.1f}ms")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)


# ============================================================================
# CLI for analysis (can be run as: python -m megatron.rl.rl_profiling ...)
# ============================================================================

def main():
    """Command-line interface for profile analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RL Profiling Analysis")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a single run")
    analyze_parser.add_argument("profile", help="Path to summary CSV file")
    analyze_parser.add_argument("--top", type=int, default=10, help="Top N bottlenecks")
    
    # compare command
    compare_parser = subparsers.add_parser("compare", help="Compare multiple runs")
    compare_parser.add_argument("profiles", nargs="+", help="Paths to summary CSV files")
    compare_parser.add_argument("--names", nargs="+", help="Names for each run")
    compare_parser.add_argument("--output", help="Output file path")
    
    # list command
    list_parser = subparsers.add_parser("list", help="List iterations in JSONL file")
    list_parser.add_argument("profile", help="Path to JSONL profile file")
    list_parser.add_argument("--last", type=int, default=10, help="Show last N iterations")
    
    args = parser.parse_args()
    
    if args.command == "analyze":
        print(analyze_bottlenecks(args.profile, args.top))
    elif args.command == "compare":
        print(compare_runs(args.profiles, args.names, args.output))
    elif args.command == "list":
        profiles = load_profile_jsonl(args.profile)
        for p in profiles[-args.last:]:
            print(f"Iteration {p['iteration']}: {p['elapsed_time_ms']:.1f}ms")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
