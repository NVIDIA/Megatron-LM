# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Compute statistical bounds for golden values from multiple test runs.

This script aggregates results from multiple parallel runs of a functional test
and computes statistics (min, max, mean, std) for each metric at each step.
The output can be used to determine appropriate tolerances for test validation.

Usage:
    # Step 1: Run batch tests (from megatron-rl directory):
    ./tests/functional_tests/shell_test_utils/run_batch_ci_tests.sh \\
        test_cases/gpt/gpt_grpo_tp4_pp1_dp2_8b_correctness_and_throughput.sh 10

    # Step 2: Wait for jobs to complete, then compute statistics:
    python tests/functional_tests/python_test_utils/compute_golden_statistics.py \\
        --results-dir batch_test_logs_gpt_grpo_*/ \\
        --output golden_values_stats.json \\
        --recommend-tolerances

    # The script parses .out log files to find where each run wrote its results.
    # Each .out file should contain: "This test wrote results into /opt/megatron-lm/runs/<uuid>"
    # The container path /opt/megatron-lm maps to the workspace root on the host.

    # Or specify individual JSON files directly:
    python compute_golden_statistics.py \\
        --result-files runs/abc123/golden_values.json runs/def456/golden_values.json \\
        --output golden_values_stats.json
"""

import argparse
import glob
import json
import logging
import math
import os
import sys
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def find_result_json_files(results_dir: str, workspace_root: Optional[str] = None) -> List[str]:
    """
    Find all result JSON files from a batch test run.

    The batch test infrastructure (run_batch_ci_tests.sh) writes .out log files
    to the results directory. Each .out file contains a line like:
        "This test wrote results into /opt/megatron-lm/runs/<uuid>"

    The container path /opt/megatron-lm maps to the workspace root on the host.
    This function parses the .out files to find where the JSON results are.

    Args:
        results_dir: Path to batch_test_logs_* directory containing .out files
        workspace_root: Root of the megatron workspace (defaults to cwd)
    """
    result_files = []
    results_path = Path(results_dir)

    if not results_path.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return []

    if workspace_root is None:
        # Try to find workspace root by looking for common markers
        workspace_root = os.getcwd()

    # Find all .out files from batch test runs
    out_files = list(results_path.glob("*.out"))

    if not out_files:
        logger.warning(f"No .out files found in {results_dir}")
        # Fall back to searching for JSON files directly
        return _find_json_files_directly(results_dir)

    logger.info(f"Found {len(out_files)} .out files to parse")

    for out_file in out_files:
        json_path = _extract_result_path_from_log(out_file, workspace_root)
        if json_path and os.path.exists(json_path):
            result_files.append(json_path)
        elif json_path:
            logger.warning(f"Result file not found: {json_path} (from {out_file.name})")

    return result_files


def _extract_result_path_from_log(out_file: Path, workspace_root: str) -> Optional[str]:
    """
    Parse a .out log file to find the result JSON path.

    Looks for the line: "This test wrote results into /opt/megatron-lm/runs/<uuid>"
    and converts the container path to the host path.
    """
    try:
        with open(out_file, 'r', errors='ignore') as f:
            content = f.read()
    except IOError as e:
        logger.warning(f"Failed to read {out_file}: {e}")
        return None

    # Look for the output path marker
    marker = "This test wrote results into "
    for line in content.split('\n'):
        if marker in line:
            # Extract the path after the marker
            idx = line.find(marker)
            output_path = line[idx + len(marker) :].strip()

            # Convert container path to host path
            # /opt/megatron-lm/... -> <workspace_root>/...
            if output_path.startswith("/opt/megatron-lm/"):
                host_path = output_path.replace("/opt/megatron-lm/", "")
                output_path = os.path.join(workspace_root, host_path)

            # Find JSON result files in this directory (search recursively)
            output_dir = Path(output_path)
            if output_dir.exists() and output_dir.is_dir():
                # Look for result JSON files with various naming patterns
                # Search recursively since files may be in subdirectories (e.g., 1/, 2/)
                patterns = [
                    "**/golden_values*.json",
                    "**/generations*.json",
                    "**/test_results*.json",
                ]

                for pattern in patterns:
                    json_files = list(output_dir.glob(pattern))
                    if json_files:
                        # Return the first match
                        logger.debug(f"Found result file: {json_files[0]}")
                        return str(json_files[0])

                # Fallback: any JSON file in subdirectories
                json_files = list(output_dir.glob("**/*.json"))
                if json_files:
                    logger.debug(f"Found result file (fallback): {json_files[0]}")
                    return str(json_files[0])

            logger.debug(f"Output directory not found or empty: {output_path}")
            return None

    logger.debug(f"No output path marker found in {out_file.name}")
    return None


def _find_json_files_directly(results_dir: str) -> List[str]:
    """
    Fallback: search for JSON files directly in the results directory.

    This is used when .out files don't contain the expected markers.
    """
    result_files = []
    results_path = Path(results_dir)

    # Look for golden_values*.json files in subdirectories
    patterns = ["**/golden_values*.json", "**/test_results*.json", "**/*_output.json"]

    for pattern in patterns:
        matches = list(results_path.glob(pattern))
        result_files.extend([str(p) for p in matches])

    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for f in result_files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)

    return unique_files


def load_result_file(filepath: str) -> Optional[Dict[str, Any]]:
    """Load a single result JSON file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        data = json.loads(content)

        # Handle JSONL format (single line)
        if isinstance(data, str):
            data = json.loads(data)

        return data
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to load {filepath}: {e}")
        return None


def _detect_result_format(data: Dict[str, Any]) -> str:
    """
    Detect whether the result file is from a training test or inference test.

    Returns:
        "training" - TensorBoard metrics format: {"metric_name": {"values": {...}}}
        "inference" - Generation output format: {"request_id": {"latency": ..., ...}}
        "unknown" - Unrecognized format
    """
    if not data:
        return "unknown"

    # Check first key's value structure
    first_key = next(iter(data.keys()))
    first_value = data[first_key]

    if isinstance(first_value, dict):
        if 'values' in first_value:
            return "training"
        if 'latency' in first_value or 'generated_text' in first_value:
            return "inference"

    return "unknown"


def _is_valid_numeric(value) -> bool:
    """Check if a value is a valid (non-NaN) numeric value."""
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return False

    if isinstance(value, (int, float)):
        return not math.isnan(value)

    return False


def _to_float(value) -> Optional[float]:
    """Convert value to float, returning None for invalid/NaN values."""
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return None

    if isinstance(value, (int, float)):
        if math.isnan(value):
            return None
        return float(value)

    return None


def _aggregate_training_results(
    data: Dict[str, Any], aggregated: Dict[str, Dict[str, List[float]]], run_index: int
) -> None:
    """Aggregate results from training test format."""
    for metric_name, metric_data in data.items():
        if not isinstance(metric_data, dict) or 'values' not in metric_data:
            continue

        if metric_name not in aggregated:
            aggregated[metric_name] = {}

        values = metric_data['values']
        for step, value in values.items():
            # Skip non-numeric or NaN values
            float_val = _to_float(value)
            if float_val is None:
                continue

            if step not in aggregated[metric_name]:
                aggregated[metric_name][step] = []

            aggregated[metric_name][step].append(float_val)

        # For metrics that use median-based comparison in the test (iteration-time,
        # mem-allocated-bytes, mem-max-allocated-bytes), also store all values from
        # this run so we can compute per-run medians later.
        # IMPORTANT: Store values in step order to match the test's index-based slicing.
        if metric_name in ['iteration-time', 'mem-allocated-bytes', 'mem-max-allocated-bytes']:
            all_values_key = f"_all_values_run_{run_index}"
            if all_values_key not in aggregated[metric_name]:
                aggregated[metric_name][all_values_key] = []

            # Sort by step number to ensure consistent ordering for index-based slicing
            sorted_steps = sorted(
                values.keys(), key=lambda x: int(x) if x.isdigit() else float('inf')
            )
            for step in sorted_steps:
                float_val = _to_float(values[step])
                if float_val is None:
                    continue
                aggregated[metric_name][all_values_key].append(
                    float_val
                )  # Just the value, not tuple


def _aggregate_inference_results(
    data: Dict[str, Any], aggregated: Dict[str, Dict[str, List[float]]], run_index: int
) -> None:
    """
    Aggregate results from inference test format.

    Extracts metrics like latency, step_count, and logprob statistics
    from generation outputs.
    """
    # Metrics to extract per request
    latencies = []
    step_counts = []
    prompt_logprob_means = []
    generated_logprob_means = []

    for request_id, request_data in data.items():
        if not isinstance(request_data, dict):
            continue

        # Extract latency
        if 'latency' in request_data:
            latencies.append(float(request_data['latency']))

        # Extract step count
        if 'step_count' in request_data:
            step_counts.append(float(request_data['step_count']))

        # Extract mean of prompt logprobs (as a consistency metric)
        if 'prompt_logprobs' in request_data and request_data['prompt_logprobs']:
            logprobs = request_data['prompt_logprobs']
            if isinstance(logprobs, list) and len(logprobs) > 0:
                prompt_logprob_means.append(sum(logprobs) / len(logprobs))

        # Extract mean of generated logprobs
        if 'generated_log_probs' in request_data and request_data['generated_log_probs']:
            logprobs = request_data['generated_log_probs']
            if isinstance(logprobs, list) and len(logprobs) > 0:
                generated_logprob_means.append(sum(logprobs) / len(logprobs))

    # Store aggregated metrics using run_index as the "step"
    run_key = str(run_index)

    if latencies:
        if 'latency' not in aggregated:
            aggregated['latency'] = {}
        if 'mean' not in aggregated['latency']:
            aggregated['latency']['mean'] = []
        aggregated['latency']['mean'].append(sum(latencies) / len(latencies))

        if 'total' not in aggregated['latency']:
            aggregated['latency']['total'] = []
        aggregated['latency']['total'].append(sum(latencies))

    if step_counts:
        if 'step_count' not in aggregated:
            aggregated['step_count'] = {}
        if 'mean' not in aggregated['step_count']:
            aggregated['step_count']['mean'] = []
        aggregated['step_count']['mean'].append(sum(step_counts) / len(step_counts))

    if prompt_logprob_means:
        if 'prompt_logprob_mean' not in aggregated:
            aggregated['prompt_logprob_mean'] = {}
        if 'mean' not in aggregated['prompt_logprob_mean']:
            aggregated['prompt_logprob_mean']['mean'] = []
        aggregated['prompt_logprob_mean']['mean'].append(
            sum(prompt_logprob_means) / len(prompt_logprob_means)
        )

    if generated_logprob_means:
        if 'generated_logprob_mean' not in aggregated:
            aggregated['generated_logprob_mean'] = {}
        if 'mean' not in aggregated['generated_logprob_mean']:
            aggregated['generated_logprob_mean']['mean'] = []
        aggregated['generated_logprob_mean']['mean'].append(
            sum(generated_logprob_means) / len(generated_logprob_means)
        )


def aggregate_results(result_files: List[str]) -> Dict[str, Dict[str, List[float]]]:
    """
    Aggregate results from multiple JSON files.

    Supports both training test format (TensorBoard metrics) and
    inference test format (generation outputs).

    Returns:
        Dict mapping metric_name -> step/key -> list of values across all runs
    """
    aggregated: Dict[str, Dict[str, List[float]]] = {}
    loaded_count = 0
    detected_format = None

    for idx, filepath in enumerate(result_files):
        data = load_result_file(filepath)
        if data is None:
            continue

        loaded_count += 1

        # Detect format from first file
        file_format = _detect_result_format(data)
        if detected_format is None:
            detected_format = file_format
            logger.info(f"Detected result format: {file_format}")

        if file_format == "training":
            _aggregate_training_results(data, aggregated, idx)
        elif file_format == "inference":
            _aggregate_inference_results(data, aggregated, idx)
        else:
            logger.warning(f"Unknown format in {filepath}, skipping")

    logger.info(f"Successfully loaded {loaded_count} of {len(result_files)} result files")
    return aggregated


def compute_statistics(aggregated: Dict[str, Dict[str, List[float]]]) -> Dict[str, Any]:
    """
    Compute statistics for each metric at each step.

    Returns:
        Dict with structure:
        {
            "metric_name": {
                "num_samples": N,
                "values": {
                    "step": {
                        "min": ...,
                        "max": ...,
                        "mean": ...,
                        "std": ...,
                        "samples": [...]  # original values
                    }
                }
            }
        }
    """
    stats: Dict[str, Any] = {}

    for metric_name, step_values in aggregated.items():
        # Determine number of samples (should be consistent across steps)
        # Skip internal keys used for median calculations
        regular_steps = {k: v for k, v in step_values.items() if not k.startswith("_")}
        sample_counts = [len(vals) for vals in regular_steps.values()]
        num_samples = max(sample_counts) if sample_counts else 0

        metric_stats = {"num_samples": num_samples, "values": {}}

        for step, values in regular_steps.items():
            if len(values) == 0:
                continue

            step_stats = {
                "min": min(values),
                "max": max(values),
                "mean": mean(values),
                "std": stdev(values) if len(values) > 1 else 0.0,
                "count": len(values),
            }

            # Include original samples for debugging
            step_stats["samples"] = values

            metric_stats["values"][step] = step_stats

        stats[metric_name] = metric_stats

    return stats


def compute_recommended_tolerances(
    stats: Dict[str, Any],
    aggregated: Dict[str, Dict[str, List[float]]],
    confidence_multiplier: float = 3.0,
    start_step: int = 1,
) -> Dict[str, Dict[str, float]]:
    """
    Compute recommended tolerances for each metric based on observed variance.

    For metrics that use median-based comparison in the test (iteration-time,
    mem-allocated-bytes, mem-max-allocated-bytes), computes variance of per-run
    medians rather than per-step variance.

    Args:
        stats: Output from compute_statistics()
        aggregated: Raw aggregated data (needed for median calculations)
        confidence_multiplier: Number of standard deviations for bounds (default 3.0 for ~99.7% coverage)
        start_step: First step to include in tolerance calculation (skips warmup steps)

    Returns:
        Dict mapping metric_name -> {
            "relative_tolerance": recommended relative tolerance,
            "absolute_tolerance": recommended absolute tolerance (for near-zero values),
            "max_observed_relative_variance": max(|value - mean| / |mean|) across all samples
        }
    """
    tolerances = {}

    # Metrics that use median-based comparison in the test (iteration-time)
    median_based_metrics = ['iteration-time']
    # Metrics that use max-based comparison in the test (memory)
    max_based_metrics = ['mem-allocated-bytes', 'mem-max-allocated-bytes']

    for metric_name, metric_data in stats.items():
        max_relative_variance = 0.0
        max_absolute_variance = 0.0
        steps_included = 0

        # For median-based metrics, compute variance of per-run medians
        if metric_name in median_based_metrics and metric_name in aggregated:
            run_medians = []

            # Find all run data keys
            for key in aggregated[metric_name].keys():
                if key.startswith("_all_values_run_"):
                    run_data = aggregated[metric_name][key]
                    # Use index-based slicing to match test behavior:
                    # [start_step:] skips the first `start_step` items
                    filtered_values = run_data[start_step:]

                    if filtered_values:
                        run_median = median(filtered_values)
                        run_medians.append(run_median)

            if run_medians:
                median_mean = mean(run_medians)

                # Compute relative variance of medians
                if abs(median_mean) > 1e-9:
                    for m in run_medians:
                        rel_var = abs(m - median_mean) / abs(median_mean)
                        max_relative_variance = max(max_relative_variance, rel_var)
                else:
                    for m in run_medians:
                        max_absolute_variance = max(max_absolute_variance, abs(m))

                steps_included = len(run_medians)

                logger.debug(
                    f"{metric_name}: computed variance from {len(run_medians)} run medians, "
                    f"mean={median_mean:.4f}, max_rel_var={max_relative_variance:.4%}"
                )

        # For max-based metrics (memory), compute variance of per-run max values
        elif metric_name in max_based_metrics and metric_name in aggregated:
            run_maxes = []

            # Find all run data keys
            for key in aggregated[metric_name].keys():
                if key.startswith("_all_values_run_"):
                    run_data = aggregated[metric_name][key]
                    # Skip first value (warmup), take max of rest
                    filtered_values = run_data[1:] if len(run_data) > 1 else run_data

                    if filtered_values:
                        run_max = max(filtered_values)
                        run_maxes.append(run_max)

            if run_maxes:
                max_mean = mean(run_maxes)

                # Compute relative variance of max values
                if abs(max_mean) > 1e-9:
                    for m in run_maxes:
                        rel_var = abs(m - max_mean) / abs(max_mean)
                        max_relative_variance = max(max_relative_variance, rel_var)
                else:
                    for m in run_maxes:
                        max_absolute_variance = max(max_absolute_variance, abs(m))

                steps_included = len(run_maxes)

                logger.debug(
                    f"{metric_name}: computed variance from {len(run_maxes)} run maxes, "
                    f"mean={max_mean:.4f}, max_rel_var={max_relative_variance:.4%}"
                )
        else:
            # Standard per-step variance calculation for other metrics
            for step, step_stats in metric_data["values"].items():
                # Skip warmup steps - try to parse step as int, skip if < start_step
                try:
                    step_num = int(step)
                    if step_num < start_step:
                        continue
                except (ValueError, TypeError):
                    # Non-numeric step key (e.g., "mean" for inference metrics) - include it
                    pass

                steps_included += 1
                mean_val = step_stats["mean"]

                # Compute observed relative variance
                if abs(mean_val) > 1e-9:
                    # For non-zero means, compute relative variance
                    for sample in step_stats["samples"]:
                        rel_var = abs(sample - mean_val) / abs(mean_val)
                        max_relative_variance = max(max_relative_variance, rel_var)
                else:
                    # For near-zero means, track absolute variance
                    for sample in step_stats["samples"]:
                        max_absolute_variance = max(max_absolute_variance, abs(sample))

        # Recommend tolerance with safety margin
        # Use observed variance * confidence_multiplier, with a minimum of 0.1%
        recommended_relative = max(max_relative_variance * confidence_multiplier, 0.001)

        # Round to reasonable precision
        recommended_relative = round(recommended_relative, 4)

        tolerances[metric_name] = {
            "relative_tolerance": recommended_relative,
            "absolute_tolerance": max(max_absolute_variance * confidence_multiplier, 1e-6),
            "max_observed_relative_variance": round(max_relative_variance, 6),
            "max_observed_absolute_variance": round(max_absolute_variance, 6),
            "steps_included": steps_included,
        }

    return tolerances


def format_summary(stats: Dict[str, Any], tolerances: Dict[str, Dict[str, float]]) -> str:
    """Format a human-readable summary of the statistics."""
    lines = []
    lines.append("=" * 70)
    lines.append("Golden Values Statistics Summary")
    lines.append("=" * 70)

    for metric_name in sorted(stats.keys()):
        metric_data = stats[metric_name]
        tol = tolerances.get(metric_name, {})

        lines.append(f"\n{metric_name}:")
        lines.append(f"  Samples: {metric_data['num_samples']}")
        lines.append(f"  Steps: {len(metric_data['values'])}")

        if tol:
            lines.append(
                f"  Max observed relative variance: {tol.get('max_observed_relative_variance', 'N/A'):.4%}"
            )
            lines.append(
                f"  Recommended relative tolerance: {tol.get('relative_tolerance', 'N/A'):.2%}"
            )
            lines.append(
                f"  Recommended absolute tolerance: {tol.get('absolute_tolerance', 'N/A'):.2e}"
            )

        # Show a few example steps
        values = metric_data["values"]
        example_steps = list(values.keys())[:3]
        if example_steps:
            lines.append("  Example steps:")
            for step in example_steps:
                s = values[step]
                lines.append(
                    f"    Step {step}: mean={s['mean']:.6g}, std={s['std']:.6g}, "
                    f"range=[{s['min']:.6g}, {s['max']:.6g}]"
                )

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compute statistical bounds for golden values from multiple test runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--results-dir",
        type=str,
        help="Directory containing batch test results (searches for JSON files)",
    )
    input_group.add_argument(
        "--result-files",
        type=str,
        nargs="+",
        help="Explicit list of result JSON files to aggregate",
    )

    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Output path for statistics JSON file"
    )

    parser.add_argument(
        "--recommend-tolerances",
        action="store_true",
        help="Compute and display recommended tolerances based on observed variance",
    )

    parser.add_argument(
        "--confidence-multiplier",
        type=float,
        default=1.5,
        help="Multiplier for observed max variance when computing recommended tolerance. "
        "Example: if max observed variance is 5%% and multiplier is 1.5, recommended tolerance is 7.5%%. "
        "Use higher values (2-3) for more safety margin. Default: 1.5",
    )

    parser.add_argument(
        "--min-samples",
        type=int,
        default=2,
        help="Minimum number of samples required to compute statistics (default: 2)",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    parser.add_argument(
        "--workspace-root",
        type=str,
        default=None,
        help="Root of the megatron workspace (where runs/ directory is located). "
        "Defaults to current working directory.",
    )

    parser.add_argument(
        "--start-step",
        type=int,
        default=0,
        help="Number of initial steps to skip (index-based, matching test behavior). "
        "Uses Python slicing [start_step:] so --start-step 10 skips first 10 items. "
        "Default: 0 (include all). Set to match THROUGHPUT_TEST_PARAMS.--start_step from model_config.yaml.",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Find or use result files
    if args.results_dir:
        result_files = find_result_json_files(args.results_dir, args.workspace_root)
        if not result_files:
            logger.error(f"No result JSON files found in {args.results_dir}")
            logger.info("Make sure the batch tests have completed and results are available.")
            logger.info(
                "The script looks for .out files and parses them to find the result JSON paths."
            )
            logger.info(
                "Each .out file should contain: 'This test wrote results into /opt/megatron-lm/runs/<uuid>'"
            )
            sys.exit(1)
        logger.info(f"Found {len(result_files)} result files from {args.results_dir}")
    else:
        result_files = args.result_files
        # Verify files exist
        for f in result_files:
            if not os.path.exists(f):
                logger.error(f"Result file not found: {f}")
                sys.exit(1)

    if args.verbose:
        for f in result_files:
            logger.debug(f"  - {f}")

    # Aggregate results
    aggregated = aggregate_results(result_files)

    if not aggregated:
        logger.error("No valid results found to aggregate")
        sys.exit(1)

    # Check minimum samples
    for metric_name, step_values in aggregated.items():
        for step, values in step_values.items():
            if len(values) < args.min_samples:
                logger.warning(
                    f"{metric_name} step {step}: only {len(values)} samples "
                    f"(minimum {args.min_samples} recommended)"
                )

    # Compute statistics
    stats = compute_statistics(aggregated)

    # Compute recommended tolerances (excluding warmup steps)
    if args.start_step > 1:
        logger.info(f"Excluding steps < {args.start_step} from tolerance calculation (warmup)")
    tolerances = compute_recommended_tolerances(
        stats, aggregated, args.confidence_multiplier, start_step=args.start_step
    )

    # Build output
    output = {
        "metadata": {
            "num_runs": len(result_files),
            "result_files": result_files,
            "confidence_multiplier": args.confidence_multiplier,
            "start_step": args.start_step,
        },
        "statistics": stats,
        "recommended_tolerances": tolerances,
    }

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Statistics written to {args.output}")

    # Print summary
    if args.recommend_tolerances or args.verbose:
        print(format_summary(stats, tolerances))

        print("\nRecommended tolerance settings:")
        print("-" * 50)
        # Training test metrics
        training_metrics = [
            "lm-loss",
            "lm loss",
            "iteration-time",
            "mem-allocated-bytes",
            "mem-max-allocated-bytes",
        ]
        # Inference test metrics
        inference_metrics = [
            "latency",
            "step_count",
            "prompt_logprob_mean",
            "generated_logprob_mean",
        ]

        for metric_name in training_metrics + inference_metrics:
            if metric_name in tolerances:
                tol = tolerances[metric_name]
                var_name = metric_name.upper().replace('-', '_').replace(' ', '_')
                print(
                    f"{var_name}_RELATIVE_TOLERANCE = "
                    f"{tol['relative_tolerance']}  # {tol['relative_tolerance']:.2%}"
                )
                print(f"{var_name}_ABSOLUTE_TOLERANCE = " f"{tol['absolute_tolerance']:.2e}")


if __name__ == "__main__":
    main()
