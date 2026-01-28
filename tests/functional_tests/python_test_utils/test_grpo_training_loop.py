# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
import logging
from statistics import median
from typing import Any, Dict, List, Tuple

import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tolerance settings for all metrics.
# These tolerances account for hardware variance (different GPU silicon,
# driver versions, CUDA/cuDNN differences) while still catching real regressions.
# Tolerances can be tuned using compute_golden_statistics.py to analyze variance
# across multiple runs on different hardware.

# LM Loss tolerances
LM_LOSS_RELATIVE_TOLERANCE = 0.01  # 1% relative tolerance
LM_LOSS_ABSOLUTE_TOLERANCE = 1e-6  # For values near zero

# Iteration time tolerances (performance metric, higher variance expected)
ITERATION_TIME_RELATIVE_TOLERANCE = 0.15  # 15% relative tolerance

# Memory allocation tolerances
MEM_ALLOCATED_BYTES_RELATIVE_TOLERANCE = 0.10  # 10% relative tolerance
MEM_MAX_ALLOCATED_BYTES_RELATIVE_TOLERANCE = 0.10  # 10% relative tolerance


def validate_with_tolerance(
    golden_values: Dict[str, Any],
    current_values: Dict[str, Any],
    relative_tolerance: float,
    absolute_tolerance: float = 1e-9,
    metric_name: str = "metric",
) -> Tuple[bool, List[str]]:
    """
    Validate that current values are within tolerance of golden values.

    Args:
        golden_values: Dict mapping step -> expected value
        current_values: Dict mapping step -> actual value
        relative_tolerance: Maximum allowed relative difference (e.g., 0.01 for 1%)
        absolute_tolerance: Tolerance for values near zero
        metric_name: Name of metric for error messages

    Returns:
        Tuple of (passed: bool, mismatches: List[str])
    """
    mismatches = []

    for step, golden_val in golden_values.items():
        if step not in current_values:
            mismatches.append(f"Step {step}: missing in current run")
            continue

        current_val = current_values[step]

        # Handle the case where golden value is zero or near-zero
        if golden_val == 0 or abs(golden_val) < absolute_tolerance:
            if abs(current_val) > absolute_tolerance:
                mismatches.append(f"Step {step}: expected ~0, got {current_val}")
        else:
            # Calculate relative difference
            rel_diff = abs(current_val - golden_val) / abs(golden_val)
            if rel_diff > relative_tolerance:
                mismatches.append(
                    f"Step {step}: {current_val} differs from golden {golden_val} "
                    f"by {rel_diff:.4%} (tolerance: {relative_tolerance:.2%})"
                )

    # Check for extra steps in current that aren't in golden
    extra_steps = set(current_values.keys()) - set(golden_values.keys())
    if extra_steps:
        logger.info(f"{metric_name}: Ignoring extra steps in current run: {extra_steps}")

    return len(mismatches) == 0, mismatches


def test_grpo_training_loop(
    golden_values_path: str, test_values_path: str, model_config_path: str
) -> None:
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)
        metrics = model_config["METRICS"]
        if "THROUGHPUT_TEST_PARAMS" in model_config:
            throughput_test_params = model_config["THROUGHPUT_TEST_PARAMS"]
            start_step = throughput_test_params["--start_step"]
        else:
            start_step = 1

    with open(golden_values_path, 'r') as f1, open(test_values_path, 'r') as f2:
        golden_values_content = f1.read()
        tensorboard_content = f2.read()

    output_groundtruth = json.loads(golden_values_content)

    if isinstance(output_groundtruth, str):
        # Handle JSONL output, assume only one line in this case.
        output_groundtruth = json.loads(output_groundtruth)

    output_current = json.loads(tensorboard_content)
    if isinstance(output_current, str):
        # Handle JSONL output, assume only one line in this case.
        output_current = json.loads(output_current)

    # Allow current run to have extra metrics not in golden values
    # (only compare metrics defined in golden values)
    extra_in_current = set(output_current.keys()) - set(output_groundtruth.keys())
    if extra_in_current:
        logger.info(f"Ignoring extra metrics in current run: {extra_in_current}")

    assert set(output_groundtruth.keys()).issubset(
        set(output_current.keys())
    ), f"Some IDs from groundtruth are missing in current: {output_groundtruth.keys()} vs {output_current.keys()}"
    if set(output_groundtruth.keys()) != set(output_current.keys()):
        logger.warning(
            f"Some IDs from groundtruth are missing in output, only the subset of ids in groundtruth will be tested: {output_groundtruth.keys()} vs {output_current.keys()}"
        )
    assert len(output_groundtruth) > 0, "No test performed for output"

    if "iteration-time" in metrics and "iteration-time" in output_current:

        # First warmup iteration is excluded from iteration-time statistics.
        iteration_time_sampled = median(
            [l for l in output_current["iteration-time"]['values'].values()][start_step:]
        )
        iteration_time_golden = median(
            [l for l in output_groundtruth["iteration-time"]['values'].values()][start_step:]
        )

        lower_bound = (1 - ITERATION_TIME_RELATIVE_TOLERANCE) * iteration_time_golden
        upper_bound = (1 + ITERATION_TIME_RELATIVE_TOLERANCE) * iteration_time_golden
        assert lower_bound <= iteration_time_sampled <= upper_bound, (
            f"Iteration time {iteration_time_sampled} ms not within "
            f"{ITERATION_TIME_RELATIVE_TOLERANCE:.0%} of golden value ~{iteration_time_golden} ms. "
            f"Sampled: {output_current['iteration-time']} ms. "
            f"Please update golden values in the functional tests if this is expected."
        )

        output_groundtruth.pop('iteration-time')

    if "lm-loss" in metrics and "lm-loss" in output_current:

        # Validate lm-loss values with tolerance to account for hardware variance.
        # Previously required exact matching, but this caused flaky failures due to
        # floating-point differences across different GPU hardware.
        golden_lm_loss_values = output_groundtruth["lm-loss"]['values']
        current_lm_loss_values = output_current["lm-loss"]['values']

        passed, mismatches = validate_with_tolerance(
            golden_lm_loss_values,
            current_lm_loss_values,
            relative_tolerance=LM_LOSS_RELATIVE_TOLERANCE,
            absolute_tolerance=LM_LOSS_ABSOLUTE_TOLERANCE,
            metric_name="lm-loss",
        )

        if not passed:
            error_msg = (
                f"LM loss values outside tolerance ({LM_LOSS_RELATIVE_TOLERANCE:.1%}):\n"
                + "\n".join(f"  - {m}" for m in mismatches)
                + f"\n\nGolden: {golden_lm_loss_values}\n"
                + f"Current: {current_lm_loss_values}\n"
                + "Please update golden values in the functional tests if this is expected."
            )
            assert False, error_msg

        output_groundtruth.pop('lm-loss')

    if "mem-allocated-bytes" in metrics and "mem-allocated-bytes" in output_current:

        # Use max instead of median - we care about worst-case memory usage
        # Skip first step (warmup) which may have different memory characteristics
        current_values = [l for l in output_current["mem-allocated-bytes"]['values'].values()][1:]
        golden_values = [l for l in output_groundtruth["mem-allocated-bytes"]['values'].values()][
            1:
        ]

        mem_allocated_bytes_sampled = max(current_values)
        mem_allocated_bytes_golden = max(golden_values)

        upper_bound = (1 + MEM_ALLOCATED_BYTES_RELATIVE_TOLERANCE) * mem_allocated_bytes_golden
        assert mem_allocated_bytes_sampled <= upper_bound, (
            f"Max mem allocated bytes {mem_allocated_bytes_sampled} bytes exceeds "
            f"{MEM_ALLOCATED_BYTES_RELATIVE_TOLERANCE:.0%} above golden max {mem_allocated_bytes_golden} bytes. "
            f"Upper bound: {upper_bound} bytes. "
            f"Please update golden values in the functional tests if this is expected."
        )

        output_groundtruth.pop('mem-allocated-bytes')

    if "mem-max-allocated-bytes" in metrics and "mem-max-allocated-bytes" in output_current:

        # Use max - we care that peak memory doesn't exceed the golden peak
        # Skip first step (warmup) which may have different memory characteristics
        current_values = [l for l in output_current["mem-max-allocated-bytes"]['values'].values()][
            1:
        ]
        golden_values = [
            l for l in output_groundtruth["mem-max-allocated-bytes"]['values'].values()
        ][1:]

        mem_max_allocated_bytes_sampled = max(current_values)
        mem_max_allocated_bytes_golden = max(golden_values)

        upper_bound = (
            1 + MEM_MAX_ALLOCATED_BYTES_RELATIVE_TOLERANCE
        ) * mem_max_allocated_bytes_golden
        assert mem_max_allocated_bytes_sampled <= upper_bound, (
            f"Max mem-max-allocated bytes {mem_max_allocated_bytes_sampled} bytes exceeds "
            f"{MEM_MAX_ALLOCATED_BYTES_RELATIVE_TOLERANCE:.0%} above golden max {mem_max_allocated_bytes_golden} bytes. "
            f"Upper bound: {upper_bound} bytes. "
            f"Please update golden values in the functional tests if this is expected."
        )

        output_groundtruth.pop('mem-max-allocated-bytes')
