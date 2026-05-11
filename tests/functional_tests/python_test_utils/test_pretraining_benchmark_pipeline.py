# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _is_metric_key(key: str) -> bool:
    return ":" in key


def _resolve_metric_config(golden_values: Dict) -> Dict:
    """Support flat or platform/device-classified benchmark gold values."""
    current = golden_values
    while current and not any(_is_metric_key(key) for key in current.keys()):
        current = current[next(iter(current))]
    return current


def _extract_metrics_from_lines(lines: Iterable[str], metric_keys: List[str]) -> Dict[str, List[float]]:
    results = {key: [] for key in metric_keys}

    for line in lines:
        if "iteration" not in line:
            continue

        for part in line.split("|"):
            part = part.strip()
            for key in metric_keys:
                if part.startswith(key.rstrip(":")):
                    match = re.search(r":\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)", part)
                    if match:
                        results[key].append(float(match.group(1)))

    return results


def _read_benchmark_metrics(logs_dir: str, metric_keys: List[str]) -> Dict[str, List[float]]:
    stdout_logs = sorted(Path(logs_dir).glob("**/stdout.log"))
    assert stdout_logs, f"No stdout.log files found under {logs_dir}"

    selected_metrics = None
    selected_count = -1
    selected_log = None

    for log_path in stdout_logs:
        with log_path.open(errors="replace") as f:
            metrics = _extract_metrics_from_lines(f, metric_keys)

        count = max((len(values) for values in metrics.values()), default=0)
        if count > selected_count:
            selected_metrics = metrics
            selected_count = count
            selected_log = log_path

    logger.info("Using benchmark log: %s", selected_log)
    assert selected_metrics is not None and selected_count > 0, (
        f"No benchmark metrics found in stdout logs under {logs_dir}"
    )
    return selected_metrics


def _stable_average(values: List[float], warmup_steps: int) -> float:
    stable_values = values[warmup_steps:] if len(values) > warmup_steps else values
    assert stable_values, "No benchmark values left after warmup filtering"
    return float(np.mean(stable_values))


def test_pretraining_benchmark_pipeline(golden_values_path: str, logs_dir: str):
    with open(golden_values_path) as f:
        metric_config = _resolve_metric_config(json.load(f))

    metric_keys = list(metric_config.keys())
    actual_metrics = _read_benchmark_metrics(logs_dir, metric_keys)

    failed_metrics = []

    for metric_name in metric_keys:
        golden_entry = metric_config[metric_name]
        golden_values = golden_entry.get("values", [])
        threshold = golden_entry.get("threshold", {})
        threshold_type = threshold.get("type", "upper_bound")
        tolerance = threshold.get("tolerance", 0.1)
        actual_values = actual_metrics.get(metric_name, [])

        assert actual_values, f"No values extracted for benchmark metric {metric_name}"
        assert golden_values, f"No baseline values configured for benchmark metric {metric_name}"

        # Match FlagScale benchmark behavior: ignore early warmup values before comparing averages.
        warmup_steps = min(5, len(golden_values))
        actual_avg = _stable_average(actual_values, warmup_steps)
        golden_avg = _stable_average(golden_values, warmup_steps)

        logger.info("Metric: %s", metric_name)
        logger.info("Actual values: %s", actual_values)
        logger.info("Golden values: %s", golden_values)
        logger.info("Actual average after warmup: %.4f", actual_avg)
        logger.info("Golden average after warmup: %.4f", golden_avg)

        if threshold_type == "upper_bound":
            limit = golden_avg * (1 + tolerance)
            passed = actual_avg <= limit
            logger.info("Upper limit: %.4f", limit)
        elif threshold_type == "lower_bound":
            limit = golden_avg * (1 - tolerance)
            passed = actual_avg >= limit
            logger.info("Lower limit: %.4f", limit)
        else:
            raise ValueError(f"Unsupported benchmark threshold type: {threshold_type}")

        logger.info("Benchmark check for %s: %s", metric_name, "PASSED" if passed else "FAILED")
        if not passed:
            failed_metrics.append(metric_name)

    assert not failed_metrics, f"Benchmark regression detected for: {', '.join(failed_metrics)}"
