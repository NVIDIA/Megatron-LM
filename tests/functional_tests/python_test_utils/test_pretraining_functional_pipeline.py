# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math

import yaml

from tests.functional_tests.python_test_utils.common import GoldenValueMetric


def test_functional_pipeline(
    actual_values_first_run: dict[str, GoldenValueMetric],
    actual_values_second_run: dict[str, GoldenValueMetric],
    train_iters: int,
    model_config_path: str,
) -> None:
    """Require complete, finite metrics from training and checkpoint resume."""
    with open(model_config_path) as config_file:
        model_config = yaml.safe_load(config_file)

    metrics = model_config.get("METRICS")
    assert isinstance(metrics, list) and metrics, "METRICS must be a nonempty list"
    assert train_iters > 0, "train_iters must be positive"

    for run_name, actual_values, start_step in (
        ("first run", actual_values_first_run, 1),
        ("second run", actual_values_second_run, train_iters // 2 + 1),
    ):
        expected_steps = set(range(start_step, train_iters + 1))
        for metric in metrics:
            assert metric in actual_values, f"{run_name}: missing metric {metric!r}"
            values = actual_values[metric].values
            observed_steps = set(values)
            assert observed_steps == expected_steps, (
                f"{run_name}: {metric!r} has incomplete or unexpected steps; "
                f"missing={sorted(expected_steps - observed_steps)}, "
                f"unexpected={sorted(observed_steps - expected_steps)}"
            )
            for step, value in values.items():
                assert (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                ), f"{run_name}: {metric!r} at step {step} must be finite and numeric, got {value!r}"
