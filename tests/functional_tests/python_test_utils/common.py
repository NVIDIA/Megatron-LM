# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import enum
import glob
import json
import logging
import math
import os
import pathlib
from typing import Dict, List, Optional, Union

import numpy as np
import pydantic
from tensorboard.backend.event_processing import event_accumulator

# By default TB tries to be smart about what to load in memory to avoid OOM
# Since we expect every step to be there when we do our comparisons, we explicitly
# set the size guidance to 0 so that we load everything. It's okay given our tests
# are small/short.
SIZE_GUIDANCE = {event_accumulator.TENSORS: 0, event_accumulator.SCALARS: 0}

logger = logging.getLogger(__name__)


class TypeOfTestResult(enum.Enum):
    APPROXIMATE = 1
    DETERMINISTIC = 2


class Test(pydantic.BaseModel):
    pass


class NotApproximateError(Exception):
    """Raised if comparison is not within approximate bounds"""


class NotDeterminsticError(Exception):
    """Raised if comparison is not within approximate bounds"""


class ApproximateTest(Test):
    atol: Union[int, float] = 0
    rtol: float = 1e-5

    @property
    def type_of_test_result(self) -> TypeOfTestResult:
        return TypeOfTestResult.APPROXIMATE

    def error_message(self, metric_name: str) -> NotApproximateError:
        return NotApproximateError(f"Approximate comparison of {metric_name}: FAILED")


class DeterministicTest(Test):
    @property
    def rtol(self) -> float:
        return 0.0

    @property
    def atol(self) -> Union[int, float]:
        return 0

    @property
    def type_of_test_result(self) -> TypeOfTestResult:
        return TypeOfTestResult.DETERMINISTIC

    def error_message(self, metric_name: str) -> NotDeterminsticError:
        return NotDeterminsticError(f"Exact comparison of {metric_name}: FAILED")


class GoldenValueMetric(pydantic.BaseModel):
    start_step: int
    end_step: int
    step_interval: int
    values: Dict[int, Union[int, float, str]]

    def __repr__(self):
        return f"Values ({self.start_step},{self.end_step},{self.step_interval}): {', '.join([str(f'({step}, {value})') for step, value in self.values.items()])}"


class GoldenValues(pydantic.RootModel):
    root: Dict[str, GoldenValueMetric]


def _infer_step_interval(steps: List[int], start_idx: int, default: int) -> int:
    """Infer the cadence of observed samples, ignoring the special start sample."""
    cadence_steps = sorted(step for step in steps if step != start_idx)
    if len(cadence_steps) < 2:
        return default

    return math.gcd(
        *(
            current_step - previous_step
            for previous_step, current_step in zip(cadence_steps, cadence_steps[1:])
        )
    )


class MissingTensorboardLogsError(Exception):
    """Raised if TensorboardLogs not found"""


class UndefinedMetricError(Exception):
    """Raised of golden values metric has no test definition"""


class SkipMetricError(Exception):
    """Raised if metric shall be skipped"""


def _load_event_accumulators_with_scalars(
    files: List[str],
) -> List[event_accumulator.EventAccumulator]:
    """Loads event-file accumulators that contain scalar data, preserving order.

    A resumed training phase can emit a header-only TensorBoard event file with
    zero scalars before the file that holds the actual metrics (for example when
    the fault-tolerance launcher initializes a SummaryWriter ahead of logging).
    Dropping scalar-less files keeps positional ``index`` selection aligned with
    real run data instead of latching onto an empty file and yielding no metrics.

    Args:
        files: Event-file paths, ordered oldest-first.

    Returns:
        Reloaded accumulators that expose at least one scalar tag, in input order.
    """
    accumulators = []
    for event_file in files:
        ea = event_accumulator.EventAccumulator(event_file, size_guidance=SIZE_GUIDANCE)
        ea.Reload()
        if ea.Tags()["scalars"]:
            accumulators.append(ea)
    return accumulators


def read_tb_logs_as_list(
    path, index: int = 0, train_iters: int = 50, start_idx: int = 1, step_size: int = 5
) -> Optional[Dict[str, GoldenValueMetric]]:
    """Reads a TensorBoard Events file from the input path, and returns the
    summary specified as input as a list.

    Args:
        path: str, path to the dir where the events file is located.
        summary_name: str, name of the summary to read from the TB logs.

    Returns:
        summary_list: list, the values in the read summary list, formatted as a list.
    """
    if step_size <= 0:
        raise ValueError(f"step_size must be positive, got {step_size}")

    files = glob.glob(f"{path}/events*tfevents*")
    files += glob.glob(f"{path}/results/events*tfevents*")

    if not files:
        logger.error(f"File not found matching: {path}/events* || {path}/results/events*")
        return None

    files.sort(key=lambda x: os.path.getmtime(os.path.join(path, pathlib.Path(x).name)))

    accumulators = _load_event_accumulators_with_scalars(files)

    if not accumulators:
        logger.error(f"No event file with scalar data found at: {path}")
        return None

    if index != -1:
        if index >= len(accumulators):
            logger.error(
                f"Requested event-file index {index} but only {len(accumulators)} "
                f"event file(s) with scalar data found at: {path}"
            )
            return None
        accumulators = [accumulators[index]]

    summaries = {}
    for ea in accumulators:
        for scalar_name in ea.Tags()["scalars"]:
            if scalar_name in summaries:
                for x in ea.Scalars(scalar_name):
                    if x.step not in summaries[scalar_name]:
                        summaries[scalar_name][x.step] = round(x.value, 5)

            else:
                summaries[scalar_name] = {
                    x.step: round(x.value, 5) for x in ea.Scalars(scalar_name)
                }

    golden_values = {}

    for metric, values in summaries.items():
        values = {
            step: values[step]
            for step in sorted(values)
            if 1 <= step <= train_iters
            and (step == start_idx or (step > start_idx and step % step_size == 0))
        }
        if not values:
            logger.warning(
                "Skipping metric %r because it has no observed values "
                "on the requested sampling grid",
                metric,
            )
            continue

        steps = list(values)

        golden_values[metric] = GoldenValueMetric(
            start_step=steps[0],
            end_step=steps[-1],
            step_interval=_infer_step_interval(steps, start_idx, step_size),
            values=values,
        )

    return golden_values


def read_golden_values_from_json(
    golden_values_path: Union[str, pathlib.Path],
) -> Dict[str, GoldenValueMetric]:
    with open(golden_values_path) as f:
        if os.path.exists(golden_values_path):
            with open(golden_values_path) as f:
                return GoldenValues(**json.load(f)).root

        raise ValueError(f"File {golden_values_path} not found!")


def _filter_checks(
    checks: List[Union[ApproximateTest, DeterministicTest]], filter_for_type_of_check
):
    return [test for test in checks if test.type_of_test_result == filter_for_type_of_check]


def pipeline(
    compare_approximate_results: bool,
    golden_values: Dict[str, GoldenValueMetric],
    actual_values: Dict[str, GoldenValueMetric],
    checks: Dict[str, List[Union[ApproximateTest, DeterministicTest]]],
):
    all_test_passed = True
    failed_metrics = []

    for metric_name, metric_thresholds in checks.items():
        if metric_name not in list(actual_values.keys()):
            raise MissingTensorboardLogsError(
                f"Metric {metric_name} not found in Tensorboard logs! Please modify `model_config.yaml` to record it."
            )

        for test in metric_thresholds:
            if (
                compare_approximate_results
                and test.type_of_test_result == TypeOfTestResult.DETERMINISTIC
            ):
                continue

            try:
                golden_value = golden_values[metric_name]
                if not golden_value.values:
                    raise MissingTensorboardLogsError(
                        f"Metric {metric_name} has no values in the golden file."
                    )

                golden_value_list = list(golden_value.values.values())
                actual_value_list = [
                    actual_values[metric_name].values.get(value_step, "nan")
                    for value_step in golden_value.values
                ]

                if metric_name == "iteration-time":
                    finite_golden_steps = [
                        value_step
                        for value_step, value in sorted(golden_value.values.items())
                        if not isinstance(value, str) and math.isfinite(value)
                    ]
                    if not finite_golden_steps:
                        raise MissingTensorboardLogsError(
                            "Metric iteration-time has no finite values."
                        )

                    max_golden_step = finite_golden_steps[-1]
                    steady_window = range(5, 21) if max_golden_step <= 25 else range(30, 46)
                    comparison_steps = [
                        value_step
                        for value_step in finite_golden_steps
                        if value_step in steady_window
                    ]
                    if not comparison_steps:
                        comparison_steps = [
                            value_step
                            for value_step in finite_golden_steps
                            if value_step >= steady_window.start
                        ][:4]
                    if not comparison_steps:
                        raise MissingTensorboardLogsError(
                            "Metric iteration-time has no finite values after its warmup window."
                        )

                    golden_value_list = [
                        golden_value.values[value_step] for value_step in comparison_steps
                    ]
                    actual_value_list = [
                        actual_values[metric_name].values.get(value_step, "nan")
                        for value_step in comparison_steps
                    ]
                    golden_value_list = [
                        np.median([np.inf if isinstance(v, str) else v for v in golden_value_list])
                    ]
                    actual_value_list = [
                        np.median([np.inf if isinstance(v, str) else v for v in actual_value_list])
                    ]
                    total_steps_evaluated = 1
                else:
                    total_steps_evaluated = len(golden_value.values)

                    actual_value_list = [
                        np.inf if isinstance(v, str) else v for v in actual_value_list
                    ]
                    golden_value_list = [
                        np.inf if isinstance(v, str) else v for v in golden_value_list
                    ]

                actual = np.array(actual_value_list)
                golden = np.array(golden_value_list)

                # Tolerance check
                is_close = np.isclose(actual, golden, rtol=test.rtol, atol=test.atol)

                if (
                    test.type_of_test_result == TypeOfTestResult.DETERMINISTIC
                    or total_steps_evaluated == 1
                ):
                    passing = bool(np.all(is_close))
                else:
                    num_failing_steps_allowed = min(max(total_steps_evaluated // 100, 1), 50)
                    passing = np.mean(is_close) >= 1 - (
                        num_failing_steps_allowed / total_steps_evaluated
                    )

                if not passing:
                    logger.info(
                        "Actual values: %s", ", ".join([str(v) for v in (*actual_value_list,)])
                    )
                    logger.info(
                        "Golden values: %s", ", ".join([str(v) for v in (*golden_value_list,)])
                    )
                    raise test.error_message(metric_name)

                result = f"{test.type_of_test_result.name} test for metric {metric_name}: PASSED"
                result_code = 0

            except (NotApproximateError, NotDeterminsticError, MissingTensorboardLogsError) as e:
                result = str(e)
                result_code = 1
            except SkipMetricError:
                logger.info(f"{test.type_of_test_result.name} test for {metric_name}: SKIPPED")
                continue

            log_emitter = logger.info if result_code == 0 else logger.error
            log_emitter(result)
            if result_code == 1:
                all_test_passed = False
                failed_metrics.append(metric_name)

    assert all_test_passed, f"The following metrics failed: {', '.join(failed_metrics)}"
