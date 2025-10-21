import enum
import glob
import json
import logging
import os
import pathlib
from typing import Callable, Dict, List, Optional, Union

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


class MissingTensorboardLogsError(Exception):
    """Raised if TensorboardLogs not found"""


class UndefinedMetricError(Exception):
    """Raised of golden values metric has no test definition"""


class SkipMetricError(Exception):
    """Raised if metric shall be skipped"""


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
    files = glob.glob(f"{path}/events*tfevents*")
    files += glob.glob(f"{path}/results/events*tfevents*")

    if not files:
        logger.error(f"File not found matching: {path}/events* || {path}/results/events*")
        return None

    files.sort(key=lambda x: os.path.getmtime(os.path.join(path, pathlib.Path(x).name)))
    accumulators = []

    if index == -1:
        for event_file in files:
            ea = event_accumulator.EventAccumulator(event_file, size_guidance=SIZE_GUIDANCE)
            ea.Reload()
            accumulators.append(ea)
    else:
        event_file = files[index]
        ea = event_accumulator.EventAccumulator(event_file, size_guidance=SIZE_GUIDANCE)
        ea.Reload()
        accumulators.append(ea)

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
        # Add missing values
        values = {
            k: (values[k] if k in values else "nan")
            for k in range(1, train_iters + 1)
            if k == start_idx or (k > start_idx and int(k) % step_size == 0)
        }

        golden_values[metric] = GoldenValueMetric(
            start_step=min(values.keys()),
            end_step=max(values.keys()),
            step_interval=step_size,
            values=values,
        )

    return golden_values


def read_golden_values_from_json(
    golden_values_path: Union[str, pathlib.Path]
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
                golden_value_list = list(golden_value.values.values())
                actual_value_list = [
                    value
                    for value_step, value in actual_values[metric_name].values.items()
                    if value_step in golden_value.values.keys()
                ]

                if metric_name == "iteration-time":
                    actual_value_list = [
                        np.median([np.inf if type(v) is str else v for v in actual_value_list])
                    ]
                    golden_value_list = [
                        np.median([np.inf if type(v) is str else v for v in golden_value_list])
                    ]
                    total_steps_evaluated = 1
                else:
                    total_steps_evaluated = golden_value.end_step / golden_value.step_interval + 1

                    actual_value_list = [np.inf if type(v) is str else v for v in actual_value_list]
                    golden_value_list = [np.inf if type(v) is str else v for v in golden_value_list]

                actual = np.array(actual_value_list)
                golden = np.array(golden_value_list)

                # Tolerance check
                is_close = np.isclose(actual, golden, rtol=test.rtol, atol=test.atol)

                num_failing_steps_allowed = min(max(total_steps_evaluated // 100, 1), 50)
                passing = np.mean(is_close) >= (num_failing_steps_allowed / total_steps_evaluated)

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
