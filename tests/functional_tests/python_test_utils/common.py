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


def approximate_threshold(rtol: float) -> Callable:
    def _func(y_pred: List[Union[float, int]], y_true: List[Union[float, int]]):
        return np.mean([np.mean(y_pred), np.mean(y_true)]) * rtol

    return _func


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
    atol: Optional[Union[int, float]] = 0
    atol_func: Optional[Callable] = None
    rtol: float = 1e-5

    @property
    def type_of_test_result(self) -> TypeOfTestResult:
        return TypeOfTestResult.APPROXIMATE

    def error_message(self, metric_name: str) -> NotApproximateError:
        return NotApproximateError(f"Approximate comparison of {metric_name}: FAILED")


class DeterministicTest(Test):
    @property
    def atol(self) -> Union[int, float]:
        return 0

    atol_func: Optional[Callable] = None

    @property
    def rtol(self) -> float:
        return 0.0

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

    # for metric_name, golden_value in golden_values.items():
    #     logger.info(
    #         f"Extracted {golden_value.end_step} values of {metric_name} from Tensorboard logs. Here are the sampled values: {golden_value.values}"
    #     )

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
    tensorboard_logs: Dict[str, GoldenValueMetric],
    checks: Dict[str, List[Union[ApproximateTest, DeterministicTest]]],
):

    all_test_passed = True
    failed_metrics = []

    for golden_value_key, golden_value in golden_values.items():

        try:
            if golden_value_key not in list(tensorboard_logs.keys()):
                raise MissingTensorboardLogsError(
                    f"Metric {golden_value_key} not found in Tensorboard logs! Please modify `model_config.yaml` to record it."
                )

            if golden_value_key not in checks or (golden_value_key in checks and len(checks) == 0):
                logger.debug(
                    "For metric `%s`, no check was defined. Will fall back to `DeterminsticTest` with exact thresholds.",
                    golden_value_key,
                )
                test = DeterministicTest()
            else:
                # For approximate tests, we cannot use deterministic
                if compare_approximate_results is True:
                    tests = _filter_checks(checks[golden_value_key], TypeOfTestResult.APPROXIMATE)

                # For deterministic, we can fall back to approximate
                else:
                    tests = _filter_checks(
                        checks[golden_value_key], TypeOfTestResult.DETERMINISTIC
                    ) or _filter_checks(checks[golden_value_key], TypeOfTestResult.APPROXIMATE)

                if len(tests) != 1:
                    raise SkipMetricError(
                        f"No {'approximate' if compare_approximate_results is True else 'deterministic'} check found for {golden_value_key}: SKIPPED"
                    )

                test = tests[0]

            golden_value_list = list(golden_value.values.values())
            actual_value_list = [
                value
                for value_step, value in tensorboard_logs[golden_value_key].values.items()
                if value_step in golden_value.values.keys()
            ]

            if golden_value_key == "iteration-time":
                actual_value_list = actual_value_list[3:-1]
                golden_value_list = golden_value_list[3:-1]
                logger.info(
                    "For metric `%s`, the first 3 and the last scalars are removed from the list to reduce noise.",
                    golden_value_key,
                )

            actual_value_list = [np.inf if type(v) is str else v for v in actual_value_list]
            golden_value_list = [np.inf if type(v) is str else v for v in golden_value_list]

            if not np.allclose(
                actual_value_list,
                golden_value_list,
                rtol=test.rtol,
                atol=(
                    test.atol_func(actual_value_list, golden_value_list)
                    if test.atol_func is not None
                    else test.atol
                ),
            ):
                logger.info("Actual values: %s", ", ".join([str(v) for v in actual_value_list]))
                logger.info("Golden values: %s", ", ".join([str(v) for v in golden_value_list]))
                raise test.error_message(golden_value_key)

            result = f"{test.type_of_test_result.name} test for metric {golden_value_key}: PASSED"
            result_code = 0

        except (NotApproximateError, NotDeterminsticError, MissingTensorboardLogsError) as e:
            result = str(e)
            result_code = 1
        except SkipMetricError:
            logger.info(f"{test.type_of_test_result.name} test for {golden_value_key}: SKIPPED")
            continue

        log_emitter = logger.info if result_code == 0 else logger.error
        log_emitter(result)
        if result_code == 1:
            all_test_passed = False
            failed_metrics.append(golden_value_key)

    assert all_test_passed, f"The following metrics failed: {', '.join(failed_metrics)}"
