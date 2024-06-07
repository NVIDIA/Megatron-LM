import json
import os

import pytest

from .common import TYPE_OF_TEST_TO_METRIC, TypeOfTest, read_tb_logs_as_list

LOGS_DIR = os.getenv("LOGS_DIR")
EXPECTED_METRICS_FILE = os.getenv("EXPECTED_METRICS_FILE")
ALLOW_NONDETERMINISTIC = bool(int(os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO")))

with open(EXPECTED_METRICS_FILE) as f:
    if os.path.exists(EXPECTED_METRICS_FILE):
        with open(EXPECTED_METRICS_FILE) as f:
            EXPECTED_METRICS = json.load(f)
    else:
        print(f"File {EXPECTED_METRICS_FILE} not found!")


# If we require a variation of tests for any of the other pipelines we can just inherit this class.
@pytest.mark.parametrize("expected_metric", EXPECTED_METRICS.keys())
class TestCIPipeline:
    margin_loss, margin_time = 0.05, 0.1
    expected = EXPECTED_METRICS

    def _test_helper(self, metric_type, test_type):
        if self.expected is None:
            raise FileNotFoundError("Expected data is none")
        expected = self.expected[metric_type]
        expected_list = expected["values"]
        print(f"The list of expected values: {expected_list}")
        try:
            actual_list = read_tb_logs_as_list(LOGS_DIR)[metric_type]
        except KeyError as e:
            raise KeyError(
                f"Required metric {metric_type} not found in TB logs. Please make sure your model exports this metric as its required by the test case/golden values file"
            ) from e
        assert (
            actual_list is not None
        ), f"No TensorBoard events file was found in the logs for {metric_type}."
        actual_list_sliced = actual_list[
            expected["start_step"] : expected["end_step"] : expected["step_interval"]
        ]
        print(f"The list of actual values: {actual_list_sliced}")
        for i, (expected_val, actual_val) in enumerate(
            zip(expected_list, actual_list_sliced)
        ):
            step = i * expected["step_interval"]
            print(f"Checking step {step} against expected {i}")
            if test_type == TypeOfTest.APPROX:
                assert (
                    actual_val
                    == pytest.approx(expected=expected_val, rel=self.margin_loss)
                ), f"Metrics {metric_type} at step {step} should be approximately {expected_val} but it is {actual_val}."
            else:
                assert (
                    actual_val == expected_val
                ), f"The value at step {step} should be {expected_val} but it is {actual_val}."

    @pytest.mark.skipif(ALLOW_NONDETERMINISTIC, reason="Nondeterministic is allowed.")
    def test_deterministic(self, expected_metric):
        if expected_metric in TYPE_OF_TEST_TO_METRIC[TypeOfTest.DETERMINISTIC]:
            self._test_helper(expected_metric, TypeOfTest.DETERMINISTIC)

    @pytest.mark.skipif(
        not ALLOW_NONDETERMINISTIC, reason="Nondeterministic is not allowed."
    )
    def test_approx(self, expected_metric):
        if expected_metric in TYPE_OF_TEST_TO_METRIC[TypeOfTest.APPROX]:
            self._test_helper(expected_metric, TypeOfTest.APPROX)

    # @TODO: This is inactive, do we want to activate it?
    def iteration_timing_node(self):
        expected_iteration_timing_avg = self.expected["train_step_timing_avg"]
        iteration_time = read_tb_logs_as_list(LOGS_DIR)["iteration-time"]
        idx = len(iteration_time) // 3
        iteration_time_avg = sum(iteration_time[idx:]) / len(iteration_time[idx:])
        assert (
            expected_iteration_timing_avg
            == pytest.approx(expected=iteration_time_avg, rel=self.margin_time)
        ), f"The time per global step must be approximately {expected_iteration_timing_avg} but it is {iteration_time_avg}."
