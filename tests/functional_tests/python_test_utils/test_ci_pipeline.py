import os
import json
import pytest
import sys
import glob
from .common import read_tb_logs_as_list, TypeOfTest

LOGS_DIR = os.getenv('LOGS_DIR')
EXPECTED_METRICS_FILE = os.getenv('EXPECTED_METRICS_FILE')
ALLOW_NONDETERMINISTIC = os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO")


# If we require a variation of tests for any of the other pipelines we can just inherit this class.
class TestCIPipeline:

    margin_loss, margin_time = 0.05, 0.1
    expected = None
    allow_nondeterministic = bool(int(ALLOW_NONDETERMINISTIC))

    def _setup(self):
        if os.path.exists(EXPECTED_METRICS_FILE):
            with open(EXPECTED_METRICS_FILE) as f:
                self.expected = json.load(f)
        else:
            print(f"File {EXPECTED_METRICS_FILE} not found!")

    def _get_actual(self, loss_type):
        return read_tb_logs_as_list(LOGS_DIR, loss_type)

    def _test_helper(self, loss_type, test_type):
        if self.expected is None:
            raise FileNotFoundError(f"Expected data is none")
        expected = self.expected[loss_type]
        expected_list = expected["values"]
        print(f"The list of expected values: {expected_list}")
        actual_list = self._get_actual(loss_type)
        assert actual_list is not None, f"No TensorBoard events file was found in the logs for {loss_type}."
        actual_list_sliced = actual_list[expected["start_step"]:expected["end_step"]:expected["step_interval"]]
        print(f"The list of actual values: {actual_list_sliced}")
        for i, (expected_val, actual_val) in enumerate(zip(expected_list, actual_list_sliced)):
            step = i * expected["step_interval"]
            print(f"Checking step {step} against expected {i}")
            if test_type == TypeOfTest.APPROX:
                assert actual_val == pytest.approx(expected=expected_val, rel=self.margin_loss), f"The loss at step {step} should be approximately {expected_val} but it is {actual_val}."
            else:
                assert actual_val == expected_val, f"The value at step {step} should be {expected_val} but it is {actual_val}."

    @pytest.mark.skipif(allow_nondeterministic, reason="Nondeterministic is allowed.")
    def test_lm_loss_deterministic(self):
        # Expected training loss curve at different global steps.
        self._setup()
        self._test_helper("lm loss", TypeOfTest.DETERMINISTIC)

    @pytest.mark.skipif(not allow_nondeterministic, reason="Nondeterministic is not allowed.")
    def test_lm_loss_approx(self):
        # Expected training loss curve at different global steps.
        self._setup()
        self._test_helper("lm loss", TypeOfTest.APPROX)

    @pytest.mark.skipif(allow_nondeterministic, reason="Nondeterministic is allowed.")
    def test_num_zeros_deterministic(self):
        # Expected validation loss curve at different global steps.
        self._setup()
        self._test_helper("num-zeros", TypeOfTest.DETERMINISTIC)

    def iteration_timing_node(self):
        expected_iteration_timing_avg = self.expected["train_step_timing_avg"]
        iteration_time = read_tb_logs_as_list(LOGS_DIR, "iteration-time")
        idx = len(iteration_time)//3
        iteration_time_avg = sum(iteration_time[idx:])/len(iteration_time[idx:])
        assert expected_iteration_timing_avg == pytest.approx(expected=iteration_time_avg, rel=self.margin_time), f"The time per global step must be approximately {expected_iteration_timing_avg} but it is {iteration_time_avg}."
