import json
import os

import numpy as np
import pytest
import scipy.stats as ss
from scipy.integrate import trapezoid

from .common import TypeOfTest, read_tb_logs_as_list

LOGS_DIR = os.getenv("LOGS_DIR")
EXPECTED_METRICS_FILE = os.getenv("EXPECTED_METRICS_FILE")


# If we require a variation of tests for any of the other pipelines we can just inherit this class.
class TestFP8CIPipeline:
    margin_loss, margin_time = 0.2, 0.1
    auc_threshold, correlation_threshold = 0.01, 0.999
    expected = None

    def _setup(self):
        if os.path.exists(EXPECTED_METRICS_FILE):
            with open(EXPECTED_METRICS_FILE) as f:
                self.expected = json.load(f)
            if self.expected is None:
                raise FileNotFoundError("Expected data is none")

    def _get_actual(self, loss_type):
        actual_list = read_tb_logs_as_list(LOGS_DIR)[loss_type]
        assert (
            actual_list is not None
        ), f"No TensorBoard events file was found in the logs for {loss_type}."
        return actual_list

    def _margin_test_helper(self, loss_type):
        expected = self.expected[loss_type]
        expected_list = np.array(expected["values"])
        actual_list = self._get_actual(loss_type)
        actual_list_sliced = np.array(
            actual_list[
                expected["start_step"] : expected["end_step"] : expected[
                    "step_interval"
                ]
            ]
        )

        max_diff_index = np.argmax(np.abs(actual_list_sliced - expected_list))
        max_diff = np.abs(
            actual_list_sliced[max_diff_index] - expected_list[max_diff_index]
        )

        print(
            f"[INFO - margin]: maximum absolute difference for {loss_type} is {max_diff} at index {max_diff_index}, "
            f"Actual: {actual_list_sliced[max_diff_index]}, Expected: {expected_list[max_diff_index]}"
        )
        assert np.allclose(
            actual_list_sliced, expected_list, rtol=1e-5, atol=self.margin_loss
        ), f"Actual is not equal to Expected for {loss_type}"

    def _auc_test_helper(self, loss_type):
        expected = self.expected[loss_type]
        expected_list = np.array(expected["values"])
        actual_list = self._get_actual(loss_type)
        actual_list_sliced = np.array(
            actual_list[
                expected["start_step"] : expected["end_step"] : expected[
                    "step_interval"
                ]
            ]
        )

        def compute_auc(y_values):
            x_values = np.arange(0, len(y_values), 1)
            area = trapezoid(y_values, x_values)
            return round(area, 5)

        baseline_area = compute_auc(expected_list)
        current_area = compute_auc(actual_list_sliced)
        diff = abs(baseline_area - current_area)

        print(
            f"[INFO - AUC]: AUC diff: {diff * 100 / baseline_area} %, current: {current_area}, baseline: {baseline_area}"
        )
        assert (baseline_area <= 0) or (diff <= self.auc_threshold * baseline_area)

    def _correlation_test_helper(self, loss_type):
        expected = self.expected[loss_type]
        expected_list = np.array(expected["values"])
        actual_list = self._get_actual(loss_type)
        actual_list_sliced = np.array(
            actual_list[
                expected["start_step"] : expected["end_step"] : expected[
                    "step_interval"
                ]
            ]
        )
        corr = ss.pearsonr(actual_list_sliced, expected_list).statistic

        print(f"[INFO - Corr]: Corr: {corr}")
        assert corr > self.correlation_threshold

    @pytest.mark.xfail
    def test_lm_loss_margin(self):
        self._setup()
        self._margin_test_helper("lm loss")

    def test_lm_loss_auc(self):
        self._setup()
        self._auc_test_helper("lm loss")

    @pytest.mark.xfail
    def test_lm_loss_correlation(self):
        self._setup()
        self._correlation_test_helper("lm loss")

    def iteration_timing_node(self):
        expected_iteration_timing_avg = self.expected["train_step_timing_avg"]
        iteration_time = read_tb_logs_as_list(LOGS_DIR)["iteration-time"]
        idx = len(iteration_time) // 3
        iteration_time_avg = sum(iteration_time[idx:]) / len(iteration_time[idx:])
        assert (
            expected_iteration_timing_avg
            == pytest.approx(expected=iteration_time_avg, rel=self.margin_time)
        ), f"The time per global step must be approximately {expected_iteration_timing_avg} but it is {iteration_time_avg}."
