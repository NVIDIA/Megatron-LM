import json
import os
from typing import List, Union

import numpy as np
import pytest

from .common import (
    ALLOW_NONDETERMINISTIC,
    LOGS_DIR,
    METRIC_TO_THRESHOLD,
    TYPE_OF_TEST_TO_METRIC,
    TypeOfTest,
    load_expected_data,
    read_tb_logs_as_list,
)


@pytest.fixture(params=load_expected_data().items())
def expected_data(request):
    return request.param


# If we require a variation of tests for any of the other pipelines we can just inherit this class.
class TestCIPipeline:
    allow_nondeterministic = ALLOW_NONDETERMINISTIC

    # Replace symbol in namespace to fix function call result for lifetime of
    # this class.

    def _test_helper(self, metric_type: str, metric_dict: List[Union[int, float]], test_type):
        expected_list = metric_dict['values']
        print(f"The list of expected values: {expected_list} for metric {metric_type}")

        try:
            actual_list = read_tb_logs_as_list(LOGS_DIR)[metric_type]
        except KeyError as e:
            raise KeyError(
                f"Required metric {metric_type} not found in TB logs. Please make sure your model exports this metric as its required by the test case/golden values file"
            ) from e

        if actual_list is None:
            raise ValueError(f"No values of {metric_type} found in TB logs.")
        
        
        actual_list_sliced = actual_list[
            metric_dict["start_step"] : metric_dict["end_step"] : metric_dict["step_interval"]
        ]
        print(f"The list of actual values: {actual_list_sliced}")

        if metric_type == "iteration-time":
            actual_list_sliced = actual_list_sliced[3:]
            expected_list = expected_list[3:]
            print(f"Removing first items of values for metric_type iteration-time")
        
        if test_type == TypeOfTest.DETERMINISTIC:
            assert np.allclose(
                actual_list_sliced, expected_list, rtol=0, atol=0
            ), f"Actual is not equal to Expected for {metric_type}"
        elif test_type == TypeOfTest.APPROX:
            assert np.allclose(
                actual_list_sliced, expected_list, rtol=1e-5, atol=METRIC_TO_THRESHOLD[metric_type]
            ), f"Actual is not equal to Expected for {metric_type}"
        else:
            raise ValueError(f"Unexpected test_type {test_type} provided")

    def test_approx(self, expected_data):
        expected_metric, expected_values = expected_data

        if expected_metric in TYPE_OF_TEST_TO_METRIC[TypeOfTest.APPROX]:
            self._test_helper(expected_metric, expected_values, TypeOfTest.APPROX)
        else:
            print(f"Skipping metric {expected_metric} for approximate as it is deterministic only.")

    @pytest.mark.skipif(allow_nondeterministic, reason="Cannot expect exact results")
    def test_deterministic(self, expected_data):
        expected_metric, expected_values = expected_data

        if expected_metric in TYPE_OF_TEST_TO_METRIC[TypeOfTest.DETERMINISTIC]:
            self._test_helper(expected_metric, expected_values, TypeOfTest.DETERMINISTIC)
        else:
            print(f"Skipping metric {expected_metric} for deterministic as it is approximate only.")
            
    # # @TODO: This is inactive, do we want to activate it?
    # def iteration_timing_node(self):
    #     expected_iteration_timing_avg = self.expected["train_step_timing_avg"]
    #     iteration_time = read_tb_logs_as_list(LOGS_DIR)["iteration-time"]
    #     idx = len(iteration_time) // 3
    #     iteration_time_avg = sum(iteration_time[idx:]) / len(iteration_time[idx:])
    #     assert (
    #         expected_iteration_timing_avg
    #         == pytest.approx(expected=iteration_time_avg, rel=self.margin_time)
    #     ), f"The time per global step must be approximately {expected_iteration_timing_avg} but it is {iteration_time_avg}."

# if deterministic, then also approx
# if not determinstic, then also aprox

