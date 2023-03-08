import os
import json
import pytest
import sys
import glob
from tensorboard.backend.event_processing import event_accumulator

LOGS_DIR = os.getenv('LOGS_DIR')
EXPECTED_METRICS_FILE = os.getenv('EXPECTED_METRICS_FILE')

import enum

class TypeOfTest(enum.Enum):
    APPROX = 1
    DETERMINISTIC = 2


def read_tb_logs_as_list(path, summary_name):
    """Reads a TensorBoard Events file from the input path, and returns the
    summary specified as input as a list.

    Arguments:
    path: str, path to the dir where the events file is located.
    summary_name: str, name of the summary to read from the TB logs.
    Output:
    summary_list: list, the values in the read summary list, formatted as a list.
    """
    files = glob.glob(f"{path}/events*tfevents*")
    files += glob.glob(f"{path}/results/events*tfevents*")
    files.sort(key=lambda x: os.path.getmtime(os.path.join(path, x)))
    if files:
        event_file = files[0]
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()
        summary = ea.Scalars(summary_name)
        summary_list = [round(x.value, 5) for x in summary]
        print(summary_list)
        return summary_list
    raise FileNotFoundError(f"File not found matching: {path}/events*")


# If we require a variation of tests for any of the other pipelines we can just inherit this class.
class TestCIPipeline:

    margin_loss, margin_time = 0.05, 0.1
    expected = None
    if os.path.exists(EXPECTED_METRICS_FILE):
        with open(EXPECTED_METRICS_FILE) as f:
            expected = json.load(f)

    def _test_helper(self, loss_type, test_type):
        if self.expected is None:
            raise FileNotFoundError("Expected data is none")
        expected = self.expected[loss_type]
        expected_list = expected["values"]
        actual_list = read_tb_logs_as_list(LOGS_DIR, loss_type)
        assert actual_list is not None, f"No TensorBoard events file was found in the logs for {loss_type}."
        for i, step in enumerate(range(expected["start_step"], expected["end_step"], expected["step_interval"])):
            if test_type == TypeOfTest.APPROX:
                assert actual_list[step] == pytest.approx(expected=expected_list[i], rel=self.margin_loss), f"{self.job_name} : The loss at step {step} should be approximately {expected_list[i]} but it is {actual_list[step]}."
            else:
                assert actual_list[step] == expected_list[i], f"The value at step {step} should be {expected_list[i]} but it is {actual_list[step]}."

    @pytest.mark.xfail
    def test_lm_loss_deterministic(self):
        # Expected training loss curve at different global steps.
        self._test_helper("lm loss", TypeOfTest.DETERMINISTIC)

    def test_lm_loss_approx(self):
        # Expected training loss curve at different global steps.
        self._test_helper("lm loss", TypeOfTest.APPROX)

    def test_num_zeros_deterministic(self):
        # Expected validation loss curve at different global steps.
        self._test_helper("num-zeros", TypeOfTest.DETERMINISTIC)
    
    def iteration_timing_node(self):
        expected_iteration_timing_avg = self.expected["train_step_timing_avg"]
        iteration_time = read_tb_logs_as_list(LOGS_DIR, "iteration-time")
        idx = len(iteration_time)//3   
        iteration_time_avg = sum(iteration_time[idx:])/len(iteration_time[idx:])
        assert expected_iteration_timing_avg == pytest.approx(expected=iteration_time_avg, rel=self.margin_time), f"The time per global step must be approximately {expected_iteration_timing_avg} but it is {iteration_time_avg}."
