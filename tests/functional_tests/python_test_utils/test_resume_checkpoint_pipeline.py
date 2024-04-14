import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
import glob
import json
import shutil
import sys

import pytest
from tensorboard.backend.event_processing import event_accumulator

from tests.functional_tests.python_test_utils.common import TypeOfTest

LOGS_DIR = os.getenv('LOGS_DIR')
STEP_INTERVAL = 5

def read_tb_logs_as_list(path, summary_name, index):
    files = glob.glob(f"{path}/events*tfevents*")
    files += glob.glob(f"{path}/results/events*tfevents*")
    files.sort(key=lambda x: os.path.getmtime(os.path.join(path, x)))
    if files:
        event_file = files[index]
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()
        summary = ea.Scalars(summary_name)
        summary_list = [round(x.value, 5) for x in summary]
        print(summary_list)
        return summary_list
    raise FileNotFoundError(f"File not found matching: {path}/events*")    

def collect_train_test_metrics(logs_dir, index):
    train_loss_list = read_tb_logs_as_list(logs_dir, "lm loss", index)
    train_loss_list = [round(elem,3) for elem in train_loss_list]
    train_metrics = {
        "lm loss": train_loss_list[0:len(train_loss_list):STEP_INTERVAL],
    } 
    str_train_metrics = str(train_metrics).replace("'", "\"")
    print(f"\n ----------- The following are the metrics for ----------")
    print(f"\n {str_train_metrics}", flush=True)
    return train_metrics

class TestCIPipeline:

    margin_loss = 0.005
    train_metrics_100 = collect_train_test_metrics(LOGS_DIR, 0)
    train_metrics_50_to_100 = collect_train_test_metrics(LOGS_DIR, 1)

    def _test_helper(self, loss_type, test_type):
        expected = self.train_metrics_100[loss_type]
        assert len(expected) == 100 // STEP_INTERVAL, \
            f"Train metrics from first run (before checkpoint load) should have {100 // STEP_INTERVAL} elements"
        print('expected : '  + str(expected))
        actual = self.train_metrics_50_to_100[loss_type]
        assert len(actual) == 50 // STEP_INTERVAL, \
            f"Train metrics from second run (after checkpoint load) should have {50 // STEP_INTERVAL} elements"
        print('actual : '  + str(actual))
        start_idx_expected = len(expected) - len(actual)
        print('start_idx_expected:', start_idx_expected)
        # Here we will just be comparing values of actual and second half (50-100) of expected
        for i, (expected_val, actual_val) in enumerate(zip(expected[start_idx_expected:], actual)):
            step = start_idx_expected + i * STEP_INTERVAL
            if test_type == TypeOfTest.APPROX:
                assert actual_val == pytest.approx(expected=expected_val, rel=self.margin_loss), f"The loss at step {step} should be approximately {expected_val} but it is {actual_val}."
            else:
                assert actual_val == expected_val, f"The value at step {step} should be {expected_val} but it is {actual_val}."

    # def test_lm_loss_deterministic(self):
    #     self._test_helper("lm loss", TypeOfTest.DETERMINISTIC)

    def test_lm_loss_approx(self):
        self._test_helper("lm loss", TypeOfTest.APPROX)
