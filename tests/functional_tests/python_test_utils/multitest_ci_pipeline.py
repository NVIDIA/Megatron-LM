import os
import json
import pytest
import sys
import glob
from .common import read_tb_logs_as_list, TypeOfTest
from .test_ci_pipeline import TestCIPipeline

LOGS_DIR = os.getenv('LOGS_DIR')
EXPECTED_METRICS_DIR = os.getenv('EXPECTED_METRICS_DIR')


class TestBulkCIPipeline(TestCIPipeline):

    margin_loss, margin_time = 0.05, 0.1

    def _setup(self, config_name):
        self.config_name = config_name
        baseline_filename = config_name + '.json'

        filepath = os.path.join(EXPECTED_METRICS_DIR, baseline_filename)
        if os.path.exists(filepath):
            with open(filepath) as f:
                self.expected = json.load(f)
        else:
            raise FileNotFoundError(f"{baseline_filename} does not exist")

    def _get_actual(self, loss_type):
        return read_tb_logs_as_list(LOGS_DIR+'/'+self.config_name, loss_type)

    @pytest.mark.parametrize("config_name", os.listdir(LOGS_DIR))
    def test_lm_loss_deterministic(self, config_name):
        # Expected training loss curve at different global steps.
        self._setup(config_name)
        self._test_helper("lm loss", TypeOfTest.DETERMINISTIC)

    @pytest.mark.parametrize("config_name", os.listdir(LOGS_DIR))
    def test_lm_loss_approx(self, config_name):
        # Expected training loss curve at different global steps.
        self._setup(config_name)
        self._test_helper("lm loss", TypeOfTest.APPROX)

    @pytest.mark.parametrize("config_name", os.listdir(LOGS_DIR))
    def test_num_zeros_deterministic(self, config_name):
        # Expected validation loss curve at different global steps.
        self._setup(config_name)
        self._test_helper("num-zeros", TypeOfTest.DETERMINISTIC)
