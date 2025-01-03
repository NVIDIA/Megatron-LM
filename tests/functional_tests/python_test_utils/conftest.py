from typing import Dict, List, Union

import pytest

from tests.functional_tests.python_test_utils import common


def pytest_addoption(parser):
    """
    Additional command-line arguments passed to pytest.
    """
    parser.addoption(
        "--allow-nondeterministic-algo",
        action="store_true",
        default=False,
        help="If set, test system checks for approximate results.",
    )
    parser.addoption("--golden-values-path", action="store", help="Path to golden values")
    parser.addoption(
        "--train-iters", action="store", default=100, help="Number of train iters", type=int
    )
    parser.addoption("--tensorboard-path", action="store", help="Path to tensorboard records")
    parser.addoption("--model-config-path", action="store", help="Path to model_config.yaml")


@pytest.fixture
def compare_approximate_results(request) -> bool:
    """Simple fixture returning whether to check against results approximately."""
    return request.config.getoption("--allow-nondeterministic-algo") is True


@pytest.fixture
def golden_values(request):
    """Simple fixture returning golden values."""
    return common.read_golden_values_from_json(request.config.getoption("--golden-values-path"))


@pytest.fixture
def train_iters(request):
    """Simple fixture returning number of train iters."""
    return request.config.getoption("--train-iters")


@pytest.fixture
def tensorboard_logs(request, train_iters):
    """Simple fixture returning tensorboard metrics."""
    return common.read_tb_logs_as_list(
        request.config.getoption("--tensorboard-path"), train_iters=train_iters
    )


@pytest.fixture
def tensorboard_path(request):
    """Simple fixture returning path to tensorboard logs."""
    return request.config.getoption("--tensorboard-path")


@pytest.fixture
def model_config_path(request):
    """Simple fixture returning path to model_config.yaml."""
    return request.config.getoption("--model-config-path")
