# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pathlib

import pytest
import yaml

from tests.functional_tests.python_test_utils import common

YAML_DIR = pathlib.Path(__file__).parent / ".." / "functional_tests" / "test_cases"

# Default metrics a functional test validates against when its model_config.yaml
# does not set an explicit METRICS list. Sourced from the functional-test harness
# (single source of truth in common.DEFAULT_METRICS).
DEFAULT_METRICS = common.DEFAULT_METRICS

# Mapping from a validated tensorboard metric to the MODEL_ARGS flag that must be
# enabled for that metric to be recorded. Metrics not listed here (e.g. losses,
# generated tokens, logprobs) are always logged and need no extra flag.
METRIC_TO_REQUIRED_ARG = {
    "num-zeros": "--log-num-zeros-in-grad",
    "mem-allocated-bytes": "--log-memory-to-tensorboard",
    "mem-max-allocated-bytes": "--log-memory-to-tensorboard",
    "iteration-time": "--log-timers-to-tensorboard",
}


def get_yaml_files(directory):
    """Retrieve all YAML files from the specified directory."""
    return list([file for file in directory.rglob("model_config.yaml") if file is not None])


def load_yaml(file_path):
    """Load a YAML file and return its content as a Python dictionary."""
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def get_effective_metrics(model_config):
    """Return the metrics a test validates against.

    Uses the explicit METRICS list when present (an empty list means the test
    validates no metrics), otherwise falls back to DEFAULT_METRICS.
    """
    if "METRICS" in model_config:
        return model_config["METRICS"] or []
    return DEFAULT_METRICS


def get_required_args(model_config):
    """Return the sorted MODEL_ARGS flags required by the test's metrics."""
    return sorted(
        {
            METRIC_TO_REQUIRED_ARG[metric]
            for metric in get_effective_metrics(model_config)
            if metric in METRIC_TO_REQUIRED_ARG
        }
    )


@pytest.mark.parametrize("yaml_file", get_yaml_files(YAML_DIR))
def test_model_config_tracks_tested_metrics(yaml_file):
    """Test that each YAML file tracks exactly the metrics it validates against.

    A functional test only needs to enable the logging flags for the metrics in
    its effective METRICS list (explicit METRICS, or the default otherwise). This
    keeps the safety net that a test tracks what it later checks, without forcing
    every test to enable unrelated metrics (e.g. num-zeros for a memory-only test).
    """
    if any(k in str(yaml_file) for k in ["gpt3-nemo", "ckpt_converter", "gpt-nemo", "inference"]):
        pytest.skip("Skipping `test_model_config_tracks_tested_metrics`")

    model_config = load_yaml(yaml_file)

    assert "MODEL_ARGS" in model_config, (
        f"Please add a `MODEL_ARGS` section to `{yaml_file.parent.name}/model_config.yaml` "
        "so its metrics get tracked."
    )

    for required_arg in get_required_args(model_config):
        assert (
            required_arg in model_config["MODEL_ARGS"]
            and model_config["MODEL_ARGS"][required_arg] is True
        ), (
            f"Please add argument `{required_arg}` to "
            f"`{yaml_file.parent.name}/model_config.yaml` so that its metric gets tracked."
        )
