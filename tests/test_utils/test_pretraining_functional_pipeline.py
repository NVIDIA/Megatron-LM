# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from copy import deepcopy

import pytest
import yaml

from tests.functional_tests.python_test_utils import (
    test_pretraining_functional_pipeline as functional,
)
from tests.functional_tests.python_test_utils.common import GoldenValueMetric

METRICS = ["lm loss", "mtp_1 loss"]
RUNS = ["actual_values_first_run", "actual_values_second_run"]


@pytest.fixture
def pipeline_args(tmp_path):
    config_path = tmp_path / "model_config.yaml"
    config_path.write_text(yaml.safe_dump({"METRICS": METRICS}))
    return {
        "actual_values_first_run": {
            name: GoldenValueMetric(
                start_step=1,
                end_step=4,
                step_interval=1,
                values={step: 2.0 for step in range(1, 5)},
            )
            for name in METRICS
        },
        "actual_values_second_run": {
            name: GoldenValueMetric(
                start_step=3,
                end_step=4,
                step_interval=1,
                values={step: 3.0 for step in range(3, 5)},
            )
            for name in METRICS
        },
        "train_iters": 4,
        "model_config_path": str(config_path),
    }


def test_complete_finite_metrics_do_not_require_a_baseline_or_modify_values(pipeline_args):
    original = deepcopy(pipeline_args)

    functional.test_functional_pipeline(**pipeline_args)

    assert pipeline_args == original


@pytest.mark.parametrize("metrics", [None, [], "lm loss"])
def test_metrics_must_be_configured_and_nonempty(pipeline_args, metrics):
    with open(pipeline_args["model_config_path"], "w") as config_file:
        yaml.safe_dump({"METRICS": metrics} if metrics is not None else {}, config_file)

    with pytest.raises(AssertionError, match="METRICS must be a nonempty list"):
        functional.test_functional_pipeline(**pipeline_args)


@pytest.mark.parametrize("run", RUNS)
@pytest.mark.parametrize("metric", METRICS)
def test_missing_metric_fails(pipeline_args, run, metric):
    del pipeline_args[run][metric]

    with pytest.raises(AssertionError, match=f"missing metric {metric!r}"):
        functional.test_functional_pipeline(**pipeline_args)


@pytest.mark.parametrize("run", RUNS)
@pytest.mark.parametrize("metric", METRICS)
def test_missing_step_fails(pipeline_args, run, metric):
    del pipeline_args[run][metric].values[3]

    with pytest.raises(AssertionError, match=r"missing=\[3\]"):
        functional.test_functional_pipeline(**pipeline_args)


@pytest.mark.parametrize("run", RUNS)
def test_unexpected_step_fails(pipeline_args, run):
    pipeline_args[run]["lm loss"].values[0] = 2.0

    with pytest.raises(AssertionError, match=r"unexpected=\[0\]"):
        functional.test_functional_pipeline(**pipeline_args)


@pytest.mark.parametrize("run", RUNS)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), "nan", True])
def test_nonfinite_or_nonnumeric_values_fail(pipeline_args, run, value):
    pipeline_args[run]["mtp_1 loss"].values[3] = value

    with pytest.raises(AssertionError, match="at step 3 must be finite and numeric"):
        functional.test_functional_pipeline(**pipeline_args)
