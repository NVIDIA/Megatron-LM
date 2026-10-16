# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json

from click.testing import CliRunner

from tests.functional_tests.python_test_utils import get_test_results_from_tensorboard_logs
from tests.functional_tests.python_test_utils.common import GoldenValueMetric


def test_non_finite_value_does_not_leave_partial_output(monkeypatch, tmp_path):
    output_path = tmp_path / "golden_values.json"
    non_finite_metric = GoldenValueMetric(
        start_step=1, end_step=1, step_interval=1, values={1: float("nan")}
    )
    monkeypatch.setattr(
        get_test_results_from_tensorboard_logs.common,
        "read_tb_logs_as_list",
        lambda *args, **kwargs: {"lm loss": non_finite_metric},
    )

    result = CliRunner().invoke(
        get_test_results_from_tensorboard_logs.collect_train_test_metrics,
        ["--logs-dir", str(tmp_path), "--train-iters", "1", "--output-path", str(output_path)],
    )

    assert result.exit_code != 0
    assert "Out of range float values are not JSON compliant" in str(result.exception)
    assert not output_path.exists()


def test_unselected_non_finite_metric_is_not_serialized(monkeypatch, tmp_path):
    output_path = tmp_path / "golden_values.json"
    finite_metric = GoldenValueMetric(start_step=1, end_step=1, step_interval=1, values={1: 1.0})
    non_finite_metric = GoldenValueMetric(
        start_step=1, end_step=1, step_interval=1, values={1: float("nan")}
    )
    monkeypatch.setattr(
        get_test_results_from_tensorboard_logs.common,
        "read_tb_logs_as_list",
        lambda *args, **kwargs: {"lm loss": finite_metric, "learning-rate": non_finite_metric},
    )

    result = CliRunner().invoke(
        get_test_results_from_tensorboard_logs.collect_train_test_metrics,
        ["--logs-dir", str(tmp_path), "--train-iters", "1", "--output-path", str(output_path)],
    )

    assert result.exit_code == 0
    assert set(json.loads(output_path.read_text())) == {"lm loss"}
