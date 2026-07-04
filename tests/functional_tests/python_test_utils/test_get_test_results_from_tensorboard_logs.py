# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for the ``get_test_results_from_tensorboard_logs`` CLI and the
``common`` helpers it relies on that were previously uncovered.

These tests avoid writing real TensorBoard event files by monkeypatching
``common.read_tb_logs_as_list`` (exercised separately via its callers); the
focus here is the metric-filtering / output-writing behaviour of the
``collect_train_test_metrics`` click command and the JSON (de)serialisation
helpers in ``common``.
"""

import json

import pytest
from click.testing import CliRunner

from tests.functional_tests.python_test_utils import common
from tests.functional_tests.python_test_utils import (
    get_test_results_from_tensorboard_logs as gtr,
)

# ── helpers ─────────────────────────────────────────────────────


def make_metric(values: dict, step_interval: int = 5) -> common.GoldenValueMetric:
    steps = sorted(values)
    return common.GoldenValueMetric(
        start_step=steps[0], end_step=steps[-1], step_interval=step_interval, values=values
    )


# ── common.read_golden_values_from_json / GoldenValues ───────────────────


class TestReadGoldenValuesFromJson:
    def test_round_trip(self, tmp_path):
        metric = make_metric({1: 1.0, 5: 2.0, 10: 3.0})
        payload = {"lm loss": json.loads(metric.model_dump_json())}
        path = tmp_path / "golden.json"
        path.write_text(json.dumps(payload))

        result = common.read_golden_values_from_json(str(path))

        assert set(result.keys()) == {"lm loss"}
        loaded = result["lm loss"]
        assert loaded.start_step == 1
        assert loaded.end_step == 10
        assert loaded.step_interval == 5
        assert loaded.values == {1: 1.0, 5: 2.0, 10: 3.0}

    def test_accepts_pathlib_path(self, tmp_path):
        metric = make_metric({1: 0.5})
        path = tmp_path / "golden.json"
        path.write_text(json.dumps({"num-zeros": json.loads(metric.model_dump_json())}))

        result = common.read_golden_values_from_json(path)

        assert result["num-zeros"].values == {1: 0.5}

    def test_missing_file_raises(self, tmp_path):
        missing = tmp_path / "does_not_exist.json"
        with pytest.raises((FileNotFoundError, ValueError)):
            common.read_golden_values_from_json(str(missing))


class TestGoldenValueMetricRepr:
    def test_repr_contains_bounds_and_values(self):
        metric = make_metric({1: 1.0, 5: 2.0})
        text = repr(metric)
        assert "(1,5,5)" in text
        assert "(1, 1.0)" in text
        assert "(5, 2.0)" in text


class TestGoldenValuesRootModel:
    def test_round_trip_multiple_metrics(self):
        metrics = {
            "lm loss": make_metric({1: 1.0}),
            "num-zeros": make_metric({1: 0.0}),
        }
        gv = common.GoldenValues(root=metrics)
        dumped = json.loads(gv.model_dump_json())
        reloaded = common.GoldenValues(**dumped).root
        assert set(reloaded.keys()) == {"lm loss", "num-zeros"}


# ── collect_train_test_metrics CLI ─────────────────────────────────


class TestCollectTrainTestMetrics:
    def _invoke(self, monkeypatch, summaries, extra_args=None):
        """Run the CLI with ``read_tb_logs_as_list`` stubbed to ``summaries``.

        Returns ``(result, captured_kwargs)`` where ``captured_kwargs`` are the
        keyword arguments the CLI passed through to ``read_tb_logs_as_list``.
        """
        captured = {}

        def fake_read(path, **kwargs):
            captured["path"] = path
            captured.update(kwargs)
            return summaries

        monkeypatch.setattr(common, "read_tb_logs_as_list", fake_read)
        runner = CliRunner()
        args = ["--logs-dir", "/tmp/logs", "--train-iters", "50"]
        if extra_args:
            args += extra_args
        result = runner.invoke(gtr.collect_train_test_metrics, args)
        return result, captured

    def test_writes_only_whitelisted_metrics(self, monkeypatch, tmp_path):
        summaries = {
            "lm loss": make_metric({1: 1.0}),
            "num-zeros": make_metric({1: 0.0}),
            "not-a-golden-metric": make_metric({1: 42.0}),
        }
        out = tmp_path / "out.json"
        result, _ = self._invoke(
            monkeypatch, summaries, extra_args=["--output-path", str(out)]
        )

        assert result.exit_code == 0, result.output
        written = json.loads(out.read_text())
        assert set(written.keys()) == {"lm loss", "num-zeros"}
        assert "not-a-golden-metric" not in written
        # The dumped metric is a GoldenValueMetric.model_dump() shape.
        assert written["lm loss"]["values"] == {"1": 1.0}

    def test_no_output_path_does_not_write(self, monkeypatch, tmp_path):
        summaries = {"lm loss": make_metric({1: 1.0})}
        result, _ = self._invoke(monkeypatch, summaries)
        assert result.exit_code == 0, result.output
        # No file was requested, so nothing to assert beyond a clean exit.

    def test_no_logs_found_warns_and_writes_nothing(self, monkeypatch, tmp_path):
        out = tmp_path / "out.json"
        result, _ = self._invoke(
            monkeypatch, None, extra_args=["--output-path", str(out)]
        )
        assert result.exit_code == 0, result.output
        assert not out.exists()

    def test_normal_test_uses_index_zero(self, monkeypatch):
        result, captured = self._invoke(monkeypatch, {"lm loss": make_metric({1: 1.0})})
        assert result.exit_code == 0, result.output
        assert captured["index"] == 0

    def test_convergence_test_uses_index_minus_one(self, monkeypatch):
        result, captured = self._invoke(
            monkeypatch,
            {"lm loss": make_metric({1: 1.0})},
            extra_args=["--is-convergence-test"],
        )
        assert result.exit_code == 0, result.output
        assert captured["index"] == -1

    def test_step_size_is_forwarded(self, monkeypatch):
        result, captured = self._invoke(
            monkeypatch,
            {"lm loss": make_metric({1: 1.0})},
            extra_args=["--step-size", "10"],
        )
        assert result.exit_code == 0, result.output
        assert captured["step_size"] == 10
