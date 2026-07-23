# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit tests for the functional_tests python_test_utils common IO helpers and
the golden-value collection CLI.

These live under ``tests/unit_tests/`` so the unit-test CI harness (which
collects tests from that tree) discovers them; they exercise the modules in
``tests/functional_tests/python_test_utils`` and complement the ``pipeline`` /
``Test`` coverage in ``tests/functional_tests/python_test_utils/test_common.py``.
They are pure CPU tests: TensorBoard event accumulators are faked so no real
training run or GPU is needed.
"""

import json

import pytest
from click.testing import CliRunner

from tests.functional_tests.python_test_utils import common
from tests.functional_tests.python_test_utils.common import (
    GoldenValueMetric,
    GoldenValues,
    read_golden_values_from_json,
    read_tb_logs_as_list,
)
from tests.functional_tests.python_test_utils.get_test_results_from_tensorboard_logs import (
    collect_train_test_metrics,
)


class _Scalar:
    def __init__(self, step, value):
        self.step = step
        self.value = value


class _FakeEventAccumulator:
    def __init__(self, scalars):
        self._scalars = {name: [_Scalar(s, v) for s, v in pts] for name, pts in scalars.items()}

    def Reload(self):  # noqa: N802
        return self

    def Tags(self):  # noqa: N802
        return {"scalars": list(self._scalars.keys())}

    def Scalars(self, name):  # noqa: N802
        return self._scalars[name]


def _patch_tb(monkeypatch, files, accumulators):
    mapping = dict(zip(files, accumulators))
    glob_results = iter([list(files), []])
    monkeypatch.setattr(common.glob, "glob", lambda pattern: next(glob_results, []))
    monkeypatch.setattr(common.os.path, "getmtime", lambda path: 0)
    monkeypatch.setattr(
        common.event_accumulator,
        "EventAccumulator",
        lambda event_file, size_guidance: mapping[event_file],
    )


class TestGoldenValueModels:
    def test_repr_lists_bounds_and_pairs(self):
        metric = GoldenValueMetric(
            start_step=1, end_step=3, step_interval=1, values={1: 1.5, 3: 2.5}
        )
        text = repr(metric)
        assert "(1,3,1)" in text
        assert "(1, 1.5)" in text
        assert "(3, 2.5)" in text

    def test_root_model_round_trips(self):
        payload = {
            "lm loss": {
                "start_step": 1,
                "end_step": 5,
                "step_interval": 5,
                "values": {"1": 1.0, "5": 0.5},
            }
        }
        parsed = GoldenValues(**payload).root
        assert set(parsed) == {"lm loss"}
        assert parsed["lm loss"].values == {1: 1.0, 5: 0.5}


class TestReadGoldenValuesFromJson:
    def test_reads_valid_file(self, tmp_path):
        path = tmp_path / "golden.json"
        path.write_text(
            json.dumps(
                {
                    "lm loss": {
                        "start_step": 1,
                        "end_step": 5,
                        "step_interval": 5,
                        "values": {"1": 2.0, "5": 1.0},
                    }
                }
            )
        )
        result = read_golden_values_from_json(str(path))
        assert isinstance(result["lm loss"], GoldenValueMetric)
        assert result["lm loss"].values == {1: 2.0, 5: 1.0}

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(OSError):
            read_golden_values_from_json(str(tmp_path / "nope.json"))


class TestLoadEventAccumulatorsWithScalars:
    def test_drops_scalarless_and_preserves_order(self, monkeypatch):
        files = ["a", "b", "c"]
        accs = [
            _FakeEventAccumulator({}),
            _FakeEventAccumulator({"lm loss": [(1, 1.0)]}),
            _FakeEventAccumulator({"lm loss": [(1, 2.0)]}),
        ]
        mapping = dict(zip(files, accs))
        monkeypatch.setattr(
            common.event_accumulator,
            "EventAccumulator",
            lambda event_file, size_guidance: mapping[event_file],
        )
        loaded = common._load_event_accumulators_with_scalars(files)
        assert loaded == [accs[1], accs[2]]


class TestReadTbLogsAsList:
    def test_returns_none_when_no_event_files(self, monkeypatch):
        monkeypatch.setattr(common.glob, "glob", lambda pattern: [])
        assert read_tb_logs_as_list("/some/path") is None

    def test_returns_none_when_no_scalar_data(self, monkeypatch):
        _patch_tb(monkeypatch, ["a"], [_FakeEventAccumulator({})])
        assert read_tb_logs_as_list("/some/path") is None

    def test_returns_none_when_index_out_of_range(self, monkeypatch):
        _patch_tb(monkeypatch, ["a"], [_FakeEventAccumulator({"lm loss": [(1, 1.0)]})])
        assert read_tb_logs_as_list("/some/path", index=5) is None

    def test_backfills_missing_steps_with_nan(self, monkeypatch):
        _patch_tb(monkeypatch, ["a"], [_FakeEventAccumulator({"lm loss": [(1, 1.23456789)]})])
        result = read_tb_logs_as_list(
            "/some/path", index=0, train_iters=10, start_idx=1, step_size=5
        )
        metric = result["lm loss"]
        assert metric.start_step == 1
        assert metric.end_step == 10
        assert metric.values[1] == 1.23457
        assert metric.values[5] == "nan"
        assert metric.values[10] == "nan"

    def test_index_minus_one_merges_without_overwrite(self, monkeypatch):
        _patch_tb(
            monkeypatch,
            ["a", "b"],
            [
                _FakeEventAccumulator({"lm loss": [(1, 1.0)]}),
                _FakeEventAccumulator({"lm loss": [(1, 9.0), (5, 5.0)]}),
            ],
        )
        result = read_tb_logs_as_list(
            "/some/path", index=-1, train_iters=5, start_idx=1, step_size=5
        )
        values = result["lm loss"].values
        assert values[1] == 1.0
        assert values[5] == 5.0


class TestCollectTrainTestMetrics:
    def test_filters_metrics_and_writes_output(self, monkeypatch, tmp_path):
        def fake_read(path, index, train_iters, start_idx, step_size):
            return {
                "lm loss": GoldenValueMetric(
                    start_step=1, end_step=1, step_interval=5, values={1: 1.0}
                ),
                "unwanted-metric": GoldenValueMetric(
                    start_step=1, end_step=1, step_interval=5, values={1: 2.0}
                ),
            }

        monkeypatch.setattr(common, "read_tb_logs_as_list", fake_read)
        output = tmp_path / "out.json"
        result = CliRunner().invoke(
            collect_train_test_metrics,
            ["--logs-dir", str(tmp_path), "--train-iters", "50", "--output-path", str(output)],
        )
        assert result.exit_code == 0, result.output
        written = json.loads(output.read_text())
        assert set(written) == {"lm loss"}
        assert written["lm loss"]["values"] == {"1": 1.0}

    def test_is_convergence_test_selects_last_index(self, monkeypatch, tmp_path):
        captured = {}

        def fake_read(path, index, train_iters, start_idx, step_size):
            captured["index"] = index
            return {}

        monkeypatch.setattr(common, "read_tb_logs_as_list", fake_read)
        result = CliRunner().invoke(
            collect_train_test_metrics,
            ["--logs-dir", str(tmp_path), "--train-iters", "50", "--is-convergence-test"],
        )
        assert result.exit_code == 0, result.output
        assert captured["index"] == -1

    def test_no_logs_found_does_not_write_output(self, monkeypatch, tmp_path):
        monkeypatch.setattr(common, "read_tb_logs_as_list", lambda *a, **k: None)
        output = tmp_path / "out.json"
        result = CliRunner().invoke(
            collect_train_test_metrics,
            ["--logs-dir", str(tmp_path), "--train-iters", "50", "--output-path", str(output)],
        )
        assert result.exit_code == 0, result.output
        assert not output.exists()
