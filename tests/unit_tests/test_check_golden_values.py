# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for ``tools/check_golden_values.py`` (CPU only)."""

import json

import pytest

from tools import check_golden_values

DETERMINISTIC_CONFIG = """ENV_VARS:
  NVTE_ALLOW_NONDETERMINISTIC_ALGO: 0
  NCCL_ALGO: Ring
MODEL_ARGS:
  --deterministic-mode: true
TEST_TYPE: regular
"""

APPROXIMATE_CONFIG = """ENV_VARS:
  NVTE_ALLOW_NONDETERMINISTIC_ALGO: 0
  NON_DETERMINSTIC_RESULTS: 1
MODEL_ARGS:
  --deterministic-mode: true
TEST_TYPE: regular
"""

FULL = check_golden_values.FULL_PRECISION
LEGACY = check_golden_values.LEGACY_PRECISION


def _metric(values, precision=None):
    steps = sorted(values)
    block = {
        "start_step": steps[0],
        "end_step": steps[-1],
        "step_interval": 1,
        "values": {str(step): value for step, value in values.items()},
    }
    if precision is not None:
        block["value_precision"] = precision
    return block


def _write_case(tmp_path, model_config, golden_values, name="golden_values_dev_dgx_h100.json"):
    (tmp_path / "model_config.yaml").write_text(model_config)
    golden_file = tmp_path / name
    golden_file.write_text(json.dumps(golden_values))
    return golden_file


def test_precision_markers_mirror_the_pipeline_enum():
    common = pytest.importorskip("tests.functional_tests.python_test_utils.common")
    assert FULL == common.ValuePrecision.FULL.value
    assert LEGACY == common.ValuePrecision.ROUNDED_5_DECIMAL_PLACES.value


class TestComparesDeterministically:
    def test_default_is_deterministic(self):
        assert check_golden_values.compares_deterministically(DETERMINISTIC_CONFIG)

    @pytest.mark.parametrize(
        "line",
        [
            "  NON_DETERMINSTIC_RESULTS: 1",
            "  NVTE_ALLOW_NONDETERMINISTIC_ALGO: 1",
            "  NVTE_ALLOW_NONDETERMINISTIC_ALGO: '1'",
            "  NVTE_ALLOW_NONDETERMINISTIC_ALGO: 1 # comment",
            "  SKIP_PYTEST: 1",
        ],
    )
    def test_opt_out_env_vars(self, line):
        config = f"ENV_VARS:\n{line}\nMODEL_ARGS:\n  --train-iters: 50\n"
        assert not check_golden_values.compares_deterministically(config)

    def test_zero_does_not_opt_out(self):
        assert check_golden_values.compares_deterministically(
            "ENV_VARS:\n  NON_DETERMINSTIC_RESULTS: 0\n"
        )


class TestFindLegacyPrecisionMetrics:
    def test_unmarked_metric_is_legacy(self):
        golden = {"lm loss": _metric({1: 10.96462, 2: 10.95232}), "num-zeros": _metric({1: 1.0})}
        assert check_golden_values.find_legacy_precision_metrics(golden) == ["lm loss", "num-zeros"]

    def test_explicit_legacy_marker_is_legacy(self):
        golden = {"lm loss": _metric({1: 10.96462}, LEGACY)}
        assert check_golden_values.find_legacy_precision_metrics(golden) == ["lm loss"]

    def test_full_marker_passes(self):
        golden = {"lm loss": _metric({1: 10.964620590209961, 2: 10.952320098876953}, FULL)}
        assert check_golden_values.find_legacy_precision_metrics(golden) == []

    def test_full_marker_passes_regardless_of_value_spelling(self):
        # A full-precision value can have a short decimal spelling; only the marker counts.
        golden = {"lm loss": _metric({1: 0.5, 2: 10.96462, 3: 3.0}, FULL)}
        assert check_golden_values.find_legacy_precision_metrics(golden) == []

    def test_only_unmarked_metrics_are_reported(self):
        golden = {"lm loss": _metric({1: 0.5}, FULL), "num-zeros": _metric({1: 1.0})}
        assert check_golden_values.find_legacy_precision_metrics(golden) == ["num-zeros"]

    def test_other_layouts_are_ignored(self):
        assert check_golden_values.find_legacy_precision_metrics({"req-0": {"tokens": [1]}}) == []
        assert check_golden_values.find_legacy_precision_metrics([1, 2, 3]) == []


class TestMain:
    def test_legacy_deterministic_case_fails(self, tmp_path):
        golden_file = _write_case(
            tmp_path, DETERMINISTIC_CONFIG, {"lm loss": _metric({1: 10.96462, 2: 10.95232})}
        )
        assert check_golden_values.main([str(golden_file)]) == 1

    def test_allow_legacy_precision_overrides(self, tmp_path):
        golden_file = _write_case(
            tmp_path, DETERMINISTIC_CONFIG, {"lm loss": _metric({1: 10.96462, 2: 10.95232})}
        )
        assert check_golden_values.main(["--allow-legacy-precision", str(golden_file)]) == 0

    def test_legacy_approximate_case_passes(self, tmp_path):
        golden_file = _write_case(
            tmp_path, APPROXIMATE_CONFIG, {"lm loss": _metric({1: 10.96462, 2: 10.95232})}
        )
        assert check_golden_values.main([str(golden_file)]) == 0

    def test_full_precision_deterministic_case_passes(self, tmp_path):
        golden_file = _write_case(
            tmp_path,
            DETERMINISTIC_CONFIG,
            {
                "lm loss": _metric({1: 10.964620590209961, 2: 0.5}, FULL),
                "num-zeros": _metric({1: 3.0}, FULL),
            },
        )
        assert check_golden_values.main([str(golden_file)]) == 0

    def test_missing_model_config_skips_precision_check(self, tmp_path):
        golden_file = tmp_path / "golden_values_dev_dgx_h100.json"
        golden_file.write_text(json.dumps({"lm loss": _metric({1: 10.96462, 2: 10.95232})}))
        assert check_golden_values.main([str(golden_file)]) == 0

    def test_non_finite_values_fail(self, tmp_path):
        golden_file = _write_case(
            tmp_path, APPROXIMATE_CONFIG, {"lm loss": _metric({1: "nan", 2: 10.95232}, FULL)}
        )
        assert check_golden_values.main([str(golden_file)]) == 1

    def test_unreadable_file_fails(self, tmp_path):
        assert check_golden_values.main([str(tmp_path / "missing.json")]) == 1
