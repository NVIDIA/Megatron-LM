# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
import json
import sys

import pytest
import yaml

from tests.performance_tests.shell_test_utils import compare_to_baseline


@pytest.fixture
def comparison(tmp_path, monkeypatch):
    baseline = {
        "batch_1": {
            "dataset": "gsm8k",
            "batch_size": 1,
            "num_output_tokens": 128,
            "num_iters": 5,
            "num_input_tokens_avg": 60.2,
            "throughput_tok_per_sec": 100.0,
            "avg_latency_ms": 100.0,
        }
    }
    results = copy.deepcopy(baseline)
    config = {
        "TOLERANCE_PCT": 10,
        "UPPER_TOLERANCE_PCT": 20,
        "METRICS": ["throughput_tok_per_sec", "avg_latency_ms"],
    }

    def run():
        (tmp_path / "results.json").write_text(json.dumps(results))
        (tmp_path / "baseline.json").write_text(json.dumps({"h100": baseline}))
        (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "compare_to_baseline.py",
                "--results",
                str(tmp_path / "results.json"),
                "--baseline",
                str(tmp_path / "baseline.json"),
                "--config",
                str(tmp_path / "config.yaml"),
                "--platform",
                "h100",
            ],
        )
        return compare_to_baseline.main()

    return results["batch_1"], baseline["batch_1"], config, run


@pytest.mark.parametrize(
    "field,value",
    [("dataset", "synthetic"), ("batch_size", 8), ("num_output_tokens", 256), ("num_iters", 10)],
)
def test_metadata_mismatch_blocks_numeric_comparison(comparison, capsys, field, value):
    results, _, _, run = comparison
    results[field] = value
    # Invalid numeric values prove that the mismatched batch never reaches the metric check.
    results["throughput_tok_per_sec"] = None
    assert run() == 1
    output = capsys.readouterr().out
    assert f"metadata {field!r} differs" in output
    assert "INCOMPARABLE" in output
    assert "measured=" not in output


@pytest.mark.parametrize("field", ["dataset", "batch_size", "num_output_tokens", "num_iters"])
@pytest.mark.parametrize("side", ["results", "baseline"])
def test_missing_metadata_fails_clearly(comparison, capsys, field, side):
    results, baseline, _, run = comparison
    del {"results": results, "baseline": baseline}[side][field]
    assert run() == 1
    assert f"metadata {field!r} missing from {side}" in capsys.readouterr().out


@pytest.mark.parametrize(
    "metric,value,expected",
    [
        ("throughput_tok_per_sec", 90.0, 0),
        ("throughput_tok_per_sec", 89.9, 1),
        ("throughput_tok_per_sec", 120.0, 0),
        ("throughput_tok_per_sec", 120.1, 1),
        ("avg_latency_ms", 110.0, 0),
        ("avg_latency_ms", 110.1, 1),
    ],
)
def test_matching_metadata_preserves_performance_gates(comparison, capsys, metric, value, expected):
    results, _, _, run = comparison
    results[metric] = value
    assert run() == expected
    assert "INCOMPARABLE" not in capsys.readouterr().out


def test_configured_tolerances_remain_effective(comparison):
    results, _, config, run = comparison
    config.update(TOLERANCE_PCT=5, UPPER_TOLERANCE_PCT=15)
    results["throughput_tok_per_sec"] = 94.0
    assert run() == 1
    results["throughput_tok_per_sec"] = 116.0
    assert run() == 1


def test_average_input_length_is_not_an_equality_gate(comparison):
    results, _, _, run = comparison
    results["num_input_tokens_avg"] = 66.2
    assert run() == 0
