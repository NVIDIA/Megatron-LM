# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Check coverage subset accounting independently of GPU execution."""

import json

import pytest

from tests.test_utils.python_scripts.analyze_functional_coverage import (
    load_baseline,
    plan_reduction,
)


def _test(lines, branches=(), seconds=1):
    return {
        "lines": set(lines),
        "branches": set(branches),
        "metadata": {"elapsed_seconds": seconds},
    }


def test_removes_overlap_but_preserves_platform_specific_paths():
    tests = {
        ("h100", "gpt", "one"): _test(range(100), range(20)),
        ("h100", "gpt", "two"): _test(range(100), range(20), seconds=10),
        ("gb200", "gpt", "only"): _test(range(100), range(20)),
    }
    plan = plan_reduction(tests, 0.2, 0.05)
    assert plan["removed"] == ["h100/gpt/two"]
    assert all(group["lost"] == 0 for group in plan["coverage_loss"].values())


def test_loss_is_cumulative_and_strictly_less_than_five_percent():
    tests = {
        ("h100", "gpt", "main"): _test(range(95)),
        **{("h100", "gpt", str(i)): _test([95 + i]) for i in range(5)},
    }
    plan = plan_reduction(tests, 0.9, 0.05)
    assert plan["removed_count"] == 4
    assert plan["target_met"] is False
    assert plan["coverage_loss"]["h100/lines"]["fraction"] == 0.04


def test_branch_loss_blocks_line_redundant_removal():
    tests = {
        ("h100", "gpt", "one"): _test(range(100), [1]),
        ("h100", "gpt", "two"): _test(range(100), [2]),
    }
    assert plan_reduction(tests, 0.2, 0.05)["removed_count"] == 0


def test_missing_platform_matrix_rejects_baseline(tmp_path):
    with pytest.raises(ValueError, match="Expected one h100 matrix"):
        load_baseline(tmp_path)


def test_failed_attempt_cannot_count_as_coverage(tmp_path):
    for platform in ("h100", "gb200"):
        (tmp_path / f"integration-tests-{platform}.json").write_text(
            json.dumps([{"model": "gpt", "test_case": "one"}])
        )
    attempt = tmp_path / "attempt-0"
    attempt.mkdir()
    (attempt / "metadata.json").write_text(
        json.dumps(
            {
                "platform": "dgx_h100",
                "model": "gpt",
                "test_case": "one",
                "collection_status": "collected",
                "workload_exit_code": 1,
            }
        )
    )
    with pytest.raises(ValueError, match="Missing successful coverage"):
        load_baseline(tmp_path)
