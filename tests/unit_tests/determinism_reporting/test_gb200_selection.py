# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Check scheduling and file selection with the actual CI parser, without a GPU."""

import subprocess
import tomllib
from pathlib import Path

from tests.test_utils.python_scripts.recipe_parser import load_and_flatten
from tests.unit_tests import find_test_cases
from tests.unit_tests.find_test_cases import expand_pattern, file_has_marker, is_child_of_bucket


def test_gb200_replay_buckets_are_scheduled_and_excluded_from_catch_all(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[3]
    monkeypatch.chdir(root)
    rows = load_and_flatten("tests/test_utils/recipes/gb200/unit-tests.yaml")
    buckets = {row.spec["test_case"]: row.spec for row in rows}
    kernels = "tests/unit_tests/determinism/kernels/**/*.py"
    models = "tests/unit_tests/determinism/correctness/**/*.py"
    general = "tests/unit_tests/**/*.py"
    assert {kernels, models, general} <= buckets.keys()
    for bucket in (kernels, models):
        spec = buckets[bucket]
        assert (spec["gpus"], spec["nodes"], spec["scope"], spec["tag"]) == (
            4,
            1,
            "unit-tests",
            "latest",
        )
        assert is_child_of_bucket(bucket, general)
        marked = [
            file
            for file in expand_pattern(bucket)
            if Path(file).name.startswith("test_") and file_has_marker(file, "launch_on_gb200")
        ]
        assert marked
        script = spec["script"].format(**{**spec, "assets_dir": str(tmp_path)})
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)
        assert "--require-verified" in script
        assert "REQUIRED_CASES=(--require-author-checks)" in script
        assert "--determinism-branch-coverage" in script
        assert "--require-branches" in script
        assert "--require-parallelism" in script
        assert "--require-case '*fp8-mxfp8*'" in script
        assert "--require-case '*fp4-nvfp4*'" in script


def test_h100_branch_reports_share_the_latest_ci_coverage_mode(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[3]
    monkeypatch.chdir(root)
    config = tomllib.loads((root / "pyproject.toml").read_text())
    assert config["tool"]["coverage"]["run"]["branch"] is True
    rows = load_and_flatten("tests/test_utils/recipes/h100/unit-tests.yaml")
    selected = [
        row.spec
        for row in rows
        if "/determinism/" in row.spec["test_case"] and row.spec["tag"] == "latest"
    ]
    assert {spec["test_case"] for spec in selected} == {
        "tests/unit_tests/determinism/kernels/**/*.py",
        "tests/unit_tests/determinism/correctness/**/*.py",
    }
    for spec in selected:
        script = spec["script"].format(**{**spec, "assets_dir": str(tmp_path)})
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)
        assert "--require-branches" in script
        assert "--require-parallelism" in script
        assert "--determinism-evidence-scope=$EVIDENCE_SCOPE" in script
        assert "REQUIRED_VIEWS=(--require-author-checks)" in script


def test_gb200_replay_and_m2n_buckets_keep_each_marked_test_in_one_process(monkeypatch, capsys):
    root = Path(__file__).resolve().parents[3]
    monkeypatch.chdir(root)
    rows = load_and_flatten("tests/test_utils/recipes/gb200/unit-tests.yaml")
    buckets = sorted({row.spec["test_case"] for row in rows})
    general = "tests/unit_tests/**/*.py"
    m2n = "tests/unit_tests/resharding/test_nccl_m2n_copy_service.py"
    assert {
        general,
        m2n,
        "tests/unit_tests/determinism/kernels/**/*.py",
        "tests/unit_tests/determinism/correctness/**/*.py",
    } <= set(buckets)
    # Replace only the yq extraction; execute the actual selector for each recipe.
    monkeypatch.setattr(find_test_cases, "get_test_cases", lambda path: buckets)
    selections = []
    for bucket in buckets:
        monkeypatch.setattr(find_test_cases.sys, "argv", ["find_test_cases.py", bucket, "gb200"])
        find_test_cases.main()
        ignored = {line.removeprefix("--ignore=") for line in capsys.readouterr().out.splitlines()}
        selected = {
            file
            for file in expand_pattern(bucket)
            if Path(file).name.startswith("test_") and file not in ignored
        }
        if bucket == m2n:
            assert selected == {m2n}
        assert all(selected.isdisjoint(previous) for previous in selections)
        selections.append(selected)
    assert set().union(*selections) == {
        file
        for file in expand_pattern(general)
        if Path(file).name.startswith("test_") and file_has_marker(file, "launch_on_gb200")
    }
