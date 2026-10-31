# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise marker selection and file ownership without importing GPU tests."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests.unit_tests import gb200_placement

ROOT = Path(__file__).parents[2]
sys.modules.setdefault("gb200_placement", gb200_placement)
SPEC = importlib.util.spec_from_file_location(
    "placement_finder", ROOT / "tests/unit_tests/find_test_cases.py"
)
finder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(finder)


@pytest.mark.parametrize(
    "enabled,expression,passed",
    [
        ("true", "launch_on_gb200", 3),
        ("true", "not gb200_only", 4),
        ("false", "launch_on_gb200", 1),
        ("false", "not flaky_in_dev", 6),
    ],
)
def test_real_pytest_marker_selection(tmp_path, enabled, expression, passed):
    files = {
        "ssm/test_gated_delta_net2.py": "def test_portable(): pass\n",
        "inference/engines/test_dynamic_engine.py": (
            "class TestDynamicInferenceEngine:\n"
            "    def test_simple(self): pass\n"
            "    def test_suspend_resume_cycle(self): pass\n"
            "class TestFutureEngineTopology:\n"
            "    def test_eight_ranks(self): pass\n"
        ),
        "test_unmigrated.py": (
            "import pytest\n"
            "@pytest.mark.launch_on_gb200\n"
            "def test_existing_additive_marker(): pass\n"
            "def test_h100_only(): pass\n"
        ),
    }
    for relative, source in files.items():
        path = tmp_path / "tests/unit_tests" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\nmarkers =\n"
        "    launch_on_gb200: existing GB200 coverage\n"
        "    flaky_in_dev: flaky case\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "tests.unit_tests.gb200_placement_plugin",
            "-q",
            "-m",
            expression,
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "UNIT_TEST_GB200_ENABLED": enabled,
        },
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"{passed} passed" in result.stdout
    assert "skipped" not in result.stdout


@pytest.mark.parametrize("enabled", [False, True])
def test_gb200_files_have_exactly_one_active_bucket(monkeypatch, capsys, enabled):
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("UNIT_TEST_GB200_ENABLED", str(enabled).lower())
    recipe = yaml.safe_load((ROOT / "tests/test_utils/recipes/gb200/unit-tests.yaml").read_text())
    buckets = [case for product in recipe["products"] for case in product["test_case"]]
    monkeypatch.setattr(finder, "get_test_cases", lambda _: buckets)
    owners = {}
    for bucket in gb200_placement.active_buckets(buckets, "gb200", enabled):
        monkeypatch.setattr(sys, "argv", ["find_test_cases.py", bucket, "gb200"])
        finder.main()
        ignored = {line.removeprefix("--ignore=") for line in capsys.readouterr().out.splitlines()}
        for path in set(finder.expand_pattern(bucket)) - ignored:
            if Path(path).name.startswith("test_"):
                owners.setdefault(path, []).append(bucket)
    assert all(len(buckets) == 1 for buckets in owners.values()), owners
    expected = {
        str(path.relative_to(ROOT))
        for path in (ROOT / "tests/unit_tests").rglob("test_*.py")
        if "launch_on_gb200" in path.read_text()
    }
    if enabled:
        expected |= gb200_placement.candidate_files()
    assert set(owners) == expected


def test_h100_fallback_and_migration_keep_mixed_buckets():
    recipe = yaml.safe_load((ROOT / "tests/test_utils/recipes/h100/unit-tests.yaml").read_text())
    original = [case for product in recipe["products"] for case in product["test_case"]]
    migrated = gb200_placement.active_buckets(original, "h100", True)
    assert set(original) - set(migrated) == set(gb200_placement.FULL_BUCKETS)
    assert gb200_placement.active_buckets(original, "h100", False) == original
    for bucket in (
        "tests/unit_tests/inference/engines/test_dynamic_engine.py",
        "tests/unit_tests/transformer/**/*.py",
        "tests/unit_tests/a2a_overlap/**/*.py",
    ):
        assert bucket in migrated


@pytest.mark.parametrize(
    "tag,environment", [("latest", "dev"), ("latest", "lts"), ("legacy", "dev")]
)
def test_runner_only_excludes_portable_cases_in_latest_dev(tag, environment):
    source = (ROOT / "tests/unit_tests/run_ci_test.sh").read_text()
    marker_setup = source.split("cd $TEST_PATH\n", 1)[1].split("export BUCKET", 1)[0]
    result = subprocess.run(
        [
            "bash",
            "-eu",
            "-c",
            marker_setup + '\nprintf "%s\\n" "$MARKER_ARG" "$UNIT_TEST_GB200_ENABLED"',
        ],
        env={
            **os.environ,
            "TAG": tag,
            "ENVIRONMENT": environment,
            "PLATFORM": "h100",
            "UNIT_TEST_GB200_ENABLED": "true",
        },
        text=True,
        capture_output=True,
        check=True,
    )
    assert ("not gb200_only" in result.stdout) == (tag == "latest" and environment == "dev")
