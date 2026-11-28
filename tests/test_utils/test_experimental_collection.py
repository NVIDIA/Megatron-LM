# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_utils.python_scripts import experimental_collection

REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN = "tests.test_utils.python_scripts.experimental_collection"


@pytest.fixture
def mini_suite(tmp_path):
    suite = tmp_path / "suite"
    suite.mkdir()
    reports = tmp_path / "reports"
    reports.mkdir()
    (suite / "pytest.ini").write_text("[pytest]\nmarkers = experimental\n")

    def run(source, *, rank=0, extra_args=(), conftest="", report_dir=reports):
        (suite / "test_example.py").write_text(source)
        (suite / "conftest.py").write_text(conftest)
        env = {
            **os.environ,
            "PYTHONPATH": str(REPO_ROOT),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "RANK": str(rank),
            experimental_collection.REPORT_DIR_ENV: str(report_dir),
        }
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                PLUGIN,
                "-m",
                "not experimental",
                *extra_args,
            ],
            cwd=suite,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    return run, reports


def _decision(reports, ranks):
    return subprocess.run(
        [sys.executable, str(Path(experimental_collection.__file__)), str(reports), str(ranks)],
        capture_output=True,
        text=True,
        check=False,
    ).returncode


def test_plain_collection_can_skip(mini_suite):
    run, reports = mini_suite
    result = run("def test_production(): pass\n")
    assert result.returncode == 0, result.stdout + result.stderr
    assert _decision(reports, 1) == 0


@pytest.mark.parametrize(
    "source",
    [
        "import pytest\npytestmark = pytest.mark.experimental\ndef test_experimental(): pass\n",
        "import pytest\n@pytest.mark.experimental\nclass TestExperimental:\n"
        "    def test_experimental(self): pass\ndef test_production(): pass\n",
    ],
)
def test_inherited_markers_survive_production_deselection(mini_suite, source):
    run, reports = mini_suite
    result = run(source)
    assert result.returncode in (0, 5), result.stdout + result.stderr
    assert "deselected" in result.stdout
    assert _decision(reports, 1) == 1


@pytest.mark.parametrize("wrapper_order", ["tryfirst", "trylast"])
def test_late_wrapper_can_mark_an_already_deselected_item(mini_suite, wrapper_order):
    run, reports = mini_suite
    result = run(
        "def test_production(): pass\ndef test_late_experimental(): pass\n",
        extra_args=("-k", "production"),
        conftest=f"""
import pytest

@pytest.hookimpl(hookwrapper=True, {wrapper_order}=True)
def pytest_collection_modifyitems(items):
    original_items = tuple(items)
    yield
    for item in original_items:
        if item.name == "test_late_experimental":
            item.add_marker(pytest.mark.experimental)
""",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed, 1 deselected" in result.stdout
    assert _decision(reports, 1) == 1


def test_rank_specific_collections_require_the_union(mini_suite):
    run, reports = mini_suite
    source = """
import os
import pytest

def test_production(): pass

if os.environ["RANK"] == "1":
    @pytest.mark.experimental
    def test_rank_one_experimental(): pass
"""
    assert run(source, rank=0).returncode == 0
    assert _decision(reports, 2) == 1
    assert run(source, rank=1).returncode == 0
    assert _decision(reports, 2) == 1
    assert run("def test_production(): pass\n", rank=1).returncode == 0
    assert _decision(reports, 2) == 0


@pytest.mark.parametrize(
    "invalid_report",
    [
        None,
        "{",
        "[]",
        "null",
        '{"rank":1,"has_experimental":0}',
        '{"rank":true,"has_experimental":false}',
        '{"rank":0,"has_experimental":false}',
        '{"rank":1}',
        '{"rank":1,"has_experimental":true}',
    ],
)
def test_missing_or_invalid_report_keeps_experimental_phase(tmp_path, invalid_report):
    (tmp_path / "rank-0.json").write_text(json.dumps({"rank": 0, "has_experimental": False}))
    if invalid_report is not None:
        (tmp_path / "rank-1.json").write_text(invalid_report)
    assert _decision(tmp_path, 2) == 1


@pytest.mark.parametrize("ranks", ["0", "-1", "invalid"])
def test_invalid_expected_rank_count_cannot_skip(tmp_path, ranks):
    assert _decision(tmp_path, ranks) != 0


@pytest.mark.parametrize(
    "source,conftest",
    [
        ("raise RuntimeError('collection failed')\n", ""),
        (
            "def test_production(): pass\n",
            "def pytest_collection_modifyitems(items):\n    raise RuntimeError('hook failed')\n",
        ),
    ],
)
def test_collection_failure_does_not_report_safe_to_skip(mini_suite, source, conftest):
    run, reports = mini_suite
    result = run(source, conftest=conftest)
    assert result.returncode != 0
    assert _decision(reports, 1) == 1


def test_report_write_failure_preserves_production_result(mini_suite, tmp_path):
    run, reports = mini_suite
    unavailable_dir = tmp_path / "not-a-directory"
    unavailable_dir.write_text("")
    result = run("def test_production(): pass\n", report_dir=unavailable_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Experimental collection report unavailable" in result.stderr
    assert _decision(reports, 1) == 1


@pytest.mark.parametrize(
    "scenario,expected_phases",
    [
        ("plain", ["prod", "prod"]),
        ("experimental", ["prod", "experimental", "prod", "experimental"]),
        ("missing-second-report", ["prod", "prod", "experimental"]),
        ("unavailable-report-dir", ["prod", "experimental", "prod", "experimental"]),
    ],
)
def test_full_runner_uses_fresh_reports_each_repeat(tmp_path, scenario, expected_phases):
    """Exercise the real shell runner with CPU pytest in place of torchrun."""
    suite = tmp_path / "suite"
    (suite / "tests/unit_tests").mkdir(parents=True)
    (suite / "tests/test_utils/python_scripts").mkdir(parents=True)
    (suite / "tests/unit_tests/find_test_cases.py").write_text("")
    (suite / "tests/test_utils/python_scripts/experimental_collection.py").symlink_to(
        experimental_collection.__file__
    )
    (suite / "pytest.ini").write_text("[pytest]\nmarkers = experimental\n")
    (suite / "tests/unit_tests/conftest.py").write_text(
        'def pytest_addoption(parser):\n'
        '    parser.addoption("--experimental", action="store_true")\n'
        'def pytest_sessionfinish(session, exitstatus):\n'
        '    if exitstatus == 5: session.exitstatus = 0\n'
    )
    source = "def test_production(): pass\n"
    if scenario == "experimental":
        source += "import pytest\n@pytest.mark.experimental\ndef test_experimental(): pass\n"
    (suite / "tests/unit_tests/test_example.py").write_text(source)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text("""#!/bin/bash
set -euo pipefail
shift 3  # run --no-sync python
if [[ "$1 ${2:-}" != "-m torch.distributed.run" ]]; then
    exec "$SMOKE_PYTHON" "$@"
fi
phase=experimental
if [[ " $* " == *" -m coverage "* ]]; then phase=prod; fi
printf '%s %s\n' "$phase" "${UNIT_TEST_EXPERIMENTAL_REPORT_DIR:-}" >> "$SMOKE_LAUNCHES"
if [[ "$SMOKE_SCENARIO" == missing-second-report && "$phase" == prod ]]; then
    count=0
    while read -r launched_phase rest; do
        if [[ "$launched_phase" == prod ]]; then count=$((count + 1)); fi
    done < "$SMOKE_LAUNCHES"
    if [[ "$count" -gt 1 ]]; then unset UNIT_TEST_EXPERIMENTAL_REPORT_DIR; fi
fi
while [[ "$1 ${2:-}" != "-m pytest" ]]; do shift; done
shift 2
RANK=0 "$SMOKE_PYTHON" -m pytest -p "$SMOKE_PLUGIN" -c "$SMOKE_REPO/pytest.ini" "$@"
""")
    uv.chmod(0o755)
    coverage = bin_dir / "coverage"
    coverage.write_text("#!/bin/bash\nexit 0\n")
    coverage.chmod(0o755)
    if scenario == "unavailable-report-dir":
        mktemp = bin_dir / "mktemp"
        mktemp.write_text("#!/bin/bash\nexit 1\n")
        mktemp.chmod(0o755)
    launches = tmp_path / "launches"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "PYTHONPATH": str(REPO_ROOT),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "UNIT_TESTMON_MODE": "full",
        "GPUS_PER_NODE": "1",
        "NUM_NODES": "1",
        "SMOKE_REPO": str(suite),
        "SMOKE_PYTHON": sys.executable,
        "SMOKE_PLUGIN": PLUGIN,
        "SMOKE_LAUNCHES": str(launches),
        "SMOKE_SCENARIO": scenario,
    }
    result = subprocess.run(
        [
            "bash",
            "-c",
            'cd() { if [[ "$1" == /opt/megatron-lm ]]; then builtin cd "$SMOKE_REPO"; '
            'else builtin cd "$@"; fi; }; export -f cd; exec bash "$@"',
            "runner-smoke",
            str(REPO_ROOT / "tests/unit_tests/run_ci_test.sh"),
            "--tag",
            "latest",
            "--environment",
            "dev",
            "--bucket",
            "tests/unit_tests",
            "--unit-test-repeat",
            "2",
            "--log-dir",
            str(tmp_path / "logs"),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    phase_reports = [line.split() for line in launches.read_text().splitlines()]
    assert [entry[0] for entry in phase_reports] == expected_phases
    if scenario != "unavailable-report-dir":
        production_dirs = [entry[1] for entry in phase_reports if entry[0] == "prod"]
        assert len(set(production_dirs)) == 2
