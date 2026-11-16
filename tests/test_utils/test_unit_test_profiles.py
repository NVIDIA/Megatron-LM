# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise profiled unit launches with real pytest processes and CPU-only workers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
RUNNER = ROOT / "tests/unit_tests/run_ci_test.sh"


def _functions(*names: str) -> str:
    source = RUNNER.read_text()
    return "\n".join(
        name + "() {" + source.split(name + "() {", 1)[1].split("\n}\n", 1)[0] + "\n}"
        for name in names
    )


@pytest.fixture
def project(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "pytest.ini").write_text("[pytest]\nmarkers = experimental: experimental test\n")
    (tmp_path / "tests/conftest.py").write_text(
        "def pytest_addoption(parser):\n"
        "    parser.addoption('--experimental', action='store_true')\n"
    )
    return tmp_path


def _run(project, script, *, profiles="delayed blockwise", ranks=1, mode="full", extra_args=""):
    # Replace only the distributed launcher. Each worker still runs the original
    # command in a separate interpreter, including pytest, coverage, and Testmon.
    launcher = r'''
uv() {
    [[ "$1 $2 $3" == "run --no-sync python" ]] || return 90
    shift 3
    if [[ "${1:-} ${2:-}" == "-m torch.distributed.run" ]]; then
        shift 2
        [[ "$1" == --nproc_per_node ]] || return 91
        local ranks="$2" rank
        shift 2
        if [[ "${1:-}" == --no-python ]]; then
            shift
            for ((rank = 0; rank < ranks; rank++)); do
                RANK="$rank" WORLD_SIZE="$ranks" "$@" || return $?
            done
        else
            for ((rank = 0; rank < ranks; rank++)); do
                RANK="$rank" WORLD_SIZE="$ranks" "$TEST_PYTHON" "$@" || return $?
            done
        fi
    else
        "$TEST_PYTHON" "$@"
    fi
}
uv "$@"
'''
    executable_dir = project / "bin"
    executable_dir.mkdir(exist_ok=True)
    launcher_path = executable_dir / "uv"
    launcher_path.write_text("#!/bin/bash\nset -euo pipefail\n" + launcher)
    launcher_path.chmod(0o755)
    return subprocess.run(
        ["bash", "-e", "-u", "-o", "pipefail"],
        input="\n".join(
            (
                f"UNIT_PROFILES=({profiles})",
                f"UNIT_PYTEST_ARGS=({extra_args})",
                'DISTRIBUTED_ARGS=(--nproc_per_node "$GPUS_PER_NODE")',
                "UNIT_LAUNCH=0",
                'mkdir -p "$UNIT_RESULTS_DIR"',
                _functions("run_profiled_command", "validate_profile_results"),
                script,
            )
        ),
        cwd=project,
        env={
            **os.environ,
            "TEST_PYTHON": sys.executable,
            "TESTMON_WRAPPER": str(ROOT / "tests/unit_tests/testmon_selector.py"),
            "PYTHONPATH": str(project),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PATH": str(executable_dir) + os.pathsep + os.environ["PATH"],
            "GPUS_PER_NODE": str(ranks),
            "UNIT_RESULTS_DIR": str(project / "results"),
            "UNIT_TESTMON_CACHE_DIR": str(project / "cache"),
            "UNIT_TESTMON_MODE": mode,
            "UNIT_RESULTS_MODE": mode,
        },
        text=True,
        capture_output=True,
        timeout=60,
    )


def _assert_success(result):
    assert result.returncode == 0, result.stdout + result.stderr


def _events(project):
    return [json.loads(line) for line in (project / "events").read_text().splitlines()]


def _write_precision_tests(project):
    (project / "tests/test_precision.py").write_text(
        "import json\nimport os\nfrom pathlib import Path\nimport pytest\n\n"
        "seen = set()\n\n"
        "def record(name):\n"
        "    with Path('events').open('a') as output:\n"
        "        output.write(json.dumps([name, os.getpid()]) + '\\n')\n\n"
        "def test_admission():\n    record('admission')\n\n"
        "@pytest.mark.parametrize('recipe', ['delayed', 'blockwise', 'mxfp8'])\n"
        "def test_precision(recipe):\n"
        "    assert not seen, 'precision profiles shared process state'\n"
        "    seen.add(recipe)\n"
        "    record(recipe)\n"
    )


def test_precision_profiles_preserve_process_isolation_and_admission_checks(project):
    _write_precision_tests(project)
    result = _run(
        project,
        'run_profiled_command prod full uv run --no-sync python -m torch.distributed.run '
        '"${DISTRIBUTED_ARGS[@]}" -m pytest -q -c pytest.ini tests/test_precision.py\n'
        "validate_profile_results",
    )
    _assert_success(result)
    recorded = _events(project)
    assert len(recorded) == 3
    events = dict(recorded)
    assert set(events) == {"admission", "delayed", "blockwise"}
    assert events["admission"] == events["delayed"]
    assert events["delayed"] != events["blockwise"]


def test_empty_selected_profile_does_not_prevent_the_next_profile(project):
    _write_precision_tests(project)
    result = _run(
        project,
        'run_profiled_command prod selected uv run --no-sync python -m torch.distributed.run '
        '"${DISTRIBUTED_ARGS[@]}" -m pytest -q -c pytest.ini '
        "'tests/test_precision.py::test_precision[blockwise]'\n"
        "validate_profile_results",
        mode="preselected",
    )
    _assert_success(result)
    assert [name for name, _ in _events(project)] == ["blockwise"]


@pytest.mark.parametrize("ranks", (4, 6))
def test_disaggregation_keeps_experimental_flag_and_requires_every_rank_report(project, ranks):
    (project / "tests/test_disagg.py").write_text(
        "import json\nimport os\nfrom pathlib import Path\n\n"
        "def test_disagg(pytestconfig):\n"
        "    assert pytestconfig.getoption('--experimental')\n"
        "    with Path('events').open('a') as output:\n"
        "        output.write(json.dumps(int(os.environ['RANK'])) + '\\n')\n"
    )
    result = _run(
        project,
        "for iteration in 1 2; do\n"
        'run_profiled_command prod full uv run --no-sync python -m torch.distributed.run '
        '"${DISTRIBUTED_ARGS[@]}" -m pytest -q -c pytest.ini tests/test_disagg.py\n'
        "done\n"
        "validate_profile_results\n"
        f'rm "$UNIT_RESULTS_DIR/default/prod-2/rank-{ranks - 1}.xml"\n'
        "if validate_profile_results; then exit 92; fi",
        profiles="default",
        ranks=ranks,
        extra_args="--experimental",
    )
    _assert_success(result)
    assert _events(project) == list(range(ranks)) * 2
    assert f"rank {ranks - 1} is missing its report in prod-2" in result.stderr


@pytest.mark.parametrize("body", ("pass", "pytest.skip('unexpected skip')"), ids=("empty", "skip"))
def test_full_profiles_reject_empty_or_skipped_execution(project, body):
    (project / "tests/test_empty.py").write_text(
        f"import pytest\n\ndef test_example():\n    {body}\n"
    )
    selection = "-k nonexistent" if body == "pass" else ""
    result = _run(
        project,
        'run_profiled_command prod full uv run --no-sync python -m torch.distributed.run '
        '"${DISTRIBUTED_ARGS[@]}" -m pytest -q -c pytest.ini '
        f"tests/test_empty.py {selection}\n"
        "validate_profile_results",
        profiles="default",
    )
    assert result.returncode != 0
    assert "executed no tests" in result.stderr or "has skipped or failed tests" in result.stderr


def test_testmon_profile_selections_accumulate_without_mutating_baseline(project):
    pytest.importorskip("testmon")
    source = "def delayed():\n    return 1 + 2\n\ndef blockwise():\n    return 3 + 4\n"
    (project / "app.py").write_text(source)
    (project / "tests/test_dependencies.py").write_text(
        "import app\nimport pytest\n\n"
        "@pytest.mark.parametrize('recipe, expected', [('delayed', 3), ('blockwise', 7)])\n"
        "def test_precision(recipe, expected):\n"
        "    assert getattr(app, recipe)() == expected\n"
    )
    command = (
        'uv run --no-sync python -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}" '
        '"$TESTMON_WRAPPER" --mode {mode} --cache-dir "$UNIT_TESTMON_CACHE_DIR" '
        "--phase prod -- -q -c pytest.ini tests/test_dependencies.py"
    )
    baseline = _run(
        project,
        "run_profiled_command prod baseline " + command.replace("{mode}", "baseline") + "\n"
        "validate_profile_results",
        mode="baseline",
    )
    _assert_success(baseline)
    phase = project / "cache/prod"
    before = {path.name: path.read_bytes() for path in phase.iterdir() if path.is_file()}

    for changed, expected in (
        (source.replace("1 + 2", "2 + 1").replace("3 + 4", "4 + 3"), {"delayed-3", "blockwise-7"}),
        (source.replace("1 + 2", "2 + 1"), {"delayed-3"}),
    ):
        (project / "app.py").write_text(changed)
        selected = _run(
            project,
            "run_profiled_command prod select " + command.replace("{mode}", "select"),
            mode="enforce",
        )
        _assert_success(selected)
        actual = (
            (project / "cache/.testmon-work/prod/rank-0/selected-tests").read_text().splitlines()
        )
        assert set(actual) == {
            f"tests/test_dependencies.py::test_precision[{recipe}]" for recipe in expected
        }
        assert {
            path.name: path.read_bytes() for path in phase.iterdir() if path.is_file()
        } == before
