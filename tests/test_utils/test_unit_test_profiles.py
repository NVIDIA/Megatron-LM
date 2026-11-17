# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise public unit-test launchers with real pytest and CPU-only workers."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
FP8 = "tests/unit_tests/inference/test_batch_invariant_fp8.py"
COORDINATOR = "tests/unit_tests/inference/engines/test_disagg_coordinator_pairwise.py"
TRANSFER = "tests/unit_tests/inference/test_disagg_transfer_pairwise.py"


@pytest.fixture
def project(tmp_path):
    for relative in (
        "tests/unit_tests/run_ci_test.sh",
        "tests/unit_tests/find_test_cases.py",
        "tests/unit_tests/testmon_selector.py",
        "tests/unit_tests/testmon_cache.py",
        "tests/unit_tests/testmon_mandatory.py",
        "tests/unit_tests/testmon_mandatory_tests.yaml",
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\nmarkers =\n    experimental: experimental test\n"
        "    launch_on_gb200: GB200 test\n"
    )
    (tmp_path / ".coveragerc").write_text("[run]\nparallel = true\n")
    (tmp_path / "tests/conftest.py").write_text(
        "def pytest_addoption(parser):\n"
        "    parser.addoption('--experimental', action='store_true')\n"
    )
    (tmp_path / "megatron/core").mkdir(parents=True)
    (tmp_path / "megatron/__init__.py").touch()
    (tmp_path / "megatron/core/__init__.py").touch()
    (tmp_path / "megatron/core/precision.py").write_text(
        "def delayed():\n    return 1 + 2\n\ndef blockwise():\n    return 3 + 4\n"
        "\ndef mxfp8():\n    return 4 + 5\n"
    )
    (tmp_path / "nixl.py").write_text("_api = object()\n")
    executable_dir = tmp_path / "bin"
    executable_dir.mkdir()
    launcher = r'''
import argparse
import os
import subprocess
import sys
from pathlib import Path

if sys.argv[1:3] == ["pip", "install"]:
    import testmon
    sys.exit(0)
command = sys.argv[4:]
if command[:2] == ["-m", "torch.distributed.run"]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nproc_per_node", type=int)
    for option in ("nnodes", "master_addr", "master_port", "node_rank", "log-dir", "tee", "redirects"):
        parser.add_argument("--" + option)
    parser.add_argument("--no-python", action="store_true")
    args, command = parser.parse_known_args(command[2:])
    counter = Path("worker-launches")
    launch = int(counter.read_text()) + 1 if counter.exists() else 1
    counter.write_text(str(launch))
    for rank in range(args.nproc_per_node):
        environment = dict(os.environ, RANK=str(rank), WORLD_SIZE=str(args.nproc_per_node))
        if launch == 3 and rank == args.nproc_per_node - 1:
            environment["TEST_DROP_REPORT"] = os.environ.get("TEST_MISSING_REPORT", "")
        result = subprocess.run(
            command if args.no_python else [sys.executable, *command], env=environment, check=False
        )
        if result.returncode not in (0, 5):
            sys.exit(result.returncode)
    sys.exit(0)
if os.environ.get("TEST_DROP_REPORT"):
    command = [argument for argument in command if not argument.startswith("--junitxml=")]
os.execv(sys.executable, [sys.executable, *command])
'''
    launcher_path = executable_dir / "uv"
    launcher_path.write_text(f"#!{sys.executable}\n" + launcher)
    launcher_path.chmod(0o755)
    discovery = executable_dir / "yq"
    discovery.write_text("#!/bin/sh\nexit 0\n")
    discovery.chmod(0o755)
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache/changed-files").write_text("")
    return tmp_path


def _run(project, bucket=FP8, *, mode="full", repeat=1, **env):
    arguments = [
        "--tag",
        "latest",
        "--environment",
        "dev",
        "--bucket",
        bucket,
        "--unit-test-repeat",
        str(repeat),
        "--log-dir",
        str(project / "logs"),
    ]
    return subprocess.run(
        ["bash", str(project / "tests/unit_tests/run_ci_test.sh"), *arguments],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONPATH": str(project),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PATH": str(project / "bin") + os.pathsep + os.environ["PATH"],
            "UNIT_TEST_ROOT": str(project),
            "UNIT_TESTMON_CACHE_DIR": str(project / "cache"),
            "UNIT_TESTMON_MODE": mode,
            **env,
        },
        text=True,
        capture_output=True,
        timeout=60,
    )


def _assert_success(result):
    assert result.returncode == 0, result.stdout + result.stderr


def _events(project):
    path = project / "events"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _write_precision_tests(project):
    (project / FP8).parent.mkdir(parents=True, exist_ok=True)
    (project / FP8).write_text(
        "import json\nimport os\nfrom pathlib import Path\nimport pytest\n\n"
        "from megatron.core import precision\n\n"
        "def record(name):\n"
        "    with Path('events').open('a') as output:\n"
        "        output.write(json.dumps([name, os.getpid()]) + '\\n')\n\n"
        "def test_admission():\n    record('admission')\n\n"
        "@pytest.mark.parametrize('recipe, expected', [('delayed', 3), ('blockwise', 7),\n"
        "    pytest.param('mxfp8', 9, marks=pytest.mark.launch_on_gb200)])\n"
        "def test_precision(recipe, expected):\n"
        "    assert getattr(precision, recipe)() == expected\n"
        "    record(recipe)\n"
    )


def test_precision_profiles_preserve_process_isolation_and_admission_checks(project):
    _write_precision_tests(project)
    result = _run(project)
    _assert_success(result)
    recorded = _events(project)
    assert len(recorded) == 3
    events = dict(recorded)
    assert set(events) == {"admission", "delayed", "blockwise"}
    assert events["admission"] == events["delayed"]
    assert events["delayed"] != events["blockwise"]


@pytest.mark.parametrize("bucket,ranks", [(COORDINATOR, 4), (TRANSFER, 6)])
def test_disaggregation_keeps_experimental_flag_and_requires_every_rank_report(
    project, bucket, ranks
):
    (project / bucket).parent.mkdir(parents=True, exist_ok=True)
    (project / bucket).write_text(
        "import json\nimport os\nfrom pathlib import Path\n\n"
        "def test_disagg(pytestconfig):\n"
        "    assert pytestconfig.getoption('--experimental')\n"
        "    with Path('events').open('a') as output:\n"
        "        output.write(json.dumps(int(os.environ['RANK'])) + '\\n')\n"
    )
    result = _run(project, bucket, repeat=2)
    _assert_success(result)
    assert _events(project) == list(range(ranks)) * 2
    (project / "worker-launches").unlink()
    result = _run(project, bucket, repeat=2, TEST_MISSING_REPORT="1")
    assert result.returncode != 0
    assert _events(project) == list(range(ranks)) * 4
    assert f"rank {ranks - 1} is missing its report in prod-3" in result.stderr


@pytest.mark.parametrize("body", ("pass", "pytest.skip('unexpected skip')"), ids=("empty", "skip"))
def test_full_profiles_reject_empty_or_skipped_execution(project, body):
    (project / FP8).parent.mkdir(parents=True, exist_ok=True)
    (project / FP8).write_text(
        f"import pytest\n\ndef test_mxfp8():\n    {body}\n"
        if body == "pass"
        else f"import pytest\n\ndef test_delayed():\n    {body}\n"
    )
    result = _run(project)
    assert result.returncode != 0
    assert "executed no tests" in result.stderr or "has skipped or failed tests" in result.stderr


@pytest.mark.parametrize(
    "changed_profiles", [("delayed",), ("blockwise",), ("delayed", "blockwise")]
)
def test_testmon_profile_selections_accumulate_without_mutating_baseline(project, changed_profiles):
    pytest.importorskip("testmon")
    _write_precision_tests(project)
    baseline = _run(project, mode="baseline")
    _assert_success(baseline)
    phase = project / "cache/prod"
    before = {path.name: path.read_bytes() for path in phase.iterdir() if path.is_file()}
    source_path = project / "megatron/core/precision.py"
    source = source_path.read_text()
    for profile, original, changed in (
        ("delayed", "1 + 2", "2 + 1"),
        ("blockwise", "3 + 4", "4 + 3"),
    ):
        if profile in changed_profiles:
            source = source.replace(original, changed)
    source_path.write_text(source)
    (project / "events").unlink()
    selected = _run(project, mode="enforce")
    _assert_success(selected)
    assert sorted(name for name, _ in _events(project)) == sorted(changed_profiles)
    assert "selective tests passed" in (project / "cache/summary.md").read_text()
    assert {path.name: path.read_bytes() for path in phase.iterdir() if path.is_file()} == before
