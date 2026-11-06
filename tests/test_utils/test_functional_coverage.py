# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU integration tests for coverage across worker processes and workload attempts."""

import json
import os
import subprocess
import sys
import sysconfig
import venv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
COLLECTOR = ROOT / "tests/test_utils/python_scripts/functional_coverage.py"


@pytest.fixture
def workload(tmp_path):
    """Create a disposable interpreter and fake model without GPUs or torch."""
    venv.EnvBuilder(with_pip=False).create(tmp_path / "venv")
    python = tmp_path / "venv/bin/python"
    model = tmp_path / "megatron"
    model.mkdir()
    (model / "__init__.py").write_text("")
    (model / "model.py").write_text(
        "def forward(rank):\n"
        "    if rank == 0:\n"
        "        return 'first'\n"
        "    return 'last'\n"
    )
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import os\n"
        "from multiprocessing import get_context\n"
        "from megatron.model import forward\n"
        "if __name__ == '__main__':\n"
        "    rank = int(os.environ['RANK'])\n"
        "    forward(rank)\n"
        "    child = get_context('spawn').Process(target=forward, args=(rank,))\n"
        "    child.start()\n"
        "    child.join()\n"
        "    assert child.exitcode == 0\n"
    )
    metadata = tmp_path / "manifest.json"
    metadata.write_text(json.dumps({"platform": "cpu", "model": "gpt", "test_case": "fake"}))
    env = {
        **os.environ,
        # Make the test runner's installed coverage package visible, but use
        # the disposable interpreter's own site-packages for its startup hook.
        "PYTHONPATH": os.pathsep.join([str(tmp_path), sysconfig.get_path("purelib")]),
    }
    env.pop("COVERAGE_PROCESS_START", None)
    return python, metadata, env


@pytest.mark.parametrize("exit_code", [0, 7])
def test_collects_all_ranks_phases_and_preserves_failure(tmp_path, workload, exit_code):
    """Training and multiprocessing children contribute data even on a failed attempt."""
    python, metadata, env = workload
    script = tmp_path / "run.sh"
    script.write_text(
        "set -eu\n"
        "for phase in 1 2; do\n"
        "  for rank in 0 1; do\n"
        f'    RANK="$rank" RUN_NUMBER="$phase" REPEAT=1 "{python}" worker.py\n'
        "  done\n"
        "done\n"
        f"exit {exit_code}\n"
    )
    for attempt in range(2):
        output = tmp_path / f"attempt-{attempt}"
        result = subprocess.run(
            [
                str(python),
                str(COLLECTOR),
                "--script",
                str(script),
                "--metadata",
                str(metadata),
                "--output",
                str(output),
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == exit_code, result.stdout + result.stderr
        manifest = json.loads((output / "metadata.json").read_text())
        assert manifest["workload_exit_code"] == exit_code
        assert manifest["collection_status"] == "collected"
        assert manifest["observed_ranks"] == [0, 1]
        assert len(manifest["raw_data_files"]) >= 8
        assert len(manifest["contexts"]) == 4
        report = json.loads((output / "coverage.json").read_text())
        assert report["meta"]["branch_coverage"] is True
        assert report["files"]["megatron/model.py"]["missing_lines"] == []
        assert report["files"]["megatron/model.py"]["missing_branches"] == []
        assert not list((tmp_path / "venv").rglob("_mcore_functional_coverage.pth"))


def test_missing_coverage_fails_collection(tmp_path, workload):
    """A shell-only success must not masquerade as a coverage baseline."""
    python, metadata, env = workload
    script = tmp_path / "empty.sh"
    script.write_text("exit 0\n")
    output = tmp_path / "attempt-0"
    result = subprocess.run(
        [
            str(python),
            str(COLLECTOR),
            "--script",
            str(script),
            "--metadata",
            str(metadata),
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    manifest = json.loads((output / "metadata.json").read_text())
    assert manifest["workload_exit_code"] == 0
    assert manifest["collection_status"] == "error"
