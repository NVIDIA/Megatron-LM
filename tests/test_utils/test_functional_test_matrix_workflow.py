# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[2]


@pytest.fixture
def jobs():
    return yaml.safe_load((ROOT / ".github/workflows/cicd-main.yml").read_text())["jobs"]


@pytest.mark.parametrize("platform", ["h100", "gb200"])
@pytest.mark.parametrize("head_sha", ["", "a" * 40])
@pytest.mark.parametrize("cadence", ["", "pr"])
@pytest.mark.parametrize("scope", ["L0", "L1"])
def test_functional_parse_shell_passes_selection_and_publishes_matrix(
    tmp_path, jobs, platform, head_sha, cadence, scope
):
    step = next(
        step
        for step in jobs[f"cicd-parse-integration-tests-{platform}"]["steps"]
        if step.get("id") == "main"
    )
    binaries = tmp_path / "bin"
    binaries.mkdir()
    (binaries / "python").symlink_to(sys.executable)
    generator = tmp_path / "tests/test_utils/python_scripts/generate_functional_test_matrix.py"
    generator.parent.mkdir(parents=True)
    matrix = [{"model": "gpt", "test_case": "changed", "scope": "L2", "cadence": ""}]
    generator.write_text(
        "import argparse, json\n"
        "from pathlib import Path\n"
        "parser = argparse.ArgumentParser()\n"
        "for option in ('scope', 'platform', 'cadence'):\n"
        "    parser.add_argument('--' + option, required=True)\n"
        "parser.add_argument('--base-ref')\n"
        "parser.add_argument('--head-ref')\n"
        "Path('arguments.json').write_text(json.dumps(vars(parser.parse_args())))\n"
        f"print(json.dumps({matrix!r}, separators=(',', ':')))\n"
    )
    output = tmp_path / "outputs"
    result = subprocess.run(
        ["bash", "-e", "-u", "-o", "pipefail"],
        input=step["run"],
        text=True,
        capture_output=True,
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{binaries}{os.pathsep}{os.environ['PATH']}",
            "SCOPE": scope,
            "CADENCE": cadence,
            "HEAD_SHA": head_sha,
            "GITHUB_OUTPUT": str(output),
        },
    )
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "arguments.json").read_text()) == {
        "scope": scope,
        "platform": f"dgx_{platform}",
        "cadence": cadence,
        "base_ref": "HEAD^1" if head_sha and scope == "L0" else None,
        "head_ref": (head_sha or None) if scope == "L0" else None,
    }
    name, value = output.read_text().strip().split("=", 1)
    assert name == f"integration-tests-{platform}"
    assert json.loads(value) == matrix


@pytest.mark.parametrize("platform", ["h100", "gb200"])
def test_functional_runner_uses_each_selected_scope_and_cadence(jobs, platform):
    inputs = next(
        step["with"]
        for step in jobs[f"cicd-integration-tests-latest-{platform}"]["steps"]
        if step.get("uses") == "./.github/actions"
    )
    assert inputs["is_unit_test"] == "false"
    assert inputs["scope"] == "${{ matrix.scope }}"
    assert inputs["cadence"] == "${{ matrix.cadence }}"
    assert inputs["n_repeat"] == "${{ needs.configure.outputs.n_repeat }}"
    assert inputs["lightweight"] == "${{ needs.configure.outputs.lightweight }}"

    unit_job = "cicd-unit-tests-latest" + ("-gb200" if platform == "gb200" else "")
    unit_inputs = next(
        step["with"] for step in jobs[unit_job]["steps"] if step.get("uses") == "./.github/actions"
    )
    assert unit_inputs["is_unit_test"] == "true"
    assert unit_inputs["test_case"] == "${{ matrix.bucket }}"
    assert "scope" not in unit_inputs
    assert "cadence" not in unit_inputs
