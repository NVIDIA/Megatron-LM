# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import shlex
import subprocess
from pathlib import Path

import pytest

from tests.test_utils.python_scripts import recipe_parser


@pytest.mark.parametrize("cadence", ["pr", "mergegroup"])
def test_packed_train_eval_runner_selects_one_workload_for_both_modes(cadence, tmp_path):
    # Match the launcher's singular test_case lookup, including build filtering.
    workloads = recipe_parser.load_workloads(
        container_image="none",
        container_tag="none",
        scope="L1",
        model="gpt",
        test_case="gpt_packed_train_eval",
        environment="dev",
        platform="dgx_h100",
        cadence=cadence,
    )
    workloads = [workload for workload in workloads if workload.type != "build"]
    assert len(workloads) == 1

    workload = workloads[0]
    magic_values = {
        **workload.spec,
        "assets_dir": str(tmp_path / "assets"),
        "artifacts_dir": str(tmp_path / "artifacts"),
        "n_repeat": 2,
    }
    magic_values["name"] = workload.spec["name"].format(**magic_values)
    script = workload.spec["script"].format(**magic_values)

    # Exercise the rendered shell loops without launching GPU training. Only
    # translate the container mount and replace uv with a command recorder.
    repo_dir = Path(__file__).resolve().parents[2]
    script = script.replace("cd /opt/megatron-lm", f"cd {shlex.quote(str(repo_dir))}")
    call_log = tmp_path / "calls.log"
    subprocess.run(
        ["bash", "-c", 'uv() { printf "%s\\n" "$*" >> "$RECIPE_CALL_LOG"; }\n' + script],
        check=True,
        env={**os.environ, "RECIPE_CALL_LOG": str(call_log)},
    )

    calls = [shlex.split(line) for line in call_log.read_text().splitlines()]
    assert calls == [
        [
            "run",
            "--no-sync",
            "python",
            "tests/functional_tests/test_cases/gpt/gpt_packed_train_eval/run_test.py",
            "--packing-mode",
            mode,
            "--output-path",
            str(tmp_path / "assets" / f"packed-{mode}-{repeat}"),
        ]
        for repeat in (1, 2)
        for mode in ("static", "dynamic")
    ]
