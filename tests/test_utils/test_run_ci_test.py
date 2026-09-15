# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
import os
import subprocess
from pathlib import Path

import pytest

from tests.test_utils.python_scripts.recipe_parser import load_and_flatten

RUN_CI_TEST = Path("tests/functional_tests/shell_test_utils/run_ci_test.sh")


@pytest.fixture
def run_harness(tmp_path):
    """Run the real shell control flow with stubbed training and metric commands."""
    script_dir = tmp_path / RUN_CI_TEST.parent
    script_dir.mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_yq = bin_dir / "yq"
    fake_yq.write_text("""#!/bin/bash
if [[ "$1" != "-i" ]]; then cat > /dev/null; fi
case "$1" in
    -i) exit 0 ;;
    .TEST_TYPE) echo "$HARNESS_TEST_TYPE" ;;
    .TEST_EVALUATION*) echo pass ;;
    '.MODE '*) echo pretraining ;;
    .ENV_VARS.ENABLE_LIGHTWEIGHT_MODE*) echo false ;;
    .ENV_VARS.N_REPEAT*) echo "$N_REPEAT" ;;
    .MODEL_ARGS*) echo 20 ;;
    *) echo 0 ;;
esac
""")
    fake_yq.chmod(0o755)
    fake_uv = bin_dir / "uv"
    fake_uv.write_text("""#!/usr/bin/env python3
import json
import os
import sys
with open(os.environ["HARNESS_CALLS"], "a") as stream:
    stream.write(json.dumps(sys.argv[1:]) + "\\n")
if "pytest" in sys.argv and os.environ.get("HARNESS_FAIL_VALIDATION") == "1":
    sys.exit(1)
""")
    fake_uv.chmod(0o755)
    (script_dir / "_run_training.sh").write_text("""#!/bin/bash
set -e
printf '%s:%s\\n' "$REPEAT" "$RUN_NUMBER" >> "$HARNESS_TRAINING_CALLS"
if [[ "$HARNESS_FAIL_TRAINING" == "1" ]]; then exit 1; fi
if [[ "$RUN_NUMBER" == "2" ]]; then
    test "$(cat "$CHECKPOINT_LOAD_PATH/latest_checkpointed_iteration.txt")" = 10
    test -d "$CHECKPOINT_LOAD_PATH/iter_0000010"
    test ! -d "$CHECKPOINT_LOAD_PATH/iter_0000020"
fi
mkdir -p "$CHECKPOINT_SAVE_PATH/iter_0000010" "$CHECKPOINT_SAVE_PATH/iter_0000020"
""")
    script = script_dir / RUN_CI_TEST.name
    script.write_text(
        RUN_CI_TEST.read_text()
        .replace("/usr/local/bin/yq", str(fake_yq))
        .replace("/tmp/checkpoints/", str(tmp_path / "scratch_checkpoints") + "/")
    )
    config = tmp_path / "model_config.yaml"
    config.write_text("TEST_TYPE: ckpt-resume\n")
    calls_path = tmp_path / "calls.jsonl"
    training_path = tmp_path / "training.txt"

    def run(test_type="ckpt-resume", golden="", fail_validation=False, fail_training=False):
        arguments = {
            "TRAINING_SCRIPT_PATH": "unused.py",
            "TRAINING_PARAMS_PATH": str(config),
            "GOLDEN_VALUES_PATH": golden,
            "OUTPUT_PATH": str(tmp_path / "output"),
            "TENSORBOARD_PATH": str(tmp_path / "tensorboard"),
            "CHECKPOINT_SAVE_PATH": str(tmp_path / "save"),
            "CHECKPOINT_LOAD_PATH": str(tmp_path / "load"),
            "DATA_PATH": str(tmp_path / "data"),
            "DATA_CACHE_PATH": str(tmp_path / "cache"),
            "N_REPEAT": "2",
            "NUM_NODES": "1",
            "NODE_RANK": "0",
            "SLURM_NODEID": "0",
            "ENABLE_LIGHTWEIGHT_MODE": "false",
            "RECORD_CHECKPOINTS": "false",
            "RUN_CI_BARRIER_DIR": str(tmp_path / "barriers"),
        }
        result = subprocess.run(
            ["bash", str(script), *(f"{key}={value}" for key, value in arguments.items())],
            capture_output=True,
            text=True,
            check=False,
            env={
                **os.environ,
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
                "HARNESS_TEST_TYPE": test_type,
                "HARNESS_CALLS": str(calls_path),
                "HARNESS_TRAINING_CALLS": str(training_path),
                "HARNESS_FAIL_VALIDATION": str(int(fail_validation)),
                "HARNESS_FAIL_TRAINING": str(int(fail_training)),
            },
        )
        calls = (
            [json.loads(line) for line in calls_path.read_text().splitlines()]
            if calls_path.exists()
            else []
        )
        training = training_path.read_text().splitlines() if training_path.exists() else []
        return result, calls, training

    return run


@pytest.mark.parametrize("golden", ["", "/reference/golden_values_dev_dgx_gb200.json"])
def test_checkpoint_resume_runs_all_repetitions_and_checks(run_harness, golden):
    result, calls, training = run_harness(golden=golden)
    assert result.returncode == 0, result.stderr
    assert training == ["1:1", "1:2", "2:1", "2:2"]
    validators = [call for call in calls if "pytest" in call]
    assert len(validators) == (4 if golden else 2)
    filename = Path(golden).name if golden else "actual_values.json"
    extractions = [call for call in calls if "--output-path" in call]
    assert [Path(call[call.index("--output-path") + 1]).name for call in extractions] == [
        filename,
        filename.removesuffix(".json") + "_2nd.json",
    ] * 2
    for call in validators:
        if "--golden-values-path" in call:
            assert golden
            assert call[call.index("--golden-values-path") + 1] == golden
        else:
            checks = [Path(arg).name for arg in call if arg.endswith(".py")]
            assert checks == ([] if golden else ["test_pretraining_functional_pipeline.py"]) + [
                "test_pretraining_resume_checkpoint_pipeline.py"
            ]
            assert Path(call[call.index("--actual-values-first-run-path") + 1]).name == filename
            assert Path(call[call.index("--actual-values-second-run-path") + 1]).name == (
                filename.removesuffix(".json") + "_2nd.json"
            )


def test_regular_test_still_requires_a_golden_path(run_harness):
    result, calls, training = run_harness(test_type="regular")
    assert result.returncode != 0
    assert "Providing $GOLDEN_VALUES_PATH is mandatory" in result.stdout
    assert not calls
    assert not training


@pytest.mark.parametrize("golden", ["", "/reference/golden.json"])
def test_validation_failure_is_not_skipped(run_harness, golden):
    result, _, training = run_harness(golden=golden, fail_validation=True)
    assert result.returncode != 0
    assert training == ["1:1", "1:2"]


def test_training_failure_is_not_skipped(run_harness):
    result, _, training = run_harness(fail_training=True)
    assert result.returncode != 0
    assert training == ["1:1"]


@pytest.mark.parametrize("puzzle", [False, True])
def test_nemotron_launch_resolves_golden_path_without_changing_lightning(puzzle):
    workloads = load_and_flatten("tests/test_utils/recipes/gb200/nemotron.yaml")
    spec = next(
        workload.spec
        for workload in workloads
        if ("puzzle" in workload.spec["test_case"]) == puzzle
    )
    script = spec["script"].format(
        **spec, assets_dir="/test/assets", artifacts_dir="/test/artifacts"
    )
    # Execute only argument construction, without setup, training, or remote access.
    argument_script = script[script.index("GOLDEN_VALUES_PATH=") :]
    argument_script = argument_script[: argument_script.index("bash ./tests/")]
    result = subprocess.run(
        ["bash", "-c", argument_script + '\nprintf "%s\\n" "${ARGUMENTS[@]}"'],
        capture_output=True,
        text=True,
        check=True,
    )
    arguments = dict(line.split("=", 1) for line in result.stdout.splitlines())
    assert arguments["TRAINING_SCRIPT_PATH"] == (
        "examples/hybrid/puzzle.py" if puzzle else "pretrain_hybrid.py"
    )
    assert arguments["GOLDEN_VALUES_PATH"] == (
        ""
        if puzzle
        else f"./tests/functional_tests/test_cases/nemotron/{spec['test_case']}/golden_values_dev_dgx_gb200.json"
    )
