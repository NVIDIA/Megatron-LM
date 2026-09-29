# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

RUN_CI_TEST = (
    Path(__file__).resolve().parents[1] / "functional_tests/shell_test_utils/run_ci_test.sh"
)


def write_executable(path, source):
    path.write_text(source)
    path.chmod(0o755)


@pytest.mark.parametrize("test_type", ["regular", "ckpt-resume"])
@pytest.mark.parametrize("actual_filename", [None, "golden_values_dev_dgx_gb300.json"])
def test_actual_paths_are_independent_of_golden_reference(tmp_path, test_type, actual_filename):
    shell_dir = tmp_path / "repo/tests/functional_tests/shell_test_utils"
    shell_dir.mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls_path = tmp_path / "uv-calls.jsonl"
    training_calls_path = tmp_path / "training-calls.txt"

    fake_yq = bin_dir / "yq"
    write_executable(
        fake_yq,
        f"#!{sys.executable}\n" """import os
import sys

if sys.argv[1] == '-i':
    sys.exit(0)
sys.stdin.read()
values = {
    '.TEST_TYPE': os.environ['TEST_TYPE'],
    '.TEST_EVALUATION // "pass"': 'pass',
    '.ENV_VARS.ENABLE_LIGHTWEIGHT_MODE // "false"': 'false',
    '.ENV_VARS.N_REPEAT // "1"': '1',
    '.MODE // "pretraining"': 'pretraining',
    '.MODEL_ARGS."--exit-interval" // "100"': '4',
    '.ENV_VARS.NVTE_ALLOW_NONDETERMINISTIC_ALGO': '0',
    '.ENV_VARS.NON_DETERMINSTIC_RESULTS // "0"': '0',
    '.ENV_VARS.SKIP_PYTEST': '0',
}
print(values[sys.argv[1]])
""",
    )
    script = shell_dir / "run_ci_test.sh"
    script.write_text(RUN_CI_TEST.read_text().replace("/usr/local/bin/yq", str(fake_yq)))
    write_executable(
        shell_dir / "_run_training.sh",
        '#!/bin/bash\nprintf "%s\\n" "$RUN_NUMBER" >> "$TRAINING_CALLS"\n',
    )
    write_executable(
        bin_dir / "uv",
        f"#!{sys.executable}\n" """import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
with open(os.environ['UV_CALLS'], 'a') as stream:
    stream.write(json.dumps(args) + '\\n')
assert args[:2] == ['run', '--no-sync'], args
if args[2] == 'python':
    assert args[3].endswith('/get_test_results_from_tensorboard_logs.py'), args
    output = Path(args[args.index('--output-path') + 1])
    output.write_text('{"actual": true}')
elif args[2] == 'pytest':
    for flag in ('--golden-values-path', '--actual-values-path',
                 '--actual-values-first-run-path', '--actual-values-second-run-path'):
        if flag in args:
            assert Path(args[args.index(flag) + 1]).is_file(), args
else:
    raise AssertionError(args)
""",
    )

    config = tmp_path / "model_config.yaml"
    config.write_text("{}\n")
    reference = tmp_path / "golden_values_dev_dgx_gb200.json"
    reference_contents = '{"reference": true}\n'
    reference.write_text(reference_contents)
    output_dir = tmp_path / "output"
    actual_path = output_dir / (actual_filename or reference.name)
    arguments = {
        "TRAINING_SCRIPT_PATH": "pretrain_gpt.py",
        "TRAINING_PARAMS_PATH": str(config),
        "GOLDEN_VALUES_PATH": str(reference),
        "OUTPUT_PATH": str(output_dir),
        "TENSORBOARD_PATH": str(tmp_path / "tensorboard"),
        "CHECKPOINT_SAVE_PATH": str(tmp_path / "save"),
        "CHECKPOINT_LOAD_PATH": str(tmp_path / "load"),
        "DATA_PATH": str(tmp_path / "data"),
        "DATA_CACHE_PATH": str(tmp_path / "cache"),
        "N_REPEAT": "1",
    }
    if actual_filename is not None:
        arguments["ACTUAL_VALUES_PATH"] = str(actual_path)

    result = subprocess.run(
        ["bash", str(script), *(f"{key}={value}" for key, value in arguments.items())],
        env={
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "TEST_TYPE": test_type,
            "UV_CALLS": str(calls_path),
            "TRAINING_CALLS": str(training_calls_path),
            "NUM_NODES": "1",
            "NODE_RANK": "0",
            "ENABLE_LIGHTWEIGHT_MODE": "false",
            "RECORD_CHECKPOINTS": "false",
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
    expected_runs = ["1", "2"] if test_type == "ckpt-resume" else ["1"]
    assert training_calls_path.read_text().splitlines() == expected_runs
    assert len(calls) == 2 * len(expected_runs)

    def argument(call, flag):
        return call[call.index(flag) + 1]

    assert argument(calls[0], "--output-path") == str(actual_path)
    assert calls[1][2] == "pytest"
    assert argument(calls[1], "--golden-values-path") == str(reference)
    assert argument(calls[1], "--actual-values-path") == str(actual_path)
    expected_outputs = {actual_path.name}
    if test_type == "ckpt-resume":
        second_path = actual_path.with_name(f"{actual_path.stem}_2nd.json")
        assert argument(calls[0], "--logs-dir").endswith("/run_1")
        assert argument(calls[2], "--logs-dir").endswith("/run_2")
        assert argument(calls[2], "--output-path") == str(second_path)
        assert calls[3][2] == "pytest"
        assert argument(calls[3], "--actual-values-first-run-path") == str(actual_path)
        assert argument(calls[3], "--actual-values-second-run-path") == str(second_path)
        expected_outputs.add(second_path.name)
    assert {path.name for path in output_dir.glob("*.json")} == expected_outputs
    assert reference.read_text() == reference_contents
