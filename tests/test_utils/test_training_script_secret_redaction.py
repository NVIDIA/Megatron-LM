# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import subprocess
from pathlib import Path

RUN_CI_TEST = Path("tests/functional_tests/shell_test_utils/run_ci_test.sh")
RUN_TRAINING = Path("tests/functional_tests/shell_test_utils/_run_training.sh")
SENSITIVE_ASSIGNMENTS = {
    "WANDB_API_KEY": "wandb-value-must-not-appear",
    "CI_JOB_TOKEN": "token-value-must-not-appear",
    "INTERNAL_API_URL": "api-value-must-not-appear",
}


def assert_sensitive_values_redacted(result):
    for name, value in SENSITIVE_ASSIGNMENTS.items():
        assert value not in result.stdout
        assert value not in result.stderr
        assert f"{name}=<redacted>" in result.stdout


def test_training_wrappers_redact_sensitive_arguments_from_output():
    sensitive_args = [f"{name}={value}" for name, value in SENSITIVE_ASSIGNMENTS.items()]

    for script in (RUN_CI_TEST, RUN_TRAINING):
        result = subprocess.run(
            ["bash", str(script), *sensitive_args, "VISIBLE_ENV=visible-value"],
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0  # Intentionally omit required training arguments.
        assert_sensitive_values_redacted(result)
        assert "VISIBLE_ENV=visible-value" in result.stdout


def test_run_training_redacts_sensitive_values_from_model_config(tmp_path):
    fake_yq = tmp_path / "yq"
    fake_yq.write_text("""#!/bin/bash
if [[ "$1" == *".ENV_VARS"* ]]; then
    printf 'WANDB_API_KEY=%s\\nCI_JOB_TOKEN=%s\\nINTERNAL_API_URL=%s\\nVISIBLE_ENV=visible-value\\n' \
        "$TEST_KEY_VALUE" "$TEST_TOKEN_VALUE" "$TEST_API_VALUE"
elif [[ "$1" == ".BEFORE_SCRIPT" ]]; then
    printf 'exit 0\\n'
else
    printf 'null\\n'
fi
""")
    fake_yq.chmod(0o755)

    script = tmp_path / "run_training.sh"
    script.write_text(RUN_TRAINING.read_text().replace("/usr/local/bin/yq", str(fake_yq)))
    config = tmp_path / "model_config.yaml"
    config.write_text("ENV_VARS: {}\n")

    required_args = {
        "TRAINING_SCRIPT_PATH": "unused.py",
        "TRAINING_PARAMS_PATH": str(config),
        "OUTPUT_PATH": str(tmp_path / "output"),
        "TENSORBOARD_PATH": str(tmp_path / "tensorboard"),
        "CHECKPOINT_SAVE_PATH": str(tmp_path / "save"),
        "CHECKPOINT_LOAD_PATH": str(tmp_path / "load"),
        "DATA_PATH": str(tmp_path / "data"),
        "RUN_NUMBER": "1",
        "REPEAT": "1",
    }
    result = subprocess.run(
        ["bash", str(script), *(f"{key}={value}" for key, value in required_args.items())],
        check=False,
        capture_output=True,
        env={
            **os.environ,
            "TEST_KEY_VALUE": SENSITIVE_ASSIGNMENTS["WANDB_API_KEY"],
            "TEST_TOKEN_VALUE": SENSITIVE_ASSIGNMENTS["CI_JOB_TOKEN"],
            "TEST_API_VALUE": SENSITIVE_ASSIGNMENTS["INTERNAL_API_URL"],
        },
        text=True,
    )

    assert result.returncode == 0
    assert_sensitive_values_redacted(result)
    assert "VISIBLE_ENV=visible-value" in result.stdout
