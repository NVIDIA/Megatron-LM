# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Protect process-local tokenizer defaults without exhausting host thread limits."""

import multiprocessing
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(os.name != "posix", reason="requires process-group cleanup")
@pytest.mark.parametrize("start_method", multiprocessing.get_all_start_methods())
@pytest.mark.parametrize(
    "parallelism,enable_parallelism,expected",
    [
        pytest.param(None, False, "false", id="default"),
        pytest.param("true", False, "true", id="env-true"),
        pytest.param("false", False, "false", id="env-false"),
        pytest.param(None, True, "true", id="flag"),
        pytest.param("false", True, "false", id="env-overrides-flag"),
    ],
)
def test_preprocess_parallelism(
    tmp_path: Path,
    start_method: str,
    parallelism: str | None,
    enable_parallelism: bool,
    expected: str,
) -> None:
    """Check initialization order, process isolation, overrides, and encoded output."""
    env = dict(os.environ)
    env.pop("TOKENIZERS_PARALLELISM", None)
    if parallelism is not None:
        env["TOKENIZERS_PARALLELISM"] = parallelism
    # Keep this test CPU-only even when launched by the distributed unit runner.
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    root = Path(__file__).resolve().parents[3]
    command = [
        sys.executable,
        "-m",
        "tests.unit_tests.data.preprocess_parallelism_runner",
        start_method,
        str(tmp_path),
        expected,
    ]
    if enable_parallelism:
        command.append("--tokenizer-parallelism")
    with subprocess.Popen(
        command,
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            output, _ = process.communicate(timeout=240)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate(timeout=10)
            pytest.fail(f"Preprocessing timed out ({start_method}, {parallelism}):\n{output}")
        finally:
            # Also remove orphaned pool workers/forkservers if the controller
            # exited early. All descendants belong to this isolated session.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        assert process.returncode == 0, output
