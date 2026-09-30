# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from megatron.core.datasets.indexed_dataset import IndexedDataset


def _run_preprocessing(tmp_path, source, extra_args=(), timeout=120):
    """Run the actual CLI and clean up its workers if the bounded check times out."""
    repo = Path(__file__).resolve().parents[3]
    prefix = tmp_path / "output"
    command = [
        sys.executable,
        str(repo / "tools/preprocess_data.py"),
        "--input",
        str(source),
        "--output-prefix",
        str(prefix),
        "--tokenizer-type",
        "NullTokenizer",
        "--vocab-size",
        "128",
        "--workers",
        "2",
        *extra_args,
    ]
    with subprocess.Popen(
        command,
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            pytest.fail(f"Preprocessing hung after {timeout}s. stdout={stdout} stderr={stderr}")
    return prefix, process.returncode, stdout, stderr


@pytest.mark.skipif(os.name != "posix", reason="Preprocessing uses fork-based workers")
@pytest.mark.parametrize("failure", ["malformed_json", "missing_file"])
def test_preprocess_partition_failure_exits(tmp_path, failure):
    """A child failure must terminate the parent instead of blocking on the queue."""
    source = tmp_path / "input.jsonl"
    if failure == "malformed_json":
        source.write_text("{invalid JSON}\n", encoding="utf-8")
    _, returncode, stdout, stderr = _run_preprocessing(tmp_path, source, timeout=60)
    assert returncode != 0, (stdout, stderr)
    assert "Preprocessing worker" in stderr
    assert "exit code" in stderr


@pytest.mark.skipif(os.name != "posix", reason="Preprocessing uses fork-based workers")
def test_preprocess_partition_success_still_merges(tmp_path):
    """Successful workers still publish results and merge all partition outputs."""
    source = tmp_path / "input.jsonl"
    source.write_text(' {"text":"1 2"}\n{"text":"3 4"}\n', encoding="utf-8")
    prefix, returncode, stdout, stderr = _run_preprocessing(tmp_path, source, ("--partitions", "2"))
    assert returncode == 0, (stdout, stderr)
    dataset = IndexedDataset(str(prefix) + "_text_document")
    assert [item.tolist() for item in dataset[:]] == [[1, 2], [3, 4]]
