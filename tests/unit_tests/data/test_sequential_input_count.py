# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import glob
import gzip
import json
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
@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("empty_shard", [False, True])
def test_preprocess_sequential_input_shards(tmp_path, compressed, empty_shard):
    """Compressed and empty shards preserve the document order and partition counts."""
    suffix = ".jsonl.gz" if compressed else ".jsonl"
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    shards = ([[]] if empty_shard else []) + [[[1, 2], [3, 4]], [[5, 6]]]
    for index, shard in enumerate(shards):
        source = source_dir / f"{index}{suffix}"
        opener = gzip.open if compressed else open
        with opener(source, "wt", encoding="utf-8") as writer:
            for tokens in shard:
                writer.write(json.dumps({"text": " ".join(map(str, tokens))}) + "\n")
    expected = []
    for filename in glob.glob(str(source_dir / f"*{suffix}")):
        opener = gzip.open if compressed else open
        with opener(filename, "rt", encoding="utf-8") as reader:
            expected.extend(
                [int(token) for token in json.loads(line)["text"].split()] for line in reader
            )
    prefix, returncode, stdout, stderr = _run_preprocessing(
        tmp_path, source_dir / f"*{suffix}", ("--partitions", "2", "--keep-sequential-samples")
    )
    assert returncode == 0, (stdout, stderr)
    dataset = IndexedDataset(str(prefix) + "_text_document")
    assert [item.tolist() for item in dataset[:]] == expected
    first_partition = IndexedDataset(str(prefix) + "_0_text_document")
    assert [item.tolist() for item in first_partition[:]] == expected[:2]
