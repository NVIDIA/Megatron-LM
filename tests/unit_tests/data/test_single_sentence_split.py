# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
import os
import pickle
import signal
import subprocess
import sys
from pathlib import Path

import pytest
from nltk.tokenize.punkt import PunktParameters, PunktSentenceTokenizer, save_punkt_params

from megatron.core.datasets.indexed_dataset import IndexedDataset


def _run_preprocessing(tmp_path, source, extra_args=(), timeout=120):
    """Run the actual CLI and clean up its workers if the bounded check times out."""
    repo = Path(__file__).resolve().parents[3]
    prefix = tmp_path / "output"
    runner = """
import os
import runpy
import sys
from unittest.mock import patch

import nltk

nltk.data.path = [os.environ.pop("MEGATRON_TEST_NLTK_DATA")]
sys.argv = sys.argv[1:]
with patch("nltk.download", return_value=True):
    runpy.run_path(sys.argv[0], run_name="__main__")
"""
    command = [
        sys.executable,
        "-c",
        runner,
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
        env={
            **{key: value for key, value in os.environ.items() if key != "NLTK_DATA"},
            "MEGATRON_TEST_NLTK_DATA": str(tmp_path / "nltk_data"),
        },
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            pytest.fail(f"Preprocessing hung after {timeout}s. stdout={stdout} stderr={stderr}")
    return prefix, process.returncode, stdout, stderr


@pytest.mark.skipif(os.name != "posix", reason="Preprocessing uses fork-based workers")
def test_single_partition_sentence_split_produces_dataset(tmp_path):
    """A fresh sentence-split input must be tokenized in the same invocation."""
    # Generate real NLTK resources for both its legacy pickle and current tabular loader.
    nltk_data = tmp_path / "nltk_data" / "tokenizers"
    punkt_dir = nltk_data / "punkt"
    punkt_dir.mkdir(parents=True)
    parameters = PunktParameters()
    with (punkt_dir / "english.pickle").open("wb") as writer:
        pickle.dump(PunktSentenceTokenizer(parameters), writer)
    punkt_tab_dir = nltk_data / "punkt_tab" / "english"
    punkt_tab_dir.mkdir(parents=True)
    save_punkt_params(parameters, dir=str(punkt_tab_dir))
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps({"text": "1 2 3"}) + "\n", encoding="utf-8")
    prefix, returncode, stdout, stderr = _run_preprocessing(
        tmp_path, source, ("--split-sentences",)
    )
    assert returncode == 0, (stdout, stderr)
    dataset_prefix = str(prefix) + "_text_sentence"
    assert IndexedDataset.exists(dataset_prefix), (stdout, stderr)
    dataset = IndexedDataset(dataset_prefix)
    assert dataset.document_indices.tolist() == [0, 1]
    assert dataset[0].tolist() == [1, 2, 3]
