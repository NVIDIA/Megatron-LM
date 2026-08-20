# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import hashlib
import numpy as np
import runpy
import sys
import tempfile

from megatron.core.datasets.indexed_dataset import IndexedDataset


def get_sha256(filename):
    sha256_hash = hashlib.sha256()
    with open(filename, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def test_preprocess_data_fast_hf():
    """Test HF tokenizer data preprocessing"""
    with tempfile.TemporaryDirectory() as temp_dir:

        # default script args
        default = f"{temp_dir}/default"
        default_args = [
            "--input",
            "/opt/data/datasets/dclm/dclm.jsonl",
            "--output-prefix",
            default,
            "--tokenizer-type",
            "HuggingFaceTokenizer",
            "--tokenizer-model",
            "/opt/data/tokenizers/huggingface",
            "--tokenizer-hf-no-include-special-tokens",
            "--append-eod",
            "--workers",
            "2",
            "--log-interval",
            "1000",
        ]
        sys.argv = ["/opt/megatron-lm/tools/preprocess_data.py"] + default_args
        runpy.run_path("/opt/megatron-lm/tools/preprocess_data.py", run_name="__main__")

        # fast script args
        fast = f"{temp_dir}/fast"
        fast_args = [
            "--input",
            "/opt/data/datasets/dclm/dclm.jsonl",
            "--output-prefix",
            fast,
            "--tokenizer-type",
            "HuggingFaceTokenizer",
            "--tokenizer-model",
            "/opt/data/tokenizers/huggingface",
            "--tokenizer-hf-no-include-special-tokens",
            "--use-gigatoken",
            "--append-eod",
        ]
        sys.argv = ["/opt/megatron-lm/tools/preprocess_data_fast.py"] + fast_args
        runpy.run_path("/opt/megatron-lm/tools/preprocess_data_fast.py", run_name="__main__")

        # Compare the output files
        default_ds_path = f"{default}_text_document"
        fast_ds_path = f"{fast}_text_document"
        default_ds = IndexedDataset(default_ds_path)
        fast_ds = IndexedDataset(fast_ds_path)

        assert len(default_ds) == len(fast_ds)
        for doc1, doc2 in zip(default_ds, fast_ds):
            assert len(doc1) == len(doc2)
            assert np.array_equal(doc1, doc2)

        # Verify hash is the same for bin/idx files
        assert get_sha256(f"{default_ds_path}.idx") == get_sha256(f"{fast_ds_path}.idx")
        assert get_sha256(f"{default_ds_path}.bin") == get_sha256(f"{fast_ds_path}.bin")


def test_preprocess_data_fast_megatron():
    """Test Megatron tokenizer data preprocessing"""
    with tempfile.TemporaryDirectory() as temp_dir:

        # default script args
        default = f"{temp_dir}/default"
        default_args = [
            "--input",
            "/opt/data/datasets/dclm/dclm.jsonl",
            "--output-prefix",
            default,
            "--tokenizer-type",
            "GPT2BPETokenizer",
            "--vocab-file",
            "/opt/data/tokenizers/megatron/gpt2-vocab.json",
            "--merge-file",
            "/opt/data/tokenizers/megatron/gpt2-merges.txt",
            "--append-eod",
            "--workers",
            "2",
            "--log-interval",
            "1000",
        ]
        sys.argv = ["/opt/megatron-lm/tools/preprocess_data.py"] + default_args
        runpy.run_path("/opt/megatron-lm/tools/preprocess_data.py", run_name="__main__")

        # fast script args
        fast = f"{temp_dir}/fast"
        fast_args = [
            "--input",
            "/opt/data/datasets/dclm/dclm.jsonl",
            "--output-prefix",
            fast,
            "--tokenizer-type",
            "GPT2BPETokenizer",
            "--vocab-file",
            "/opt/data/tokenizers/megatron/gpt2-vocab.json",
            "--merge-file",
            "/opt/data/tokenizers/megatron/gpt2-merges.txt",
            "--use-gigatoken",
            "--append-eod",
        ]
        sys.argv = ["/opt/megatron-lm/tools/preprocess_data_fast.py"] + fast_args
        runpy.run_path("/opt/megatron-lm/tools/preprocess_data_fast.py", run_name="__main__")

        # Compare the output files
        default_ds_path = f"{default}_text_document"
        fast_ds_path = f"{fast}_text_document"
        default_ds = IndexedDataset(default_ds_path)
        fast_ds = IndexedDataset(fast_ds_path)

        assert len(default_ds) == len(fast_ds)
        for doc1, doc2 in zip(default_ds, fast_ds):
            assert len(doc1) == len(doc2)
            assert np.array_equal(doc1, doc2)

        # Verify hash is the same for bin/idx files
        assert get_sha256(f"{default_ds_path}.idx") == get_sha256(f"{fast_ds_path}.idx")
        assert get_sha256(f"{default_ds_path}.bin") == get_sha256(f"{fast_ds_path}.bin")


if __name__ == "__main__":
    test_preprocess_data_fast_hf()
    test_preprocess_data_fast_megatron()
