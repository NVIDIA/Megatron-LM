# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for ModelOpt Hugging Face finetuning dataset helpers."""

import sys
from argparse import ArgumentParser
from pathlib import Path

import pytest
import torch

pytest.importorskip("datasets")
pytest.importorskip("modelopt")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
transformers = pytest.importorskip("transformers")

_MODEL_OPT_EXAMPLE_DIR = Path(__file__).parents[3] / "examples" / "post_training" / "modelopt"
sys.path.insert(0, str(_MODEL_OPT_EXAMPLE_DIR))

from finetune import SFTDataset
from utils import get_eos_token_id

from megatron.post_training.arguments import add_modelopt_args


class _Tokenizer:
    eos_token = "<|return|>"
    eos_token_id = 3

    def __init__(self, converted_id):
        self.converted_id = converted_id

    def __len__(self):
        return 10

    def convert_tokens_to_ids(self, _token):
        return self.converted_id


def test_finetune_data_files_argument_accepts_multiple_files():
    parser = ArgumentParser()
    add_modelopt_args(parser)

    args = parser.parse_args(
        [
            "--finetune-data-split",
            "chat",
            "--finetune-data-files",
            "data/chat-00000.parquet",
            "data/chat-00001.parquet",
        ]
    )

    assert args.finetune_data_split == "chat"
    assert args.finetune_data_files == ["data/chat-00000.parquet", "data/chat-00001.parquet"]


def test_materialize_data_files_resolves_local_dataset_directory(tmp_path):
    data_file = tmp_path / "train.jsonl"
    data_file.write_text('{"messages": []}\n')

    resolved = SFTDataset._materialize_data_files(str(tmp_path), ["train.jsonl"])

    assert resolved == [str(data_file)]


def test_materialize_data_files_downloads_explicit_hub_file(monkeypatch):
    calls = []

    def _download(**kwargs):
        calls.append(kwargs)
        return "/cache/chat.parquet"

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _download)

    resolved = SFTDataset._materialize_data_files(
        "nvidia/Nemotron-Post-Training-Dataset-v2", ["data/chat.parquet"]
    )

    assert resolved == ["/cache/chat.parquet"]
    assert calls == [
        {
            "repo_id": "nvidia/Nemotron-Post-Training-Dataset-v2",
            "filename": "data/chat.parquet",
            "repo_type": "dataset",
            "token": None,
        }
    ]


def test_materialize_data_files_rejects_wildcards():
    with pytest.raises(ValueError, match="explicit files"):
        SFTDataset._materialize_data_files("org/dataset", ["data/chat-*.parquet"])


def test_infer_data_files_builder_rejects_mixed_formats():
    with pytest.raises(ValueError, match="single format"):
        SFTDataset._infer_data_files_builder(["train.jsonl", "train.parquet"])


def test_load_parquet_data_files_ignores_huggingface_schema_metadata(tmp_path):
    data_file = tmp_path / "train.parquet"
    table = pa.table({"text": ["sample"]}).replace_schema_metadata(
        {b"huggingface": b"unsupported feature metadata"}
    )
    pq.write_table(table, data_file)

    dataset = SFTDataset._load_parquet_data_files([str(data_file)])

    assert dataset["text"] == ["sample"]


@pytest.mark.parametrize(
    ("input_ids", "expected"),
    [
        ([1, 2], [1, 2]),
        ([[1, 2]], [1, 2]),
        (torch.tensor([[1, 2]]), [1, 2]),
        (transformers.BatchEncoding({"input_ids": [[1, 2]]}), [1, 2]),
    ],
)
def test_normalize_input_ids(input_ids, expected):
    assert SFTDataset._normalize_input_ids(input_ids) == expected


def test_normalize_input_ids_rejects_batches():
    with pytest.raises(ValueError, match="one tokenized chat sample"):
        SFTDataset._normalize_input_ids([[1, 2], [3, 4]])


def test_get_eos_token_id_prefers_valid_converted_id():
    assert get_eos_token_id(_Tokenizer(converted_id=7)) == 7


def test_get_eos_token_id_falls_back_from_out_of_vocab_chat_id():
    assert get_eos_token_id(_Tokenizer(converted_id=199999)) == 3


def test_get_eos_token_id_rejects_out_of_vocab_fallback():
    tokenizer = _Tokenizer(converted_id=199999)
    tokenizer.eos_token_id = 10

    with pytest.raises(ValueError, match="outside the tokenizer vocabulary"):
        get_eos_token_id(tokenizer)
