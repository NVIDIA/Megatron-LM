# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Arguments for ModelOpt finetuning."""

from argparse import ArgumentParser

import pytest

pytest.importorskip("datasets")
pytest.importorskip("modelopt")
pytest.importorskip("pyarrow")
pytest.importorskip("pyarrow.parquet")
pytest.importorskip("transformers")

from megatron.post_training.arguments import add_modelopt_args


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
