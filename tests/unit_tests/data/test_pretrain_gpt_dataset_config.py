# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import sys
from types import SimpleNamespace

from megatron.training.arguments import parse_args


def test_core_gpt_dataset_config_forwards_sft_mock_dataset_config(monkeypatch):
    import pretrain_gpt

    config_json = (
        '{"mode":"distribution","type":"lognormal","format":"thd",'
        '"min_seq_len":128,"max_seq_len":2048,"mean_seq_len":512,'
        '"lognormal_sigma":1.2}'
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["pretrain_gpt.py", "--seq-length", "2048", "--sft-mock-dataset-config-json", config_json],
    )
    args = parse_args()
    args.data_parallel_size = args.world_size // (
        args.tensor_model_parallel_size
        * args.pipeline_model_parallel_size
        * args.context_parallel_size
    )
    monkeypatch.setattr(pretrain_gpt, "build_tokenizer", lambda _: SimpleNamespace(vocab_size=2048))
    monkeypatch.setattr(pretrain_gpt, "get_blend_and_blend_per_split", lambda _: (None, None))

    config = pretrain_gpt.core_gpt_dataset_config_from_args(args)

    assert config.sft_mock_dataset_config_json == config_json
