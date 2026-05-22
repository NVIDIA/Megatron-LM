# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import argparse

import pytest

from megatron.training import arguments


def test_add_megatron_arguments_registers_training_parser_groups():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    returned_parser = arguments.add_megatron_arguments(parser)
    group_titles = {group.title for group in parser._action_groups}

    assert returned_parser is parser
    assert "network size" in group_titles
    assert "training" in group_titles
    assert "learning rate and weight decay" in group_titles
    assert "checkpointing" in group_titles
    assert "distributed init" in group_titles
    assert "validation" in group_titles
    assert "data and dataloader" in group_titles
    assert "tokenizer" in group_titles


def test_parser_accepts_representative_training_arguments():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    arguments.add_megatron_arguments(parser)

    parsed = parser.parse_args(
        [
            "--num-layers",
            "2",
            "--hidden-size",
            "16",
            "--num-attention-heads",
            "4",
            "--seq-length",
            "8",
            "--max-position-embeddings",
            "8",
            "--micro-batch-size",
            "1",
            "--global-batch-size",
            "1",
            "--lr",
            "0.001",
            "--min-lr",
            "0.0001",
            "--lr-decay-style",
            "cosine",
            "--dataloader-type",
            "single",
            "--tokenizer-type",
            "NullTokenizer",
            "--bf16",
            "--use-distributed-optimizer",
        ]
    )

    assert parsed.num_layers == 2
    assert parsed.hidden_size == 16
    assert parsed.num_attention_heads == 4
    assert parsed.micro_batch_size == 1
    assert parsed.global_batch_size == 1
    assert parsed.lr == 0.001
    assert parsed.min_lr == 0.0001
    assert parsed.lr_decay_style == "cosine"
    assert parsed.dataloader_type == "single"
    assert parsed.tokenizer_type == "NullTokenizer"
    assert parsed.bf16
    assert parsed.use_distributed_optimizer


def test_parse_args_sets_rank_and_world_size_from_environment(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "program",
            "--num-layers",
            "2",
            "--hidden-size",
            "16",
            "--num-attention-heads",
            "4",
            "--seq-length",
            "8",
            "--max-position-embeddings",
            "8",
            "--micro-batch-size",
            "1",
            "--global-batch-size",
            "1",
        ],
    )
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "8")

    parsed = arguments.parse_args()

    assert parsed.rank == 3
    assert parsed.world_size == 8
    assert parsed.enable_msc


def test_parse_args_allows_extra_provider_and_unknown_args(monkeypatch):
    def extra_provider(parser):
        parser.add_argument("--custom-flag", type=int, default=0)
        return parser

    monkeypatch.setattr("sys.argv", ["program", "--custom-flag", "7", "--unknown-flag"])

    parsed = arguments.parse_args(extra_args_provider=extra_provider, ignore_unknown_args=True)

    assert parsed.custom_flag == 7


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("[0,1]*2", [0, 1, 0, 1]),
        ("([1]+[0])*2", [1, 0, 1, 0]),
        ("[1,0,0]", [1, 0, 0]),
    ],
)
def test_eval_pattern_accepts_safe_list_expressions(pattern, expected):
    assert arguments._eval_pattern(pattern) == expected


def test_eval_pattern_rejects_unsafe_expression():
    with pytest.raises(ValueError, match="Invalid pattern"):
        arguments._eval_pattern("[import('os').system('echo unsafe')]")


def test_frequency_and_tuple_helpers():
    assert arguments.no_rope_freq_type(None) is None
    assert arguments.no_rope_freq_type(2) == 2
    assert arguments.no_rope_freq_type("2") == 2
    assert arguments.no_rope_freq_type("[1,0]") == [1, 0]

    assert arguments.moe_freq_type(3) == 3
    assert arguments.moe_freq_type("3") == 3
    assert arguments.moe_freq_type("[1,0,1]") == [1, 0, 1]

    assert arguments.la_freq_type(None) is None
    assert arguments.la_freq_type(4) == 4
    assert arguments.la_freq_type("4") == 4
    assert arguments.la_freq_type("[1,1,0]") == [1, 1, 0]

    assert arguments.tuple_type(None) is None
    assert arguments.tuple_type((1, 2)) == (1, 2)
    assert arguments.tuple_type("1,2,3") == (1, 2, 3)
    assert arguments.tuple_type("(4,5)") == (4, 5)
