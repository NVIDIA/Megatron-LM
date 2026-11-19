# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""SFT batches carry no attention mask, so argument validation turns its TP broadcast off."""

import sys

from megatron.training.arguments import parse_args, validate_args


def _args(monkeypatch, **overrides):
    monkeypatch.setattr(sys, 'argv', ['test_sft_attention_mask_args.py'])
    args = parse_args()
    args.num_layers = 4
    args.vocab_size = 256
    args.hidden_size = 64
    args.num_attention_heads = 4
    args.max_position_embeddings = 128
    args.seq_length = 128
    args.micro_batch_size = 1
    args.train_iters = 1
    args.lr = 1e-4
    args.tokenizer_type = 'NullTokenizer'
    args.mock_data = True
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_sft_turns_the_attention_mask_broadcast_off(monkeypatch):
    args = _args(monkeypatch, sft=True)
    assert args.create_attention_mask_in_dataloader

    assert not validate_args(args).create_attention_mask_in_dataloader


def test_attention_mask_broadcast_stays_on_without_sft(monkeypatch):
    assert validate_args(_args(monkeypatch)).create_attention_mask_in_dataloader
