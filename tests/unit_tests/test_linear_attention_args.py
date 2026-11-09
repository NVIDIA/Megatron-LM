# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""validate_args must reject gated delta net variants that have no layer pattern.

Without --hybrid-layer-pattern, the GDN layers are placed by --linear-attention-freq. If it is
missing, GPT models fail when they are built, and other models at the first FLOPs count.
"""

import sys

import pytest

from megatron.training.arguments import parse_args, validate_args

_GUARD = "requires --linear-attention-freq unless --hybrid-layer-pattern is set"


def _gdn_args(monkeypatch, **overrides):
    monkeypatch.setattr(sys, "argv", ["test_linear_attention_args.py"])
    args = parse_args()
    # parse_args reads WORLD_SIZE from the environment; pin it so the test does not depend
    # on how the job was launched.
    args.world_size = 1
    args.num_layers = 2
    args.hidden_size = 128
    args.num_attention_heads = 4
    args.max_position_embeddings = 1024
    args.seq_length = 1024
    args.micro_batch_size = 1
    args.global_batch_size = 1
    args.train_iters = 1
    args.lr = 1e-4
    args.tokenizer_type = "NullTokenizer"
    args.vocab_size = 1024
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@pytest.mark.parametrize("variant", ["gdn", "gdn2"])
def test_gdn_variant_without_linear_attention_freq_is_rejected(monkeypatch, variant):
    args = _gdn_args(monkeypatch, experimental_attention_variant=variant)
    with pytest.raises(ValueError, match=_GUARD):
        validate_args(args)


@pytest.mark.parametrize("variant", ["gdn", "gdn2"])
def test_gdn_variant_with_linear_attention_freq_validates(monkeypatch, variant):
    args = _gdn_args(monkeypatch, experimental_attention_variant=variant, linear_attention_freq=2)
    validate_args(args)
    assert args.linear_attention_freq == 2
