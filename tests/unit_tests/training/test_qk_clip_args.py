# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""validate_args rejects --qk-clip and --log-max-attention-logit with --transformer-impl local.

clip_qk reads core_attention.current_max_attn_logits, which only TEDotProductAttention keeps.
With the local layer spec the first training step used to fail with an AttributeError.
"""

import sys

import pytest

from megatron.training.arguments import parse_args, validate_args

MODEL_ARGV = (
    "--num-layers 2 --hidden-size 128 --num-attention-heads 8 --micro-batch-size 1 "
    "--global-batch-size 8 --seq-length 32 --max-position-embeddings 32"
).split()


def _validated(monkeypatch, extra):
    monkeypatch.setattr(sys, "argv", ["test_qk_clip_args.py", *MODEL_ARGV, *extra])
    args = parse_args()
    # parse_args reads WORLD_SIZE from the environment; pin it so the test does not depend
    # on how the job was launched.
    args.world_size = 1
    return validate_args(args)


@pytest.mark.parametrize("flag", ["--qk-clip", "--log-max-attention-logit"])
def test_max_attention_logit_options_reject_local_transformer_impl(monkeypatch, flag):
    with pytest.raises(ValueError, match="not supported with --transformer-impl local"):
        _validated(monkeypatch, [flag, "--transformer-impl", "local"])


@pytest.mark.parametrize("flag", ["--qk-clip", "--log-max-attention-logit"])
def test_max_attention_logit_options_accept_transformer_engine(monkeypatch, flag):
    args = _validated(monkeypatch, [flag, "--transformer-impl", "transformer_engine"])
    assert args.transformer_impl == "transformer_engine"
