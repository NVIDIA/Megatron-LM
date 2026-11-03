# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Startup guards for full-iteration CUDA graphs with packed (THD) inputs.

Full-iteration capture records ``get_batch`` too, so packed inputs are accepted only on the
``--sft`` path under the static THD contract, at CP = 1 and PP = 1. Run with four ranks so the
CP = 2 and PP = 2 cases pass the world-size checks and reach these guards.
"""

import os
import sys

import pytest

from megatron.training.arguments import parse_args, validate_args

pytestmark = pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) % 4 != 0, reason="requires a multiple of four ranks"
)


def _args(monkeypatch, **overrides):
    monkeypatch.setattr(sys, 'argv', ['test_full_iteration_thd_args.py'])
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
    args.cuda_graph_impl = 'full_iteration'
    args.check_for_nan_in_loss_and_grad = False
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


STATIC_THD = dict(
    sft=True,
    pad_packed_seq_alignment='max',
    max_seqlen_per_dp_cp_rank=128,
    thd_max_packed_sequences=8,
)


def test_static_thd_contract_at_cp1_is_accepted(monkeypatch):
    args = validate_args(_args(monkeypatch, **STATIC_THD))
    assert args.cuda_graph_impl == 'full_iteration' and args.sft


def test_dynamic_thd_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="static THD"):
        validate_args(_args(monkeypatch, sft=True))


@pytest.mark.parametrize(
    "overrides, match",
    [
        (dict(context_parallel_size=2), "--context-parallel-size 1 only"),
        (dict(pipeline_model_parallel_size=2), "--pipeline-model-parallel-size 1 only"),
        (dict(thd_tail_padding_policy='extend_last'), "append_dummy_seq"),
    ],
)
def test_static_thd_outside_part_a_scope_is_rejected(monkeypatch, overrides, match):
    with pytest.raises(ValueError, match=match):
        validate_args(_args(monkeypatch, **STATIC_THD, **overrides))


def test_sequence_packing_schedulers_are_rejected(monkeypatch):
    args = _args(
        monkeypatch,
        use_varlen_dataset=True,
        pad_packed_seq_alignment='max',
        max_seqlen_per_dp_cp_rank=128,
        thd_max_packed_sequences=8,
    )
    with pytest.raises(ValueError, match="sequence-packing schedulers"):
        validate_args(args)


def test_dynamic_context_parallel_is_rejected(monkeypatch):
    args = _args(
        monkeypatch,
        sft=True,
        dynamic_context_parallel=True,
        calculate_per_token_loss=True,
        pad_packed_seq_alignment='max',
        max_seqlen_per_dp_cp_rank=128,
        thd_max_packed_sequences=8,
    )
    with pytest.raises(ValueError, match="dynamic-context-parallel"):
        validate_args(args)
