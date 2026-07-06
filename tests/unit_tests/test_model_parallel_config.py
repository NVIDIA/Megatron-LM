# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys

import pytest

from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.transformer import TransformerConfig
from megatron.training.arguments import (
    _validate_dynamic_context_parallel_topology,
    parse_args,
    validate_args,
)


def test_te_cross_entropy_loss_fusion_warns_in_model_parallel_config():
    with pytest.warns(UserWarning, match="known stability issues"):
        config = ModelParallelConfig(cross_entropy_loss_fusion=True, cross_entropy_fusion_impl='te')

    assert config.cross_entropy_loss_fusion
    assert config.cross_entropy_fusion_impl == 'te'


def test_native_cross_entropy_loss_fusion_is_allowed():
    config = ModelParallelConfig(cross_entropy_loss_fusion=True, cross_entropy_fusion_impl='native')

    assert config.cross_entropy_loss_fusion
    assert config.cross_entropy_fusion_impl == 'native'


def test_te_cross_entropy_loss_fusion_is_disabled_by_training_args(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['test_model_parallel_config.py'])
    args = parse_args()
    args.num_layers = 2
    args.hidden_size = 128
    args.num_attention_heads = 4
    args.max_position_embeddings = 1024
    args.seq_length = 1024
    args.micro_batch_size = 1
    # Let validate_args derive a global batch size that is valid for the
    # active data-parallel size in distributed unit-test jobs.
    args.train_iters = 1
    args.lr = 1e-4
    args.tokenizer_type = 'NullTokenizer'
    args.vocab_size = 1024
    args.cross_entropy_loss_fusion = True
    args.cross_entropy_fusion_impl = 'te'

    with pytest.raises(AssertionError, match="Transformer Engine cross entropy loss fusion"):
        validate_args(args)


def test_dynamic_context_parallel_rejects_moe_transformer_config():
    with pytest.raises(ValueError, match="does not support MoE"):
        TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=1,
            dynamic_context_parallel=True,
            num_moe_experts=2,
            moe_ffn_hidden_size=16,
        )


def test_hybrid_context_parallel_uses_default_dynamic_scheduler():
    with pytest.warns(DeprecationWarning, match="deprecated"):
        config = ModelParallelConfig(hybrid_context_parallel=True)

    assert config.dynamic_context_parallel
    assert not config.hybrid_context_parallel
    assert config.sequence_packing_scheduler == 'default_dynamic_cp'


def test_dynamic_context_parallel_rejects_moe_training_args(monkeypatch):
    monkeypatch.setenv('WORLD_SIZE', '2')
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'test_model_parallel_config.py',
            '--dynamic-context-parallel',
            '--calculate-per-token-loss',
            '--max-seqlen-per-dp-cp-rank',
            '1024',
            '--num-experts',
            '2',
        ],
    )
    args = parse_args()
    args.num_layers = 2
    args.hidden_size = 128
    args.num_attention_heads = 4
    args.max_position_embeddings = 1024
    args.seq_length = 1024
    args.micro_batch_size = 1
    args.train_iters = 1
    args.lr = 1e-4
    args.tokenizer_type = 'NullTokenizer'
    args.vocab_size = 1024

    with pytest.raises(AssertionError, match="does not support MoE"):
        validate_args(args)


def test_dynamic_context_parallel_requires_power_of_two_domain():
    with pytest.raises(AssertionError, match="world size to be a power of two"):
        _validate_dynamic_context_parallel_topology(dp_cp_size=6, min_cp_size=2)


def test_dynamic_context_parallel_allows_full_non_power_of_two_domain():
    _validate_dynamic_context_parallel_topology(dp_cp_size=3, min_cp_size=3)
