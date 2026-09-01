# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys

import pytest

from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.training.arguments import parse_args, validate_args


def test_dynamic_cp_selects_default_scheduler_and_variable_lengths():
    config = ModelParallelConfig(dynamic_context_parallel=True, max_seqlen_per_dp_cp_rank=4096)

    assert config.sequence_packing_scheduler == "default_dynamic_cp"
    assert config.variable_seq_lengths is True


def test_dynamic_cp_rejects_non_dynamic_scheduler():
    with pytest.raises(ValueError, match="requires.*default_dynamic_cp"):
        ModelParallelConfig(
            dynamic_context_parallel=True,
            sequence_packing_scheduler="dp_balanced",
            max_seqlen_per_dp_cp_rank=4096,
        )


def test_default_dynamic_cp_scheduler_requires_dynamic_cp():
    with pytest.raises(ValueError, match="requires.*dynamic_context_parallel=True"):
        ModelParallelConfig(
            sequence_packing_scheduler="default_dynamic_cp", max_seqlen_per_dp_cp_rank=4096
        )


def test_hybrid_cp_alias_normalizes_to_dynamic_cp():
    with pytest.warns(DeprecationWarning, match="deprecated"):
        config = ModelParallelConfig(hybrid_context_parallel=True, max_seqlen_per_dp_cp_rank=4096)

    assert config.dynamic_context_parallel is True
    assert config.hybrid_context_parallel is False
    assert config.sequence_packing_scheduler == "default_dynamic_cp"


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
