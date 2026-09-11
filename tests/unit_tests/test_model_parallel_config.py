# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys

import pytest

from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.arguments import parse_args, validate_args


def test_te_cross_entropy_loss_fusion_warns_in_model_parallel_config():
    with pytest.warns(UserWarning, match="known stability issues"):
        config = ModelParallelConfig(cross_entropy_loss_fusion=True, cross_entropy_fusion_impl='te')

    assert config.cross_entropy_loss_fusion
    assert config.cross_entropy_fusion_impl == 'te'


def test_native_cross_entropy_loss_fusion_is_allowed():
    config = ModelParallelConfig(cross_entropy_loss_fusion=True, cross_entropy_fusion_impl='native')

    assert config.cross_entropy_loss_fusion
    assert config.cross_entropy_fusion_impl == 'native'


def test_invalid_thd_tail_padding_policy_is_rejected_during_config_initialization():
    with pytest.raises(ValueError, match="thd_tail_padding_policy must be"):
        ModelParallelConfig(thd_tail_padding_policy="bogus")


def test_pipeline_p2p_fixed_shape_requires_sequence_packing():
    with pytest.raises(ValueError, match="requires a sequence_packing_scheduler"):
        ModelParallelConfig(
            pipeline_p2p_fixed_shape=True,
            max_seqlen_per_dp_cp_rank=2048,
            pad_packed_seq_alignment="max",
        )


def test_pipeline_p2p_fixed_shape_requires_max_padding():
    with pytest.raises(ValueError, match="requires pad_packed_seq_alignment='max'"):
        ModelParallelConfig(
            pipeline_p2p_fixed_shape=True,
            sequence_packing_scheduler="dp_balanced",
            max_seqlen_per_dp_cp_rank=2048,
            pad_packed_seq_alignment=128,
        )


def test_pipeline_p2p_fixed_shape_rejects_dynamic_context_parallel():
    with pytest.raises(ValueError, match="not supported with dynamic_context_parallel"):
        ModelParallelConfig(
            pipeline_p2p_fixed_shape=True,
            dynamic_context_parallel=True,
            max_seqlen_per_dp_cp_rank=2048,
            pad_packed_seq_alignment="max",
        )


def test_pipeline_p2p_fixed_shape_accepts_static_max_padding():
    config = ModelParallelConfig(
        pipeline_p2p_fixed_shape=True,
        sequence_packing_scheduler="dp_balanced",
        max_seqlen_per_dp_cp_rank=2048,
        pad_packed_seq_alignment="max",
    )
    assert config.pipeline_p2p_fixed_shape


def test_contiguous_context_parallel_rejects_bshd_inputs():
    with pytest.raises(
        ValueError,
        match="cp_partition_mode='contiguous'.*requires THD.*BSHD inputs are not supported",
    ):
        TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            context_parallel_size=2,
            cp_partition_mode="contiguous",
        )


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
