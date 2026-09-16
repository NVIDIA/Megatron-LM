# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys

import pytest
import torch

import megatron.core.transformer.transformer_config as transformer_config_module
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


def test_pipeline_p2p_fixed_shape_requires_max_seqlen_when_alignment_unset():
    """Both fields unset must not slip through the alignment membership test.

    ``None not in ("max", None)`` is False, so without an explicit guard this configuration would
    validate and then derive the pipeline buffer from a None sequence length.
    """
    with pytest.raises(ValueError, match="requires max_seqlen_per_dp_cp_rank to be set"):
        ModelParallelConfig(pipeline_p2p_fixed_shape=True, sequence_packing_scheduler="dp_balanced")


def test_pipeline_p2p_fixed_shape_requires_max_seqlen_with_max_alignment():
    """Same hole reached via a set alignment; here the pre-existing padding guard fires first.

    Matched on the field name rather than one guard's wording so the test pins the outcome (no
    fixed-shape config without a concrete padded length) instead of which check happens to run.
    """
    with pytest.raises(ValueError, match="max_seqlen_per_dp_cp_rank"):
        ModelParallelConfig(
            pipeline_p2p_fixed_shape=True,
            sequence_packing_scheduler="dp_balanced",
            pad_packed_seq_alignment="max",
        )


@pytest.mark.parametrize("vpp_size", [1, 2])
def test_pipeline_p2p_fixed_shape_rejects_virtual_pipeline_parallelism(vpp_size):
    """vpp_size=1 must be rejected too: get_forward_backward_func() picks the interleaved
    schedule on `is not None`, so even a size of 1 bypasses get_tensor_shapes()."""
    with pytest.raises(ValueError, match="not supported with virtual pipeline"):
        ModelParallelConfig(
            pipeline_p2p_fixed_shape=True,
            sequence_packing_scheduler="dp_balanced",
            max_seqlen_per_dp_cp_rank=2048,
            pad_packed_seq_alignment="max",
            pipeline_model_parallel_size=4,
            virtual_pipeline_model_parallel_size=vpp_size,
            pipeline_dtype=torch.bfloat16,
        )


def test_pipeline_p2p_fixed_shape_requires_tp_divisible_max_seqlen():
    with pytest.raises(ValueError, match="to be divisible by"):
        ModelParallelConfig(
            pipeline_p2p_fixed_shape=True,
            sequence_packing_scheduler="dp_balanced",
            max_seqlen_per_dp_cp_rank=2049,
            pad_packed_seq_alignment="max",
            tensor_model_parallel_size=2,
            sequence_parallel=True,
        )


def test_pipeline_p2p_fixed_shape_allows_indivisible_max_seqlen_without_sequence_parallel():
    """Without sequence parallelism the pipeline buffer is not scattered along TP."""
    config = ModelParallelConfig(
        pipeline_p2p_fixed_shape=True,
        sequence_packing_scheduler="dp_balanced",
        max_seqlen_per_dp_cp_rank=2049,
        pad_packed_seq_alignment="max",
        tensor_model_parallel_size=2,
    )
    assert config.pipeline_p2p_fixed_shape


def test_pipeline_p2p_fixed_shape_warns_when_mtp_standalone():
    with pytest.warns(UserWarning, match="no effect when mtp_standalone"):
        ModelParallelConfig(
            pipeline_p2p_fixed_shape=True,
            sequence_packing_scheduler="dp_balanced",
            max_seqlen_per_dp_cp_rank=2048,
            pad_packed_seq_alignment="max",
            mtp_standalone=True,
        )


def test_pipeline_p2p_fixed_shape_rejects_layout_derived_vpp():
    """Flexible layouts derive VPP after ModelParallelConfig.__post_init__ has returned."""
    with pytest.raises(ValueError, match="not supported with virtual pipeline"):
        TransformerConfig(
            num_layers=4,
            hidden_size=64,
            num_attention_heads=4,
            pipeline_model_parallel_size=2,
            pipeline_model_parallel_layout=[
                ["embedding"],
                ["decoder", "decoder"],
                ["decoder", "decoder"],
                ["loss"],
            ],
            pipeline_dtype=torch.bfloat16,
            pipeline_p2p_fixed_shape=True,
            sequence_packing_scheduler="dp_balanced",
            max_seqlen_per_dp_cp_rank=2048,
            pad_packed_seq_alignment="max",
        )


def test_pipeline_p2p_fixed_shape_warns_for_layout_derived_standalone_mtp(monkeypatch):
    """Flexible layouts also derive mtp_standalone after base-config validation."""
    monkeypatch.setattr(transformer_config_module, "is_te_min_version", lambda _: True)
    with pytest.warns(UserWarning, match="no effect when mtp_standalone"):
        config = TransformerConfig(
            num_layers=4,
            hidden_size=64,
            num_attention_heads=4,
            pipeline_model_parallel_size=4,
            pipeline_model_parallel_layout=[
                ["embedding", "decoder"],
                ["decoder"],
                ["decoder", "decoder", "mtp"],
                ["loss"],
            ],
            pipeline_dtype=torch.bfloat16,
            pipeline_p2p_fixed_shape=True,
            sequence_packing_scheduler="dp_balanced",
            max_seqlen_per_dp_cp_rank=2048,
            pad_packed_seq_alignment="max",
            mtp_num_layers=1,
        )
    assert config.mtp_standalone


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
