# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version


def _make_overlap_config(mtp_num_layers: int | None) -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=2,
        expert_model_parallel_size=2,
        moe_token_dispatcher_type="alltoall",
        overlap_moe_expert_parallel_comm=True,
        bf16=True,
        mtp_num_layers=mtp_num_layers,
    )


@pytest.mark.parametrize("mtp_num_layers", [None, 0, 1])
def test_ep_a2a_overlap_accepts_supported_mtp_layer_counts(mtp_num_layers: int | None):
    config = _make_overlap_config(mtp_num_layers)

    assert config.mtp_num_layers == mtp_num_layers


@pytest.mark.parametrize("mtp_num_layers", [-1, 2])
def test_ep_a2a_overlap_rejects_unsupported_mtp_layer_counts(mtp_num_layers: int):
    with pytest.raises(AssertionError, match="MTP supports at most one layer"):
        _make_overlap_config(mtp_num_layers)


requires_te_2_9 = pytest.mark.skipif(
    not is_te_min_version("2.9.0"),
    reason="sequence packing requires Transformer Engine >= 2.9.0",
)


def _make_packing_config(**kwargs) -> TransformerConfig:
    defaults = dict(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        sequence_packing_scheduler="dp_balanced",
        max_seqlen_per_dp_cp_rank=4096,
    )
    defaults.update(kwargs)
    return TransformerConfig(**defaults)


@requires_te_2_9
def test_sequence_packing_dense_config_passes():
    # Dense models have no MoE dispatcher; the (unused) allgather default
    # must not fail sequence-packing validation.
    config = _make_packing_config()
    assert config.variable_seq_lengths is True


@requires_te_2_9
def test_sequence_packing_moe_requires_alltoall_dispatcher():
    with pytest.raises(AssertionError, match="alltoall"):
        _make_packing_config(num_moe_experts=2, moe_token_dispatcher_type="allgather")


@requires_te_2_9
def test_sequence_packing_moe_alltoall_dispatcher_passes():
    config = _make_packing_config(num_moe_experts=2, moe_token_dispatcher_type="alltoall")
    assert config.variable_seq_lengths is True


def test_sequence_packing_rejects_unknown_scheduler():
    # Raised by ModelParallelConfig.__post_init__ before any TE check runs.
    with pytest.raises(ValueError, match="Unsupported scheduler"):
        _make_packing_config(sequence_packing_scheduler="bogus")


def test_sequence_packing_requires_max_seqlen_per_dp_cp_rank():
    with pytest.raises(ValueError, match="max_seqlen_per_dp_cp_rank"):
        _make_packing_config(max_seqlen_per_dp_cp_rank=None)
