# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from megatron.core.transformer.transformer_config import TransformerConfig


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


def _make_scheduled_release_config(moe_ncclep_zero_copy: bool) -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=2,
        expert_model_parallel_size=2,
        moe_token_dispatcher_type="alltoall",
        overlap_moe_expert_parallel_comm=True,
        ep_overlap_use_scheduled_tensor_release=True,
        moe_ncclep_zero_copy=moe_ncclep_zero_copy,
        bf16=True,
    )


def test_scheduled_tensor_release_accepts_staged_ncclep_buffers():
    config = _make_scheduled_release_config(moe_ncclep_zero_copy=False)

    assert config.ep_overlap_use_scheduled_tensor_release


def test_scheduled_tensor_release_rejects_ncclep_zero_copy():
    with pytest.raises(ValueError, match="moe_ncclep_zero_copy is not supported"):
        _make_scheduled_release_config(moe_ncclep_zero_copy=True)
