# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor
from megatron.core.transformer.moe import moonep_manager
from megatron.core.transformer.moe.experts import TEGroupedMLP, _parse_te_expert_idx
from megatron.core.transformer.moe.moonep_manager import MoonEPManager, expert_weight_shapes
from megatron.core.transformer.transformer_config import TransformerConfig


def _moonep_config_kwargs(**overrides):
    kwargs = dict(
        num_layers=2,
        hidden_size=256,
        num_attention_heads=4,
        num_moe_experts=8,
        moe_ffn_hidden_size=128,
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=1,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type='moonep',
        gradient_accumulation_fusion=True,
        params_dtype=torch.bfloat16,
        gated_linear_unit=True,
        add_bias_linear=False,
    )
    kwargs.update(overrides)
    return kwargs


def test_moonep_transformer_config_accepts_supported_configuration():
    config = TransformerConfig(**_moonep_config_kwargs())

    assert config.moe_token_dispatcher_type == 'moonep'
    assert config.moe_moonep_num_sms == 32
    assert config.moe_moonep_token_padding == 128


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"moe_grouped_gemm": False}, "moe_grouped_gemm"),
        ({"add_bias_linear": True}, "add_bias_linear"),
        ({"gated_linear_unit": False}, "gated expert MLP"),
        ({"expert_tensor_parallel_size": 2}, "expert_tensor_parallel_size"),
        ({"params_dtype": torch.float16}, "BF16"),
        ({"moe_expert_capacity_factor": 1.0}, "dropless"),
    ],
)
def test_moonep_transformer_config_rejects_unsupported_configuration(overrides, match):
    with pytest.raises(ValueError, match=match):
        TransformerConfig(**_moonep_config_kwargs(**overrides))


def test_expert_weight_shapes_uses_fused_gated_fc1():
    config = TransformerConfig(**_moonep_config_kwargs())

    assert expert_weight_shapes(config) == ((256, 256), (256, 128))


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("linear_fc1.weight0", 0),
        ("linear_fc1.bias12", 12),
        ("linear_fc1._extra_state", None),
        ("linear_fc2.weight1", None),
        ("linear_fc1.weight", None),
    ],
)
def test_parse_te_expert_idx(key, expected):
    assert _parse_te_expert_idx(key, "linear_fc1") == expected


def test_moonep_checkpoint_filter_drops_replica_entries_in_singleton_mode():
    experts = TEGroupedMLP.__new__(TEGroupedMLP)
    torch.nn.Module.__init__(experts)
    experts.ep_group = SimpleNamespace(size=lambda: 2, rank=lambda: 1)
    master_value = object()
    extra_state = object()

    filtered = experts._filter_replica_checkpoint_entries(
        {
            "linear_fc1.weight0": master_value,
            "linear_fc1.weight1": object(),
            "linear_fc1._extra_state": extra_state,
        },
        "linear_fc1",
        ep_axis=0,
        num_local_master_experts=1,
        fix_metadata=False,
    )

    assert filtered == {"linear_fc1.weight0": master_value, "linear_fc1._extra_state": extra_state}


def test_moonep_checkpoint_filter_restores_logical_expert_metadata():
    experts = TEGroupedMLP.__new__(TEGroupedMLP)
    torch.nn.Module.__init__(experts)
    experts.ep_group = SimpleNamespace(size=lambda: 2, rank=lambda: 1)
    master = ShardedTensor.from_rank_offsets(
        "linear_fc2.weight0", torch.zeros(2, 3), (0, 3, 6), prepend_axis_num=1
    )
    replica = ShardedTensor.from_rank_offsets(
        "linear_fc2.weight1", torch.zeros(2, 3), (0, 4, 6), prepend_axis_num=1
    )
    extra_state = ShardedObject("linear_fc2._extra_state", object(), (1,), (0,))

    filtered = experts._filter_replica_checkpoint_entries(
        {
            "linear_fc2.weight0": master,
            "linear_fc2.weight1": replica,
            "linear_fc2._extra_state": extra_state,
        },
        "linear_fc2",
        ep_axis=0,
        num_local_master_experts=1,
        fix_metadata=True,
    )

    assert set(filtered) == {"linear_fc2.weight0", "linear_fc2._extra_state"}
    assert filtered["linear_fc2.weight0"].global_shape == (2, 2, 3)
    assert filtered["linear_fc2.weight0"].global_offset == (1, 0, 0)
    assert filtered["linear_fc2._extra_state"] is extra_state


def _manager_stub(rank, num_ranks, num_global_experts):
    manager = MoonEPManager.__new__(MoonEPManager)
    manager.rank = rank
    manager.num_ranks = num_ranks
    manager.num_global_experts = num_global_experts
    manager.num_local_master_experts = num_global_experts // num_ranks
    manager.num_prefetch_slots = manager.num_local_master_experts
    return manager


def test_local_master_slice_selects_this_ranks_home_group():
    manager = _manager_stub(rank=2, num_ranks=4, num_global_experts=8)

    assert manager.local_master_slice == slice(4, 6)


def test_local_tokens_per_expert_takes_home_group_then_prefetch_slots():
    manager = _manager_stub(rank=1, num_ranks=2, num_global_experts=4)
    # Rows: experts 0..3 then 2 prefetch slots; rank 1 owns experts 2 and 3.
    counts = torch.tensor([0, 0, 5, 7, 3, 4], dtype=torch.int32)
    cu_seqlens = torch.cumsum(counts, dim=0).to(torch.int32)

    assert manager.local_tokens_per_expert(cu_seqlens).tolist() == [5, 7, 3, 4]


def test_local_tokens_per_expert_handles_an_empty_home_group():
    manager = _manager_stub(rank=0, num_ranks=2, num_global_experts=4)
    counts = torch.tensor([0, 0, 0, 0, 6, 0], dtype=torch.int32)
    cu_seqlens = torch.cumsum(counts, dim=0).to(torch.int32)

    assert manager.local_tokens_per_expert(cu_seqlens).tolist() == [0, 0, 6, 0]


def test_destroy_moonep_managers_closes_and_clears_registry():
    manager_a = SimpleNamespace(close=Mock())
    manager_b = SimpleNamespace(close=Mock())
    moonep_manager._MOONEP_MANAGER_REGISTRY.update({1: manager_a, 2: manager_b})

    moonep_manager.destroy_moonep_managers()

    manager_a.close.assert_called_once_with()
    manager_b.close.assert_called_once_with()
    assert moonep_manager._MOONEP_MANAGER_REGISTRY == {}


def test_moonep_manager_close_is_idempotent():
    manager = MoonEPManager.__new__(MoonEPManager)
    manager._buffer = SimpleNamespace(destroy=Mock())
    manager._layers = {}
    manager._shared_chunks = {}
    manager._closed = False

    buffer = manager._buffer
    manager.close()
    manager.close()

    buffer.destroy.assert_called_once_with()
    assert manager._closed
