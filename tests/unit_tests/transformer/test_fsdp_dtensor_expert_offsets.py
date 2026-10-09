# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Expert offsets in FSDP-DTensor names come from the expert-parallel layout the caller passes.

The helpers read the global expert-parallel rank and size when the caller passes none. A model on
its own grid then named its local experts by another grid's layout, so its optimizer state was
saved under other experts' keys.
"""

import contextlib
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.fsdp_dtensor_checkpoint import (
    expert_param_local_key,
    get_ep_layer_offset,
    get_ep_rank_and_size,
    get_global_unique_param_name,
    handle_experts_in_state_dict,
)

NUM_EXPERTS = 8
ATTENTION_KEY = 'decoder.layers.0.self_attention.linear_qkv.weight'


@contextlib.contextmanager
def forbid_global_expert_parallel_reads():
    """Make the global expert-parallel rank and size accessors raise."""
    with contextlib.ExitStack() as stack:
        for name in (
            'get_expert_model_parallel_rank',
            'get_expert_model_parallel_world_size',
            'get_expert_model_parallel_group',
        ):
            stack.enter_context(
                mock.patch.object(parallel_state, name, side_effect=AssertionError(f"read {name}"))
            )
        yield


def sequential_expert_key(expert_index):
    return f'decoder.layers.0.mlp.experts.local_experts.{expert_index}.linear_fc1.weight'


def grouped_expert_key(expert_index):
    return f'decoder.layers.0.mlp.experts.linear_fc2.weight{expert_index}'


def expert_model(num_local_experts, num_moe_experts):
    """A model whose parameters are named like a SequentialMLP's local experts."""
    experts = torch.nn.Module()
    experts.local_experts = torch.nn.ModuleList(
        torch.nn.Linear(2, 2, bias=False) for _ in range(num_local_experts)
    )
    mlp = torch.nn.Module()
    mlp.experts = experts
    layer = torch.nn.Module()
    layer.mlp = mlp
    decoder = torch.nn.Module()
    decoder.layers = torch.nn.ModuleList([layer])
    model = torch.nn.Module()
    model.decoder = decoder
    model.config = SimpleNamespace(num_moe_experts=num_moe_experts)
    return model


class TestExpertOffsetsFromIntegers:

    def test_offset(self):
        with forbid_global_expert_parallel_reads():
            assert get_ep_layer_offset(NUM_EXPERTS, ep_rank=3, ep_size=4) == 6

    @pytest.mark.parametrize("kwargs", [{"ep_rank": 1}, {"ep_size": 4}])
    def test_rank_and_size_go_together(self, kwargs):
        with pytest.raises(ValueError, match="together"):
            get_ep_layer_offset(NUM_EXPERTS, **kwargs)

    def test_state_dict_and_local_keys(self):
        state_dict = {sequential_expert_key(1): 'a', grouped_expert_key(0): 'b', ATTENTION_KEY: 'c'}

        with forbid_global_expert_parallel_reads():
            renamed = handle_experts_in_state_dict(state_dict, NUM_EXPERTS, ep_rank=3, ep_size=4)
            local_key = expert_param_local_key(
                sequential_expert_key(7), NUM_EXPERTS, ep_rank=3, ep_size=4
            )

        assert renamed == {
            sequential_expert_key(7): 'a',
            grouped_expert_key(6): 'b',
            ATTENTION_KEY: 'c',
        }
        assert local_key == sequential_expert_key(1)

    def test_global_unique_param_name(self):
        model = expert_model(num_local_experts=2, num_moe_experts=NUM_EXPERTS)
        param = model.decoder.layers[0].mlp.experts.local_experts[1].weight

        with forbid_global_expert_parallel_reads():
            name = get_global_unique_param_name([model], param, ep_rank=3, ep_size=4)

        assert name == 'decoder.layers.0.mlp.experts.local_experts.7.weight'

    def test_rank_and_size_from_a_collection_without_ep(self):
        assert get_ep_rank_and_size(None) == (None, None)
        assert get_ep_rank_and_size(ProcessGroupCollection()) == (None, None)
        # An explicit None marks expert parallelism as off.
        assert get_ep_rank_and_size(ProcessGroupCollection(ep=None)) == (0, 1)
