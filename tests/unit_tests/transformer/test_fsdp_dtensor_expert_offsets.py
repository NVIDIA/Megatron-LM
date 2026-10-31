# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Expert offsets in FSDP-DTensor names come from the expert-parallel layout the caller passes.

The helpers read the global expert-parallel rank and size when the caller passes none. A model on
its own grid then named its local experts by another grid's layout, so its optimizer state was
saved under other experts' keys. That fallback warns, and a model without experts reads no
expert-parallel layout at all.
"""

import contextlib
import re
import warnings
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core import parallel_state, process_groups_config
from megatron.core.process_groups_config import ProcessGroupCollection, ProcessGroupFallbackWarning
from megatron.core.transformer.fsdp_dtensor_checkpoint import (
    expert_param_local_key,
    get_ep_layer_offset,
    get_ep_rank_and_size,
    get_global_unique_param_name,
    get_model_ep_rank_and_size,
    handle_experts_in_state_dict,
)
from tests.unit_tests.test_utilities import Utils, forbid_global_process_groups

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


@pytest.fixture(autouse=True)
def fresh_fallback_warnings(monkeypatch):
    """Each test observes the first process-group fallback warning of every owner."""
    monkeypatch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())


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

    def test_rank_and_size_from_a_wrapped_model(self):
        model = torch.nn.Module()
        model.pg_collection = ProcessGroupCollection(ep=None)
        wrapper = torch.nn.Module()
        wrapper.module = model
        # The wrapper chain is walked to the model that owns the collection.
        assert get_model_ep_rank_and_size(wrapper) == (0, 1)

    def test_model_without_a_collection_keeps_the_global_layout(self):
        # Models built without a collection (mocks, legacy model classes) name their experts by
        # the global layout, as before.
        wrapper = torch.nn.Module()
        wrapper.module = torch.nn.Linear(1, 1)
        assert get_model_ep_rank_and_size(wrapper) == (None, None)


class TestWithoutExperts:

    @pytest.mark.parametrize("num_experts", [None, 0])
    def test_reads_no_expert_parallel_layout(self, num_experts):
        """A model without experts has offset 0 and needs no expert-parallel rank or size."""
        state_dict = {ATTENTION_KEY: 'c'}
        model = expert_model(num_local_experts=0, num_moe_experts=num_experts)
        layer = model.decoder.layers[0]
        layer.self_attention = torch.nn.Module()
        layer.self_attention.linear_qkv = torch.nn.Linear(2, 2, bias=False)
        param = layer.self_attention.linear_qkv.weight

        with forbid_global_process_groups(), warnings.catch_warnings():
            warnings.simplefilter("error", ProcessGroupFallbackWarning)
            assert get_ep_layer_offset(num_experts) == 0
            assert handle_experts_in_state_dict(state_dict, num_experts) == state_dict
            name = get_global_unique_param_name([model], param)

        assert name == ATTENTION_KEY


class TestExpertOffsetsFromTheGlobalGrid:
    """Without ep_rank and ep_size, the helpers warn and use the global expert-parallel layout."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_fallback_warns_and_uses_the_global_layout(self):
        if Utils.world_size < 2 or Utils.world_size % 2:
            pytest.skip(f"needs an even world size of at least 2, got {Utils.world_size}")
        Utils.initialize_model_parallel(expert_model_parallel_size=2)
        ep_rank = parallel_state.get_expert_model_parallel_rank()
        ep_size = parallel_state.get_expert_model_parallel_world_size()

        with (
            forbid_global_process_groups(forbid_shim=False),
            pytest.warns(
                ProcessGroupFallbackWarning,
                match=re.escape("get_ep_layer_offset was called without `ep_rank/ep_size`"),
            ),
        ):
            offset = get_ep_layer_offset(NUM_EXPERTS)

        assert ep_size == 2
        assert offset == ep_rank * (NUM_EXPERTS // ep_size)

    def test_fallback_without_an_expert_parallel_group_raises(self):
        Utils.destroy_model_parallel()
        if parallel_state.get_expert_model_parallel_group(check_initialized=False) is not None:
            pytest.skip("the global grid of an earlier test is still initialized")

        with (
            pytest.warns(ProcessGroupFallbackWarning),
            pytest.raises(RuntimeError, match="has not created the expert-parallel group"),
        ):
            get_ep_layer_offset(NUM_EXPERTS)
