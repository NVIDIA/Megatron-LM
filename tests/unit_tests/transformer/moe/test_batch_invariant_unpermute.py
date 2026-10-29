# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Batch-invariant MoE unpermute takes the expert-parallel size from its caller.

The batch-invariant combine sums the experts of each EP rank, rounds that partial to fp32, and
adds the partials in rank order, so its result depends on how the experts are split across EP
ranks. One token routed to four experts whose outputs are [1e20, 1, -1e20, 1] makes the split
visible: one shard sums to 1, two shards sum to 0 (each 1 is lost next to 1e20 before the
partials cancel), and four shards sum to 1.
"""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.custom_layers.batch_invariant_kernels import set_batch_invariant_mode
from megatron.core.transformer.moe.moe_utils import unpermute
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

_HIDDEN = 4
_NUM_EXPERTS = 4
_SUM_BY_EP_SIZE = {1: 1.0, 2: 0.0, 4: 1.0}


def _expert_outputs():
    """Permuted rows for the token; row k comes from expert k."""
    rows = torch.tensor([[1e20], [1.0], [-1e20], [1.0]], device="cuda", dtype=torch.float32)
    return rows.expand(_NUM_EXPERTS, _HIDDEN)


def _inverse_map():
    """Top-k slot k of the token reads permuted row k, produced by expert k."""
    slots = torch.arange(_NUM_EXPERTS, device="cuda", dtype=torch.int64).unsqueeze(0)
    return torch.stack((slots, slots))


def _unpermute(**kwargs):
    return unpermute(
        _expert_outputs(),
        torch.zeros(_NUM_EXPERTS, device="cuda", dtype=torch.int64),
        torch.Size((1, _HIDDEN)),
        routing_map=torch.ones(1, _NUM_EXPERTS, device="cuda", dtype=torch.bool),
        batch_invariant_inverse_map=_inverse_map(),
        **kwargs,
    )


def _forbid_global_process_groups(monkeypatch):
    """Make the global group, rank and world-size accessors and the collection shim fail."""

    def forbidden(*args, **kwargs):
        pytest.fail("the global parallel state was read")

    for name in dir(parallel_state):
        if name.startswith("get_") and name.endswith(
            ("_rank", "_ranks", "_world_size", "_group", "_groups")
        ):
            monkeypatch.setattr(parallel_state, name, forbidden)
    monkeypatch.setattr(ProcessGroupCollection, "use_mpu_process_groups", forbidden)


@pytest.mark.parametrize("ep_size", sorted(_SUM_BY_EP_SIZE))
def test_unpermute_splits_experts_by_the_given_ep_size(monkeypatch, ep_size):
    with set_batch_invariant_mode(True), monkeypatch.context() as patch:
        _forbid_global_process_groups(patch)
        output = _unpermute(ep_size=ep_size)

    expected = torch.full((1, _HIDDEN), _SUM_BY_EP_SIZE[ep_size], device="cuda")
    torch.testing.assert_close(output, expected, rtol=0.0, atol=0.0)


def test_unpermute_without_ep_size_uses_the_global_ep_size(monkeypatch):
    monkeypatch.setattr(parallel_state, "get_expert_model_parallel_world_size", lambda: 2)
    with set_batch_invariant_mode(True):
        output = _unpermute()

    expected = torch.full((1, _HIDDEN), _SUM_BY_EP_SIZE[2], device="cuda")
    torch.testing.assert_close(output, expected, rtol=0.0, atol=0.0)


class TestAlltoAllDispatcherBatchInvariantCombine:
    """The AlltoAll dispatcher combines by the size of its own EP group."""

    def setup_method(self, method):
        # The global grid has no expert parallelism; the dispatcher gets 2-rank EP groups.
        Utils.initialize_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_combine_postprocess_uses_the_dispatcher_ep_size(self, monkeypatch):
        if Utils.world_size < 2 or Utils.world_size % 2:
            pytest.skip("needs an even world size of at least 2")
        ep_group, _ = torch.distributed.new_subgroups_by_enumeration(
            [[rank, rank + 1] for rank in range(0, Utils.world_size, 2)]
        )
        config = TransformerConfig(
            num_layers=1,
            hidden_size=_HIDDEN,
            num_attention_heads=1,
            num_moe_experts=_NUM_EXPERTS,
            moe_token_dispatcher_type="alltoall",
        )
        num_local_experts = _NUM_EXPERTS // ep_group.size()
        first_local_expert = ep_group.rank() * num_local_experts
        dispatcher = MoEAlltoAllTokenDispatcher(
            num_local_experts=num_local_experts,
            local_expert_indices=list(
                range(first_local_expert, first_local_expert + num_local_experts)
            ),
            config=config,
            pg_collection=ProcessGroupCollection(ep=ep_group, expt_tp=None, tp_ep=ep_group),
        )
        # The state dispatch leaves behind for one token routed to all four experts.
        dispatcher.hidden_shape = torch.Size((1, 1, _HIDDEN))
        dispatcher.hidden_shape_before_permute = torch.Size((1, _HIDDEN))
        dispatcher.routing_map = torch.ones(1, _NUM_EXPERTS, device="cuda", dtype=torch.bool)
        dispatcher.reversed_local_input_permutation_mapping = torch.zeros(
            _NUM_EXPERTS, device="cuda", dtype=torch.int64
        )
        dispatcher.batch_invariant_inverse_permutation_mapping = _inverse_map()

        with set_batch_invariant_mode(True), monkeypatch.context() as patch:
            _forbid_global_process_groups(patch)
            output = dispatcher.combine_postprocess(_expert_outputs())

        expected = torch.full((1, 1, _HIDDEN), _SUM_BY_EP_SIZE[2], device="cuda")
        torch.testing.assert_close(output, expected, rtol=0.0, atol=0.0)
