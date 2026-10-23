# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Host-side activation ownership for the native PP/VPP scheduling order."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core.pipeline_parallel.schedules import get_pp_rank_microbatches, get_schedule_table
from megatron.core.transformer.moe.megakernel.dist_moe import runtime
from megatron.core.transformer.moe.megakernel.dist_moe.pipeline import (
    DistMoEPipelineSchedule,
    begin_forward,
    plan_activation_slots,
)


@pytest.mark.parametrize("pp_size,pp_rank", [(2, 0), (2, 1), (4, 0), (4, 3)])
@pytest.mark.parametrize("chunks", [1, 2, 3, 4])
@pytest.mark.parametrize("microbatches,group", [(4, 2), (6, 3), (8, 4)])
def test_slot_plan_matches_native_chunk_queues(pp_size, pp_rank, chunks, microbatches, group):
    communicator = SimpleNamespace(
        pp_group=SimpleNamespace(size=lambda: pp_size, rank=lambda: pp_rank),
        virtual_pipeline_model_parallel_size=chunks if chunks > 1 else None,
    )
    _, _, warmup, _ = get_pp_rank_microbatches(
        microbatches, chunks, group, p2p_communicator=communicator
    )
    table = get_schedule_table(microbatches, chunks, group)
    plan = plan_activation_slots(table, warmup)
    queues = [[] for _ in range(chunks)]
    live_slots = set()
    peak = 0
    for direction, chunk, microbatch in plan.events:
        slot = plan.slots[chunk, microbatch]
        if direction == "forward":
            assert slot not in live_slots
            live_slots.add(slot)
            queues[chunk].append(microbatch)
            peak = max(peak, len(live_slots))
        else:
            assert queues[chunk].pop(0) == microbatch
            live_slots.remove(slot)
    assert not live_slots and not any(queues)
    assert peak == plan.num_slots
    assert plan == plan_activation_slots(table, warmup)


def test_stage_microbatch_slots_depend_on_pipeline_rank():
    table = get_schedule_table(4, 2, 2)
    first = plan_activation_slots(table, 4)
    last = plan_activation_slots(table, 2)
    assert (first.num_slots, last.num_slots) == (5, 3)
    assert first.slots[0, 0] == first.slots[1, 3]
    assert first.slots[1, 0] == first.slots[0, 3]


def test_plans_are_cached_and_only_color_chunks_in_the_pool():
    table = get_schedule_table(4, 2, 2)
    first = plan_activation_slots(table, 4, tracked_chunks={0})
    second = plan_activation_slots(table, 4, tracked_chunks={1})
    assert (first.num_slots, second.num_slots) == (4, 2)
    assert first.events == second.events
    assert all(chunk == 0 for chunk, _ in first.slots)
    assert first is plan_activation_slots(table, 4, tracked_chunks={0})
    with pytest.raises(TypeError):
        first.slots[0, 0] = 99


class _Layer(torch.nn.Module):
    def __init__(self, chunk, context) -> None:
        super().__init__()
        self._chunk_index = chunk
        self.context = context
        self._runtime_pool = context
        self.config = SimpleNamespace(pipeline_model_parallel_size=2)


@pytest.fixture
def layers(monkeypatch):
    context = SimpleNamespace(select_activation_slot=Mock())
    chunks = [torch.nn.ModuleList([_Layer(chunk, context) for _ in range(2)]) for chunk in range(2)]
    monkeypatch.setattr(runtime, "_backends", lambda models: [m for chunk in models for m in chunk])
    monkeypatch.setattr(runtime, "configure_dist_moe_activation_slots", Mock())
    return chunks


def test_non_dist_duck_typed_model_keeps_native_schedule_behavior():
    model = SimpleNamespace(config=SimpleNamespace(), parameters=lambda: ())
    scope = DistMoEPipelineSchedule([model], [(0, 0)], 0)
    assert not scope.modules
    assert scope.forward_step(lambda **kwargs: "output", current_microbatch=0) == "output"
    assert scope.backward_step(lambda: "gradient", 0, 0) == "gradient"
    scope.finish()


def _forward(scope, chunks, chunk, microbatch):
    def execute(**kwargs):
        return [begin_forward(module) for module in chunks[chunk]]

    return scope.forward_step(execute, vp_stage=chunk, current_microbatch=microbatch)


def _backward(scope, tickets, chunk, microbatch):
    def execute():
        for ticket in reversed(tickets):
            ticket.before_backward()
            ticket.complete()

    scope.backward_step(execute, chunk, microbatch)


def test_non_lifo_backward_and_early_slot_reuse(layers):
    # F(A), F(B), B(A), F(C), B(B), B(C), with two saved layers per invocation.
    scope = DistMoEPipelineSchedule(layers[:1], [(i, 0) for i in range(3)], 1)
    first = _forward(scope, layers, 0, 0)
    second = _forward(scope, layers, 0, 1)
    _backward(scope, first, 0, 0)
    third = _forward(scope, layers, 0, 2)
    assert scope.plan.slots[0, 0] == scope.plan.slots[0, 2]
    assert all(ticket.completed for ticket in first)
    assert not any(ticket.completed for ticket in second)
    _backward(scope, second, 0, 1)
    _backward(scope, third, 0, 2)
    scope.finish()
    assert all(module._pipeline_schedule is None for module in layers[0])


def test_graph_recording_binding_is_read_only_and_includes_schedule_order(layers):
    keys = []
    for microbatches in (2, 2, 3):
        scope = DistMoEPipelineSchedule(layers[:1], [(i, 0) for i in range(microbatches)], 1)
        tickets = {}
        first_key = None
        for direction, chunk, microbatch in scope.plan.events:
            if direction == "forward":
                with scope._step(direction, chunk, microbatch):
                    select = layers[0][0].context.select_activation_slot
                    calls_before = select.call_count
                    key = scope.graph_key(layers[0][0])
                    assert scope.graph_key(layers[0][1]) == key
                    assert scope.leases[chunk, microbatch] == {}
                    assert select.call_count == calls_before
                    tickets[chunk, microbatch] = [begin_forward(module) for module in layers[0]]
                    assert scope.graph_key(layers[0][0]) == key
                    if first_key is None:
                        first_key = key
            else:
                _backward(scope, tickets.pop((chunk, microbatch)), chunk, microbatch)
        scope.finish()
        keys.append(first_key)
    assert keys[0] == keys[1]
    # Reusing slot zero is not sufficient: a changed schedule invalidates graph-pool ordering.
    assert keys[0][1:] == keys[2][1:]
    assert keys[0] != keys[2]


def test_interleaved_chunks_and_schedule_change(layers):
    for table, warmup in ((get_schedule_table(4, 2, 2), 4), (get_schedule_table(6, 2, 3), 5)):
        scope = DistMoEPipelineSchedule(layers, table, warmup)
        tickets = {}
        for direction, chunk, microbatch in scope.plan.events:
            if direction == "forward":
                tickets[chunk, microbatch] = _forward(scope, layers, chunk, microbatch)
            else:
                _backward(scope, tickets.pop((chunk, microbatch)), chunk, microbatch)
        scope.finish()
        assert not tickets
    assert runtime.configure_dist_moe_activation_slots.call_count == 2


def test_different_pools_and_empty_chunks_do_not_reserve_unused_slots(layers):
    second_context = SimpleNamespace(select_activation_slot=Mock())
    for module in layers[1]:
        module.context = module._runtime_pool = second_context
    scope = DistMoEPipelineSchedule(layers, get_schedule_table(4, 2, 2), 4)
    capacities = runtime.configure_dist_moe_activation_slots.call_args.args[1]
    assert capacities == {id(layers[0][0]._runtime_pool): 4, id(layers[1][0]._runtime_pool): 2}
    tickets = {}
    for direction, chunk, microbatch in scope.plan.events:
        if direction == "forward":
            tickets[chunk, microbatch] = _forward(scope, layers, chunk, microbatch)
        else:
            _backward(scope, tickets.pop((chunk, microbatch)), chunk, microbatch)
    scope.finish()
    layers[1] = []
    scope = DistMoEPipelineSchedule(layers, get_schedule_table(4, 2, 2), 4)
    for direction, chunk, microbatch in scope.plan.events:
        if direction == "forward":
            tickets[chunk, microbatch] = _forward(scope, layers, chunk, microbatch)
        else:
            _backward(scope, tickets.pop((chunk, microbatch)), chunk, microbatch)
    scope.finish()


def test_layer_order_is_checked_before_backward_execution(layers):
    scope = DistMoEPipelineSchedule(layers[:1], [(0, 0)], 0)
    tickets = _forward(scope, layers, 0, 0)
    with pytest.raises(RuntimeError, match="reverse layer order"):
        scope.backward_step(tickets[0].before_backward, 0, 0)
    assert not tickets[0].started
    assert not tickets[1].completed
    with pytest.raises(RuntimeError, match="previous pipeline schedule"):
        DistMoEPipelineSchedule(layers[:1], [(0, 0)], 0)


def test_incomplete_backward_does_not_release_slot(layers):
    scope = DistMoEPipelineSchedule(layers[:1], [(0, 0)], 0)
    _forward(scope, layers, 0, 0)
    with pytest.raises(RuntimeError, match="unfinished layers"):
        scope.backward_step(lambda: None, 0, 0)
    assert (0, 0) in scope.leases
    with pytest.raises(RuntimeError, match="did not drain"):
        scope.finish()


def test_eval_creates_no_training_lease_and_allows_training_afterwards(layers):
    scope = DistMoEPipelineSchedule(layers, get_schedule_table(4, 2, 2), 8, forward_only=True)
    for _, chunk, microbatch in scope.plan.events:

        def execute(**kwargs):
            assert not torch.is_grad_enabled()
            assert all(begin_forward(module) is None for module in layers[chunk])

        scope.forward_step(execute, vp_stage=chunk, current_microbatch=microbatch)
        assert not scope.leases
    scope.finish()
    assert torch.is_grad_enabled()
    scope = DistMoEPipelineSchedule(layers[:1], [(0, 0)], 0)
    _backward(scope, _forward(scope, layers, 0, 0), 0, 0)
    scope.finish()


def test_pipeline_backend_requires_schedule(layers):
    with pytest.raises(RuntimeError, match="native pipeline schedule"):
        begin_forward(layers[0][0])
    layers[0][0].config.pipeline_model_parallel_size = 1
    assert begin_forward(layers[0][0]) is None


def test_wrong_chunk_and_duplicate_layer_are_rejected(layers):
    scope = DistMoEPipelineSchedule(layers, get_schedule_table(2, 2, 2), 4)
    with scope._step("forward", 0, 0):
        with pytest.raises(RuntimeError, match="another pipeline chunk"):
            begin_forward(layers[1][0])
        begin_forward(layers[0][0])
        with pytest.raises(RuntimeError, match="twice"):
            begin_forward(layers[0][0])
        begin_forward(layers[0][1])
