# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.async_txn import (
    assert_ep_phase_tag,
    broadcast_ep_accepted_counts,
    broadcast_ep_sampled_tokens,
    broadcast_ep_stop_word_finished_ids,
    resolve_ep_decode_broadcast_plan,
)
from megatron.core.inference.batch_dimensions_utils import InferenceBatchDimensions
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines.async_zmq_communicator import (
    AsyncZMQCommunicator,
    ZMQCollectiveError,
)
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
from megatron.core.inference.ep_async_protocol import EPAsyncPhase, EPAsyncStepProtocol
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)


class FakeEPGroup:
    def __init__(self, *, size=2, rank=1):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


class FakeStepCommunicator:
    world_size = 4
    protocol_mismatch_count = 0

    def __init__(self):
        self.async_calls = []
        self.sync_calls = []

    async def all_reduce_max(self, *values, async_op=True, phase=None, step_id=None):
        self.async_calls.append((phase, step_id, values))
        return values[0] if len(values) == 1 else values

    def sync_all_reduce_max(self, *values, phase=None, step_id=None):
        self.sync_calls.append((phase, step_id, values))
        return values[0] if len(values) == 1 else values


class FakeGraphSlot:
    def __init__(self, slot_id):
        self.slot_id = slot_id

    def cuda_graph_key(self, base_key):
        return (*base_key, ("slot", self.slot_id))


def _make_graph_key_context(*, async_scheduling=True, decode_only=True, active_slot_id=1):
    context = object.__new__(DynamicInferenceContext)
    context.async_scheduling = async_scheduling
    context._using_cuda_graph_this_step = True
    context.padded_active_request_count = 1
    context.padded_active_token_count = 1
    context.padded_batch_dimensions = InferenceBatchDimensions(
        token_count=1,
        prefill_req_count=0 if decode_only else 1,
        decode_req_count=1,
    )
    context.async_decode_slot_ring = SimpleNamespace(
        slots=(FakeGraphSlot(0), FakeGraphSlot(1))
    )
    context.active_decode_slot_id = active_slot_id
    return context


def test_cuda_graph_cache_key_includes_active_async_decode_slot():
    context = _make_graph_key_context(active_slot_id=1)

    key = DynamicInferenceContext.cuda_graph_cache_key(context)

    assert key == ("decode", 1, 1, ("slot", 1))


def test_cuda_graph_cache_key_stays_shape_only_for_non_decode_or_non_async():
    non_decode = _make_graph_key_context(decode_only=False)
    non_async = _make_graph_key_context(async_scheduling=False)

    assert DynamicInferenceContext.cuda_graph_cache_key(non_decode) == non_decode.padded_batch_dimensions
    assert DynamicInferenceContext.cuda_graph_cache_key(non_async) == non_async.padded_batch_dimensions


def test_token_broadcast_gives_identical_survivor_set_across_ranks():
    local_tokens = torch.tensor([99, 4, 99], dtype=torch.int64)
    canonical_tokens = torch.tensor([5, 4, 7], dtype=torch.int64)
    termination_id = torch.tensor([7, 7, 7], dtype=torch.int64)

    def broadcast(tensor, src, group):
        assert src == 0
        assert group.rank() == 1
        tensor.copy_(canonical_tokens)

    broadcast_ep_sampled_tokens(local_tokens, 3, FakeEPGroup(), broadcast_fn=broadcast)

    assert local_tokens.tolist() == [5, 4, 7]
    assert (local_tokens != termination_id).tolist() == [True, True, False]


def test_stop_word_id_broadcast_gives_identical_finish_mask():
    active_request_ids = [101, 102, 103]
    canonical_finish_mask = torch.tensor([0, 1, 0], dtype=torch.int32)

    def broadcast(tensor, src, group):
        tensor.copy_(canonical_finish_mask)

    finished_ids = broadcast_ep_stop_word_finished_ids(
        active_request_ids,
        finished_request_ids=set(),
        group=FakeEPGroup(),
        broadcast_fn=broadcast,
    )

    assert finished_ids == {102}


def test_forced_argmax_tie_uses_canonical_sampled_tokens():
    local_tie_break = torch.tensor([31, 32], dtype=torch.int64)
    canonical_tie_break = torch.tensor([41, 41], dtype=torch.int64)

    def broadcast(tensor, src, group):
        tensor.copy_(canonical_tie_break)

    broadcast_ep_sampled_tokens(local_tie_break, 2, FakeEPGroup(), broadcast_fn=broadcast)

    assert local_tie_break.tolist() == [41, 41]


def test_mtp_accepted_count_broadcast_gives_identical_speculative_prefix():
    local_counts = torch.tensor([0, 2, 1, 9], dtype=torch.int64)
    canonical_counts = torch.tensor([2, 1, 0], dtype=torch.int64)

    def broadcast(tensor, src, group):
        tensor.copy_(canonical_counts)

    broadcast_ep_accepted_counts(local_counts, 3, FakeEPGroup(), broadcast_fn=broadcast)

    assert local_counts.tolist() == [2, 1, 0, 9]


def test_ep_decode_broadcast_plan_selects_nonzero_real_source_rank():
    group = FakeEPGroup(size=4, rank=3)

    def sync_all_reduce_max(local_count, local_src_max, local_neg_src_min):
        assert (local_count, local_src_max, local_neg_src_min) == (1, 3, -3)
        # Ranks 0-2 are dummy ranks and rank 3 owns the real coordinator state.
        values_by_rank = [
            (0, -1, -5),
            (0, -1, -5),
            (0, -1, -5),
            (local_count, local_src_max, local_neg_src_min),
        ]
        return tuple(max(values[index] for values in values_by_rank) for index in range(3))

    plan = resolve_ep_decode_broadcast_plan(
        1,
        group,
        has_real_work=True,
        sync_all_reduce_max_fn=sync_all_reduce_max,
    )

    assert plan.active_request_count == 1
    assert plan.src_group_rank == 3
    assert plan.has_real_work is True


def test_ep_decode_broadcast_plan_gives_dummy_same_source_and_count():
    group = FakeEPGroup(size=4, rank=0)

    def sync_all_reduce_max(local_count, local_src_max, local_neg_src_min):
        assert (local_count, local_src_max, local_neg_src_min) == (0, -1, -5)
        values_by_rank = [
            (local_count, local_src_max, local_neg_src_min),
            (0, -1, -5),
            (0, -1, -5),
            (2, 3, -3),
        ]
        return tuple(max(values[index] for values in values_by_rank) for index in range(3))

    plan = resolve_ep_decode_broadcast_plan(
        0,
        group,
        has_real_work=False,
        sync_all_reduce_max_fn=sync_all_reduce_max,
    )

    assert plan.active_request_count == 2
    assert plan.src_group_rank == 3
    assert plan.has_real_work is True


def test_ep_decode_broadcast_plan_rejects_multiple_real_sources():
    group = FakeEPGroup(size=4, rank=1)

    def sync_all_reduce_max(local_count, local_src_max, local_neg_src_min):
        values_by_rank = [
            (0, -1, -5),
            (local_count, local_src_max, local_neg_src_min),
            (0, -1, -5),
            (1, 3, -3),
        ]
        return tuple(max(values[index] for values in values_by_rank) for index in range(3))

    with pytest.raises(RuntimeError, match="exactly one real source rank"):
        resolve_ep_decode_broadcast_plan(
            1,
            group,
            has_real_work=True,
            sync_all_reduce_max_fn=sync_all_reduce_max,
        )


def _make_dummy_controller(*, num_speculative_tokens):
    calls = []
    controller = object.__new__(TextGenerationController)
    controller.num_speculative_tokens = num_speculative_tokens
    controller.model_config = SimpleNamespace(moe_pad_experts_for_cuda_graph_inference=False)
    controller.inference_wrapped_model = SimpleNamespace(
        inference_context=SimpleNamespace(reset=lambda: calls.append("reset"))
    )
    controller._decide_ep_step_begin = lambda has_real_work: SimpleNamespace(
        reuse_pending_forward=True
    )
    controller._dynamic_step_context_init = lambda is_dummy_forward=False: (_ for _ in ()).throw(
        AssertionError("non-MTP reuse should not run a base dummy forward")
    )
    controller._dynamic_step_forward_logits = lambda input_ids, position_ids: calls.append(
        "forward"
    )
    controller._dummy_decode_sample_collective = lambda: calls.append("decode_result_collective")
    controller._dummy_decode_async_child_forward_if_planned = lambda: calls.append("child_forward")
    controller._dummy_serial_mtp_forward = lambda: calls.append("serial_mtp")
    controller._clear_ep_decode_broadcast_plan = lambda: calls.append("clear_plan")
    controller._clear_ep_step_begin_decision = lambda: calls.append("clear_step")
    controller._dummy_context_h2d_done_event = None
    return controller, calls


def test_non_mtp_dummy_reuse_skips_decode_result_collective():
    controller, calls = _make_dummy_controller(num_speculative_tokens=0)

    controller.dummy_forward()

    assert calls == ["child_forward", "serial_mtp", "clear_plan", "clear_step", "reset"]


def test_mtp_dummy_reuse_keeps_decode_result_collective_before_mtp_and_child():
    controller, calls = _make_dummy_controller(num_speculative_tokens=2)

    controller.dummy_forward()

    assert calls == [
        "decode_result_collective",
        "serial_mtp",
        "child_forward",
        "clear_plan",
        "clear_step",
        "reset",
    ]


def test_dummy_forward_fences_metadata_h2d_before_context_reset():
    controller, calls = _make_dummy_controller(num_speculative_tokens=0)

    class Event:
        def synchronize(self):
            calls.append("h2d_wait")

    controller._dummy_context_h2d_done_event = Event()

    controller.dummy_forward()

    assert calls == [
        "child_forward",
        "serial_mtp",
        "clear_plan",
        "clear_step",
        "h2d_wait",
        "reset",
    ]
    assert controller._dummy_context_h2d_done_event is None


def test_dummy_rank_mirrors_sync_replacement_and_mtp_phase_collectives():
    calls = []

    def broadcast(tensor, src, group):
        calls.append((src, tensor.dtype, tensor.numel()))

    broadcast_ep_sampled_tokens(
        torch.tensor([11, 12], dtype=torch.int64), 2, FakeEPGroup(), broadcast_fn=broadcast
    )
    broadcast_ep_stop_word_finished_ids(
        [101, 102], {102}, FakeEPGroup(), broadcast_fn=broadcast
    )
    broadcast_ep_accepted_counts(
        torch.tensor([1, 0], dtype=torch.int64), 2, FakeEPGroup(), broadcast_fn=broadcast
    )

    assert calls == [
        (0, torch.int64, 2),
        (0, torch.int32, 2),
        (0, torch.int64, 2),
    ]


def test_ep_protocol_does_not_exchange_layout_identifiers():
    request_ids = {101, 102}
    kv_block_ids = {909, 910}
    mamba_slots = {808, 809}
    captured = []

    def broadcast(tensor, src, group):
        captured.append(tensor.clone())

    broadcast_ep_sampled_tokens(
        torch.tensor([11, 12], dtype=torch.int64), 2, FakeEPGroup(), broadcast_fn=broadcast
    )
    broadcast_ep_stop_word_finished_ids(
        [101, 102], {102}, FakeEPGroup(), broadcast_fn=broadcast
    )
    broadcast_ep_accepted_counts(
        torch.tensor([1, 0], dtype=torch.int64), 2, FakeEPGroup(), broadcast_fn=broadcast
    )

    exchanged_values = set()
    for tensor in captured:
        exchanged_values.update(int(value) for value in tensor.tolist())

    assert exchanged_values.isdisjoint(request_ids)
    assert exchanged_values.isdisjoint(kv_block_ids)
    assert exchanged_values.isdisjoint(mamba_slots)


def test_phase_tag_mismatch_raises_explicit_error_instead_of_hanging():
    def all_gather(gathered, local, group):
        gathered[0].copy_(local)
        gathered[1].copy_(torch.tensor([local[0] + 1, local[1], local[2]], dtype=local.dtype))

    with pytest.raises(RuntimeError, match="EP async transaction phase mismatch"):
        assert_ep_phase_tag(
            "sample",
            step_id=4,
            active_request_count=2,
            group=FakeEPGroup(),
            device=torch.device("cpu"),
            all_gather_fn=all_gather,
        )


def test_zmq_collective_payload_rejects_wrong_phase():
    msg = AsyncZMQCommunicator._pack_values_message("ep_graph_shape", 4, (1, 0))

    with pytest.raises(ZMQCollectiveError, match="phase mismatch"):
        AsyncZMQCommunicator._unpack_values_message(
            msg,
            expected_phase="ep_async_child_handoff",
            expected_step_id=4,
            expected_count=2,
        )


def test_controller_ep_sync_helper_uses_named_phase_when_supported():
    calls = []

    class PhaseAwareCommunicator:
        def sync_all_reduce_max(self, *values, phase=None):
            calls.append((phase, values))
            return values

    result = TextGenerationController._sync_all_reduce_max_with_phase(
        PhaseAwareCommunicator(), "ep_async_child_handoff", 1, 3
    )

    assert result == (1, 3)
    assert calls == [("ep_async_child_handoff", (1, 3))]


def test_ep_async_step_protocol_keeps_child_phases_in_active_work_step():
    async def run_test():
        communicator = FakeStepCommunicator()
        protocol = EPAsyncStepProtocol(communicator)

        consensus = await protocol.establish_work_consensus(1, False, async_op=False)
        decode_result = protocol.sync_all_reduce_max(EPAsyncPhase.DECODE_POST_FORWARD, 1, 2, -2)
        handoff_result = protocol.sync_all_reduce_max(EPAsyncPhase.ASYNC_CHILD_HANDOFF, 1, 2, -2)
        graph_result = protocol.sync_all_reduce_max(EPAsyncPhase.GRAPH_SHAPE, 1, 0)
        await protocol.complete_work_step(async_op=False)

        assert consensus.step_id == 0
        assert decode_result == (1, 2, -2)
        assert handoff_result == (1, 2, -2)
        assert graph_result == (1, 0)
        assert communicator.sync_calls == [
            ("ep_decode_post_forward", 0, (1, 2, -2)),
            ("ep_async_child_handoff", 0, (1, 2, -2)),
            ("ep_graph_shape", 0, (1, 0)),
        ]
        assert communicator.async_calls == [
            ("ep_work_consensus", 0, (1, 0)),
            ("ep_work_consensus_ack", 0, (1,)),
            ("ep_step_complete", 0, (1,)),
            ("ep_step_complete_ack", 0, (1,)),
        ]

    asyncio.run(run_test())


def test_ep_async_step_protocol_reuse_pending_forward_skips_dummy_base_forward():
    communicator = FakeStepCommunicator()
    protocol = EPAsyncStepProtocol(communicator)
    protocol.ensure_work_step()

    decision = protocol.decide_step_begin(
        has_real_work=True,
        has_pending_forward=True,
        pending_forward_reusable=True,
    )

    assert decision.reuse_pending_forward is True
    assert decision.discard_pending_forward is False
    assert communicator.sync_calls == [
        ("ep_step_begin", 0, (1, 1, 1, 0, 0)),
        ("ep_step_begin_ack", 0, (1,)),
    ]


def test_ep_async_step_protocol_discards_pending_forward_if_real_rank_missing_child():
    class MissingChildCommunicator(FakeStepCommunicator):
        def sync_all_reduce_max(self, *values, phase=None, step_id=None):
            self.sync_calls.append((phase, step_id, values))
            if phase == "ep_step_begin":
                # Some EP rank has a pending child, but a real rank is missing it.
                return (1, 1, 1, 0, 1)
            return values[0] if len(values) == 1 else values

    communicator = MissingChildCommunicator()
    protocol = EPAsyncStepProtocol(communicator)
    protocol.ensure_work_step()

    decision = protocol.decide_step_begin(
        has_real_work=True,
        has_pending_forward=False,
        pending_forward_reusable=False,
    )

    assert decision.reuse_pending_forward is False
    assert decision.discard_pending_forward is True


def test_ep_work_step_sends_coordinator_reply_after_step_completion():
    async def run_test():
        events = []
        engine = object.__new__(DynamicInferenceEngine)
        engine.ep_world_size = 4
        engine.use_synchronous_zmq_collectives = True

        class Protocol:
            def ensure_work_step(self):
                events.append("ensure")

            async def complete_work_step(self, *, async_op=True):
                events.append(("complete", async_op))

        async def async_step(*, send_coordinator_replies=True):
            events.append(("step", send_coordinator_replies))
            return {"finished_request_records": ["record"]}

        engine.ep_async_step_protocol = Protocol()
        engine.async_step = async_step
        engine._send_finished_records_to_coordinator = lambda records: events.append(
            ("reply", records)
        )

        await DynamicInferenceEngine._run_ep_work_step(engine, local_pending=1)

        assert events == [
            "ensure",
            ("step", False),
            ("complete", False),
            ("reply", ["record"]),
        ]

    asyncio.run(run_test())


def test_controller_ep_sync_helper_prefers_step_protocol_when_available():
    calls = []
    controller = object.__new__(TextGenerationController)

    class Protocol:
        enabled = True

        def sync_all_reduce_max(self, phase, *values):
            calls.append((phase, values))
            return values

    controller._ep_async_protocol = Protocol()

    result = TextGenerationController._sync_ep_protocol_all_reduce_max(
        controller, EPAsyncPhase.ASYNC_CHILD_HANDOFF, 1, 3
    )

    assert result == (1, 3)
    assert calls == [(EPAsyncPhase.ASYNC_CHILD_HANDOFF, (1, 3))]


def test_ep_async_step_protocol_starts_cached_consensus_work_steps():
    communicator = FakeStepCommunicator()
    protocol = EPAsyncStepProtocol(communicator)

    step_id = protocol.ensure_work_step()
    result = protocol.sync_all_reduce_max(EPAsyncPhase.ASYNC_CHILD_HANDOFF, 1, 0, 0)

    assert step_id == 0
    assert result == (1, 0, 0)
    assert communicator.sync_calls == [("ep_async_child_handoff", 0, (1, 0, 0))]
