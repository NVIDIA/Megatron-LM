# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Cache ownership and capacity tests for disaggregated state handoff."""

import asyncio
from collections import deque
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core.inference.config import PrefixCachingEvictionPolicy
from megatron.core.inference.contexts.kv_block_allocator import KVBlockAllocator
from megatron.core.inference.disaggregation.decode_admission import (
    additional_decode_blocks,
    admit_prefilled_decode,
    can_admit_prefilled_decode,
)
from megatron.core.inference.disaggregation.handoff_completion_tracker import (
    HandoffCompletionTracker,
)
from megatron.core.inference.disaggregation.inference_state_handoff import (
    InferenceStateHandoffMixin,
)
from megatron.core.inference.disaggregation.pending_handoff_imports import (
    DeferredKvHandoff,
    PendingKvImport,
    PendingSSMImport,
)
from megatron.core.inference.inference_request import (
    DynamicInferenceRequest,
    DynamicInferenceRequestRecord,
    Status,
    compute_block_hashes_batched,
)
from megatron.core.inference.sampling_params import SamplingParams


class _PendingHandle:
    def poll(self):
        return False

    def wait(self):
        return None


class _TransferAgent:
    def __init__(self):
        self.calls = []
        self.is_push = False

    def begin_pull_blocks(self, peer_meta, src_block_ids, dst_block_ids):
        self.calls.append((peer_meta, list(src_block_ids), list(dst_block_ids)))
        return _PendingHandle()


class _KvAllocator:
    enable_prefix_caching = True

    def __init__(self):
        self.next_block = 10
        self.capacity_available = True
        self.releases = []
        self.registered_parent_hashes = []
        self.block_ref_counts = torch.zeros(256, dtype=torch.int32)
        self.kv_hash_to_block_id = {}

    def is_memory_available(self, _count, potential_matched_count=0):
        assert potential_matched_count >= 0
        return self.capacity_available

    def allocate_memory_blocks(self, count):
        blocks = torch.arange(self.next_block, self.next_block + count, dtype=torch.int32)
        self.next_block += count
        self.block_ref_counts[blocks] = 1
        return blocks

    def release_memory_blocks(self, blocks):
        assert torch.all(self.block_ref_counts[blocks] > 0)
        self.block_ref_counts[blocks] -= 1
        self.releases.append(blocks.tolist())

    def retain_memory_blocks(self, block_ids):
        if block_ids:
            self.block_ref_counts[torch.tensor(block_ids, dtype=torch.int64)] += 1

    def register_kv_block_hashes(self, block_ids, block_hashes, parent_hashes=None):
        self.kv_hash_to_block_id.update(zip(block_hashes, block_ids))
        self.registered_parent_hashes.extend(parent_hashes or [])

    def update_timestamps(self, block_ids):
        return None


class _MambaMetadata:
    def __init__(self, available):
        self.mamba_state_free_slot_count = available
        self.next_slot = 20
        self.freed = []

    def allocate_slot(self):
        if self.mamba_state_free_slot_count == 0:
            return None
        slot = self.next_slot
        self.next_slot += 1
        self.mamba_state_free_slot_count -= 1
        return slot

    def free_slot(self, slot):
        self.freed.append(slot)
        self.mamba_state_free_slot_count += 1


class _SchedulerHarness:
    def schedule_waiting_requests(self):
        if not self.waiting_request_ids:
            return
        request_id = self.waiting_request_ids[0]
        block_ids = torch.tensor(self.blocks_to_bind[request_id], dtype=torch.int32)
        self.context.kv_block_allocator.block_ref_counts[block_ids] += 1
        if request_id in self.partial_admissions:
            self.get_request(request_id).finished_chunk_token_count += 4
        else:
            self.waiting_request_ids.popleft()


class _HandoffHarness(InferenceStateHandoffMixin, _SchedulerHarness):
    def __init__(self, loop, *, hybrid=False, available=0):
        self._loop = loop
        self._initialize_disaggregation_state()
        self.context = SimpleNamespace(
            block_size_tokens=4,
            num_speculative_tokens=0,
            num_prefill_requests=0,
            chunked_prefill_request_id=-1,
            total_request_count=0,
            max_requests=8,
            active_token_count=0,
            max_tokens=8,
            is_hybrid_model=hybrid,
            kv_block_allocator=_KvAllocator(),
            mamba_slot_allocator=None,
            mamba_metadata=_MambaMetadata(available) if hybrid else None,
            memory_buffer=torch.empty(1),
        )
        self._kv_transfer_agent = _TransferAgent()
        if hybrid:
            self._ssm_transfer_agents = {"conv": _TransferAgent(), "recurrent": _TransferAgent()}
        self.pg_collection = SimpleNamespace(tp=None, pp=None, mp=None)
        self.waiting_request_ids = deque()
        self.requests = {}
        self.blocks_to_bind = {}
        self.precomputed_hashes = {}
        self.partial_admissions = set()
        self.track_generated_token_events = False
        self.use_coordinator = False
        self.is_mp_coordinator = False

    async def _notify_cond_for_new_request(self):
        return None

    def add_request(self, request_id, prompt, sampling_params, precomputed_block_hashes=None):
        request = DynamicInferenceRequest(
            request_id=request_id,
            prompt_tokens=torch.tensor(prompt),
            sampling_params=sampling_params,
            precomputed_block_hashes=precomputed_block_hashes or [],
        )
        request.add_event_add_engine()
        self.requests[request_id] = request
        self.precomputed_hashes[request_id] = precomputed_block_hashes
        self.waiting_request_ids.append(request_id)
        return self._loop.create_future()

    def get_request(self, request_id):
        return self.requests[request_id]

    def _check_stop_words_for_request_post_append(self, request):
        for stop_word_ids in request.stop_word_ids or []:
            if request.generated_tokens[-len(stop_word_ids) :] == stop_word_ids:
                return True, 0
        return False, 0


def _meta(request_id):
    return {
        "request_id": request_id,
        "resume_tokens": [99],
        "ssm": {"conv": {"request_id": request_id}, "recurrent": {"request_id": request_id}},
    }


def _drain_loop(loop):
    loop.run_until_complete(asyncio.sleep(0))


def _pending_import(engine, request_id, block_id, block_hash):
    return PendingKvImport(
        request_id=request_id,
        prompt=[1, 2, 3, 4],
        sampling_params=SamplingParams(num_tokens_to_generate=2),
        local_blocks=[block_id],
        hashes=[block_hash],
        cached_prefix_block_count=0,
        handle=None,
        future=engine._loop.create_future(),
        resume_tokens=[99],
    )


def _completion_tracker(world_size=2):
    tracker = object.__new__(HandoffCompletionTracker)
    tracker.rank = 0
    tracker.world_size = world_size
    tracker.is_coordinator = True
    tracker._reports = {}
    tracker._failure_notified = set()
    tracker._socket = None
    return tracker


@pytest.fixture
def handoff_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def test_prefilled_decode_admission_uses_exact_ssm_state_without_prompt_tokens():
    metadata_types = DynamicInferenceRequest.get_metadata_types()
    context = SimpleNamespace(
        block_size_tokens=4,
        num_speculative_tokens=2,
        num_prefill_requests=0,
        chunked_prefill_request_id=-1,
        total_request_count=0,
        max_requests=2,
        active_token_count=0,
        max_tokens=8,
        is_hybrid_model=True,
        request_metadata_types=metadata_types,
        request_metadata={label: torch.empty(2, dtype=dtype) for label, dtype in metadata_types},
        request_to_kv_block_ids=torch.full((2, 4), -1, dtype=torch.int32),
        request_ids=torch.full((2,), -1, dtype=torch.int64),
        request_kv_length_offsets=torch.zeros(2, dtype=torch.int32),
        request_query_lengths=torch.zeros(2, dtype=torch.int32),
        request_output_lengths=torch.zeros(2, dtype=torch.int32),
        request_in_prefill_status_tensor=torch.ones(2, dtype=torch.int32),
        request_kv_block_counts=torch.zeros(2, dtype=torch.int32),
        request_last_kv_block_id=torch.full((2,), -1, dtype=torch.int32),
        request_last_kv_block_offset=torch.zeros(2, dtype=torch.int32),
        token_to_input_ids=torch.zeros(8, dtype=torch.int64),
        token_to_pos_ids=torch.zeros(8, dtype=torch.int64),
        token_to_request_idx=torch.zeros(8, dtype=torch.int32),
        token_to_position_in_request=torch.zeros(8, dtype=torch.int64),
        token_to_local_position_within_kv_block=torch.zeros(8, dtype=torch.int32),
        token_to_block_idx=torch.full((8,), -1, dtype=torch.int32),
    )
    context.mamba_metadata = SimpleNamespace(
        request_to_mamba_state_idx=torch.full((2,), -1, dtype=torch.int32)
    )
    request = DynamicInferenceRequest(
        request_id=7,
        prompt_tokens=torch.arange(3),
        sampling_params=SamplingParams(num_tokens_to_generate=8, termination_id=99),
    )

    context.chunked_prefill_request_id = 9
    with pytest.raises(RuntimeError, match="decode-only engine"):
        can_admit_prefilled_decode(context, 3)
    context.chunked_prefill_request_id = -1
    assert additional_decode_blocks(3, 3, 4) == 1
    admit_prefilled_decode(context, request, [10], [11], [40, 41, 42], 10)

    assert context.total_request_count == 1
    assert context.num_prefill_requests == 0
    assert context.active_token_count == 3
    assert context.request_kv_length_offsets[0].item() == 3
    assert context.request_query_lengths[0].item() == 3
    assert context.request_to_kv_block_ids[0, :2].tolist() == [10, 11]
    assert context.token_to_input_ids[:3].tolist() == [40, 41, 42]
    assert context.token_to_pos_ids[:3].tolist() == [3, 4, 5]
    assert context.token_to_block_idx[:3].tolist() == [10, 11, 11]
    assert context.mamba_metadata.request_to_mamba_state_idx[0].item() == 10
    assert request.remaining_prompt_length == 0


def test_completed_exact_ssm_handoff_enters_decode_without_waiting_queue(handoff_loop):
    engine = _HandoffHarness(handoff_loop, hybrid=True)
    engine.context.num_speculative_tokens = 0
    engine.use_coordinator = False
    engine.is_mp_coordinator = False
    request_future = handoff_loop.create_future()
    request = DynamicInferenceRequest(
        request_id=7,
        prompt_tokens=torch.arange(4),
        sampling_params=SamplingParams(num_tokens_to_generate=3, termination_id=99),
    )
    request.add_event_add_engine()

    def add_request(request_id, prompt, sampling_params, precomputed_block_hashes=None):
        assert request_id == request.request_id
        engine.requests[request_id] = request
        engine.waiting_request_ids.append(request_id)
        return request_future

    engine.add_request = add_request
    pending = PendingKvImport(
        request_id=7,
        prompt=[0, 1, 2, 3],
        sampling_params=request.sampling_params,
        local_blocks=[10],
        continuation_blocks=[11],
        hashes=[101],
        cached_prefix_block_count=0,
        handle=None,
        future=handoff_loop.create_future(),
        ssm=PendingSSMImport(handles=[], live_slot=20),
        resume_tokens=[55],
    )

    admission = (
        "megatron.core.inference.disaggregation.inference_state_handoff.admit_prefilled_decode"
    )
    with mock.patch(admission) as admit:
        engine._finalize_kv_handoff_import(pending)

    assert request.generated_tokens == [55]
    assert request.num_cached_tokens == 4
    assert list(engine.waiting_request_ids) == []
    assert pending.local_blocks == []
    assert pending.continuation_blocks == []
    admit.assert_called_once_with(engine.context, request, [10], [11], [55], ssm_state_idx=20)
    assert pending.ssm is None


def test_setup_pins_handoff_outputs_only_on_prefill():
    class _Backend:
        def __init__(self, **_kwargs):
            pass

        def export_meta(self):
            return {"transport": "test"}

    engine = object.__new__(InferenceStateHandoffMixin)
    engine._initialize_disaggregation_state()
    allocator = SimpleNamespace(
        enable_prefix_caching=True, enable_handoff_pinning=False, pool_size=8
    )
    engine.context = SimpleNamespace(
        kv_block_allocator=allocator,
        memory_buffer=torch.empty(2, 1, 8, 4, 1, 1),
        is_hybrid_model=False,
        num_attention_layers=1,
        num_attention_heads_per_partition=1,
        hidden_size_per_attention_head=1,
        block_size_tokens=4,
        num_mamba_layers=0,
    )
    model_config = SimpleNamespace(num_query_groups=1, num_attention_heads=1)
    engine.controller = SimpleNamespace(
        inference_wrapped_model=SimpleNamespace(model=SimpleNamespace(config=model_config))
    )
    engine.pg_collection = SimpleNamespace(tp=None, pp=None, mp=None)

    backend_factory = (
        "megatron.core.inference.disaggregation.inference_state_handoff."
        "construct_kv_transfer_backend_class"
    )
    pg_size = "megatron.core.inference.disaggregation.inference_state_handoff.get_pg_size"
    pg_rank = "megatron.core.inference.disaggregation.inference_state_handoff.get_pg_rank"
    with (
        mock.patch(backend_factory, return_value=_Backend),
        mock.patch(pg_size, return_value=1),
        mock.patch(pg_rank, return_value=0),
    ):
        engine.setup_kv_transfer("decode")
        assert not allocator.enable_handoff_pinning
        engine.setup_kv_transfer("prefill")
        assert allocator.enable_handoff_pinning


def test_handoff_roles_use_live_ssm_buffers_and_decode_rejects_durable_cache():
    class _Backend:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.instances.append(self)

        def export_meta(self):
            return {"transport": "test"}

        def new_registered_buffer(self, **kwargs):
            return type(self)(**kwargs)

    engine = object.__new__(InferenceStateHandoffMixin)
    engine._initialize_disaggregation_state()
    engine.context = SimpleNamespace(
        kv_block_allocator=SimpleNamespace(
            enable_prefix_caching=True, enable_handoff_pinning=False, pool_size=8
        ),
        memory_buffer=torch.empty(2, 1, 8, 4, 1, 1),
        is_hybrid_model=True,
        num_attention_layers=1,
        num_mamba_layers=1,
        num_attention_heads_per_partition=1,
        hidden_size_per_attention_head=1,
        block_size_tokens=4,
        mamba_slot_allocator=object(),
    )
    model_config = SimpleNamespace(
        num_query_groups=1, num_attention_heads=1, mamba_num_heads=2, mamba_num_groups=1
    )
    engine.controller = SimpleNamespace(
        inference_wrapped_model=SimpleNamespace(model=SimpleNamespace(config=model_config))
    )
    engine.pg_collection = SimpleNamespace(tp=None, pp=None, mp=None)

    backend_factory = (
        "megatron.core.inference.disaggregation.inference_state_handoff."
        "construct_kv_transfer_backend_class"
    )
    pg_size = "megatron.core.inference.disaggregation.inference_state_handoff.get_pg_size"
    pg_rank = "megatron.core.inference.disaggregation.inference_state_handoff.get_pg_rank"
    with (
        mock.patch(backend_factory, return_value=_Backend),
        mock.patch(pg_size, return_value=1),
        mock.patch(pg_rank, return_value=0),
    ):
        with pytest.raises(RuntimeError, match="directly into live SSM state"):
            engine.setup_kv_transfer("decode")

        _Backend.instances.clear()
        engine.context.mamba_slot_allocator = None
        engine.context.max_requests = 8
        engine.context.mamba_conv_states = torch.empty(1, 9, 18, 3)
        engine.context.mamba_ssm_states = torch.empty(1, 9, 2, 4, 5)
        engine.setup_kv_transfer("decode")

    assert _Backend.instances[1].kwargs["memory_buffer"] is engine.context.mamba_conv_states
    assert _Backend.instances[2].kwargs["memory_buffer"] is engine.context.mamba_ssm_states
    assert [backend.kwargs["expected_num_blocks"] for backend in _Backend.instances[1:]] == [9, 9]

    _Backend.instances.clear()
    engine.context.mamba_slot_allocator = object()
    with (
        mock.patch(backend_factory, return_value=_Backend),
        mock.patch(pg_size, return_value=1),
        mock.patch(pg_rank, return_value=0),
    ):
        engine.setup_kv_transfer("prefill")

    assert _Backend.instances[1].kwargs["memory_buffer"] is engine.context.mamba_conv_states
    assert _Backend.instances[2].kwargs["memory_buffer"] is engine.context.mamba_ssm_states
    assert [backend.kwargs["expected_num_blocks"] for backend in _Backend.instances[1:]] == [9, 9]


def test_capacity_miss_defers_before_any_transfer(handoff_loop):
    engine = _HandoffHarness(handoff_loop, hybrid=True, available=0)
    future = engine.add_request_with_kv_handoff(
        7, [1, 2, 3, 4, 5], SamplingParams(num_tokens_to_generate=2), _meta(7), [100, 101]
    )

    assert not future.done()
    assert engine.pending_kv_import_count == 1
    assert [item.request_id for item in engine._deferred_kv_handoffs] == [7]
    assert not engine._pending_kv_imports
    assert not engine._kv_transfer_agent.calls
    assert not engine._ssm_transfer_agents["conv"].calls
    assert engine.context.kv_block_allocator.releases == []

    engine.context.mamba_metadata.mamba_state_free_slot_count = 1
    assert engine._poll_pending_kv_imports() == 0
    _drain_loop(handoff_loop)

    assert not engine._deferred_kv_handoffs
    assert len(engine._pending_kv_imports) == 1
    assert len(engine._kv_transfer_agent.calls) == 1
    assert len(engine._ssm_transfer_agents["conv"].calls) == 1
    assert len(engine._ssm_transfer_agents["recurrent"].calls) == 1
    assert engine._pending_kv_imports[0].ssm.live_slot == 20
    assert engine.context.mamba_metadata.mamba_state_free_slot_count == 0


def test_releasing_pending_hybrid_import_returns_live_slot(handoff_loop):
    engine = _HandoffHarness(handoff_loop, hybrid=True, available=1)
    engine.add_request_with_kv_handoff(
        7, [1, 2, 3, 4], SamplingParams(num_tokens_to_generate=2), _meta(7), [100]
    )
    _drain_loop(handoff_loop)

    pending = engine._pending_kv_imports[0]
    engine._release_pending_kv_import(pending)

    assert pending.ssm is None
    assert engine.context.mamba_metadata.freed == [20]
    assert engine.context.mamba_metadata.mamba_state_free_slot_count == 1


def test_reset_cancels_capacity_queued_handoffs(handoff_loop):
    engine = _HandoffHarness(handoff_loop)
    future = engine.add_request_with_kv_handoff(
        3,
        [3] * 4,
        SamplingParams(num_tokens_to_generate=2),
        {"request_id": 3, "resume_tokens": [99]},
        [103],
    )

    engine._reset_pending_kv_imports()

    assert future.cancelled()
    assert engine.pending_kv_import_count == 0


def test_reset_waits_for_pending_prefill_pushes(handoff_loop):
    engine = _HandoffHarness(handoff_loop)
    handle = mock.Mock()
    engine._pending_kv_pushes = [(7, [handle])]

    engine._reset_pending_kv_imports()

    handle.wait.assert_called_once_with()
    assert not engine._pending_kv_pushes


def test_reset_rejects_an_active_prefill_push(handoff_loop):
    engine = _HandoffHarness(handoff_loop)
    handle = mock.Mock()
    handle.wait.side_effect = TimeoutError
    engine._pending_kv_pushes = [(7, [handle])]

    with pytest.raises(RuntimeError, match="may still access cache storage"):
        engine._reset_pending_kv_imports()

    assert engine._pending_kv_pushes == [(7, [handle])]


def test_reset_rejects_pinned_prefill_blocks(handoff_loop):
    engine = _HandoffHarness(handoff_loop)
    engine._pinned_handoff_blocks[7] = [10]

    with pytest.raises(RuntimeError, match="handoff state remains pinned"):
        engine._reset_pending_kv_imports()

    assert engine._pinned_handoff_blocks == {7: [10]}


def test_release_handoff_returns_detached_prefill_ssm_slot(handoff_loop):
    engine = _HandoffHarness(handoff_loop, hybrid=True)
    engine._pinned_handoff_ssm_slots[7] = 3

    engine.release_handoff_blocks(7)

    assert engine.context.mamba_metadata.freed == [3]
    assert engine._pinned_handoff_ssm_slots == {}


def test_reset_does_not_release_unsafe_import_destinations(handoff_loop):
    engine = _HandoffHarness(handoff_loop)
    block_id = int(engine.context.kv_block_allocator.allocate_memory_blocks(1)[0])
    pending = _pending_import(engine, 4, block_id, 104)
    pending.destinations_safe = False
    engine._pending_kv_imports.append(pending)

    with pytest.raises(RuntimeError, match="may still access cache storage"):
        engine._reset_pending_kv_imports()

    assert engine.context.kv_block_allocator.releases == []
    assert list(engine._pending_kv_imports) == [pending]


def test_prefill_handoff_pins_protect_blocks_and_restore_capacity(handoff_loop):
    engine = _HandoffHarness(handoff_loop)
    engine.context.prefix_cache_lru_clock = 0
    allocator = KVBlockAllocator(
        engine.context,
        pool_size=7,
        paused_limit=0,
        enable_prefix_caching=True,
        prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.LRU,
    )
    engine.context.kv_block_allocator = allocator
    baseline_capacity = allocator.get_allocatable_count()

    blocks = allocator.allocate_memory_blocks(4).tolist()
    first_handoff = blocks[:2]
    second_handoff = blocks[2:]
    registered_blocks = [first_handoff[0], second_handoff[0]]
    partial_tail_blocks = [first_handoff[1], second_handoff[1]]
    allocator.register_kv_block_hashes(registered_blocks, [101, 201], [0, 0])

    for request_id, request_blocks in ((11, first_handoff), (12, second_handoff)):
        allocator.retain_memory_blocks(request_blocks)
        allocator.release_memory_blocks(torch.tensor(request_blocks, dtype=torch.int32))
        engine._pinned_handoff_blocks[request_id] = request_blocks

    assert allocator.block_ref_counts[blocks].tolist() == [1, 1, 1, 1]
    assert int(allocator.get_evictable_block_count()) == 0
    assert allocator.get_allocatable_count() == baseline_capacity - 4
    assert allocator.evict_lru_blocks(1) is False
    assert allocator.allocate_memory_blocks(3) is None

    engine.release_handoff_blocks(11)
    assert allocator.get_allocatable_count() == baseline_capacity - 2
    assert allocator.block_ref_counts[first_handoff].tolist() == [0, 0]
    assert allocator.block_hashes[first_handoff[0]].item() == 101
    assert first_handoff[1] in allocator.block_bag[: allocator.pool_avail].tolist()

    engine.release_handoff_blocks(11)
    engine.release_handoff_blocks(999)
    assert allocator.get_allocatable_count() == baseline_capacity - 2

    engine.release_handoff_blocks(12)
    assert allocator.get_allocatable_count() == baseline_capacity
    assert allocator.block_ref_counts[blocks].tolist() == [0, 0, 0, 0]
    assert allocator.block_hashes[registered_blocks].tolist() == [101, 201]
    free_blocks = allocator.block_bag[: allocator.pool_avail].tolist()
    assert all(block in free_blocks for block in partial_tail_blocks)


def test_peer_failure_quarantines_an_unfinished_local_transfer(handoff_loop):
    engine = _HandoffHarness(handoff_loop)
    block_id = int(engine.context.kv_block_allocator.allocate_memory_blocks(1)[0])
    pending = _pending_import(engine, 4, block_id, 104)
    pending.handle = _PendingHandle()
    engine._pending_kv_imports.append(pending)
    engine._record_handoff_completion_notification(4, failed=True)

    engine._poll_pending_kv_imports()
    engine._admit_pending_kv_imports()

    assert isinstance(pending.future.exception(), RuntimeError)
    assert engine.context.kv_block_allocator.releases == []
    assert not engine._pending_kv_imports
    assert engine._quarantined_kv_imports == [pending]


def test_completed_handoff_keeps_notification_while_decode_is_full(handoff_loop):
    engine = _HandoffHarness(handoff_loop)
    block_id = int(engine.context.kv_block_allocator.allocate_memory_blocks(1)[0])
    pending = _pending_import(engine, 4, block_id, 104)
    pending.terminal_state_reported = True
    engine._pending_kv_imports.append(pending)
    engine._record_handoff_completion_notification(4, failed=False)
    engine.context.total_request_count = engine.context.max_requests

    with mock.patch.object(engine, "_finalize_kv_handoff_import") as finalize:
        engine._poll_pending_kv_imports()
        engine._admit_pending_kv_imports()

        finalize.assert_not_called()
        assert list(engine._pending_kv_imports) == [pending]
        assert engine._handoff_completion_notifications == {4: False}

        engine.context.total_request_count = 0
        engine._admit_pending_kv_imports()
        _drain_loop(handoff_loop)

    finalize.assert_called_once_with(pending)
    assert not engine._pending_kv_imports
    assert not engine._handoff_completion_notifications


def test_transfer_polling_defers_batch_mutation_to_scheduling(handoff_loop):
    engine = _HandoffHarness(handoff_loop)
    engine._kv_transfer_role = "decode"
    block_id = int(engine.context.kv_block_allocator.allocate_memory_blocks(1)[0])
    pending = _pending_import(engine, 4, block_id, 104)
    pending.terminal_state_reported = True
    engine._pending_kv_imports.append(pending)
    engine._record_handoff_completion_notification(4, failed=False)

    with mock.patch.object(engine, "_finalize_kv_handoff_import") as finalize:
        assert engine._poll_pending_kv_imports() == 1
        finalize.assert_not_called()

        engine.schedule_waiting_requests()
        _drain_loop(handoff_loop)

    finalize.assert_called_once_with(pending)


@pytest.mark.parametrize(
    "num_tokens_to_generate, termination_id, stop_word_ids, expected_tokens",
    [(0, -1, None, []), (1, -1, None, [55]), (3, 55, None, [55]), (3, -1, [[55]], [55])],
    ids=["sequence-limit", "generation-limit", "termination-token", "stop-word"],
)
def test_handoff_finishes_without_an_extra_decode_step(
    handoff_loop, num_tokens_to_generate, termination_id, stop_word_ids, expected_tokens
):
    engine = _HandoffHarness(handoff_loop)
    blocks = engine.context.kv_block_allocator.allocate_memory_blocks(2).tolist()
    request_future = handoff_loop.create_future()
    request = DynamicInferenceRequest(
        request_id=7,
        prompt_tokens=torch.arange(4),
        sampling_params=SamplingParams(
            num_tokens_to_generate=num_tokens_to_generate, termination_id=termination_id
        ),
    )
    request.add_event_add_engine()
    request.stop_word_ids = stop_word_ids

    def add_request(request_id, _prompt, _sampling_params, precomputed_block_hashes=None):
        engine.requests[request_id] = request
        engine.waiting_request_ids.append(request_id)
        return request_future

    engine.add_request = add_request
    pending = PendingKvImport(
        request_id=7,
        prompt=[0, 1, 2, 3],
        sampling_params=request.sampling_params,
        local_blocks=[blocks[0]],
        continuation_blocks=[blocks[1]],
        hashes=[101],
        cached_prefix_block_count=0,
        handle=None,
        future=handoff_loop.create_future(),
        resume_tokens=[55],
    )

    with (
        mock.patch(
            "megatron.core.inference.disaggregation.inference_state_handoff."
            "admit_prefilled_decode"
        ) as admit,
        mock.patch.object(engine, "_complete_handoff_request_without_forward") as complete,
    ):
        engine._finalize_kv_handoff_import(pending)

    admit.assert_not_called()
    complete.assert_called_once_with(7)
    assert request.generated_tokens == expected_tokens
    assert engine.context.kv_block_allocator.block_ref_counts[blocks].tolist() == [0, 0]
    assert pending.local_blocks == []
    assert pending.continuation_blocks == []


def test_immediate_handoff_completion_resolves_request_future(handoff_loop):
    engine = object.__new__(InferenceStateHandoffMixin)
    request = DynamicInferenceRequest(
        request_id=7,
        prompt_tokens=torch.arange(4),
        sampling_params=SamplingParams(num_tokens_to_generate=1),
        generated_tokens=[55],
    )
    request.add_event_add_engine()
    record = DynamicInferenceRequestRecord.from_request(request)
    request_future = handoff_loop.create_future()
    engine.requests = {7: SimpleNamespace(record=record, future=request_future)}
    engine.finished_request_count = 0
    engine.use_coordinator = False
    engine.is_mp_coordinator = False
    engine.controller = SimpleNamespace(
        tokenizer=object(), detokenize=mock.Mock(side_effect=["prompt", "answer"])
    )

    engine._complete_handoff_request_without_forward(7)

    assert request.status == Status.COMPLETED
    assert request.generated_length == 1
    assert request.generated_text == "answer"
    assert request_future.result() is record
    assert engine.finished_request_count == 1
    assert 7 not in engine.requests


def test_handoff_completion_waits_for_all_ranks_but_failure_is_immediate():
    tracker = _completion_tracker()
    tracker._record(7, rank=0, failed=False)
    assert tracker.drain_completed() == []

    tracker._record(7, rank=1, failed=False)
    assert tracker.drain_completed() == [(7, False)]

    tracker._record(8, rank=1, failed=True)
    assert tracker.drain_completed() == [(8, True)]
    tracker._record(8, rank=0, failed=False)
    assert tracker.drain_completed() == []
    assert 8 not in tracker._reports


def test_handoff_metadata_batches_completed_requests_across_pipeline(monkeypatch):
    engine = object.__new__(InferenceStateHandoffMixin)
    engine._initialize_disaggregation_state()
    pp_group = object()
    engine.pg_collection = SimpleNamespace(tp=None, pp=pp_group, mp=pp_group)
    engine.context = SimpleNamespace(
        block_size_tokens=4, num_speculative_tokens=0, memory_buffer=torch.empty(1)
    )
    engine._kv_peer_metas = {"global_rank": 0}
    engine._pp_kv_peer_metas = [{"global_rank": 0}, {"global_rank": 1}]
    engine._pp_ssm_peer_metas = [{}, {}]
    requests = [
        SimpleNamespace(
            request_id=7,
            prompt_tokens=torch.arange(8),
            sampling_params=SamplingParams(do_kv_handoff=True),
            disaggregated_params=None,
        ),
        SimpleNamespace(
            request_id=8,
            prompt_tokens=torch.arange(12),
            sampling_params=SamplingParams(do_kv_handoff=True),
            disaggregated_params=None,
        ),
    ]
    local_blocks = [[10, 11], [12, 13, 14]]
    pg_size = "megatron.core.inference.disaggregation.inference_state_handoff.get_pg_size"
    with (
        mock.patch(pg_size, return_value=2),
        mock.patch(
            "torch.distributed.all_gather_into_tensor",
            side_effect=AssertionError("handoff metadata must not run a collective"),
        ),
    ):
        prepared = engine._prepare_handoff_metadata_batch(
            [(request, blocks, None) for request, blocks in zip(requests, local_blocks)],
            {7: [71], 8: [81]},
            {},
        )
        for request in requests:
            engine._capture_handoff_meta(request, prepared[request.request_id])

    assert requests[0].disaggregated_params == {
        "request_id": 7,
        "block_ids": [10, 11],
        "kv_meta": {
            "resume_tokens": [71],
            "pp_metas": [
                {"tp_metas": {"global_rank": 0}, "block_ids": [10, 11]},
                {"tp_metas": {"global_rank": 1}, "block_ids": [10, 11]},
            ],
        },
    }
    assert requests[1].disaggregated_params["kv_meta"]["pp_metas"][1]["block_ids"] == [12, 13, 14]


def test_handoff_metadata_error_releases_all_finished_state():
    engine = object.__new__(InferenceStateHandoffMixin)
    engine._initialize_disaggregation_state()
    engine.pg_collection = SimpleNamespace(tp=None, pp=None, mp=None)
    engine.context = SimpleNamespace(block_size_tokens=4, num_speculative_tokens=0)
    engine._kv_peer_metas = None
    engine._release_pinned_handoff_blocks = mock.Mock()
    engine._release_pinned_handoff_ssm_slot = mock.Mock()
    regular = SimpleNamespace(
        request_id=7,
        prompt_tokens=[1, 2, 3, 4],
        sampling_params=SamplingParams(do_kv_handoff=False),
    )
    handoff = SimpleNamespace(
        request_id=8, prompt_tokens=[1, 2, 3, 4], sampling_params=SamplingParams(do_kv_handoff=True)
    )

    with (
        mock.patch(
            "megatron.core.inference.disaggregation.inference_state_handoff.get_pg_size",
            return_value=1,
        ),
        pytest.raises(RuntimeError, match="transfer setup"),
    ):
        engine._prepare_handoff_metadata_batch(
            [(regular, [10], 3), (handoff, [11], 4)],
            decode_tokens_by_request={},
            decode_log_probs_by_request={},
        )

    assert engine._release_pinned_handoff_blocks.call_args_list == [
        mock.call([10]),
        mock.call([11]),
    ]
    assert engine._release_pinned_handoff_ssm_slot.call_args_list == [mock.call(3), mock.call(4)]


def test_hybrid_handoff_keeps_partial_block_for_exact_decode_resume():
    engine = object.__new__(InferenceStateHandoffMixin)
    engine._initialize_disaggregation_state()
    engine.pg_collection = SimpleNamespace(tp=None, pp=None, mp=None)
    engine.context = SimpleNamespace(block_size_tokens=4, num_speculative_tokens=0)
    engine._kv_peer_metas = {"global_rank": 0}
    engine._pp_ssm_peer_metas = [{"conv": {"global_rank": 0}, "recurrent": {"global_rank": 0}}]
    engine._ssm_transfer_agents = {"conv": mock.Mock(), "recurrent": mock.Mock()}
    engine._release_pinned_handoff_blocks = mock.Mock()
    request = SimpleNamespace(
        request_id=7,
        prompt_tokens=torch.arange(10),
        sampling_params=SamplingParams(do_kv_handoff=True),
        disaggregated_params=None,
    )

    with mock.patch(
        "megatron.core.inference.disaggregation.inference_state_handoff.get_pg_size", return_value=1
    ):
        prepared = engine._prepare_handoff_metadata_batch(
            [(request, [10, 11, 12], 9)], {7: [99]}, {}
        )
        engine._capture_handoff_meta(request, prepared[7])

    assert request.disaggregated_params["block_ids"] == [10, 11, 12]
    assert request.disaggregated_params["kv_meta"]["resume_tokens"] == [99]
    assert engine._pinned_handoff_ssm_slots == {7: 9}
    engine._release_pinned_handoff_blocks.assert_not_called()


def test_handoff_metadata_survives_request_serialization():
    metadata = {"request_id": 7, "block_ids": [10, 11], "kv_meta": {"transport": "nixl"}}
    request = DynamicInferenceRequest(
        request_id=7,
        prompt_tokens=torch.arange(8),
        sampling_params=SamplingParams(num_tokens_to_generate=0),
        disaggregated_params=metadata,
    )

    restored = DynamicInferenceRequest.deserialize(request.serialize())

    assert restored.disaggregated_params == metadata


def test_nixl_handoff_reuses_decode_cached_prefix(handoff_loop):
    engine = _HandoffHarness(handoff_loop)
    prompt = [1] * 12
    hashes = compute_block_hashes_batched(torch.tensor(prompt), engine.context.block_size_tokens)
    cached = engine.context.kv_block_allocator.allocate_memory_blocks(2)
    engine.context.kv_block_allocator.release_memory_blocks(cached)
    engine.context.kv_block_allocator.kv_hash_to_block_id.update(zip(hashes[:2], cached.tolist()))

    kv_meta = {"request_id": 5, "resume_tokens": [99], "resume_log_probs": [-0.25]}
    engine.add_request_with_kv_handoff(
        5,
        prompt,
        SamplingParams(num_tokens_to_generate=2, return_log_probs=True, skip_prompt_log_probs=True),
        kv_meta,
        [100, 101, 102],
    )
    _drain_loop(handoff_loop)

    pending = engine._pending_kv_imports[0]
    assert engine._kv_transfer_agent.calls == [(kv_meta, [102], [12])]
    assert pending.local_blocks == [10, 11, 12]
    assert pending.cached_prefix_block_count == 2
    with mock.patch(
        "megatron.core.inference.disaggregation.inference_state_handoff.admit_prefilled_decode"
    ) as admit:
        engine._finalize_kv_handoff_import(pending)
    assert engine.precomputed_hashes[5] == hashes
    assert engine.context.kv_block_allocator.registered_parent_hashes == [hashes[1]]
    assert engine.get_request(5).generated_tokens == [99]
    assert engine.get_request(5).generated_log_probs == [-0.25]
    assert 5 not in engine.waiting_request_ids
    admit.assert_called_once_with(
        engine.context, engine.get_request(5), [10, 11, 12], [13], [99], ssm_state_idx=None
    )


def test_decode_handoff_defers_until_kv_capacity_is_available(handoff_loop):
    engine = _HandoffHarness(handoff_loop)
    engine.context.kv_block_allocator.capacity_available = False

    kv_meta = {"request_id": 8, "resume_tokens": [99]}
    future = engine.add_request_with_kv_handoff(
        8, [1] * 8, SamplingParams(num_tokens_to_generate=2), kv_meta, [100, 101]
    )

    assert engine.pending_kv_import_count == 1
    assert len(engine._deferred_kv_handoffs) == 1
    assert not engine._pending_kv_imports
    assert not engine._kv_transfer_agent.calls
    assert not future.done()

    engine.context.kv_block_allocator.capacity_available = True
    engine._poll_pending_kv_imports()
    _drain_loop(handoff_loop)

    assert not engine._deferred_kv_handoffs
    assert len(engine._pending_kv_imports) == 1
    assert engine._kv_transfer_agent.calls == [(kv_meta, [100, 101], [10, 11])]


def test_handoff_submission_failure_is_reported_to_model_parallel_coordinator(handoff_loop):
    engine = _HandoffHarness(handoff_loop)
    engine._kv_transfer_agent.begin_pull_blocks = mock.Mock(
        side_effect=RuntimeError("local handoff submission failed")
    )
    engine._handoff_completion_tracker = mock.Mock(world_size=2)
    future = handoff_loop.create_future()
    handoff = DeferredKvHandoff(
        request_id=8,
        prompt=[1] * 8,
        sampling_params=SamplingParams(num_tokens_to_generate=2),
        kv_meta={"request_id": 8, "resume_tokens": [99]},
        src_block_ids=[100, 101],
        hashes=compute_block_hashes_batched(torch.tensor([1] * 8), 4),
        num_blocks=2,
        future=future,
    )

    assert engine._try_start_kv_handoff_import(handoff)
    _drain_loop(handoff_loop)
    engine._poll_pending_kv_imports()
    engine._handoff_completion_tracker.report.assert_called_once_with(8, True)

    engine._record_handoff_completion_notification(8, failed=True)
    engine._admit_pending_kv_imports()

    assert isinstance(future.exception(), RuntimeError)
    assert engine.context.kv_block_allocator.releases == [[10, 11, 12]]
    assert not engine._pending_kv_imports


def test_nixl_handoff_trims_pipeline_stage_block_lists(handoff_loop):
    engine = _HandoffHarness(handoff_loop)
    prompt = [2] * 12
    hashes = compute_block_hashes_batched(torch.tensor(prompt), engine.context.block_size_tokens)
    cached = engine.context.kv_block_allocator.allocate_memory_blocks(1)
    engine.context.kv_block_allocator.release_memory_blocks(cached)
    engine.context.kv_block_allocator.kv_hash_to_block_id[hashes[0]] = int(cached[0])
    kv_meta = {
        "resume_tokens": [99],
        "pp_metas": [
            {"tp_metas": {"rank": 0}, "block_ids": [100, 101, 102]},
            {"tp_metas": {"rank": 1}, "block_ids": [200, 201, 202]},
        ],
    }

    engine.add_request_with_kv_handoff(
        6, prompt, SamplingParams(num_tokens_to_generate=2), kv_meta, [100, 101, 102]
    )
    _drain_loop(handoff_loop)

    submitted_meta, src_blocks, dst_blocks = engine._kv_transfer_agent.calls[0]
    assert src_blocks == [101, 102]
    assert dst_blocks == [11, 12]
    assert [stage["block_ids"] for stage in submitted_meta["pp_metas"]] == [[101, 102], [201, 202]]


def test_nixl_handoff_trims_per_rank_block_lists(handoff_loop):
    engine = _HandoffHarness(handoff_loop)
    prompt = [2] * 8
    hashes = compute_block_hashes_batched(torch.tensor(prompt), engine.context.block_size_tokens)
    cached = engine.context.kv_block_allocator.allocate_memory_blocks(1)
    engine.context.kv_block_allocator.release_memory_blocks(cached)
    engine.context.kv_block_allocator.kv_hash_to_block_id[hashes[0]] = int(cached[0])
    kv_meta = {
        "resume_tokens": [99],
        "tp_metas": [{"rank": 0, "block_ids": [100, 101]}, {"rank": 1, "block_ids": [200, 201]}],
    }

    engine.add_request_with_kv_handoff(
        6, prompt, SamplingParams(num_tokens_to_generate=2), kv_meta, [100, 101]
    )
    _drain_loop(handoff_loop)

    submitted_meta, src_blocks, dst_blocks = engine._kv_transfer_agent.calls[0]
    assert src_blocks == [101]
    assert dst_blocks == [11]
    assert [meta["block_ids"] for meta in submitted_meta["tp_metas"]] == [[101], [201]]


def test_nccl_handoff_does_not_filter_source_push(handoff_loop):
    engine = _HandoffHarness(handoff_loop)
    engine._kv_transfer_agent.is_push = True
    prompt = [3] * 8
    hashes = compute_block_hashes_batched(torch.tensor(prompt), engine.context.block_size_tokens)
    engine.context.kv_block_allocator.kv_hash_to_block_id[hashes[0]] = 4

    kv_meta = {"request_id": 9, "resume_tokens": [99]}
    engine.add_request_with_kv_handoff(
        9, prompt, SamplingParams(num_tokens_to_generate=2), kv_meta, [100, 101]
    )
    _drain_loop(handoff_loop)

    assert engine._kv_transfer_agent.calls == [(kv_meta, [100, 101], [10, 11])]


def test_decode_role_rejects_prompt_scheduling(handoff_loop):
    engine = _HandoffHarness(handoff_loop)
    engine._kv_transfer_role = "decode"
    engine.waiting_request_ids.append(3)

    with pytest.raises(RuntimeError, match="cannot schedule prompt prefill"):
        engine.schedule_waiting_requests()


def test_push_handoff_uses_final_pinned_blocks_ssm_slot():
    engine = object.__new__(InferenceStateHandoffMixin)
    engine._initialize_disaggregation_state()
    engine._kv_transfer_agent = mock.Mock()
    engine._ssm_transfer_agents = {"conv": mock.Mock(), "recurrent": mock.Mock()}
    engine._pinned_handoff_blocks[7] = [20, 21]
    engine._pinned_handoff_ssm_slots[7] = 3
    decode_metas = [
        {"ssm": {"conv": {"agent": "decode-conv"}, "recurrent": {"agent": "decode-ssm"}}}
    ]

    engine.push_handoff_kv(7, decode_metas)

    engine._kv_transfer_agent.begin_push_blocks.assert_called_once_with(
        {"tp_metas": decode_metas}, [20, 21]
    )
    engine._ssm_transfer_agents["conv"].begin_push_blocks.assert_called_once_with(
        {"tp_metas": [{"agent": "decode-conv"}]}, [3]
    )
    engine._ssm_transfer_agents["recurrent"].begin_push_blocks.assert_called_once_with(
        {"tp_metas": [{"agent": "decode-ssm"}]}, [3]
    )


def test_push_handoff_rejects_ssm_state_without_kv_blocks():
    engine = object.__new__(InferenceStateHandoffMixin)
    engine._initialize_disaggregation_state()
    engine._kv_transfer_agent = mock.Mock()
    engine._ssm_transfer_agents = {"conv": mock.Mock(), "recurrent": mock.Mock()}
    engine._pinned_handoff_ssm_slots[7] = 3

    with pytest.raises(RuntimeError, match="detached SSM state but no pinned KV blocks"):
        engine.push_handoff_kv(7, [])

    engine._kv_transfer_agent.begin_push_blocks.assert_not_called()
    engine._ssm_transfer_agents["conv"].begin_push_blocks.assert_not_called()
    engine._ssm_transfer_agents["recurrent"].begin_push_blocks.assert_not_called()


def test_push_handoff_validates_ssm_metadata_before_posting_sends():
    engine = object.__new__(InferenceStateHandoffMixin)
    engine._initialize_disaggregation_state()
    engine._kv_transfer_agent = mock.Mock()
    engine._ssm_transfer_agents = {"conv": mock.Mock(), "recurrent": mock.Mock()}
    engine._pinned_handoff_blocks[7] = [20]
    engine._pinned_handoff_ssm_slots[7] = 3

    with pytest.raises(RuntimeError, match="recurrent SSM transfer state"):
        engine.push_handoff_kv(7, [{"ssm": {"conv": {"agent": "decode"}}}])

    engine._kv_transfer_agent.begin_push_blocks.assert_not_called()
    engine._ssm_transfer_agents["conv"].begin_push_blocks.assert_not_called()


def test_hybrid_handoff_uses_mirrored_slots_without_request_collectives():
    engine = object.__new__(InferenceStateHandoffMixin)
    engine._initialize_disaggregation_state()
    tp_group = object()
    pp_group = object()
    engine.pg_collection = SimpleNamespace(tp=tp_group, pp=pp_group, mp=object())
    engine._kv_peer_metas = [{"rank": "s0t0"}, {"rank": "s0t1"}]
    engine._pp_kv_peer_metas = [engine._kv_peer_metas, [{"rank": "s1t0"}, {"rank": "s1t1"}]]
    engine._ssm_transfer_agents = {"conv": mock.Mock(), "recurrent": mock.Mock()}

    def stage_ssm_metas(stage):
        return {
            state: [{"rank": f"s{stage}t{tp}"} for tp in range(2)]
            for state in ("conv", "recurrent")
        }

    engine._pp_ssm_peer_metas = [stage_ssm_metas(0), stage_ssm_metas(1)]
    engine.context = SimpleNamespace(
        block_size_tokens=4, num_speculative_tokens=0, memory_buffer=torch.empty(1)
    )
    requests = [
        SimpleNamespace(
            request_id=2,
            prompt_tokens=[0] * 16,
            sampling_params=SamplingParams(do_kv_handoff=True),
            disaggregated_params=None,
        ),
        SimpleNamespace(
            request_id=3,
            prompt_tokens=[0] * 12,
            sampling_params=SamplingParams(do_kv_handoff=True),
            disaggregated_params=None,
        ),
    ]
    local_blocks = [[10, 11, 12, 13], [14, 15, 16]]
    live_slots = [103, 112]

    pg_size = "megatron.core.inference.disaggregation.inference_state_handoff.get_pg_size"
    with (
        mock.patch(pg_size, return_value=2),
        mock.patch(
            "torch.distributed.all_gather_into_tensor",
            side_effect=AssertionError("handoff metadata must not run a collective"),
        ),
    ):
        prepared = engine._prepare_handoff_metadata_batch(
            [
                (request, blocks, slot)
                for request, blocks, slot in zip(requests, local_blocks, live_slots)
            ],
            {2: [90], 3: [91]},
            {},
        )
        for request in requests:
            engine._capture_handoff_meta(request, prepared[request.request_id])

    assert engine._pinned_handoff_ssm_slots == {2: 103, 3: 112}
    first_ssm = requests[0].disaggregated_params["kv_meta"]["ssm"]
    assert [
        [tp_meta["block_ids"] for tp_meta in stage["tp_metas"]]
        for stage in first_ssm["conv"]["pp_metas"]
    ] == [[[103], [103]], [[103], [103]]]
    assert (
        requests[0].disaggregated_params["kv_meta"]
        is not requests[1].disaggregated_params["kv_meta"]
    )
    assert all("ssm" not in meta for meta in engine._kv_peer_metas)
