# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Cache ownership and lifecycle tests for disaggregated KV handoff."""

import asyncio
from collections import deque
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core.inference.disaggregation.inference_state_handoff import (
    InferenceStateHandoffMixin,
)
from megatron.core.inference.disaggregation.pending_handoff_imports import (
    DeferredKvHandoff,
    PendingKvImport,
)
from megatron.core.inference.inference_request import (
    DynamicInferenceRequest,
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

    def register_kv_block_hashes(self, block_ids, block_hashes, parent_hashes=None):
        self.kv_hash_to_block_id.update(zip(block_hashes, block_ids))
        self.registered_parent_hashes.extend(parent_hashes or [])

    def update_timestamps(self, block_ids):
        return None


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
    def __init__(self, loop):
        self._loop = loop
        self._initialize_disaggregation_state()
        self.context = SimpleNamespace(
            block_size_tokens=4, kv_block_allocator=_KvAllocator(), memory_buffer=torch.empty(1)
        )
        self._kv_transfer_agent = _TransferAgent()
        self.pg_collection = SimpleNamespace(mp=None)
        self.waiting_request_ids = deque()
        self.requests = {}
        self.blocks_to_bind = {}
        self.precomputed_hashes = {}
        self.partial_admissions = set()

    async def _notify_cond_for_new_request(self):
        return None

    def add_request(self, request_id, prompt, sampling_params, precomputed_block_hashes=None):
        self.requests[request_id] = SimpleNamespace(finished_chunk_token_count=0)
        self.precomputed_hashes[request_id] = precomputed_block_hashes
        self.waiting_request_ids.append(request_id)
        return self._loop.create_future()

    def get_request(self, request_id):
        return self.requests[request_id]


def _drain_loop(loop):
    loop.run_until_complete(asyncio.sleep(0))


def _pending_import(engine, request_id, block_id, block_hash):
    return PendingKvImport(
        request_id=request_id,
        prompt=[1, 2, 3, 4],
        sampling_params=SamplingParams(num_tokens_to_generate=2),
        local_blocks=[block_id],
        hashes=[block_hash],
        hashes_to_register=1,
        hash_registration_start=0,
        handle=None,
        future=engine._loop.create_future(),
    )


@pytest.fixture
def handoff_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


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
        num_attention_layers=1,
        num_attention_heads_per_partition=1,
        hidden_size_per_attention_head=1,
        block_size_tokens=4,
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

        engine.context.is_hybrid_model = True
        with pytest.raises(RuntimeError, match="require recurrent-state handoff"):
            engine.setup_kv_transfer("prefill")


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


def test_admission_collective_uses_cache_buffer_device(handoff_loop, monkeypatch):
    engine = _HandoffHarness(handoff_loop)
    pending = _pending_import(engine, 4, 10, 104)
    pending.handle = _PendingHandle()
    engine._pending_kv_imports.append(pending)
    engine.pg_collection.mp = object()
    expected_device = engine.context.memory_buffer.device
    observed_devices = []
    torch_tensor = torch.tensor

    def record_device(data, *args, device=None, **kwargs):
        observed_devices.append(device)
        return torch_tensor(data, *args, device="cpu", **kwargs)

    monkeypatch.setattr(torch, "tensor", record_device)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)

    assert engine._admission_flags() == [(False, False, None)]
    assert observed_devices == [expected_device]


def test_peer_poll_failure_fails_this_rank(handoff_loop, monkeypatch):
    engine = _HandoffHarness(handoff_loop)
    block_id = int(engine.context.kv_block_allocator.allocate_memory_blocks(1)[0])
    pending = _pending_import(engine, 4, block_id, 104)
    pending.handle = _PendingHandle()
    engine._pending_kv_imports.append(pending)
    engine.pg_collection.mp = object()

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)

    def report_peer_failure(flags, op, group):
        flags.copy_(torch.tensor([-1], dtype=flags.dtype))

    monkeypatch.setattr(torch.distributed, "all_reduce", report_peer_failure)

    with pytest.raises(RuntimeError, match="failed on a model-parallel peer"):
        engine._poll_pending_kv_imports()

    assert isinstance(pending.future.exception(), RuntimeError)
    assert engine.context.kv_block_allocator.releases == [[block_id]]


@pytest.mark.parametrize(
    ("remote_request_id", "remote_blocks", "message"),
    [(8, [20], "request_ids"), (7, [], "block_counts")],
)
def test_capture_handoff_rejects_inconsistent_pipeline_metadata(
    monkeypatch, remote_request_id, remote_blocks, message
):
    engine = object.__new__(InferenceStateHandoffMixin)
    engine._initialize_disaggregation_state()
    pp_group = object()
    engine.pg_collection = SimpleNamespace(pp=pp_group)
    engine.context = SimpleNamespace(block_size_tokens=4)
    engine._kv_peer_metas = {"global_rank": 0}
    engine._release_pinned_handoff_blocks = mock.Mock(return_value=1)
    request = SimpleNamespace(
        request_id=7, prompt_tokens=torch.arange(4), disaggregated_params=None
    )

    def gather_inconsistent_metadata(output, local_entry, group):
        assert group is pp_group
        output[:] = [
            local_entry,
            {
                "request_id": remote_request_id,
                "kv_meta": {"global_rank": 1},
                "block_ids": remote_blocks,
            },
        ]

    pg_size = "megatron.core.inference.disaggregation.inference_state_handoff.get_pg_size"
    with (
        mock.patch(pg_size, return_value=2),
        mock.patch("torch.distributed.is_initialized", return_value=True),
        mock.patch("torch.distributed.all_gather_object", side_effect=gather_inconsistent_metadata),
        pytest.raises(RuntimeError, match=message),
    ):
        engine._capture_handoff_meta(request, [10])

    engine._release_pinned_handoff_blocks.assert_called_once_with([10])


@pytest.mark.parametrize(
    ("prompt_tokens", "expected_blocks", "released_blocks"), [(10, [10, 11], [12]), (3, [], [10])]
)
def test_capture_handoff_keeps_only_complete_prompt_blocks(
    prompt_tokens, expected_blocks, released_blocks
):
    engine = object.__new__(InferenceStateHandoffMixin)
    engine._initialize_disaggregation_state()
    engine.pg_collection = SimpleNamespace(pp=None)
    engine.context = SimpleNamespace(block_size_tokens=4)
    engine._kv_peer_metas = {"global_rank": 0}
    engine._release_pinned_handoff_blocks = mock.Mock(return_value=len(released_blocks))
    request = SimpleNamespace(
        request_id=7, prompt_tokens=torch.arange(prompt_tokens), disaggregated_params=None
    )

    with mock.patch(
        "megatron.core.inference.disaggregation.inference_state_handoff.get_pg_size", return_value=1
    ):
        engine._capture_handoff_meta(request, [10, 11, 12][: len(expected_blocks) + 1])

    assert request.disaggregated_params["block_ids"] == expected_blocks
    assert engine._pinned_handoff_blocks[7] == expected_blocks
    engine._release_pinned_handoff_blocks.assert_called_once_with(released_blocks)


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

    engine.add_request_with_kv_handoff(
        5, prompt, SamplingParams(num_tokens_to_generate=2), {"request_id": 5}, [100, 101, 102]
    )
    _drain_loop(handoff_loop)

    pending = engine._pending_kv_imports[0]
    assert engine._kv_transfer_agent.calls == [({"request_id": 5}, [102], [12])]
    assert pending.local_blocks == [10, 11, 12]
    assert pending.hash_registration_start == 2
    assert pending.hashes_to_register == 1
    engine._finalize_kv_handoff_import(pending)
    assert engine.precomputed_hashes[5] == hashes
    assert engine.context.kv_block_allocator.registered_parent_hashes == [hashes[1]]


def test_decode_handoff_defers_until_kv_capacity_is_available(handoff_loop):
    engine = _HandoffHarness(handoff_loop)
    engine.context.kv_block_allocator.capacity_available = False

    future = engine.add_request_with_kv_handoff(
        8, [1] * 8, SamplingParams(num_tokens_to_generate=2), {"request_id": 8}, [100, 101]
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
    assert engine._kv_transfer_agent.calls == [({"request_id": 8}, [100, 101], [10, 11])]


def test_handoff_submission_failure_is_agreed_across_model_parallel_ranks(
    handoff_loop, monkeypatch
):
    engine = _HandoffHarness(handoff_loop)
    engine.pg_collection.mp = object()
    future = handoff_loop.create_future()
    handoff = DeferredKvHandoff(
        request_id=8,
        prompt=[1] * 8,
        sampling_params=SamplingParams(num_tokens_to_generate=2),
        kv_meta={"request_id": 8},
        src_block_ids=[100, 101],
        hashes=compute_block_hashes_batched(torch.tensor([1] * 8), 4),
        num_blocks=2,
        future=future,
    )

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)

    def report_peer_start_failure(value, op, group):
        if value.numel() == 1:
            value.zero_()

    monkeypatch.setattr(torch.distributed, "all_reduce", report_peer_start_failure)

    with pytest.raises(RuntimeError, match="submission failed on a model-parallel peer"):
        engine._try_start_kv_handoff_import(handoff)

    assert isinstance(future.exception(), RuntimeError)
    # The local receive may still be active, so its destination blocks cannot
    # return to the allocator until the engine is restarted.
    assert engine.context.kv_block_allocator.releases == []
    assert not engine._pending_kv_imports


def test_nixl_handoff_trims_pipeline_stage_block_lists(handoff_loop):
    engine = _HandoffHarness(handoff_loop)
    prompt = [2] * 12
    hashes = compute_block_hashes_batched(torch.tensor(prompt), engine.context.block_size_tokens)
    cached = engine.context.kv_block_allocator.allocate_memory_blocks(1)
    engine.context.kv_block_allocator.release_memory_blocks(cached)
    engine.context.kv_block_allocator.kv_hash_to_block_id[hashes[0]] = int(cached[0])
    kv_meta = {
        "pp_metas": [
            {"tp_metas": {"rank": 0}, "block_ids": [100, 101, 102]},
            {"tp_metas": {"rank": 1}, "block_ids": [200, 201, 202]},
        ]
    }

    engine.add_request_with_kv_handoff(
        6, prompt, SamplingParams(num_tokens_to_generate=2), kv_meta, [100, 101, 102]
    )
    _drain_loop(handoff_loop)

    submitted_meta, src_blocks, dst_blocks = engine._kv_transfer_agent.calls[0]
    assert src_blocks == [101, 102]
    assert dst_blocks == [11, 12]
    assert [stage["block_ids"] for stage in submitted_meta["pp_metas"]] == [[101, 102], [201, 202]]


def test_nccl_handoff_does_not_filter_source_push(handoff_loop):
    engine = _HandoffHarness(handoff_loop)
    engine._kv_transfer_agent.is_push = True
    prompt = [3] * 8
    hashes = compute_block_hashes_batched(torch.tensor(prompt), engine.context.block_size_tokens)
    engine.context.kv_block_allocator.kv_hash_to_block_id[hashes[0]] = 4

    engine.add_request_with_kv_handoff(
        9, prompt, SamplingParams(num_tokens_to_generate=2), {"request_id": 9}, [100, 101]
    )
    _drain_loop(handoff_loop)

    assert engine._kv_transfer_agent.calls == [({"request_id": 9}, [100, 101], [10, 11])]


def test_import_owner_survives_until_request_admission(handoff_loop):
    engine = _HandoffHarness(handoff_loop)
    first_block = int(engine.context.kv_block_allocator.allocate_memory_blocks(1)[0])
    second_block = int(engine.context.kv_block_allocator.allocate_memory_blocks(1)[0])

    engine._finalize_kv_handoff_import(_pending_import(engine, 1, first_block, 101))
    engine._finalize_kv_handoff_import(_pending_import(engine, 2, second_block, 101))

    engine.blocks_to_bind[1] = [second_block]
    engine.schedule_waiting_requests()

    ref_counts = engine.context.kv_block_allocator.block_ref_counts
    assert ref_counts[first_block] == 0
    assert ref_counts[second_block] == 2
    assert 1 not in engine._handoff_import_owners
    assert 2 in engine._handoff_import_owners


def test_chunked_admission_releases_import_owner(handoff_loop):
    engine = _HandoffHarness(handoff_loop)
    block_id = int(engine.context.kv_block_allocator.allocate_memory_blocks(1)[0])
    engine._finalize_kv_handoff_import(_pending_import(engine, 3, block_id, 103))
    engine.blocks_to_bind[3] = [block_id]
    engine.partial_admissions.add(3)

    engine.schedule_waiting_requests()

    assert list(engine.waiting_request_ids) == [3]
    assert engine.get_request(3).finished_chunk_token_count == 4
    assert engine.context.kv_block_allocator.block_ref_counts[block_id] == 1
    assert not engine._handoff_import_owners
