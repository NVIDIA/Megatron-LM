# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Cache ownership and capacity tests for disaggregated state handoff."""

import asyncio
from collections import deque
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core.inference.contexts.mamba_slot_allocator import MambaSlotCapacityError
from megatron.core.inference.disaggregation.inference_state_handoff import (
    InferenceStateHandoffMixin,
    _common_ssm_positions,
    _executable_ssm_position,
)
from megatron.core.inference.disaggregation.pending_handoff_imports import PendingKvImport
from megatron.core.inference.inference_request import compute_block_hashes_batched
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
        self.releases = []
        self.registered_parent_hashes = []
        self.block_ref_counts = torch.zeros(256, dtype=torch.int32)
        self.kv_hash_to_block_id = {}

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


class _MambaAllocator:
    def __init__(self, available):
        self.available = available
        self.next_slot = 20
        self.invalidated = []

    def allocate_slots_batch(self, block_ids):
        required = len(set(block_ids))
        if required > self.available:
            raise MambaSlotCapacityError(required=required, available=self.available)
        slots = list(range(self.next_slot, self.next_slot + required))
        self.next_slot += required
        self.available -= required
        return slots

    def invalidate_block(self, block_id):
        self.invalidated.append(block_id)


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
    def __init__(self, loop, available):
        self._loop = loop
        self._initialize_disaggregation_state()
        self.context = SimpleNamespace(
            block_size_tokens=4,
            kv_block_allocator=_KvAllocator(),
            mamba_slot_allocator=_MambaAllocator(available),
            memory_buffer=torch.empty(1),
        )
        self._kv_transfer_agent = _TransferAgent()
        self._ssm_transfer_agents = {"conv": _TransferAgent(), "recurrent": _TransferAgent()}
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


def _meta(request_id, positions):
    return {
        "request_id": request_id,
        "ssm": {
            "positions": positions,
            "conv": {"request_id": request_id},
            "recurrent": {"request_id": request_id},
        },
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


def test_ssm_handoff_selects_common_farthest_executable_checkpoint():
    assert _common_ssm_positions(
        [{"positions": [0, 1, 2]}, {"positions": [0, 2]}, {"positions": [2, 0]}]
    ) == [0, 2]
    assert _executable_ssm_position([0, 1, 2], prompt_length=10, block_size_tokens=4) == [1]


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
        num_mamba_layers=0,
        is_hybrid_model=False,
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


def test_capacity_miss_defers_before_any_transfer(handoff_loop):
    engine = _HandoffHarness(handoff_loop, available=0)
    future = engine.add_request_with_kv_handoff(
        7, [1, 2, 3, 4, 5], SamplingParams(num_tokens_to_generate=2), _meta(7, [0]), [100, 101]
    )

    assert not future.done()
    assert engine.pending_kv_import_count == 1
    assert [item.request_id for item in engine._deferred_kv_handoffs] == [7]
    assert not engine._pending_kv_imports
    assert not engine._kv_transfer_agent.calls
    assert not engine._ssm_transfer_agents["conv"].calls
    assert engine.context.kv_block_allocator.releases == [[10, 11]]

    engine.context.mamba_slot_allocator.available = 1
    assert engine._poll_pending_kv_imports() == 0
    _drain_loop(handoff_loop)

    assert not engine._deferred_kv_handoffs
    assert len(engine._pending_kv_imports) == 1
    assert len(engine._kv_transfer_agent.calls) == 1
    assert len(engine._ssm_transfer_agents["conv"].calls) == 1
    assert len(engine._ssm_transfer_agents["recurrent"].calls) == 1


def test_peer_capacity_miss_rolls_back_before_any_transfer(handoff_loop, monkeypatch):
    engine = _HandoffHarness(handoff_loop, available=1)
    engine.pg_collection.mp = object()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)

    def report_peer_capacity_miss(agreement, op, group):
        agreement.copy_(torch.tensor([0, 0, -1], dtype=agreement.dtype))

    monkeypatch.setattr(torch.distributed, "all_reduce", report_peer_capacity_miss)

    engine.add_request_with_kv_handoff(
        8, [1, 2, 3, 4, 5], SamplingParams(num_tokens_to_generate=2), _meta(8, [0]), [100, 101]
    )

    assert [item.request_id for item in engine._deferred_kv_handoffs] == [8]
    assert not engine._kv_transfer_agent.calls
    assert not engine._ssm_transfer_agents["conv"].calls
    assert engine.context.mamba_slot_allocator.invalidated == [10]
    assert engine.context.kv_block_allocator.releases == [[10, 11]]


def test_reset_cancels_capacity_queued_handoffs(handoff_loop):
    engine = _HandoffHarness(handoff_loop, available=0)
    future = engine.add_request_with_kv_handoff(
        3, [3] * 4, SamplingParams(num_tokens_to_generate=2), _meta(3, [0]), [103]
    )

    engine._reset_pending_kv_imports()

    assert future.cancelled()
    assert engine.pending_kv_import_count == 0


def test_reset_waits_for_pending_prefill_pushes(handoff_loop):
    engine = _HandoffHarness(handoff_loop, available=0)
    handle = mock.Mock()
    engine._pending_kv_pushes = [(7, [handle])]

    engine._reset_pending_kv_imports()

    handle.wait.assert_called_once_with()
    assert not engine._pending_kv_pushes


def test_reset_rejects_an_active_prefill_push(handoff_loop):
    engine = _HandoffHarness(handoff_loop, available=0)
    handle = mock.Mock()
    handle.wait.side_effect = TimeoutError
    engine._pending_kv_pushes = [(7, [handle])]

    with pytest.raises(RuntimeError, match="may still access cache storage"):
        engine._reset_pending_kv_imports()

    assert engine._pending_kv_pushes == [(7, [handle])]


def test_admission_collective_uses_cache_buffer_device(handoff_loop, monkeypatch):
    engine = _HandoffHarness(handoff_loop, available=0)
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

    assert engine._admission_flags() == [(False, None)]
    assert observed_devices == [expected_device]


def test_nixl_handoff_reuses_decode_cached_prefix(handoff_loop):
    engine = _HandoffHarness(handoff_loop, available=0)
    engine.context.mamba_slot_allocator = None
    engine._ssm_transfer_agents = {}
    prompt = [1] * 12
    hashes = compute_block_hashes_batched(
        torch.tensor(prompt), engine.context.block_size_tokens, include_partial=True
    )
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


def test_nixl_handoff_trims_pipeline_stage_block_lists(handoff_loop):
    engine = _HandoffHarness(handoff_loop, available=0)
    engine.context.mamba_slot_allocator = None
    engine._ssm_transfer_agents = {}
    prompt = [2] * 12
    hashes = compute_block_hashes_batched(
        torch.tensor(prompt), engine.context.block_size_tokens, include_partial=True
    )
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
    engine = _HandoffHarness(handoff_loop, available=0)
    engine.context.mamba_slot_allocator = None
    engine._ssm_transfer_agents = {}
    engine._kv_transfer_agent.is_push = True
    prompt = [3] * 8
    hashes = compute_block_hashes_batched(
        torch.tensor(prompt), engine.context.block_size_tokens, include_partial=True
    )
    engine.context.kv_block_allocator.kv_hash_to_block_id[hashes[0]] = 4

    engine.add_request_with_kv_handoff(
        9, prompt, SamplingParams(num_tokens_to_generate=2), {"request_id": 9}, [100, 101]
    )
    _drain_loop(handoff_loop)

    assert engine._kv_transfer_agent.calls == [({"request_id": 9}, [100, 101], [10, 11])]


def test_import_owner_survives_until_request_admission(handoff_loop):
    engine = _HandoffHarness(handoff_loop, available=0)
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
    engine = _HandoffHarness(handoff_loop, available=0)
    block_id = int(engine.context.kv_block_allocator.allocate_memory_blocks(1)[0])
    engine._finalize_kv_handoff_import(_pending_import(engine, 3, block_id, 103))
    engine.blocks_to_bind[3] = [block_id]
    engine.partial_admissions.add(3)

    engine.schedule_waiting_requests()

    assert list(engine.waiting_request_ids) == [3]
    assert engine.get_request(3).finished_chunk_token_count == 4
    assert engine.context.kv_block_allocator.block_ref_counts[block_id] == 1
    assert not engine._handoff_import_owners


def test_push_handoff_reuses_ssm_slots_advertised_during_capture():
    engine = object.__new__(InferenceStateHandoffMixin)
    engine._initialize_disaggregation_state()
    engine.context = SimpleNamespace(mamba_slot_allocator=mock.Mock())
    engine.context.mamba_slot_allocator.get_slot.side_effect = AssertionError(
        "SEND_KV must use the captured SSM slots"
    )
    engine._kv_transfer_agent = mock.Mock()
    engine._ssm_transfer_agents = {"conv": mock.Mock()}
    engine._pinned_handoff_blocks[7] = [20, 21]
    engine._pinned_handoff_ssm_slots[7] = [3]
    decode_metas = [{"ssm": {"conv": {"agent": "decode"}}}]

    engine.push_handoff_kv(7, decode_metas)

    engine._kv_transfer_agent.begin_push_blocks.assert_called_once_with(
        {"tp_metas": decode_metas}, [20, 21]
    )
    engine._ssm_transfer_agents["conv"].begin_push_blocks.assert_called_once_with(
        {"tp_metas": [{"agent": "decode"}]}, [3]
    )


def test_capture_handoff_keeps_request_ssm_metadata_independent():
    engine = object.__new__(InferenceStateHandoffMixin)
    engine._initialize_disaggregation_state()
    engine.pg_collection = SimpleNamespace(tp=None, pp=None)
    engine._kv_peer_metas = {"transport": "nccl", "global_rank": 0}
    engine._ssm_transfer_agents = {"conv": mock.Mock(), "recurrent": mock.Mock()}
    engine._ssm_peer_metas = {
        "conv": {"transport": "nccl", "state": "conv"},
        "recurrent": {"transport": "nccl", "state": "recurrent"},
    }
    engine.context = SimpleNamespace(block_size_tokens=4, mamba_slot_allocator=mock.Mock())
    first = SimpleNamespace(request_id=2, prompt_tokens=[0] * 10, disaggregated_params=None)
    second = SimpleNamespace(request_id=3, prompt_tokens=[0] * 6, disaggregated_params=None)
    engine.context.mamba_slot_allocator.get_slot.side_effect = [4, 5, 6]

    pg_size = "megatron.core.inference.disaggregation.inference_state_handoff.get_pg_size"
    with mock.patch(pg_size, return_value=1):
        engine._capture_handoff_meta(first, [10, 11])
        engine._capture_handoff_meta(second, [12])

    assert first.disaggregated_params["kv_meta"]["ssm"]["positions"] == [1]
    assert second.disaggregated_params["kv_meta"]["ssm"]["positions"] == [0]
    assert engine._pinned_handoff_ssm_slots == {2: [5], 3: [6]}
    assert first.disaggregated_params["kv_meta"] is not second.disaggregated_params["kv_meta"]
    assert "ssm" not in engine._kv_peer_metas
