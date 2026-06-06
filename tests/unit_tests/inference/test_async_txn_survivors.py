# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import torch

from megatron.core.inference.async_txn import (
    AsyncTxnDiagnostics,
    KVBlockLease,
    StepTxn,
    TxnRetireQueue,
    classify_decode_child_launch,
)
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext


class FakeEvent:
    def __init__(self, done: bool = False):
        self.done = done

    def query(self) -> bool:
        return self.done


class FakeKVAllocator:
    block_routing = {}
    paused_count = 0
    total_avail = 0

    def __init__(self, available: int = 0):
        self.available = available
        self.active_count = 16
        self.released = []

    def is_memory_available(self, num_blocks: int) -> bool:
        return num_blocks <= self.available

    def release_memory_blocks(self, block_ids: torch.Tensor) -> None:
        self.released.extend(int(block_id) for block_id in block_ids.tolist())

    def get_active_avail(self) -> int:
        return self.active_count

    def get_paused_used(self) -> int:
        return 0


class FakeLaunchContext:
    def __init__(self, *, offsets=(3,), available_blocks=0):
        self.paused_request_count = 0
        self.total_request_count = len(offsets)
        self.request_last_kv_block_offset = torch.tensor(offsets, dtype=torch.int32)
        self.request_ids = torch.arange(100, 100 + len(offsets), dtype=torch.int32)
        self.block_size_tokens = 4
        self.num_speculative_tokens = 0
        self.kv_block_allocator = FakeKVAllocator(available_blocks)
        self.chunked_prefill_request_id = -1

    def is_decode_only(self) -> bool:
        return True


def _make_bookkeeping_context() -> DynamicInferenceContext:
    context = object.__new__(DynamicInferenceContext)
    context.block_size_tokens = 4
    context.paused_request_count = 0
    context.total_request_count = 2
    context.padded_active_request_count = 2
    context.padded_active_token_count = 2
    context.is_hybrid_model = False
    context.request_ids = torch.tensor([101, 102], dtype=torch.int32)
    context.batch_dimensions = SimpleNamespace(req_count=2)
    context.padded_batch_dimensions = SimpleNamespace(req_count=2)
    context.kv_block_allocator = SimpleNamespace(dummy_block_idx=-1)

    buf = torch.zeros(2048, dtype=torch.uint8)
    context._cpu_bookkeeping_buf = buf
    offset = 0

    def view(dtype, shape):
        nonlocal offset
        element_size = torch.tensor([], dtype=dtype).element_size()
        offset += (element_size - offset % element_size) % element_size
        count = 1
        for dim in shape:
            count *= dim
        out = buf[offset : offset + count * element_size].view(dtype).view(shape)
        offset += count * element_size
        return out

    context.token_to_input_ids = view(torch.int64, (4,))
    context.token_to_pos_ids = view(torch.int64, (4,))
    context.token_to_request_idx = view(torch.int32, (4,))
    context.token_to_position_in_request = view(torch.int32, (4,))
    context.token_to_local_position_within_kv_block = view(torch.int32, (4,))
    context._staging_request_in_prefill_status = view(torch.int32, (4,))
    context._staging_request_query_lengths = view(torch.int32, (4,))
    context._staging_request_kv_length_offsets = view(torch.int32, (4,))
    context._staging_temperature = view(torch.float32, (4,))
    context._staging_top_k = view(torch.int32, (4,))
    context._staging_top_p = view(torch.float32, (4,))
    context.active_request_last_token_idxs = view(torch.int32, (4,))
    context._cpu_mha_query_lengths = view(torch.int32, (2,))
    context._cpu_mha_cu_query_seq_lengths = view(torch.int32, (3,))
    context._cpu_mha_kv_seq_lengths = view(torch.int32, (4,))
    context._cpu_mha_cu_kv_seq_lengths = view(torch.int32, (5,))
    context.token_to_block_idx = view(torch.int64, (4,))
    context._cpu_mha_block_table = view(torch.int32, (2, 2))

    context.token_to_input_ids[:2] = torch.tensor([11, 22])
    context.token_to_pos_ids[:2] = torch.tensor([1, 3])
    context.token_to_request_idx[:2] = torch.tensor([0, 1])
    context.token_to_position_in_request[:2] = torch.tensor([1, 3])
    context.token_to_local_position_within_kv_block[:2] = torch.tensor([1, 3])
    context.token_to_block_idx[:2] = torch.tensor([10, 20])
    context.request_query_lengths = torch.ones(2, dtype=torch.int32)
    context.request_kv_length_offsets = torch.tensor([1, 3], dtype=torch.int32)
    context.active_request_metadata = {
        "temperature": torch.ones(4, dtype=torch.float32),
        "top_k": torch.ones(4, dtype=torch.int32),
        "top_p": torch.zeros(4, dtype=torch.float32),
    }
    context.active_request_last_token_idxs[:2] = torch.tensor([0, 1])
    context._cpu_mha_query_lengths[:2] = torch.ones(2, dtype=torch.int32)
    context._cpu_mha_cu_query_seq_lengths[:3] = torch.tensor([0, 1, 2], dtype=torch.int32)
    context._cpu_mha_block_table[:] = torch.tensor([[10, -1], [20, -1]], dtype=torch.int32)
    context._cpu_mha_kv_seq_lengths[:2] = torch.tensor([2, 4], dtype=torch.int32)
    context._cpu_mha_cu_kv_seq_lengths[:3] = torch.tensor([0, 2, 6], dtype=torch.int32)
    return context


def _make_update_context() -> DynamicInferenceContext:
    context = object.__new__(DynamicInferenceContext)
    context.num_speculative_tokens = 0
    context.num_prefill_requests = 0
    context.paused_request_count = 0
    context.total_request_count = 2
    context.active_token_count = 2
    context.max_requests = 4
    context.max_tokens = 4
    context.block_size_tokens = 4
    context.chunked_prefill_request_id = -1
    context.paused_tokens = None
    context.paused_speculative_tokens = None
    context.is_hybrid_model = False
    context.mamba_slot_allocator = None
    context.kv_block_allocator = FakeKVAllocator()
    context.async_txn_diagnostics = AsyncTxnDiagnostics(enabled=True)
    context.async_txn_retire_queue = TxnRetireQueue(context.async_txn_diagnostics)

    context.request_ids = torch.tensor([101, 102, -1, -1], dtype=torch.int32)
    context.request_kv_length_offsets = torch.tensor([3, 1, 0, 0], dtype=torch.int32)
    context.request_in_prefill_status_tensor = torch.zeros(4, dtype=torch.int32)
    context.request_query_lengths = torch.ones(4, dtype=torch.int32)
    context.request_output_lengths = torch.full((4,), 99, dtype=torch.int32)
    context.request_to_kv_block_ids = torch.tensor(
        [[10, -1], [20, -1], [-1, -1], [-1, -1]], dtype=torch.int32
    )
    context.request_kv_block_counts = torch.tensor([1, 1, 0, 0], dtype=torch.int32)
    context.request_last_kv_block_id = torch.tensor([10, 20, -1, -1], dtype=torch.int32)
    context.request_last_kv_block_offset = torch.tensor([3, 1, 0, 0], dtype=torch.int32)
    context.request_metadata = {}
    context.token_to_input_ids = torch.zeros(8, dtype=torch.int64)
    context.token_to_pos_ids = torch.zeros(8, dtype=torch.int64)
    context.token_to_request_idx = torch.zeros(8, dtype=torch.int32)
    context.token_to_position_in_request = torch.zeros(8, dtype=torch.int64)
    context.token_to_local_position_within_kv_block = torch.zeros(8, dtype=torch.int32)
    context.token_to_block_idx = torch.zeros(8, dtype=torch.int64)
    context.reset_attention_state = lambda: None
    return context


def test_finish_in_middle_accepts_survivors():
    txn = StepTxn(step_id=4, request_ids=[101, 102, 103])
    txn.launched = True
    txn.mark_committed([101, 103], terminal_request_ids=[102])

    assert txn.is_consumable_after_commit([101, 103], terminal_request_ids=txn.terminal_request_ids)


def test_compaction_changes_row_order_but_survivor_outputs_follow_request_ids():
    txn = StepTxn(step_id=4, request_ids=[101, 102, 103])
    row_ordered_samples = torch.tensor([11, 22, 33])
    committed_request_ids = [103, 101]

    row_idxs = txn.committed_row_indices(committed_request_ids)

    assert row_ordered_samples[list(row_idxs)].tolist() == [33, 11]


def test_terminal_ignored_row_kv_blocks_are_not_reused_before_retire():
    context = _make_update_context()
    event = FakeEvent(done=False)
    txn = StepTxn(step_id=4, request_ids=[101, 102], forward_done_event=event, launched=True)

    DynamicInferenceContext.update_requests(
        context, torch.tensor([0, 1], dtype=torch.uint8), torch.tensor([7, 8]), async_txn=txn
    )

    assert context.kv_block_allocator.released == []
    event.done = True
    assert context.async_txn_retire_queue.drain_ready() == 1
    assert context.kv_block_allocator.released == [10]


def test_boundary_crosser_uses_reserved_block():
    context = _make_update_context()
    txn = StepTxn(
        step_id=4,
        request_ids=[101, 102],
        kv_block_leases=(KVBlockLease(request_id=101, block_column=1, block_id=30),),
        launched=True,
    )

    result = DynamicInferenceContext.update_requests(
        context, torch.tensor([1, 1], dtype=torch.uint8), torch.tensor([7, 8]), async_txn=txn
    )

    assert result["newly_paused_request_ids"] is None
    assert context.paused_request_count == 0
    assert context.request_to_kv_block_ids[0, 1].item() == 30
    assert context.request_last_kv_block_id[0].item() == 30
    assert context.token_to_block_idx[:2].tolist() == [30, 20]


def test_missing_reservation_prevents_launch_instead_of_post_launch_pause():
    context = FakeLaunchContext(offsets=(3,), available_blocks=0)

    result = classify_decode_child_launch(context, async_enabled=True)

    assert not result.eligible
    assert result.required_boundary_blocks == 1


def test_missing_reservation_ignores_same_step_terminal_free_headroom():
    context = FakeLaunchContext(offsets=(3, 1), available_blocks=0)

    result = classify_decode_child_launch(context, async_enabled=True)

    assert not result.eligible
    assert result.required_boundary_blocks == 1


def test_guard_failure_after_launch_does_not_accept_nonterminal_disappearance():
    txn = StepTxn(step_id=4, request_ids=[101, 102])
    txn.launched = True
    txn.mark_committed([101], terminal_request_ids=[])

    assert not txn.is_consumable_after_commit([101], terminal_request_ids=txn.terminal_request_ids)


def test_patch_plain_decode_child_uses_reserved_boundary_block():
    context = _make_bookkeeping_context()
    child = context._cpu_bookkeeping_buf.clone()
    lease = KVBlockLease(request_id=102, block_column=1, block_id=77)

    DynamicInferenceContext._build_plain_decode_child_bookkeeping(
        context, child, 2, kv_block_leases=(lease,)
    )
    child_block_idx = DynamicInferenceContext._cpu_bookkeeping_clone_view(
        context, context.token_to_block_idx, child
    )
    child_local = DynamicInferenceContext._cpu_bookkeeping_clone_view(
        context, context.token_to_local_position_within_kv_block, child
    )
    child_block_table = DynamicInferenceContext._cpu_bookkeeping_clone_view(
        context, context._cpu_mha_block_table, child
    )

    assert child_local[:2].tolist() == [2, 0]
    assert child_block_idx[:2].tolist() == [10, 77]
    assert child_block_table[1, 1].item() == 77
