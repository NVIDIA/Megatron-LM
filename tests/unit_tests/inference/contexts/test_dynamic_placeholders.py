# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.contexts.step_journal import StepJournal


class FakeGpuView:
    current_snapshot_slot_id = 0


def _fake_context():
    context = object.__new__(DynamicInferenceContext)
    context.total_request_count = 4
    context.paused_request_count = 1
    context.active_token_count = 3
    context.current_dynamic_step_id = 0
    context.step_journal = StepJournal()
    context.gpu_view = FakeGpuView()
    context.async_overlap_debug_counters = {"placeholder_count": 0}
    context.request_ids = torch.tensor([10, 11, 12, 13], dtype=torch.int32)
    context.request_kv_length_offsets = torch.tensor([7, 8, 9, 10], dtype=torch.int32)
    context.request_query_lengths = torch.tensor([1, 1, 2, 3], dtype=torch.int32)
    context.request_output_lengths = torch.tensor([16, 16, 16, 16], dtype=torch.int32)
    context.num_output_placeholders = torch.zeros(4, dtype=torch.int32)
    return context


def test_output_placeholders_adjust_sequence_and_kv_lengths():
    context = _fake_context()
    context.begin_step_journal(0)

    context.add_output_placeholders(torch.tensor([1, 2]), {11: 1, "12": 2})

    assert context.placeholder_adjusted_sequence_length(1) == 10
    assert context.placeholder_adjusted_kv_length(2) == 13
    assert context.get_placeholder_adjusted_active_sequence_lengths().tolist() == [10, 13, 13]
    assert context.async_overlap_debug_counters["placeholder_count"] == 3
    assert dict(context.step_journal.get_open_entry(0).placeholder_token_counts) == {
        "11": 1,
        "12": 2,
    }


def test_output_placeholders_consume_and_detect_underflow():
    context = _fake_context()
    context.add_output_placeholders([1, 2], [1, 1])

    context.consume_output_placeholders([1, 2], [1, 0])

    assert context.num_output_placeholders.tolist() == [0, 0, 1, 0]
    assert context.async_overlap_debug_counters["placeholder_count"] == 1
    with pytest.raises(AssertionError, match="underflow"):
        context.consume_output_placeholders([1], [1])


def test_output_placeholders_move_and_swap_with_request_slots():
    context = _fake_context()
    context.num_output_placeholders[:] = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    context.request_in_prefill_status_tensor = torch.zeros(4, dtype=torch.int32)
    context.request_to_kv_block_ids = torch.zeros((4, 1), dtype=torch.int32)
    context.request_kv_block_counts = torch.ones(4, dtype=torch.int32)
    context.request_last_kv_block_id = torch.zeros(4, dtype=torch.int32)
    context.request_last_kv_block_offset = torch.zeros(4, dtype=torch.int32)
    context.request_metadata = {}
    context.is_hybrid_model = False
    next_tokens = torch.arange(4)

    context._move_book_keeping_tensors(torch.tensor([3]), torch.tensor([1]), next_tokens)
    assert context.num_output_placeholders.tolist() == [0, 3, 2, 3]

    context._swap_book_keeping_tensors(torch.tensor([1]), torch.tensor([2]), next_tokens)
    assert context.num_output_placeholders.tolist() == [0, 2, 3, 3]


def test_finished_or_failed_request_release_clears_placeholders():
    context = _fake_context()
    context.num_output_placeholders[:] = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    context.request_to_kv_block_ids = torch.tensor([[1], [2], [-1], [3]], dtype=torch.int32)
    context.is_hybrid_model = False
    context.mamba_slot_allocator = None

    class FakeAllocator:
        def release_memory_blocks(self, block_ids):
            self.released = block_ids.tolist()

    context.kv_block_allocator = FakeAllocator()

    context.release_memory_blocks_from_request_indexes(torch.tensor([1, 3]))

    assert context.num_output_placeholders.tolist() == [0, 0, 2, 0]
    assert context.async_overlap_debug_counters["placeholder_count"] == 2
    assert context.kv_block_allocator.released == [2, 3]
