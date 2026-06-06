# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import torch

from megatron.core.inference.async_txn import (
    AsyncTxnDiagnostics,
    AsyncTxnSkipReason,
    StepTxn,
    TxnRetireQueue,
    classify_decode_child_launch,
)
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)


class FakeEvent:
    def __init__(self, done: bool = False):
        self.done = done

    def query(self) -> bool:
        return self.done


class FakeKVAllocator:
    def __init__(self):
        self.released = []

    def release_memory_blocks(self, block_ids: torch.Tensor) -> None:
        self.released.extend(int(block_id) for block_id in block_ids.tolist())


class FakeMambaMetadata:
    def __init__(self):
        self.request_to_mamba_state_idx = torch.tensor([5, 7, -1], dtype=torch.int32)
        self.request_to_mamba_state_bank = torch.tensor([0, 0, 0], dtype=torch.int32)
        self.freed_slot_ids = []
        self.free_slot_request_indices = []

    def free_slot_ids(self, slot_ids: torch.Tensor) -> None:
        self.freed_slot_ids.extend(int(slot_id) for slot_id in slot_ids.tolist() if slot_id != -1)

    def free_slots(self, request_indices: torch.Tensor) -> None:
        self.free_slot_request_indices.extend(int(idx) for idx in request_indices.tolist())
        self.free_slot_ids(self.request_to_mamba_state_idx[request_indices])
        self.request_to_mamba_state_idx[request_indices] = -1
        self.request_to_mamba_state_bank[request_indices] = 0


class FakeLaunchContext:
    def __init__(self, *, paused_request_count=0):
        self.total_request_count = 2
        self.paused_request_count = paused_request_count
        self.is_hybrid_model = True
        self.chunked_prefill_request_id = -1
        self.num_speculative_tokens = 0
        self.block_size_tokens = 4
        self.request_last_kv_block_offset = torch.tensor([0, 0], dtype=torch.int32)
        self.request_ids = torch.tensor([101, 102], dtype=torch.int32)
        self.kv_block_allocator = SimpleNamespace(is_memory_available=lambda count: True)

    def is_decode_only(self) -> bool:
        return True

    def using_cuda_graph_this_step(self) -> bool:
        return False


def _make_release_context() -> DynamicInferenceContext:
    context = object.__new__(DynamicInferenceContext)
    context.is_hybrid_model = True
    context.mamba_metadata = FakeMambaMetadata()
    context.mamba_slot_allocator = None
    context.kv_block_allocator = FakeKVAllocator()
    context.async_txn_diagnostics = AsyncTxnDiagnostics(enabled=True)
    context.async_txn_retire_queue = TxnRetireQueue(context.async_txn_diagnostics)
    context.request_to_kv_block_ids = torch.tensor(
        [[10, -1], [20, -1], [-1, -1]], dtype=torch.int32
    )
    return context


def _make_controller(context):
    controller = object.__new__(TextGenerationController)
    controller.num_speculative_tokens = 0
    controller._enable_cuda_graph = False
    controller._get_stop_word_finished_ids_callback = None
    controller.model_config = SimpleNamespace(
        expert_model_parallel_size=1,
        num_moe_experts=None,
    )
    controller.inference_wrapped_model = SimpleNamespace(inference_context=context)
    return controller


def test_finished_hybrid_row_mamba_slot_is_not_reused_before_retire():
    context = _make_release_context()
    event = FakeEvent(done=False)

    DynamicInferenceContext.release_memory_blocks_from_request_indexes(
        context,
        torch.tensor([0], dtype=torch.int64),
        retire_event=event,
        defer_release=True,
    )

    assert context.kv_block_allocator.released == []
    assert context.mamba_metadata.freed_slot_ids == []
    assert context.mamba_metadata.request_to_mamba_state_idx.tolist() == [-1, 7, -1]

    event.done = True
    assert context.async_txn_retire_queue.drain_ready() == 2
    assert context.kv_block_allocator.released == [10]
    assert context.mamba_metadata.freed_slot_ids == [5]


def test_immediate_hybrid_release_uses_existing_logical_slot_free_path():
    context = _make_release_context()

    DynamicInferenceContext.release_memory_blocks_from_request_indexes(
        context, torch.tensor([1], dtype=torch.int64)
    )

    assert context.kv_block_allocator.released == [20]
    assert context.mamba_metadata.free_slot_request_indices == [1]
    assert context.mamba_metadata.freed_slot_ids == [7]


def test_hybrid_child_launch_is_not_rejected_when_no_mutation_gate_applies():
    context = FakeLaunchContext()
    controller = _make_controller(context)
    child_txn = StepTxn(step_id=3, request_ids=[101, 102], mamba_slot_ids=(5, 7))

    reason = controller._async_child_launch_skip_reason(
        child_txn,
        return_log_probs=False,
        return_top_n_logprobs=False,
        skip_bookkeeping=False,
    )

    assert reason is None


def test_hybrid_pause_pressure_still_forces_sync_before_child_launch():
    context = FakeLaunchContext(paused_request_count=1)

    eligibility = classify_decode_child_launch(context, async_enabled=True)

    assert not eligibility.eligible
    assert eligibility.reason == AsyncTxnSkipReason.PAUSED_REQUESTS


def test_step_txn_records_logical_mamba_slots_without_bank_internals():
    txn = StepTxn(step_id=3, request_ids=[101, 102], mamba_slot_ids=(5, 7))

    assert txn.mamba_slot_ids == (5, 7)
    assert not hasattr(txn, "candidate_mamba_slot_ids")


def test_async_mamba_bank_accept_flips_only_matching_active_requests():
    context = object.__new__(DynamicInferenceContext)
    context.is_hybrid_model = True
    context.mamba_state_bank_count = 2
    context.paused_request_count = 1
    context.total_request_count = 4
    context.request_ids = torch.tensor([900, 101, 102, 103], dtype=torch.int32)
    context.mamba_metadata = SimpleNamespace(
        request_to_mamba_state_idx=torch.tensor([90, 5, 7, 9], dtype=torch.int32),
        request_to_mamba_state_bank=torch.tensor([0, 0, 1, 0], dtype=torch.int32),
    )

    assert DynamicInferenceContext._mamba_flat_indices(
        context, slice(context.paused_request_count, context.total_request_count)
    ).tolist() == [10, 15, 18]
    assert DynamicInferenceContext._mamba_flat_indices(
        context,
        slice(context.paused_request_count, context.total_request_count),
        use_candidate_bank=True,
    ).tolist() == [11, 14, 19]

    DynamicInferenceContext.accept_async_mamba_state(context, (102, 999, 101))

    assert context.mamba_metadata.request_to_mamba_state_bank.tolist() == [0, 1, 0, 0]
