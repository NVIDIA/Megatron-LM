# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for the transaction-journal API on ``DynamicInferenceContext``
(v3 plan commit 6).

Covers the validation list from the plan:
- placeholder accounting (add → commit decrements / rollback undoes)
- journal entry lifecycle (begin → commit, begin → rollback)
- max-token boundary with one in-flight placeholder
- EOS-on-prior while next is in flight (placeholder doesn't survive commit)
- cancellation with placeholder (rollback removes placeholder)
- failure with placeholder (rollback path)
- no journal entry leak after the step completes (open_step_count == 0)
"""

import pytest
import torch

from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines.async_pipeline_types import (
    Reservation,
    ReservationState,
    ResourceKind,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _build_context() -> DynamicInferenceContext:
    """Minimal CPU-leaning context. The journal/placeholder API is pure
    bookkeeping so a tiny config is enough.
    """
    return DynamicInferenceContext(
        model_config=TransformerConfig(
            params_dtype=torch.float32,
            num_layers=2,
            kv_channels=8,
            num_attention_heads=2,
        ),
        inference_config=InferenceConfig(
            max_sequence_length=128,
            buffer_size_gb=0.02,
            block_size_tokens=64,
            max_tokens=128,
            max_requests=8,
            unified_memory_level=0,
            use_flashinfer_fused_rope=None,
        ),
    )


class TestTransactionJournalAPI:
    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def test_begin_is_idempotent(self):
        ctx = _build_context()
        a = ctx.begin_step_transaction(0)
        b = ctx.begin_step_transaction(0)
        assert a is b
        assert ctx.journal.open_step_count() == 1

    def test_add_placeholders_then_commit_zeroes_out(self):
        """EOS-on-prior while next is in flight: commit decrements the slots
        the step opened; nothing leaks across step boundaries."""
        ctx = _build_context()
        ctx.add_output_placeholders(step_id=0, slot_indices=[0, 1, 2], delta=1)
        assert ctx.num_output_placeholders[:3].tolist() == [1, 1, 1]
        ctx.commit_step_transaction(0)
        assert ctx.num_output_placeholders[:3].tolist() == [0, 0, 0]
        assert ctx.journal.open_step_count() == 0

    def test_rollback_undoes_placeholder_increments(self):
        """Cancellation / failure with placeholder: rollback restores the
        per-slot counts to their pre-step values."""
        ctx = _build_context()
        ctx.num_output_placeholders[5] = 2
        ctx.add_output_placeholders(step_id=1, slot_indices=[5, 6], delta=1)
        assert ctx.num_output_placeholders[5].item() == 3
        assert ctx.num_output_placeholders[6].item() == 1
        ctx.rollback_step_transaction(1)
        assert ctx.num_output_placeholders[5].item() == 2
        assert ctx.num_output_placeholders[6].item() == 0
        assert ctx.journal.open_step_count() == 0

    def test_commit_with_partial_acceptance(self):
        """Speculative-style accounting: commit decrements by the actual
        accepted-token count rather than the full delta."""
        ctx = _build_context()
        ctx.add_output_placeholders(step_id=2, slot_indices=[0, 1], delta=4)
        assert ctx.num_output_placeholders[0].item() == 4
        ctx.commit_step_transaction(2, accepted_token_counts={0: 2, 1: 4})
        assert ctx.num_output_placeholders[0].item() == 2
        assert ctx.num_output_placeholders[1].item() == 0

    def test_record_resource_reservation(self):
        ctx = _build_context()
        r = Reservation(
            journal_id=ctx.journal.issue_journal_id(),
            resource_kind=ResourceKind.KV_BLOCK,
            resource_handle=(7, 3),
        )
        ctx.record_resource_reservation(step_id=0, reservation=r)
        entry = ctx.commit_step_transaction(0)
        assert len(entry.reservations) == 1
        assert entry.reservations[0].state is ReservationState.COMMITTED

    def test_record_snapshot_owner(self):
        ctx = _build_context()
        ctx.record_snapshot_owner(step_id=0, snapshot_buffer_id=11)
        entry = ctx.commit_step_transaction(0)
        assert entry.snapshot_buffer_id == 11

    def test_max_token_boundary_with_in_flight_placeholder(self):
        """Schedulers consult ``active_token_count_with_placeholders``; an
        in-flight placeholder narrows the available token budget."""
        ctx = _build_context()
        # Synthesize an active slot at index 0 with a 1-token in-flight
        # placeholder. ``total_active_placeholders`` requires the slot to
        # fall inside [paused_request_count, total_request_count).
        ctx.total_request_count = 1
        ctx.active_token_count = 4
        ctx.num_output_placeholders[0] = 1
        assert ctx.total_active_placeholders == 1
        assert ctx.active_token_count_with_placeholders == 5

    def test_no_journal_entry_leak_after_commit(self):
        ctx = _build_context()
        for sid in range(5):
            ctx.add_output_placeholders(step_id=sid, slot_indices=[0], delta=1)
            ctx.commit_step_transaction(sid)
        assert ctx.journal.open_step_count() == 0

    def test_rollback_clears_reservations(self):
        ctx = _build_context()
        r = Reservation(
            journal_id=ctx.journal.issue_journal_id(),
            resource_kind=ResourceKind.KV_BLOCK,
            resource_handle=42,
        )
        ctx.record_resource_reservation(step_id=0, reservation=r)
        entry = ctx.rollback_step_transaction(0)
        assert entry.reservations[0].state is ReservationState.ROLLED_BACK
