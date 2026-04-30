# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for prepare_next_step_optimistic + journal wiring (v3 plan
commit 14).

Plan validation: debug-mode test asserts the journal entry produced for
each step matches the actual mutations that happened. No journal entry
leaks at shutdown.

Tests verify that prepare_next_step_optimistic opens (idempotently) the
journal entry, records per-slot placeholders for active slots, and
records the snapshot owner. Round-trip with commit_step_transaction
zeros out the placeholders.
"""

import pytest
import torch

from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines.async_pipeline_types import DynamicStepPlan
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _build_context(num_speculative_tokens: int = 0) -> DynamicInferenceContext:
    return DynamicInferenceContext(
        model_config=TransformerConfig(
            params_dtype=torch.float32,
            num_layers=2,
            kv_channels=8,
            num_attention_heads=2,
        ),
        inference_config=InferenceConfig(
            max_sequence_length=128,
            buffer_size_gb=0.05,
            block_size_tokens=64,
            max_tokens=128,
            max_requests=8,
            unified_memory_level=0,
            use_flashinfer_fused_rope=None,
            num_speculative_tokens=num_speculative_tokens,
        ),
    )


def _seed_active(ctx, count, paused=0):
    ctx.paused_request_count = paused
    ctx.total_request_count = paused + count
    ctx.active_token_count = count


class TestPrepareNextStepOptimisticJournal:
    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def test_opens_journal_entry(self):
        ctx = _build_context()
        _seed_active(ctx, count=2)
        assert ctx.journal.open_step_count() == 0
        ctx.prepare_next_step_optimistic(step_id=5)
        assert ctx.journal.has_entry(5)

    def test_idempotent_with_engine_begin(self):
        """The engine opens the entry in async_forward (commit 6); a
        subsequent call from prepare_next_step_optimistic must not
        double-create it."""
        ctx = _build_context()
        _seed_active(ctx, count=2)
        first = ctx.begin_step_transaction(7)
        ctx.prepare_next_step_optimistic(step_id=7)
        assert ctx.journal.get_entry(7) is first
        assert ctx.journal.open_step_count() == 1

    def test_records_per_slot_placeholders(self):
        ctx = _build_context()
        _seed_active(ctx, count=3, paused=1)
        ctx.prepare_next_step_optimistic(step_id=0)
        # Slots 1, 2, 3 are the active range.
        assert ctx.num_output_placeholders[1].item() == 1
        assert ctx.num_output_placeholders[2].item() == 1
        assert ctx.num_output_placeholders[3].item() == 1
        # Paused slot stays at 0.
        assert ctx.num_output_placeholders[0].item() == 0
        entry = ctx.journal.get_entry(0)
        assert entry.placeholder_deltas == {1: 1, 2: 1, 3: 1}

    def test_records_snapshot_owner(self):
        ctx = _build_context()
        _seed_active(ctx, count=1)
        ctx.prepare_next_step_optimistic(step_id=0)
        entry = ctx.journal.get_entry(0)
        assert entry.snapshot_buffer_id == ctx._active_snapshot_slot

    def test_speculative_widens_placeholder_delta(self):
        ctx = _build_context(num_speculative_tokens=2)
        _seed_active(ctx, count=2)
        ctx.prepare_next_step_optimistic(step_id=0)
        assert ctx.num_output_placeholders[0].item() == 3  # 1 + speculative_width
        assert ctx.num_output_placeholders[1].item() == 3

    def test_commit_zeros_placeholders_no_journal_leak(self):
        """End-of-step commit decrements placeholders by the recorded
        deltas (overlap off ⇒ acceptance equals delta) and pops the entry."""
        ctx = _build_context()
        _seed_active(ctx, count=2)
        ctx.prepare_next_step_optimistic(step_id=0)
        ctx.commit_step_transaction(0)
        assert ctx.num_output_placeholders[:2].tolist() == [0, 0]
        assert ctx.journal.open_step_count() == 0

    def test_no_active_requests_no_placeholders(self):
        ctx = _build_context()
        _seed_active(ctx, count=0)
        plan = ctx.prepare_next_step_optimistic(step_id=0)
        assert plan.intended_batch_dimensions.decode_req_count == 0
        assert ctx.journal.has_entry(0)
        entry = ctx.journal.get_entry(0)
        assert entry.placeholder_deltas == {}
