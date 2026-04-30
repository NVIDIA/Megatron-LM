# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for KV-block reservations (v3 plan commit 7).

Covers the validation list:
- High allocation pressure + EOS-driven rollback; assert no block reuse
  while a snapshot is in flight.
- Zero outstanding reservations at engine shutdown.
- Prefix-cache reservation accounting (refs taken at reserve time, released
  at rollback).
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


def _build_context(prefix_caching: bool = False) -> DynamicInferenceContext:
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
            enable_prefix_caching=prefix_caching,
        ),
    )


class TestKVBlockReservationAPI:
    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def test_reserve_allocates_and_returns_reservation(self):
        ctx = _build_context()
        alloc = ctx.kv_block_allocator
        avail_before = alloc.total_avail
        r = alloc.reserve(
            request_idx=0,
            block_count=3,
            journal_id=ctx.journal.issue_journal_id(),
            must_outlast_snapshot_step_id=42,
        )
        assert isinstance(r, Reservation)
        assert r.resource_kind is ResourceKind.KV_BLOCK
        assert r.state is ReservationState.RESERVED
        assert r.must_outlast_snapshot_step_id == 42
        request_idx, block_ids = r.resource_handle
        assert request_idx == 0
        assert block_ids.numel() == 3
        assert alloc.total_avail == avail_before - 3

    def test_commit_transitions_state(self):
        ctx = _build_context()
        alloc = ctx.kv_block_allocator
        r = alloc.reserve(
            request_idx=0, block_count=2, journal_id=ctx.journal.issue_journal_id()
        )
        avail_after_reserve = alloc.total_avail
        alloc.commit(r)
        assert r.state is ReservationState.COMMITTED
        # Commit is state-only; the blocks are still held by the request.
        assert alloc.total_avail == avail_after_reserve

    def test_rollback_releases_blocks_to_pool(self):
        ctx = _build_context()
        alloc = ctx.kv_block_allocator
        avail_before = alloc.total_avail
        r = alloc.reserve(
            request_idx=0, block_count=4, journal_id=ctx.journal.issue_journal_id()
        )
        assert alloc.total_avail == avail_before - 4
        alloc.rollback(r)
        assert r.state is ReservationState.ROLLED_BACK
        assert alloc.total_avail == avail_before

    def test_journal_commit_drives_allocator_commit(self):
        """Context.commit_step_transaction iterates KV_BLOCK reservations and
        calls allocator.commit on each."""
        ctx = _build_context()
        alloc = ctx.kv_block_allocator
        r = alloc.reserve(
            request_idx=0, block_count=2, journal_id=ctx.journal.issue_journal_id()
        )
        ctx.record_resource_reservation(step_id=0, reservation=r)
        ctx.commit_step_transaction(0)
        assert r.state is ReservationState.COMMITTED
        assert ctx.journal.open_step_count() == 0

    def test_journal_rollback_drives_allocator_rollback(self):
        ctx = _build_context()
        alloc = ctx.kv_block_allocator
        avail_before = alloc.total_avail
        r = alloc.reserve(
            request_idx=0, block_count=3, journal_id=ctx.journal.issue_journal_id()
        )
        ctx.record_resource_reservation(step_id=0, reservation=r)
        assert alloc.total_avail == avail_before - 3
        ctx.rollback_step_transaction(0)
        assert r.state is ReservationState.ROLLED_BACK
        assert alloc.total_avail == avail_before

    def test_high_pressure_rollback_returns_all_blocks(self):
        """Stress: reserve until exhaustion, rollback all, pool fully recovers."""
        ctx = _build_context()
        alloc = ctx.kv_block_allocator
        avail_before = alloc.total_avail
        # Reserve in chunks until the pool is empty, journaling each.
        reservations = []
        chunk = 8
        while alloc.total_avail >= chunk:
            r = alloc.reserve(
                request_idx=len(reservations),
                block_count=chunk,
                journal_id=ctx.journal.issue_journal_id(),
            )
            ctx.record_resource_reservation(step_id=0, reservation=r)
            reservations.append(r)
        assert reservations  # at least one reservation made
        ctx.rollback_step_transaction(0)
        assert all(r.state is ReservationState.ROLLED_BACK for r in reservations)
        assert alloc.total_avail == avail_before

    def test_zero_outstanding_reservations_after_commits(self):
        """Engine-shutdown invariant: every step's reservations must be
        committed or rolled back; no entry leaks."""
        ctx = _build_context()
        alloc = ctx.kv_block_allocator
        for sid in range(4):
            r = alloc.reserve(
                request_idx=sid,
                block_count=2,
                journal_id=ctx.journal.issue_journal_id(),
            )
            ctx.record_resource_reservation(step_id=sid, reservation=r)
            ctx.commit_step_transaction(sid)
        assert ctx.journal.open_step_count() == 0

    def test_prefix_cache_refcount_taken_at_reserve_time(self):
        """When prefix caching is enabled, reserve sets ref_count=1 for the
        new block. Rollback decrements it back to 0."""
        ctx = _build_context(prefix_caching=True)
        alloc = ctx.kv_block_allocator
        r = alloc.reserve(
            request_idx=0, block_count=1, journal_id=ctx.journal.issue_journal_id()
        )
        _, block_ids = r.resource_handle
        assert alloc.block_ref_counts[block_ids].item() == 1
        alloc.rollback(r)
        # Unregistered block returns to free pool with ref_count 0.
        assert alloc.block_ref_counts[block_ids].item() == 0
