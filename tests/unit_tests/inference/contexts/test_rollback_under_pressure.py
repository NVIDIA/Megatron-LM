# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for KV-block rollback under allocator pressure (v3 plan
commit 27).

Plan validation: stress test that intentionally drives the allocator
into pressure with high stop-rate; assert no crash, all reservations
resolved. The full integration stress is exercised by the engine's
eviction-stress tests; here we verify the rollback status enum
contract.
"""

import pytest
import torch

from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.contexts.kv_block_allocator import RollbackStatus
from megatron.core.inference.engines.async_pipeline_types import ReservationState, ResourceKind
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


class TestRollbackUnderPressure:
    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def test_normal_rollback_returns_fully_released(self):
        ctx = _build_context()
        alloc = ctx.kv_block_allocator
        r = alloc.reserve(
            request_idx=0, block_count=2, journal_id=ctx.journal.issue_journal_id()
        )
        status = alloc.rollback(r)
        assert status is RollbackStatus.FULLY_RELEASED
        assert r.state is ReservationState.ROLLED_BACK

    def test_rollback_already_evicted_no_crash(self):
        """Pre-emptive release should not crash a subsequent rollback."""
        ctx = _build_context(prefix_caching=True)
        alloc = ctx.kv_block_allocator
        r = alloc.reserve(
            request_idx=0, block_count=2, journal_id=ctx.journal.issue_journal_id()
        )
        # Simulate an external release before rollback (already evicted).
        _, block_ids = r.resource_handle
        alloc.release_memory_blocks(block_ids)
        status = alloc.rollback(r)
        assert status is RollbackStatus.ALREADY_EVICTED
        assert r.state is ReservationState.ROLLED_BACK

    def test_rollback_partial_when_blocks_shared(self):
        """When prefix caching keeps a block alive via additional refs,
        rollback reports PARTIALLY_HELD instead of crashing."""
        ctx = _build_context(prefix_caching=True)
        alloc = ctx.kv_block_allocator
        r = alloc.reserve(
            request_idx=0, block_count=1, journal_id=ctx.journal.issue_journal_id()
        )
        _, block_ids = r.resource_handle
        # Bump the ref count to simulate a sibling request sharing the block.
        alloc.block_ref_counts[block_ids] += 1
        status = alloc.rollback(r)
        assert status is RollbackStatus.PARTIALLY_HELD

    def test_high_pressure_rollback_resolves_all(self):
        """Drive the allocator to exhaustion, rollback every reservation,
        and assert no crash + the allocator returns to the original avail."""
        ctx = _build_context()
        alloc = ctx.kv_block_allocator
        avail_before = alloc.total_avail
        reservations = []
        chunk = 8
        while alloc.total_avail >= chunk:
            r = alloc.reserve(
                request_idx=len(reservations),
                block_count=chunk,
                journal_id=ctx.journal.issue_journal_id(),
            )
            reservations.append(r)
        for r in reservations:
            status = alloc.rollback(r)
            assert status in (
                RollbackStatus.FULLY_RELEASED,
                RollbackStatus.ALREADY_EVICTED,
                RollbackStatus.PARTIALLY_HELD,
            )
        assert alloc.total_avail == avail_before
