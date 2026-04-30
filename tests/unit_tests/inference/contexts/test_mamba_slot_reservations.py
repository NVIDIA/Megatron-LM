# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for Mamba-slot reservations (v3 plan commit 8).

Covers the validation surface as a focused API test: reserve / commit /
rollback contract, journal-driven dispatch, and slot-pool recovery on
rollback (the no-leak invariant).
"""

import pytest
import torch

from megatron.core.inference.config import InferenceConfig, MambaInferenceStateConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines.async_pipeline_types import (
    Reservation,
    ReservationState,
    ResourceKind,
)
from megatron.core.ssm.mamba_hybrid_layer_allocation import Symbols
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _build_hybrid_context() -> DynamicInferenceContext:
    layer_type_list = [Symbols.MAMBA, Symbols.MLP, Symbols.ATTENTION, Symbols.MLP]
    mamba_inference_state_config = MambaInferenceStateConfig(
        layer_type_list,
        (16, 4),
        (4, 8, 4),
        torch.float32,
        torch.float32,
    )
    return DynamicInferenceContext(
        model_config=TransformerConfig(
            params_dtype=torch.float32,
            num_layers=4,
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
            mamba_inference_state_config=mamba_inference_state_config,
            enable_prefix_caching=True,
            prefix_caching_mamba_gb=0.005,
        ),
    )


class TestMambaSlotReservationAPI:
    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def test_reserve_returns_mamba_reservation(self):
        ctx = _build_hybrid_context()
        alloc = ctx.mamba_slot_allocator
        r = alloc.reserve(
            slot_idx=0,
            block_id=5,
            journal_id=ctx.journal.issue_journal_id(),
            must_outlast_snapshot_step_id=11,
        )
        assert isinstance(r, Reservation)
        assert r.resource_kind is ResourceKind.MAMBA_SLOT
        assert r.state is ReservationState.RESERVED
        assert r.must_outlast_snapshot_step_id == 11
        assert r.resource_handle == (0, 5)

    def test_commit_transitions_state_only(self):
        ctx = _build_hybrid_context()
        alloc = ctx.mamba_slot_allocator
        free_before = alloc.free_count
        r = alloc.reserve(slot_idx=0, block_id=5, journal_id=ctx.journal.issue_journal_id())
        alloc.commit(r)
        assert r.state is ReservationState.COMMITTED
        # commit() does not perturb the free pool.
        assert alloc.free_count == free_before

    def test_rollback_returns_slot_to_free_pool(self):
        ctx = _build_hybrid_context()
        alloc = ctx.mamba_slot_allocator
        # Simulate "this slot is owned by block 5".
        alloc.block_to_slot[5] = 7
        alloc.slot_to_block[7] = 5
        # Pull the slot off the free pool to model the allocation.
        alloc.free_count -= 1
        r = alloc.reserve(slot_idx=7, block_id=5, journal_id=ctx.journal.issue_journal_id())
        free_after_reserve = alloc.free_count

        alloc.rollback(r)
        assert r.state is ReservationState.ROLLED_BACK
        assert alloc.block_to_slot[5].item() == -1
        assert alloc.slot_to_block[7].item() == -1
        assert alloc.free_count == free_after_reserve + 1

    def test_journal_commit_drives_mamba_commit(self):
        ctx = _build_hybrid_context()
        alloc = ctx.mamba_slot_allocator
        r = alloc.reserve(slot_idx=2, block_id=3, journal_id=ctx.journal.issue_journal_id())
        ctx.record_resource_reservation(step_id=0, reservation=r)
        ctx.commit_step_transaction(0)
        assert r.state is ReservationState.COMMITTED
        assert ctx.journal.open_step_count() == 0

    def test_journal_rollback_drives_mamba_rollback(self):
        ctx = _build_hybrid_context()
        alloc = ctx.mamba_slot_allocator
        alloc.block_to_slot[3] = 2
        alloc.slot_to_block[2] = 3
        alloc.free_count -= 1
        free_before = alloc.free_count
        r = alloc.reserve(slot_idx=2, block_id=3, journal_id=ctx.journal.issue_journal_id())
        ctx.record_resource_reservation(step_id=0, reservation=r)
        ctx.rollback_step_transaction(0)
        assert r.state is ReservationState.ROLLED_BACK
        assert alloc.free_count == free_before + 1
        assert alloc.block_to_slot[3].item() == -1

    def test_no_slot_leak_after_committed_steps(self):
        """Iterate many reserve+commit cycles; free_count never drifts."""
        ctx = _build_hybrid_context()
        alloc = ctx.mamba_slot_allocator
        free_before = alloc.free_count
        for sid in range(5):
            r = alloc.reserve(
                slot_idx=sid % alloc.max_slots,
                block_id=sid,
                journal_id=ctx.journal.issue_journal_id(),
            )
            ctx.record_resource_reservation(step_id=sid, reservation=r)
            ctx.commit_step_transaction(sid)
        # Commit is state-only; free_count is unchanged.
        assert alloc.free_count == free_before
        assert ctx.journal.open_step_count() == 0
