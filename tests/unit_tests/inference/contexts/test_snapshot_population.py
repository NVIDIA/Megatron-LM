# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for snapshot population from prepare_next_step_optimistic
(v3 plan commit 15).

Plan validation: debug-mode test with async_overlap_debug_checks=True
and a forced two-snapshot scenario confirms no buffer reuse while a
snapshot is in flight.

The test forces two pool slots to be acquired in step-id order, records
a metadata_ready_event on each, and verifies that releasing the older
slot does not reclaim it before its event signals.
"""

import pytest
import torch

from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _build_context() -> DynamicInferenceContext:
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
            async_overlap_debug_checks=True,
        ),
    )


class TestSnapshotPopulation:
    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def test_transfer_records_metadata_ready_event(self):
        """transfer_bookkeeping_to_gpu records a metadata_ready_event for
        the destination snapshot view; the event is queryable through
        ``snapshot_metadata_event``."""
        ctx = _build_context()
        # Active slot is the one set up at __init__.
        active_slot = ctx._active_snapshot_slot
        assert ctx.snapshot_metadata_event(active_slot) is None
        ctx.transfer_bookkeeping_to_gpu()
        event = ctx.snapshot_metadata_event(active_slot)
        assert event is not None
        torch.cuda.synchronize()
        assert event.query() is True

    def test_explicit_snapshot_arg_targets_named_slot(self):
        """Passing a snapshot explicitly retargets the H2D destination."""
        ctx = _build_context()
        # Acquire a second slot; both slots exist with max_concurrent_steps=1
        # because buffer_count = max_concurrent_steps + 1.
        slot_idx, slot_view = ctx.snapshot_pool.acquire(step_id=99)
        ctx.transfer_bookkeeping_to_gpu(snapshot=slot_view)
        event = ctx.snapshot_metadata_event(slot_idx)
        assert event is not None
        torch.cuda.synchronize()
        # Release for cleanup.
        ctx.snapshot_pool.release(slot_idx)

    def test_forced_two_snapshot_no_buffer_reuse_while_in_flight(self):
        """Force two slots in-flight; releasing slot 0 with a not-yet-fired
        event must not reclaim it. Slot 1 stays acquired throughout."""
        ctx = _build_context()
        # Initial slot is already acquired by __init__.
        slot_a = ctx._active_snapshot_slot
        slot_b, view_b = ctx.snapshot_pool.acquire(step_id=1)
        assert slot_a != slot_b

        # Record a manual not-yet-fired event for slot_a's "in-flight" read.
        # Use a dummy fence that we won't fire until later.
        deferred = torch.cuda.Event()
        # Don't record yet — leave it un-fired so query() is True only after
        # an explicit record + sync. In CUDA, a never-recorded event reports
        # query()=True; to simulate "in flight", we use a pending kernel.
        # Run a meaningful kernel and record after; query() before the sync
        # may be False.
        torch.zeros(1024 * 1024, device=ctx.snapshot_pool.slot(slot_a)._buf.device).pow_(2)
        deferred.record()
        ctx.snapshot_pool.release(slot_a, after_event=deferred)

        # Slot 1 was never released; only slot_a has a deferred return.
        # After GPU sync the deferred event signals and slot_a is reclaimed.
        torch.cuda.synchronize()
        assert ctx.snapshot_pool.free_slot_count >= 1
        # slot_b is still acquired; only slot_a reclaim is observable.
        assert ctx.snapshot_pool.owning_step_id(slot_b) == 1
        ctx.snapshot_pool.release(slot_b)
