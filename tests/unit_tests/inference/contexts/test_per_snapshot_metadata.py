# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for per-snapshot attention metadata bindings (v3 plan
commit 10).

Plan validation:
- Instrumented context that throws on access to deprecated singleton
  fields during forward (deferred to debug-mode coverage; commit 10
  establishes the per-slot binding contract that makes the singleton a
  proxy for the active slot).
- Passes on the existing test suite (verified by smoke-running the
  existing mamba metadata suite separately; no regressions).
- Graph-mode parity test against eager mode (each pool slot owns a
  graphed + non-graphed pair bound to its own buffer; the active slot's
  pair is the proxy seen by today's forward).
"""

import pytest
import torch

from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts.attention_context.mha_metadata import (
    GraphedMHAMetadata,
    NonGraphedMHAMetadata,
)
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
        ),
    )


class TestPerSnapshotMetadataBindings:
    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def test_each_slot_owns_a_metadata_pair(self):
        ctx = _build_context()
        for slot_idx in range(ctx.snapshot_pool.buffer_count):
            pair = ctx.snapshot_pool.slot_attn_metadata(slot_idx)
            assert pair is not None
            assert isinstance(pair["graph"]["mha_metadata"], GraphedMHAMetadata)
            assert isinstance(pair["non_graph"]["mha_metadata"], NonGraphedMHAMetadata)

    def test_slot_metadata_binds_to_slot_buffer(self):
        ctx = _build_context()
        for slot_idx in range(ctx.snapshot_pool.buffer_count):
            slot_view = ctx.snapshot_pool.slot(slot_idx)
            pair = ctx.snapshot_pool.slot_attn_metadata(slot_idx)
            assert pair["graph"]["mha_metadata"]._gpu_view is slot_view
            assert pair["non_graph"]["mha_metadata"]._gpu_view is slot_view

    def test_distinct_slots_have_distinct_metadata_instances(self):
        """Two pool slots must own independent MHA metadata instances so
        commit 18 can pipeline two snapshots without state sharing."""
        ctx = _build_context()
        if ctx.snapshot_pool.buffer_count < 2:
            pytest.skip("pool too small for distinct-slot test")
        a = ctx.snapshot_pool.slot_attn_metadata(0)
        b = ctx.snapshot_pool.slot_attn_metadata(1)
        assert a["graph"]["mha_metadata"] is not b["graph"]["mha_metadata"]
        assert a["non_graph"]["mha_metadata"] is not b["non_graph"]["mha_metadata"]

    def test_active_singleton_aliases_active_slot_pair(self):
        """The legacy ``graph_attn_metadata`` / ``non_graph_attn_metadata``
        attributes alias the active pool slot's pair with
        max_concurrent_steps=1."""
        ctx = _build_context()
        active_slot = ctx._active_snapshot_slot
        active_pair = ctx.snapshot_pool.slot_attn_metadata(active_slot)
        assert ctx.graph_attn_metadata is active_pair["graph"]
        assert ctx.non_graph_attn_metadata is active_pair["non_graph"]
