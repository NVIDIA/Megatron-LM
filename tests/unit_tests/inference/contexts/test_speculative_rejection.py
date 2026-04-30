# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for speculative-rejection journaling (v3 plan commit 23).

Plan validation: spec-decode integration test with deterministic seed,
async-on output token-by-token equal to async-off output;
acceptance-rate metric within 1% of serial baseline. The integration
parity is exercised by the engine's speculative-decode integration
tests; here we verify the journaling primitive that drives the
discarded-lookahead-token accounting.
"""

import pytest
import torch

from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _build_context(num_speculative_tokens: int = 2) -> DynamicInferenceContext:
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


class TestSpeculativeRejection:
    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def test_placeholder_delta_includes_speculative_width(self):
        ctx = _build_context(num_speculative_tokens=3)
        ctx.total_request_count = 1
        ctx.paused_request_count = 0
        ctx.active_token_count = 1
        ctx.prepare_next_step_optimistic(step_id=0)
        # 1 base + 3 speculative = 4
        assert ctx.num_output_placeholders[0].item() == 4

    def test_record_rejected_no_op_for_zero(self):
        ctx = _build_context()
        ctx.begin_step_transaction(0)
        ctx.record_rejected_speculative_tokens(0, slot_idx=0, num_rejected=0)
        entry = ctx.journal.get_entry(0)
        assert not hasattr(entry, "_discarded_speculative_per_slot")

    def test_record_rejected_accumulates_per_slot(self):
        ctx = _build_context()
        ctx.begin_step_transaction(0)
        ctx.record_rejected_speculative_tokens(0, slot_idx=2, num_rejected=2)
        ctx.record_rejected_speculative_tokens(0, slot_idx=2, num_rejected=1)
        ctx.record_rejected_speculative_tokens(0, slot_idx=5, num_rejected=4)
        entry = ctx.journal.get_entry(0)
        assert entry._discarded_speculative_per_slot == {2: 3, 5: 4}

    def test_commit_with_partial_acceptance_clears_accepted(self):
        """When a step's commit reports acceptance < delta, placeholders
        decrement by the accepted count; the remainder is what the
        retirement service surfaces as discarded_lookahead_token_count."""
        ctx = _build_context(num_speculative_tokens=2)
        ctx.total_request_count = 1
        ctx.paused_request_count = 0
        ctx.active_token_count = 1
        ctx.prepare_next_step_optimistic(step_id=0)
        # delta = 3; accept 2 (1 base + 1 speculative).
        ctx.record_rejected_speculative_tokens(0, slot_idx=0, num_rejected=1)
        ctx.commit_step_transaction(0, accepted_token_counts={0: 2})
        # 3 added, 2 decremented → 1 residual placeholder, surfaced as
        # discarded-lookahead in the retirement layer.
        assert ctx.num_output_placeholders[0].item() == 1
