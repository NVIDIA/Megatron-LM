# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for chunked-prefill placeholder accounting (v3 plan
commit 22).

Plan validation: chunked-prefill stress test with many concurrent
chunked prefills overlapping decodes; correctness vs. serial. The
heavy concurrency stress is exercised by the engine's chunked-prefill
integration tests; here we verify the placeholder-delta contract on a
synthesized chunked-prefill state — non-final chunks contribute
delta=0, the standard active slots contribute 1 + speculative_width.
"""

import pytest
import torch

from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
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
            enable_chunked_prefill=True,
        ),
    )


class TestChunkedPrefillPlaceholders:
    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def test_no_chunked_prefill_uses_standard_delta(self):
        ctx = _build_context()
        ctx.total_request_count = 3
        ctx.paused_request_count = 0
        ctx.active_token_count = 3
        ctx.chunked_prefill_request_id = -1
        ctx.prepare_next_step_optimistic(step_id=0)
        assert ctx.num_output_placeholders[:3].tolist() == [1, 1, 1]

    def test_chunked_prefill_non_final_chunk_delta_zero(self):
        """The chunked-prefill slot contributes placeholder_delta=0 so the
        request remains active without consuming a token slot in the
        in-flight count."""
        ctx = _build_context()
        ctx.total_request_count = 3
        ctx.paused_request_count = 0
        ctx.active_token_count = 3
        ctx.request_ids[:3] = torch.tensor([10, 20, 30], dtype=ctx.request_ids.dtype)
        # Mark slot 2 as the chunked-prefill request (the last active slot).
        ctx.chunked_prefill_request_id = 30
        ctx.prepare_next_step_optimistic(step_id=0)
        # Slots 0 and 1 carry the standard delta; slot 2 stays at 0 because
        # its non-final chunk doesn't generate an output token this step.
        assert ctx.num_output_placeholders[:3].tolist() == [1, 1, 0]

    def test_chunked_prefill_journal_records_zero_delta(self):
        ctx = _build_context()
        ctx.total_request_count = 2
        ctx.paused_request_count = 0
        ctx.active_token_count = 2
        ctx.request_ids[:2] = torch.tensor([100, 200], dtype=ctx.request_ids.dtype)
        ctx.chunked_prefill_request_id = 100
        ctx.prepare_next_step_optimistic(step_id=0)
        entry = ctx.journal.get_entry(0)
        assert entry.placeholder_deltas[0] == 0
        assert entry.placeholder_deltas[1] == 1

    def test_speculative_widens_only_non_chunked_slots(self):
        ctx = _build_context(num_speculative_tokens=2)
        ctx.total_request_count = 2
        ctx.paused_request_count = 0
        ctx.active_token_count = 2
        ctx.request_ids[:2] = torch.tensor([5, 6], dtype=ctx.request_ids.dtype)
        ctx.chunked_prefill_request_id = 6  # slot 1 is non-final chunked.
        ctx.prepare_next_step_optimistic(step_id=0)
        # Slot 0 standard slot: 1 + speculative_width = 3.
        # Slot 1 chunked non-final: 0.
        assert ctx.num_output_placeholders[:2].tolist() == [3, 0]
