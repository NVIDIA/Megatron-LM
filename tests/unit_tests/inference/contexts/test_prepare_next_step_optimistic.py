# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for ``prepare_next_step_optimistic`` (v3 plan commit 13).

Plan validation: parity test asserts ``update_requests`` (wrapper)
produces bit-identical state before and after the refactor.

prepare_next_step_optimistic owns the sample-independent header of
update_requests: ``num_prefill_requests = 0`` and the
prefill→decode-status flip. Commits 14 and 15 layer the journal entry,
KV-block reservations, paused-request reordering, decode token-
destination indices, and snapshot-pool population on top of this method.
"""

import pytest
import torch

from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines.async_pipeline_types import DynamicStepPlan
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


class TestPrepareNextStepOptimistic:
    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def test_returns_dynamic_step_plan(self):
        ctx = _build_context()
        plan = ctx.prepare_next_step_optimistic(step_id=7)
        assert isinstance(plan, DynamicStepPlan)
        assert plan.step_id == 7
        assert plan.input_plan.speculative_width == ctx.num_speculative_tokens

    def test_resets_num_prefill_requests_to_zero(self):
        ctx = _build_context()
        ctx.num_prefill_requests = 5
        ctx.prepare_next_step_optimistic(step_id=0)
        assert ctx.num_prefill_requests == 0

    def test_flips_prefill_status_to_decode(self):
        """Bit-identical-to-today header behavior: any slot whose
        ``request_in_prefill_status_tensor`` is 1 becomes 0 after the call."""
        ctx = _build_context()
        ctx.request_in_prefill_status_tensor[:3] = 1
        ctx.request_in_prefill_status_tensor[3:5] = 0
        ctx.prepare_next_step_optimistic(step_id=0)
        assert ctx.request_in_prefill_status_tensor[:3].tolist() == [0, 0, 0]
        assert ctx.request_in_prefill_status_tensor[3:5].tolist() == [0, 0]

    def test_intended_batch_dimensions_match_active_state(self):
        ctx = _build_context()
        ctx.total_request_count = 4
        ctx.paused_request_count = 1
        ctx.active_token_count = 3
        plan = ctx.prepare_next_step_optimistic(step_id=0)
        assert plan.intended_batch_dimensions.token_count == 3
        assert plan.intended_batch_dimensions.decode_req_count == 3

    def test_idempotent_when_no_prefill_status(self):
        """No prefill-status entries → operation is observably a no-op."""
        ctx = _build_context()
        ctx.request_in_prefill_status_tensor.fill_(0)
        before = ctx.request_in_prefill_status_tensor.clone()
        ctx.prepare_next_step_optimistic(step_id=0)
        assert torch.equal(ctx.request_in_prefill_status_tensor, before)
