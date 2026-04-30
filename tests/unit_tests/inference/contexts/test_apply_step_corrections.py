# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for ``apply_step_corrections`` (v3 plan commit 12).

Plan validation: parity test asserts ``update_requests`` produces
bit-identical state before and after the refactor.

apply_step_corrections is a wrapper that (a) syncs the AsyncStepOutput,
(b) builds the active-request mask from sampled_tokens / termination_id /
length checks (logic moved from text_generation_controller), (c) runs the
stop-word callback, and (d) calls update_requests with the same arguments
the prior in-controller flow would have built. With overlap off, the
wrapped path is bit-identical to today.
"""

import pytest
import torch

from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines.async_pipeline_types import AsyncStepOutput
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


def _make_async_output(sampled: torch.Tensor) -> AsyncStepOutput:
    out = AsyncStepOutput(step_id=0)
    out.pinned_destinations["sampled_tokens"] = sampled
    out.payload_metadata["sampled_tokens"] = True
    # No event needed; cpu_view treats None event as already-synced.
    return out


class TestApplyStepCorrections:
    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def _seed_active_requests(self, ctx: DynamicInferenceContext, count: int):
        """Synthesize ``count`` active slots so the mask has something to
        operate on. Avoids spinning up a real model."""
        ctx.total_request_count = count
        ctx.paused_request_count = 0
        ctx.active_token_count = count
        ctx.request_ids[:count] = torch.arange(count, dtype=ctx.request_ids.dtype)
        ctx.request_query_lengths[:count] = 1
        ctx.request_kv_length_offsets[:count] = 1
        ctx.request_output_lengths[:count] = 64

    def test_requires_async_step_output(self):
        ctx = _build_context()
        with pytest.raises(ValueError):
            ctx.apply_step_corrections(
                step_id=0,
                async_step_output=None,
                request_metadata_termination_id=ctx.request_ids,
            )

    def test_returns_expected_dict_shape(self):
        ctx = _build_context()
        self._seed_active_requests(ctx, count=2)
        termination_id = torch.full((ctx.max_requests,), 999, dtype=torch.int64)
        sampled = torch.tensor([42, 999], dtype=torch.int64)
        out = _make_async_output(sampled)

        result = ctx.apply_step_corrections(
            step_id=0,
            async_step_output=out,
            request_metadata_termination_id=termination_id,
        )
        assert "active_request_ids" in result
        assert "finished_request_ids" in result
        assert "sample" in result
        # Slot 1 sampled the termination_id → finished.
        finished = result["finished_request_ids"]
        assert finished.numel() == 1
        assert finished[0].item() == 1

    def test_stop_word_callback_invoked(self):
        ctx = _build_context()
        self._seed_active_requests(ctx, count=2)
        termination_id = torch.full((ctx.max_requests,), 999, dtype=torch.int64)
        sampled = torch.tensor([42, 43], dtype=torch.int64)
        out = _make_async_output(sampled)

        captured = []

        def stop_cb(active_request_ids):
            captured.append(list(active_request_ids))
            return {0}  # request 0 stops via stop-word

        result = ctx.apply_step_corrections(
            step_id=0,
            async_step_output=out,
            request_metadata_termination_id=termination_id,
            stop_word_callback=stop_cb,
        )
        assert captured == [[0, 1]]
        finished = result["finished_request_ids"].tolist()
        assert 0 in finished
