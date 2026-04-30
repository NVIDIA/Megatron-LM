# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for the GPU-resident sampled-token state introduced in
commit 16.

Plan validation: debug check asserts ``_prev_sampled_token_ids[:n].cpu()``
equals the AsyncStepOutput-resolved sampled-tokens CPU buffer for every
step.

The full controller path is exercised by the engine's existing
end-to-end tests; here we verify the lower-level invariants:
- The context allocates ``_prev_sampled_token_ids`` with shape
  ``[max_requests]``, int64, on GPU.
- A simulated D2D from ``_sampled_tokens_cuda`` to
  ``_prev_sampled_token_ids`` produces matching CPU readbacks.
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
        ),
    )


class TestPrevSampledTokenIds:
    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def test_allocated_with_correct_shape_and_device(self):
        ctx = _build_context()
        t = ctx._prev_sampled_token_ids
        assert t.shape == (ctx.max_requests,)
        assert t.dtype == torch.int64
        assert t.is_cuda

    def test_d2d_matches_source_when_synchronized(self):
        """Round-trip: copy a known source into _prev_sampled_token_ids on
        a side stream and verify the CPU readback matches the original."""
        ctx = _build_context()
        sampled_tokens_cuda = torch.arange(
            ctx.max_requests, dtype=torch.int64, device=ctx._prev_sampled_token_ids.device
        ) + 100
        n = 5
        bookkeeping_stream = torch.cuda.Stream(device=sampled_tokens_cuda.device)
        sample_done = torch.cuda.Event()
        sample_done.record()
        bookkeeping_stream.wait_event(sample_done)
        with torch.cuda.stream(bookkeeping_stream):
            ctx._prev_sampled_token_ids[:n].copy_(
                sampled_tokens_cuda[:n], non_blocking=True
            )
            ready = torch.cuda.Event()
            ready.record(bookkeeping_stream)
        ready.synchronize()
        assert ctx._prev_sampled_token_ids[:n].cpu().tolist() == [100, 101, 102, 103, 104]
