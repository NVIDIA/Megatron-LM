# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for the Mamba pending-ops stream/event wiring (v3 plan
commit 24).

Plan validation: Mamba/hybrid decode stress test; correctness against
serial; no Mamba slot leak under EOS, cancellation, pause/resume,
rollback. The stress portion is exercised by the engine's hybrid
integration tests; here we verify the stream/event contract.
"""

import pytest
import torch

from megatron.core.inference.config import InferenceConfig, MambaInferenceStateConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
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


class TestMambaOpsStream:
    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def test_no_pending_ops_no_event_recorded(self):
        ctx = _build_hybrid_context()
        ctx._execute_pending_mamba_ops()
        assert ctx.mamba_ops_done_event() is None

    def test_pending_zero_ops_record_event_on_default_stream(self):
        ctx = _build_hybrid_context()
        ctx._pending_mamba_zeros = [0, 1, 2]
        ctx._execute_pending_mamba_ops()
        ev = ctx.mamba_ops_done_event()
        assert ev is not None
        torch.cuda.synchronize()
        assert ev.query() is True
        # The pending lists are drained.
        assert ctx._pending_mamba_zeros == []

    def test_pending_zero_ops_run_on_supplied_stream(self):
        ctx = _build_hybrid_context()
        ctx._pending_mamba_zeros = [3, 4]
        bookkeeping_stream = torch.cuda.Stream(device=ctx.mamba_conv_states.device)
        ctx._execute_pending_mamba_ops(gpu_bookkeeping_stream=bookkeeping_stream)
        ev = ctx.mamba_ops_done_event()
        assert ev is not None
        torch.cuda.synchronize()
        assert ev.query() is True
        assert ctx._pending_mamba_zeros == []
