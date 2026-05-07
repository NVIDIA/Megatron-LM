# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit tests for megatron.core.inference.moe.fused_moe and __init__.

Covers:
- fused_moe.py: _get_activation_func, mcore_fused_moe (bf16 path, skip_permute branches)
- __init__.py: resolve_inference_grouped_gemm_backend, InferenceGroupedGemmBackend
"""

import pytest
import torch

requires_grouped_mm = pytest.mark.skipif(
    not hasattr(torch.nn.functional, 'grouped_mm'),
    reason="Requires torch.nn.functional.grouped_mm (PyTorch >= 2.10)",
)


# ---------------------------------------------------------------------------
# __init__.py: resolve_inference_grouped_gemm_backend
# ---------------------------------------------------------------------------

class TestResolveInferenceGroupedGemmBackend:

    def test_torch_backend_returns_torch(self):
        from megatron.core.inference.moe import (
            InferenceGroupedGemmBackend,
            resolve_inference_grouped_gemm_backend,
        )
        result = resolve_inference_grouped_gemm_backend('torch', is_cuda_graphed=False)
        assert result == InferenceGroupedGemmBackend.TORCH

    def test_torch_backend_cuda_graphed_returns_torch(self):
        from megatron.core.inference.moe import (
            InferenceGroupedGemmBackend,
            resolve_inference_grouped_gemm_backend,
        )
        result = resolve_inference_grouped_gemm_backend('torch', is_cuda_graphed=True)
        assert result == InferenceGroupedGemmBackend.TORCH

    def test_te_backend_returns_te(self):
        from megatron.core.inference.moe import (
            InferenceGroupedGemmBackend,
            resolve_inference_grouped_gemm_backend,
        )
        result = resolve_inference_grouped_gemm_backend('te', is_cuda_graphed=False)
        assert result == InferenceGroupedGemmBackend.TE

    def test_te_backend_cuda_graphed_returns_te(self):
        from megatron.core.inference.moe import (
            InferenceGroupedGemmBackend,
            resolve_inference_grouped_gemm_backend,
        )
        result = resolve_inference_grouped_gemm_backend('te', is_cuda_graphed=True)
        assert result == InferenceGroupedGemmBackend.TE

    def test_auto_not_cuda_graphed_with_grouped_mm(self):
        """auto + not cuda_graphed returns TORCH when grouped_mm available."""
        from megatron.core.inference.moe import (
            InferenceGroupedGemmBackend,
            resolve_inference_grouped_gemm_backend,
        )
        if hasattr(torch.nn.functional, 'grouped_mm'):
            result = resolve_inference_grouped_gemm_backend('auto', is_cuda_graphed=False)
            assert result == InferenceGroupedGemmBackend.TORCH
        else:
            result = resolve_inference_grouped_gemm_backend('auto', is_cuda_graphed=False)
            assert result == InferenceGroupedGemmBackend.TE

    def test_auto_cuda_graphed_not_mxfp8_returns_flashinfer(self):
        """auto + cuda_graphed + not mxfp8 returns FLASHINFER."""
        from megatron.core.inference.moe import (
            InferenceGroupedGemmBackend,
            resolve_inference_grouped_gemm_backend,
        )
        result = resolve_inference_grouped_gemm_backend(
            'auto', is_cuda_graphed=True, is_mxfp8=False
        )
        assert result == InferenceGroupedGemmBackend.FLASHINFER

    def test_auto_cuda_graphed_mxfp8_returns_torch(self):
        """auto + cuda_graphed + mxfp8=True returns TORCH when scaled_grouped_mm available."""
        from megatron.core.inference.moe import (
            InferenceGroupedGemmBackend,
            resolve_inference_grouped_gemm_backend,
        )
        if hasattr(torch.nn.functional, 'scaled_grouped_mm'):
            result = resolve_inference_grouped_gemm_backend(
                'auto', is_cuda_graphed=True, is_mxfp8=True
            )
            assert result == InferenceGroupedGemmBackend.TORCH
        else:
            with pytest.raises(AssertionError, match="scaled_grouped_mm"):
                resolve_inference_grouped_gemm_backend(
                    'auto', is_cuda_graphed=True, is_mxfp8=True
                )

    def test_invalid_backend_raises(self):
        from megatron.core.inference.moe import resolve_inference_grouped_gemm_backend
        with pytest.raises(ValueError, match="Unknown inference_grouped_gemm_backend"):
            resolve_inference_grouped_gemm_backend('invalid', is_cuda_graphed=False)

    def test_enum_values(self):
        from megatron.core.inference.moe import InferenceGroupedGemmBackend
        assert InferenceGroupedGemmBackend.TORCH.value == "torch"
        assert InferenceGroupedGemmBackend.TE.value == "te"
        assert InferenceGroupedGemmBackend.FLASHINFER.value == "flashinfer"


# ---------------------------------------------------------------------------
# fused_moe.py: _get_activation_func
# ---------------------------------------------------------------------------

class TestGetActivationFunc:

    def test_squared_relu_no_quant(self):
        from megatron.core.inference.moe.activations import padded_squared_relu
        from megatron.core.inference.moe.fused_moe import ActivationType, _get_activation_func

        fn = _get_activation_func(ActivationType.SQUARED_RELU, fused_quant=False)
        assert fn is padded_squared_relu

    def test_squared_relu_with_quant(self):
        from megatron.core.inference.moe.activations import squared_relu_and_quantize_mxfp8
        from megatron.core.inference.moe.fused_moe import ActivationType, _get_activation_func

        fn = _get_activation_func(ActivationType.SQUARED_RELU, fused_quant=True)
        assert fn is squared_relu_and_quantize_mxfp8

    def test_invalid_activation_raises(self):
        from megatron.core.inference.moe.fused_moe import _get_activation_func

        class FakeType:
            pass

        with pytest.raises(ValueError, match="Unsupported activation type"):
            _get_activation_func(FakeType())


# ---------------------------------------------------------------------------
# fused_moe.py: mcore_fused_moe (BF16 path)
# ---------------------------------------------------------------------------

def _make_bf16_weights(num_experts, hidden_size, ffn_hidden_size):
    """Create stacked BF16 weight tensors for grouped_mm."""
    fc1 = torch.randn(num_experts, ffn_hidden_size, hidden_size, device='cuda', dtype=torch.bfloat16)
    fc2 = torch.randn(num_experts, hidden_size, ffn_hidden_size, device='cuda', dtype=torch.bfloat16)
    return fc1, fc2


@pytest.mark.internal
class TestMcoreFusedMoeBf16:

    @requires_grouped_mm
    def test_skip_permute_false_basic(self):
        """BF16 path with skip_permute=False: output has correct shape and dtype."""
        from megatron.core.inference.moe.fused_moe import ActivationType, mcore_fused_moe

        num_tokens, hidden_size, ffn_hidden_size = 8, 64, 32
        num_experts, topk = 4, 2
        fc1, fc2 = _make_bf16_weights(num_experts, hidden_size, ffn_hidden_size)

        hidden = torch.randn(num_tokens, hidden_size, device='cuda', dtype=torch.bfloat16)
        probs = torch.rand(num_tokens, topk, device='cuda', dtype=torch.float32)
        routing_map = torch.randint(0, num_experts, (num_tokens, topk), device='cuda')

        output = mcore_fused_moe(
            hidden_states=hidden,
            probs=probs,
            fc1_weight=fc1,
            fc2_weight=fc2,
            activation_type=ActivationType.SQUARED_RELU,
            num_local_experts=num_experts,
            local_expert_start=0,
            routing_map=routing_map,
        )
        assert output.shape == (num_tokens, hidden_size)
        assert output.dtype == torch.float32

    @requires_grouped_mm
    def test_skip_permute_true_basic(self):
        """BF16 path with skip_permute=True: tokens already permuted."""
        from megatron.core.inference.moe.fused_moe import ActivationType, mcore_fused_moe

        num_experts = 4
        hidden_size, ffn_hidden_size = 64, 32
        tokens_per_expert = torch.tensor([3, 2, 4, 1], device='cuda', dtype=torch.int32)
        total_tokens = tokens_per_expert.sum().item()
        fc1, fc2 = _make_bf16_weights(num_experts, hidden_size, ffn_hidden_size)

        # Tokens already in expert-grouped order
        hidden = torch.randn(total_tokens, hidden_size, device='cuda', dtype=torch.bfloat16)
        probs = torch.rand(total_tokens, device='cuda', dtype=torch.bfloat16)

        output = mcore_fused_moe(
            hidden_states=hidden,
            probs=probs,
            fc1_weight=fc1,
            fc2_weight=fc2,
            activation_type=ActivationType.SQUARED_RELU,
            num_local_experts=num_experts,
            local_expert_start=0,
            tokens_per_expert=tokens_per_expert,
            skip_permute=True,
        )
        assert output.shape == (total_tokens, hidden_size)

    @requires_grouped_mm
    def test_non_bf16_input_raises(self):
        """Non-BF16 input raises AssertionError."""
        from megatron.core.inference.moe.fused_moe import ActivationType, mcore_fused_moe

        num_tokens, hidden_size, ffn_hidden_size = 4, 64, 32
        num_experts, topk = 2, 1
        fc1, fc2 = _make_bf16_weights(num_experts, hidden_size, ffn_hidden_size)

        hidden = torch.randn(num_tokens, hidden_size, device='cuda', dtype=torch.float32)
        probs = torch.rand(num_tokens, topk, device='cuda', dtype=torch.float32)
        routing_map = torch.randint(0, num_experts, (num_tokens, topk), device='cuda')

        with pytest.raises(AssertionError, match="bf16"):
            mcore_fused_moe(
                hidden_states=hidden,
                probs=probs,
                fc1_weight=fc1,
                fc2_weight=fc2,
                activation_type=ActivationType.SQUARED_RELU,
                num_local_experts=num_experts,
                local_expert_start=0,
                routing_map=routing_map,
            )

    @requires_grouped_mm
    def test_skip_permute_false_requires_routing_map(self):
        """skip_permute=False without routing_map raises AssertionError."""
        from megatron.core.inference.moe.fused_moe import ActivationType, mcore_fused_moe

        num_tokens, hidden_size, ffn_hidden_size = 4, 64, 32
        num_experts = 2
        fc1, fc2 = _make_bf16_weights(num_experts, hidden_size, ffn_hidden_size)
        hidden = torch.randn(num_tokens, hidden_size, device='cuda', dtype=torch.bfloat16)
        probs = torch.rand(num_tokens, 1, device='cuda', dtype=torch.float32)

        with pytest.raises(AssertionError, match="routing_map"):
            mcore_fused_moe(
                hidden_states=hidden,
                probs=probs,
                fc1_weight=fc1,
                fc2_weight=fc2,
                activation_type=ActivationType.SQUARED_RELU,
                num_local_experts=num_experts,
                local_expert_start=0,
                routing_map=None,
                skip_permute=False,
            )

    @requires_grouped_mm
    def test_skip_permute_true_requires_tokens_per_expert(self):
        """skip_permute=True without tokens_per_expert raises AssertionError."""
        from megatron.core.inference.moe.fused_moe import ActivationType, mcore_fused_moe

        num_tokens, hidden_size, ffn_hidden_size = 4, 64, 32
        num_experts = 2
        fc1, fc2 = _make_bf16_weights(num_experts, hidden_size, ffn_hidden_size)
        hidden = torch.randn(num_tokens, hidden_size, device='cuda', dtype=torch.bfloat16)
        probs = torch.rand(num_tokens, device='cuda', dtype=torch.bfloat16)

        with pytest.raises(AssertionError, match="tokens_per_expert"):
            mcore_fused_moe(
                hidden_states=hidden,
                probs=probs,
                fc1_weight=fc1,
                fc2_weight=fc2,
                activation_type=ActivationType.SQUARED_RELU,
                num_local_experts=num_experts,
                local_expert_start=0,
                tokens_per_expert=None,
                skip_permute=True,
            )

    @requires_grouped_mm
    def test_skip_permute_true_routing_map_must_be_none(self):
        """skip_permute=True with routing_map raises AssertionError."""
        from megatron.core.inference.moe.fused_moe import ActivationType, mcore_fused_moe

        num_experts = 2
        hidden_size, ffn_hidden_size = 64, 32
        fc1, fc2 = _make_bf16_weights(num_experts, hidden_size, ffn_hidden_size)
        tpe = torch.tensor([2, 2], device='cuda', dtype=torch.int32)
        hidden = torch.randn(4, hidden_size, device='cuda', dtype=torch.bfloat16)
        probs = torch.rand(4, device='cuda', dtype=torch.bfloat16)
        routing_map = torch.zeros(4, 1, device='cuda', dtype=torch.int64)

        with pytest.raises(AssertionError, match="routing_map must be None"):
            mcore_fused_moe(
                hidden_states=hidden,
                probs=probs,
                fc1_weight=fc1,
                fc2_weight=fc2,
                activation_type=ActivationType.SQUARED_RELU,
                num_local_experts=num_experts,
                local_expert_start=0,
                routing_map=routing_map,
                tokens_per_expert=tpe,
                skip_permute=True,
            )

    @requires_grouped_mm
    def test_expert_subset(self):
        """Only local experts processed (local_expert_start != 0)."""
        from megatron.core.inference.moe.fused_moe import ActivationType, mcore_fused_moe

        num_tokens, hidden_size, ffn_hidden_size = 16, 64, 32
        total_experts, num_local = 8, 4
        local_start = 4
        topk = 2
        fc1, fc2 = _make_bf16_weights(num_local, hidden_size, ffn_hidden_size)

        hidden = torch.randn(num_tokens, hidden_size, device='cuda', dtype=torch.bfloat16)
        probs = torch.rand(num_tokens, topk, device='cuda', dtype=torch.float32)
        routing_map = torch.randint(0, total_experts, (num_tokens, topk), device='cuda')

        output = mcore_fused_moe(
            hidden_states=hidden,
            probs=probs,
            fc1_weight=fc1,
            fc2_weight=fc2,
            activation_type=ActivationType.SQUARED_RELU,
            num_local_experts=num_local,
            local_expert_start=local_start,
            routing_map=routing_map,
        )
        assert output.shape == (num_tokens, hidden_size)

    @requires_grouped_mm
    @pytest.mark.parametrize(
        "num_tokens,hidden_size,ffn_hidden_size,num_experts,topk",
        [
            (1, 64, 32, 2, 1),
            (4, 64, 32, 4, 2),
            (16, 128, 64, 8, 4),
            (32, 64, 32, 4, 2),
        ],
    )
    def test_various_sizes(self, num_tokens, hidden_size, ffn_hidden_size, num_experts, topk):
        """mcore_fused_moe works across various problem sizes."""
        from megatron.core.inference.moe.fused_moe import ActivationType, mcore_fused_moe

        fc1, fc2 = _make_bf16_weights(num_experts, hidden_size, ffn_hidden_size)
        hidden = torch.randn(num_tokens, hidden_size, device='cuda', dtype=torch.bfloat16)
        probs = torch.rand(num_tokens, topk, device='cuda', dtype=torch.float32)
        routing_map = torch.randint(0, num_experts, (num_tokens, topk), device='cuda')

        output = mcore_fused_moe(
            hidden_states=hidden,
            probs=probs,
            fc1_weight=fc1,
            fc2_weight=fc2,
            activation_type=ActivationType.SQUARED_RELU,
            num_local_experts=num_experts,
            local_expert_start=0,
            routing_map=routing_map,
        )
        assert output.shape == (num_tokens, hidden_size)
