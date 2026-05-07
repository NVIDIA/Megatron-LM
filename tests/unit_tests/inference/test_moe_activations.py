# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit tests for megatron.core.inference.moe.activations.

Covers:
- padded_squared_relu: correctness, padding-row skip, various shapes
- squared_relu_and_quantize_mxfp8: output shape/dtype, skip_padding=True/False
"""

import pytest
import torch


@pytest.mark.internal
class TestPaddedSquaredRelu:

    def test_positive_values_squared(self):
        """relu(x)^2 for positive x."""
        from megatron.core.inference.moe.activations import padded_squared_relu

        M, N = 4, 64
        x = torch.full((M, N), 2.0, device='cuda', dtype=torch.bfloat16)
        perm_map = torch.arange(M, device='cuda', dtype=torch.int32)
        out = padded_squared_relu(x, perm_map)
        expected = torch.full((M, N), 4.0, device='cuda', dtype=torch.bfloat16)
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)

    def test_negative_values_become_zero(self):
        """relu(-x)^2 = 0."""
        from megatron.core.inference.moe.activations import padded_squared_relu

        M, N = 4, 64
        x = torch.full((M, N), -3.0, device='cuda', dtype=torch.bfloat16)
        perm_map = torch.arange(M, device='cuda', dtype=torch.int32)
        out = padded_squared_relu(x, perm_map)
        assert out.sum().item() == 0.0

    def test_padding_rows_skipped(self):
        """Rows where permutation_map == -1 remain zero."""
        from megatron.core.inference.moe.activations import padded_squared_relu

        M, N = 4, 64
        x = torch.ones(M, N, device='cuda', dtype=torch.bfloat16)
        perm_map = torch.tensor([0, -1, 2, -1], device='cuda', dtype=torch.int32)
        out = padded_squared_relu(x, perm_map)
        # Real rows should have squared relu = 1.0
        assert out[0].sum().item() != 0.0
        assert out[2].sum().item() != 0.0
        # Padding rows should stay zero
        assert out[1].sum().item() == 0.0
        assert out[3].sum().item() == 0.0

    def test_mixed_positive_negative(self):
        """Mixed sign inputs: positive squared, negative zeroed."""
        from megatron.core.inference.moe.activations import padded_squared_relu

        M, N = 2, 64
        x = torch.zeros(M, N, device='cuda', dtype=torch.bfloat16)
        x[0, :32] = 2.0   # first half positive
        x[0, 32:] = -1.0  # second half negative
        x[1] = 1.5
        perm_map = torch.tensor([0, 1], device='cuda', dtype=torch.int32)
        out = padded_squared_relu(x, perm_map)
        # First row: positive part squared, negative part zero
        torch.testing.assert_close(
            out[0, :32], torch.full((32,), 4.0, device='cuda', dtype=torch.bfloat16),
            atol=1e-2, rtol=1e-2
        )
        assert out[0, 32:].sum().item() == 0.0

    @pytest.mark.parametrize("M,N", [(1, 32), (8, 64), (16, 128), (32, 256), (64, 512)])
    def test_various_shapes(self, M, N):
        """padded_squared_relu works across various (M, N) shapes."""
        from megatron.core.inference.moe.activations import padded_squared_relu

        x = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
        perm_map = torch.arange(M, device='cuda', dtype=torch.int32)
        out = padded_squared_relu(x, perm_map)
        assert out.shape == (M, N)
        assert out.dtype == torch.bfloat16
        # All outputs must be >= 0
        assert (out >= 0).all()

    def test_all_padding_rows(self):
        """When all rows are padding, output is all zeros."""
        from megatron.core.inference.moe.activations import padded_squared_relu

        M, N = 4, 64
        x = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
        perm_map = torch.full((M,), -1, device='cuda', dtype=torch.int32)
        out = padded_squared_relu(x, perm_map)
        assert out.sum().item() == 0.0

    def test_reference_correctness(self):
        """Compare against PyTorch reference."""
        from megatron.core.inference.moe.activations import padded_squared_relu

        torch.manual_seed(42)
        M, N = 16, 128
        x = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
        perm_map = torch.arange(M, device='cuda', dtype=torch.int32)
        out = padded_squared_relu(x, perm_map)
        ref = (torch.clamp(x.float(), min=0.0) ** 2).bfloat16()
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.internal
class TestSquaredReluAndQuantizeMxfp8:

    def test_output_is_mxfp8_tensor(self):
        """Returns MXFP8Tensor with fp8_e4m3fn data."""
        from megatron.core.inference.moe.activations import squared_relu_and_quantize_mxfp8
        from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

        M, K = 8, 64
        x = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
        perm_map = torch.arange(M, device='cuda', dtype=torch.int32)
        result = squared_relu_and_quantize_mxfp8(x, perm_map)
        assert isinstance(result, MXFP8Tensor)
        assert result.data.shape == (M, K)
        assert result.data.dtype == torch.float8_e4m3fn

    def test_scale_shape(self):
        """Scale tensor has the correct swizzled layout size."""
        from megatron.core.inference.moe.activations import squared_relu_and_quantize_mxfp8

        M, K = 128, 64
        x = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
        perm_map = torch.arange(M, device='cuda', dtype=torch.int32)
        result = squared_relu_and_quantize_mxfp8(x, perm_map)
        # scale_2d() should be callable without error
        scale_2d = result.scale_2d(K)
        assert scale_2d.ndim == 2

    def test_skip_padding_true(self):
        """With skip_padding=True, non-padding rows are correctly processed."""
        from megatron.core.inference.moe.activations import squared_relu_and_quantize_mxfp8

        M, K = 4, 64
        x = torch.ones(M, K, device='cuda', dtype=torch.bfloat16)
        # Rows 1 and 3 are padding
        perm_map = torch.tensor([0, -1, 2, -1], device='cuda', dtype=torch.int32)
        result = squared_relu_and_quantize_mxfp8(x, perm_map, skip_padding=True)
        assert result.data.shape == (M, K)
        # Non-padding rows: x=1.0 -> relu(1)^2 = 1.0 -> fp8 non-zero
        # (out_fp8 uses torch.empty so padding rows may have uninitialized values)
        assert result.data[0].to(torch.float32).abs().sum().item() > 0.0
        assert result.data[2].to(torch.float32).abs().sum().item() > 0.0

    def test_skip_padding_false(self):
        """With skip_padding=False, all rows are processed."""
        from megatron.core.inference.moe.activations import squared_relu_and_quantize_mxfp8
        from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

        M, K = 4, 64
        x = torch.ones(M, K, device='cuda', dtype=torch.bfloat16)
        perm_map = torch.tensor([0, -1, 2, -1], device='cuda', dtype=torch.int32)
        result = squared_relu_and_quantize_mxfp8(x, perm_map, skip_padding=False)
        assert isinstance(result, MXFP8Tensor)
        assert result.data.shape == (M, K)
        # All rows should be processed (no skipping), so all have non-zero output
        # x=1.0 -> relu(1)^2 = 1.0 -> fp8 non-zero
        for row in range(M):
            assert result.data[row].to(torch.float32).abs().sum().item() > 0.0

    def test_negative_inputs_produce_zero(self):
        """Negative activations produce zero fp8 after squared relu."""
        from megatron.core.inference.moe.activations import squared_relu_and_quantize_mxfp8

        M, K = 4, 64
        x = torch.full((M, K), -5.0, device='cuda', dtype=torch.bfloat16)
        perm_map = torch.arange(M, device='cuda', dtype=torch.int32)
        result = squared_relu_and_quantize_mxfp8(x, perm_map)
        # All negative -> relu -> 0 -> fp8 = 0
        assert result.data.to(torch.float32).sum().item() == 0.0

    @pytest.mark.parametrize("M,K", [(1, 32), (8, 64), (16, 128), (32, 256), (128, 512)])
    def test_various_shapes(self, M, K):
        """Works across various (M, K) shapes (K must be divisible by 32)."""
        from megatron.core.inference.moe.activations import squared_relu_and_quantize_mxfp8
        from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

        x = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
        perm_map = torch.arange(M, device='cuda', dtype=torch.int32)
        result = squared_relu_and_quantize_mxfp8(x, perm_map)
        assert isinstance(result, MXFP8Tensor)
        assert result.data.shape == (M, K)
