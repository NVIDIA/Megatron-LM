# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for MXFP8 quantization.

Tests cover:
- mxfp8_quantize (Triton kernel): data and swizzled scales vs PyTorch reference
- MXFP8Tensor.from_bf16: both 'triton' and 'flashinfer' backends
- MXFP8Tensor.scale_2d: reshape correctness
"""

import pytest
import torch

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.internal,
]


def ceil_div(a, b):
    return (a + b - 1) // b


# ──────────────────────────────────────────────────────────────────────
# Reference functions from PyTorch
# https://github.com/pytorch/pytorch/blob/main/torch/testing/_internal/common_quantized.py#L578
# ──────────────────────────────────────────────────────────────────────

def ref_to_mxfp(
    data_hp: torch.Tensor,
    block_size: int = 32,
    format: str = "mxfp8",
):
    if data_hp.dtype not in (torch.bfloat16, torch.float):
        raise AssertionError(f"{data_hp.dtype} is not supported yet")
    if data_hp.shape[-1] % block_size != 0:
        raise AssertionError(
            f"the last dimension of shape {data_hp.shape} must be divisible by block_size {block_size}"
        )
    if not data_hp.is_contiguous():
        raise AssertionError("unsupported: data_hp must be contiguous")

    orig_shape = data_hp.shape
    data_hp = data_hp.reshape(
        *orig_shape[:-1], orig_shape[-1] // block_size, block_size
    )

    max_abs = torch.amax(torch.abs(data_hp), -1).unsqueeze(-1)

    data_hp = data_hp.to(torch.float32)
    max_abs = max_abs.to(torch.float32)

    if format == "mxfp8":
        F8E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
        max_pos = F8E4M3_MAX
    elif format == "mxfp4":
        F4E2M1_MAX = 6.
        max_pos = F4E2M1_MAX

    # RCEIL
    def _to_mx_rceil(
        data_hp: torch.Tensor,
        max_abs: torch.Tensor,
        max_pos: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        E8M0_EXPONENT_BIAS = 127
        descale = max_abs / max_pos
        exponent = torch.where(
            torch.isnan(descale),
            0xFF,  # Handle biased exponent for nan
            (
                torch.clamp(
                    torch.ceil(torch.log2(descale)),
                    min=-E8M0_EXPONENT_BIAS,
                    max=E8M0_EXPONENT_BIAS,
                )
                + E8M0_EXPONENT_BIAS
            ).to(torch.uint8),
        )

        descale_fp = torch.where(
            exponent == 0,
            1.0,
            torch.exp2(E8M0_EXPONENT_BIAS - exponent.to(torch.float32)),
        )

        # scale and saturated cast the data elements to max of target dtype
        data_lp = torch.clamp(data_hp * descale_fp, min=-1 * max_pos, max=max_pos)
        return exponent, data_lp

    scale_e8m0_biased, data_lp = _to_mx_rceil(data_hp, max_abs, max_pos)

    # cast to target dtype
    data_lp = data_lp.to(torch.float8_e4m3fn)
    data_lp = data_lp.reshape(orig_shape)

    scale_e8m0_biased = scale_e8m0_biased.view(torch.float8_e8m0fnu)
    scale_e8m0_biased = scale_e8m0_biased.squeeze(-1)
    return scale_e8m0_biased, data_lp


def ref_swizzle(input_matrix) -> torch.Tensor:
    """Rearrange a scale matrix into cuBLAS 2D blocked (swizzled) layout.

    See: https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout

    Args:
        input_matrix: Input tensor of shape (H, W)

    Returns:
        Flattened swizzled tensor.
    """
    rows, cols = input_matrix.shape
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)

    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    padded = input_matrix
    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros((padded_rows, padded_cols), device=input_matrix.device, dtype=input_matrix.dtype)
        padded[:rows, :cols] = input_matrix

    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)

    return rearranged.flatten()


# ──────────────────────────────────────────────────────────────────────
# mxfp8_quantize (Triton kernel)
# ──────────────────────────────────────────────────────────────────────

class TestMxfp8Quantize:

    @pytest.mark.parametrize("M,K", [
        (1, 32),
        (1, 64),
        (1, 128),
        (4, 32),
        (4, 128),
        (16, 64),
        (16, 256),
        (32, 128),
        (64, 256),
        (128, 128),
        (128, 512),
        (128, 2688),     # nanov3 hidden_size
        (256, 1856),     # nanov3 moe_ffn_hidden_size
        (512, 2688),
    ])
    def test_data_matches_reference(self, M, K):
        """Quantized FP8 data matches PyTorch reference."""
        from megatron.core.inference.quantization.mxfp8_quantize import mxfp8_quantize

        torch.manual_seed(42)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

        triton_data, _ = mxfp8_quantize(x)
        _, ref_data = ref_to_mxfp(x)

        assert triton_data.shape == (M, K)
        assert triton_data.dtype == torch.float8_e4m3fn
        torch.testing.assert_close(
            triton_data.view(torch.uint8), ref_data.view(torch.uint8), atol=0, rtol=0,
        )

    @pytest.mark.parametrize("M,K", [
        (1, 32),
        (1, 64),
        (4, 128),
        (16, 256),
        (32, 128),
        (128, 128),
        (128, 512),
        (128, 2688),
        (256, 1856),
        (512, 2688),
    ])
    def test_scales_match_reference(self, M, K):
        """Swizzled scales match ref_to_mxfp scales passed through ref_swizzle."""
        from megatron.core.inference.quantization.mxfp8_quantize import mxfp8_quantize

        torch.manual_seed(42)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

        _, triton_scales = mxfp8_quantize(x)
        ref_scales_2d, _ = ref_to_mxfp(x)  # [M, K//32] e8m0

        # Swizzle the reference scales
        ref_swizzled = ref_swizzle(ref_scales_2d)

        # Compare as uint8 since e8m0 is just exponent bytes
        torch.testing.assert_close(
            triton_scales.view(torch.uint8),
            ref_swizzled.view(torch.uint8),
            atol=0, rtol=0,
        )

    @pytest.mark.parametrize("M,K", [
        (1, 32),
        (16, 128),
        (128, 2688),
    ])
    def test_all_zeros_input(self, M, K):
        """All-zero input produces all-zero FP8 data and zero scales."""
        from megatron.core.inference.quantization.mxfp8_quantize import mxfp8_quantize

        x = torch.zeros(M, K, device="cuda", dtype=torch.bfloat16)
        data, scales = mxfp8_quantize(x)
        assert (data.float() == 0).all()
        assert (scales.view(torch.uint8) == 0).all()

    @pytest.mark.parametrize("M,K", [
        (1, 32),
        (16, 128),
        (128, 256),
    ])
    def test_constant_input(self, M, K):
        """Constant input: all elements in a group have the same value."""
        from megatron.core.inference.quantization.mxfp8_quantize import mxfp8_quantize

        x = torch.full((M, K), 1.0, device="cuda", dtype=torch.bfloat16)
        data, _ = mxfp8_quantize(x)
        _, ref_data = ref_to_mxfp(x)
        torch.testing.assert_close(data.view(torch.uint8), ref_data.view(torch.uint8), atol=0, rtol=0)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_input_dtypes(self, dtype):
        """Kernel accepts bf16, fp16, and fp32 inputs."""
        from megatron.core.inference.quantization.mxfp8_quantize import mxfp8_quantize

        x = torch.randn(16, 128, device="cuda", dtype=dtype)
        data, _ = mxfp8_quantize(x)
        assert data.dtype == torch.float8_e4m3fn
        assert data.shape == (16, 128)

    @pytest.mark.parametrize("M", [1, 127, 128, 129, 255, 256, 257, 512])
    def test_various_row_counts(self, M):
        """Test row counts that are not multiples of 128 (macro tile boundary)."""
        from megatron.core.inference.quantization.mxfp8_quantize import mxfp8_quantize

        K = 128
        torch.manual_seed(42)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        data, _ = mxfp8_quantize(x)
        _, ref_data = ref_to_mxfp(x)
        torch.testing.assert_close(data.view(torch.uint8), ref_data.view(torch.uint8), atol=0, rtol=0)

    @pytest.mark.parametrize("seed", [0, 7, 42, 123, 999])
    def test_reproducible(self, seed):
        """Same input always produces same output."""
        from megatron.core.inference.quantization.mxfp8_quantize import mxfp8_quantize

        torch.manual_seed(seed)
        x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
        d1, s1 = mxfp8_quantize(x)
        d2, s2 = mxfp8_quantize(x)
        torch.testing.assert_close(d1.view(torch.uint8), d2.view(torch.uint8), atol=0, rtol=0)
        torch.testing.assert_close(s1.view(torch.uint8), s2.view(torch.uint8), atol=0, rtol=0)


# ──────────────────────────────────────────────────────────────────────
# MXFP8Tensor
# ──────────────────────────────────────────────────────────────────────

class TestMXFP8Tensor:

    @pytest.mark.parametrize("M,K", [
        (16, 128),
        (64, 256),
        (128, 2688),
    ])
    def test_from_bf16_triton(self, M, K):
        """from_bf16 with triton backend produces correct data and scales."""
        from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

        torch.manual_seed(42)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

        tensor = MXFP8Tensor.from_bf16(x, backend="triton")
        _, ref_data = ref_to_mxfp(x)

        assert tensor.data.shape == (M, K)
        assert tensor.data.dtype == torch.float8_e4m3fn
        assert tensor.backend == "triton"
        torch.testing.assert_close(tensor.data.view(torch.uint8), ref_data.view(torch.uint8), atol=0, rtol=0)

    @pytest.mark.parametrize("M,K", [
        (16, 128),
        (64, 256),
        (128, 2688),
    ])
    def test_from_bf16_flashinfer(self, M, K):
        """from_bf16 with flashinfer backend produces valid output."""
        from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor, HAVE_FLASHINFER

        if not HAVE_FLASHINFER:
            pytest.skip("FlashInfer not available")

        torch.manual_seed(42)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

        tensor = MXFP8Tensor.from_bf16(x, backend="flashinfer")
        assert tensor.data.shape == (M, K)
        assert tensor.data.dtype == torch.float8_e4m3fn
        assert tensor.backend == "flashinfer"

    def test_from_bf16_invalid_backend(self):
        """from_bf16 with invalid backend raises ValueError."""
        from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

        x = torch.randn(16, 128, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="Unknown MXFP8 quantization backend"):
            MXFP8Tensor.from_bf16(x, backend="invalid")

    @pytest.mark.parametrize("M,K", [
        (1, 32),
        (16, 128),
        (128, 2688),
        (256, 1856),
    ])
    def test_scale_2d_shape(self, M, K):
        """scale_2d returns correct shape: (-1, ceil(K//32, 4)*4)."""
        from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

        torch.manual_seed(42)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        tensor = MXFP8Tensor.from_bf16(x, backend="triton")

        scale_2d = tensor.scale_2d()
        expected_cols = ceil_div(K // 32, 4) * 4
        assert scale_2d.dim() == 2
        assert scale_2d.shape[1] == expected_cols

    @pytest.mark.parametrize("M,K", [
        (16, 128),
        (128, 2688),
    ])
    def test_scale_2d_idempotent(self, M, K):
        """Calling scale_2d twice returns the same result."""
        from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

        torch.manual_seed(42)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        tensor = MXFP8Tensor.from_bf16(x, backend="triton")

        s1 = tensor.scale_2d()
        s2 = tensor.scale_2d()
        torch.testing.assert_close(
            s1.view(torch.uint8), s2.view(torch.uint8), atol=0, rtol=0,
        )

    def test_size_method(self):
        """size() delegates to data.size()."""
        from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

        x = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
        tensor = MXFP8Tensor.from_bf16(x, backend="triton")
        assert tensor.size() == torch.Size([32, 128])
        assert tensor.size(0) == 32
        assert tensor.size(1) == 128


# ──────────────────────────────────────────────────────────────────────
# Triton vs FlashInfer cross-validation
# ──────────────────────────────────────────────────────────────────────

class TestTritonVsFlashinfer:

    @pytest.mark.parametrize("M,K", [
        (1, 32),
        (16, 128),
        (64, 256),
        (128, 2688),
        (256, 1856),
    ])
    def test_data_matches(self, M, K):
        """Triton and FlashInfer backends produce identical FP8 data."""
        from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor, HAVE_FLASHINFER

        if not HAVE_FLASHINFER:
            pytest.skip("FlashInfer not available")

        torch.manual_seed(42)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

        triton_tensor = MXFP8Tensor.from_bf16(x, backend="triton")
        flashinfer_tensor = MXFP8Tensor.from_bf16(x, backend="flashinfer")

        torch.testing.assert_close(
            triton_tensor.data.float(), flashinfer_tensor.data.float(),
            atol=0, rtol=0,
        )

    @pytest.mark.parametrize("M,K", [
        (1, 32),
        (16, 128),
        (64, 256),
        (128, 2688),
        (256, 1856),
    ])
    def test_scales_match(self, M, K):
        """Triton and FlashInfer backends produce identical swizzled scales."""
        from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor, HAVE_FLASHINFER

        if not HAVE_FLASHINFER:
            pytest.skip("FlashInfer not available")

        torch.manual_seed(42)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

        triton_tensor = MXFP8Tensor.from_bf16(x, backend="triton")
        flashinfer_tensor = MXFP8Tensor.from_bf16(x, backend="flashinfer")

        torch.testing.assert_close(
            triton_tensor.scale.view(torch.uint8),
            flashinfer_tensor.scale.view(torch.uint8),
            atol=0, rtol=0,
        )

def _make_permutation_map(M, num_padding=0):
    """Create a permutation_map with optional padding rows at the end."""
    real = torch.arange(M - num_padding, dtype=torch.int32, device="cuda")
    pad = torch.full((num_padding,), -1, dtype=torch.int32, device="cuda")
    return torch.cat([real, pad])


# ──────────────────────────────────────────────────────────────────────
# squared_relu_and_quantize_mxfp8 vs PyTorch reference
# ──────────────────────────────────────────────────────────────────────

class TestSquaredReluAndQuantizeMxfp8:
    """Compare fused squared_relu + mxfp8 quantize against PyTorch reference.

    Reference: torch.relu(x.float()).pow(2).to(bf16) -> ref_to_mxfp -> ref_swizzle.
    The fused kernel computes squared ReLU in fp32 and quantizes to MXFP8 in one pass,
    so the PyTorch fp32 reference is the correct baseline (not the unfused Triton path
    which has an intermediate bf16 roundtrip).
    """

    @pytest.mark.parametrize("M,K", [
        (1, 32),
        (4, 64),
        (16, 128),
        (32, 256),
        (64, 128),
        (128, 128),
        (128, 256),
        (128, 2688),
        (256, 1856),
        (512, 2688),
    ])
    def test_data_matches_pytorch_ref(self, M, K):
        """Fused FP8 data matches PyTorch squared ReLU + ref_to_mxfp."""
        from megatron.core.inference.moe.activations import squared_relu_and_quantize_mxfp8

        torch.manual_seed(42)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        perm_map = _make_permutation_map(M, num_padding=0)

        # PyTorch reference: squared ReLU in fp32, then downcast to bf16, then quantize
        activated_ref = torch.relu(x.float()).pow(2)
        _, ref_data = ref_to_mxfp(activated_ref)

        # Fused kernel
        fused_result = squared_relu_and_quantize_mxfp8(x, perm_map)

        torch.testing.assert_close(
            fused_result.data.view(torch.uint8), ref_data.view(torch.uint8), atol=0, rtol=0,
        )

    @pytest.mark.parametrize("M,K", [
        (1, 32),
        (16, 128),
        (128, 128),
        (128, 2688),
        (256, 1856),
    ])
    def test_scales_match_pytorch_ref(self, M, K):
        """Fused swizzled scales match PyTorch ref_to_mxfp + ref_swizzle."""
        from megatron.core.inference.moe.activations import squared_relu_and_quantize_mxfp8

        torch.manual_seed(42)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        perm_map = _make_permutation_map(M, num_padding=0)

        # PyTorch reference
        activated_ref = torch.relu(x.float()).pow(2)
        ref_scales_2d, _ = ref_to_mxfp(activated_ref)
        ref_swizzled = ref_swizzle(ref_scales_2d)

        # Fused kernel
        fused_result = squared_relu_and_quantize_mxfp8(x, perm_map)

        torch.testing.assert_close(
            fused_result.scale.view(torch.uint8),
            ref_swizzled.view(torch.uint8),
            atol=0, rtol=0,
        )

    @pytest.mark.parametrize("M,K,num_padding", [
        (32, 128, 8),
        (64, 256, 16),
        (128, 128, 32),
        (128, 2688, 64),
        (256, 1856, 128),
    ])
    def test_real_rows_match_pytorch_ref_with_padding(self, M, K, num_padding):
        """Real rows match PyTorch reference even when padding rows are present."""
        from megatron.core.inference.moe.activations import squared_relu_and_quantize_mxfp8

        torch.manual_seed(42)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        perm_map = _make_permutation_map(M, num_padding=num_padding)

        # PyTorch reference (only real rows)
        real_rows = M - num_padding
        activated_ref = torch.relu(x[:real_rows].float()).pow(2)
        _, ref_data = ref_to_mxfp(activated_ref)

        # Fused kernel
        fused_result = squared_relu_and_quantize_mxfp8(x, perm_map)

        torch.testing.assert_close(
            fused_result.data[:real_rows].view(torch.uint8), ref_data.view(torch.uint8), atol=0, rtol=0,
        )


# ──────────────────────────────────────────────────────────────────────
# permute_and_quantize_mxfp8
# ──────────────────────────────────────────────────────────────────────

class TestPermuteAndQuantizeMxfp8:
    """Compare fused permute + mxfp8 quantize against PyTorch reference.

    PyTorch reference:
    1. For each real row, quantize the source token with ref_to_mxfp
    2. Compare FP8 data per source token
    Structural checks (permutation_map, probs, offsets) verified independently.
    """

    def _make_inputs(self, num_tokens, K, topk, num_experts, seed=42):
        torch.manual_seed(seed)
        hidden = torch.randn(num_tokens, K, device="cuda", dtype=torch.bfloat16)
        probs = torch.rand(num_tokens, topk, device="cuda", dtype=torch.float32)
        routing_map = torch.randint(0, num_experts, (num_tokens, topk), device="cuda")
        return hidden, probs, routing_map

    @pytest.mark.parametrize("num_tokens,K,topk,num_experts", [
        (4, 128, 2, 4),
        (16, 128, 2, 8),
        (32, 256, 4, 8),
        (64, 128, 6, 8),
        (64, 2688, 8, 128),
        (128, 1856, 4, 32),
    ])
    def test_data_matches_pytorch_ref(self, num_tokens, K, topk, num_experts):
        """For each real row, fused FP8 data matches ref_to_mxfp of the source token."""
        from megatron.core.inference.moe.permute import permute_and_quantize_mxfp8

        hidden, probs, routing_map = self._make_inputs(num_tokens, K, topk, num_experts)

        fused_mxfp8, _, fused_perm_map, _ = permute_and_quantize_mxfp8(
            hidden, probs, routing_map, 0, num_experts, alignment=128,
        )

        # For each real row, quantize the source token with PyTorch ref and compare
        for i in range(fused_perm_map.shape[0]):
            src = fused_perm_map[i].item()
            if src < 0:
                continue
            _, ref_data = ref_to_mxfp(hidden[src].unsqueeze(0))
            torch.testing.assert_close(
                fused_mxfp8.data[i].view(torch.uint8),
                ref_data.squeeze(0).view(torch.uint8),
                atol=0, rtol=0,
                msg=f"Row {i} (src={src}) FP8 data mismatch vs PyTorch ref",
            )

    @pytest.mark.parametrize("num_tokens,K,topk,num_experts", [
        (16, 128, 2, 8),
        (32, 256, 4, 8),
        (64, 2688, 8, 128),
    ])
    def test_batch_data_matches_pytorch_ref(self, num_tokens, K, topk, num_experts):
        """Batch comparison: gather all real rows, quantize as batch, compare."""
        from megatron.core.inference.moe.permute import permute_and_quantize_mxfp8

        hidden, probs, routing_map = self._make_inputs(num_tokens, K, topk, num_experts)

        fused_mxfp8, _, fused_perm_map, _ = permute_and_quantize_mxfp8(
            hidden, probs, routing_map, 0, num_experts, alignment=128,
        )

        real_mask = fused_perm_map >= 0
        real_indices = real_mask.nonzero(as_tuple=True)[0]
        if len(real_indices) == 0:
            return

        src_tokens = fused_perm_map[real_indices].long()
        permuted_bf16 = hidden[src_tokens]

        _, ref_data = ref_to_mxfp(permuted_bf16)

        torch.testing.assert_close(
            fused_mxfp8.data[real_indices].view(torch.uint8),
            ref_data.view(torch.uint8),
            atol=0, rtol=0,
        )

    @pytest.mark.parametrize("num_tokens,K,topk,num_experts", [
        (16, 128, 2, 8),
        (32, 256, 4, 8),
        (64, 2688, 8, 128),
    ])
    def test_correct_token_count(self, num_tokens, K, topk, num_experts):
        """Number of real rows equals total (token, topk) pairs routed to local experts."""
        from megatron.core.inference.moe.permute import permute_and_quantize_mxfp8

        hidden, probs, routing_map = self._make_inputs(num_tokens, K, topk, num_experts)

        _, _, fused_perm_map, _ = permute_and_quantize_mxfp8(
            hidden, probs, routing_map, 0, num_experts, alignment=128,
        )

        real_count = (fused_perm_map >= 0).sum().item()
        # All experts are local, so every pair should appear
        assert real_count == num_tokens * topk

    @pytest.mark.parametrize("num_tokens,K,topk,num_experts,local_start,num_local", [
        (64, 128, 4, 8, 2, 3),
        (64, 256, 4, 8, 0, 4),
        (128, 128, 8, 128, 96, 32),
    ])
    def test_expert_subset(self, num_tokens, K, topk, num_experts, local_start, num_local):
        """Fused kernel correctly handles local expert subsets."""
        from megatron.core.inference.moe.permute import permute_and_quantize_mxfp8

        hidden, probs, routing_map = self._make_inputs(num_tokens, K, topk, num_experts)

        _, _, fused_perm_map, _ = permute_and_quantize_mxfp8(
            hidden, probs, routing_map, local_start, num_local, alignment=128,
        )

        real_count = (fused_perm_map >= 0).sum().item()
        local_mask = (routing_map >= local_start) & (routing_map < local_start + num_local)
        expected_count = local_mask.sum().item()
        assert real_count == expected_count

    def test_returns_mxfp8_tensor(self):
        """Result is an MXFP8Tensor with correct backend."""
        from megatron.core.inference.moe.permute import permute_and_quantize_mxfp8
        from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

        hidden, probs, routing_map = self._make_inputs(16, 128, 2, 4)
        result, _, _, _ = permute_and_quantize_mxfp8(
            hidden, probs, routing_map, 0, 4, alignment=128,
        )
        assert isinstance(result, MXFP8Tensor)
        assert result.backend == "triton"
        assert result.data.dtype == torch.float8_e4m3fn

    @pytest.mark.parametrize("alignment", [128])
    def test_offsets_aligned(self, alignment):
        """Inclusive offsets are multiples of alignment."""
        from megatron.core.inference.moe.permute import permute_and_quantize_mxfp8

        hidden, probs, routing_map = self._make_inputs(64, 128, 4, 8)
        _, _, _, offs = permute_and_quantize_mxfp8(
            hidden, probs, routing_map, 0, 8, alignment=alignment,
        )
        for i in range(offs.shape[0]):
            assert offs[i].item() % alignment == 0, (
                f"Offset {i}={offs[i].item()} not aligned to {alignment}"
            )
