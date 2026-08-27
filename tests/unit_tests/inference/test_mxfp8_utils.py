# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for MXFP8 quantization.

Tests cover:
- mxfp8_quantize (Triton kernel): data and swizzled scales vs PyTorch reference
- MXFP8Tensor.from_bf16: both 'triton' and 'flashinfer' backends
- MXFP8Tensor.scale_2d: reshape correctness
"""

import pytest
import torch

from megatron.core.inference.moe import HAVE_TE_GROUPED_MXFP8
from megatron.core.inference.quantization.mxfp8_quantize import mxfp8_quantize, mxfp8_quantize_into
from megatron.core.inference.quantization.mxfp8_tensor import HAVE_FLASHINFER, MXFP8Tensor

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


def ref_to_mxfp(data_hp: torch.Tensor, block_size: int = 32, format: str = "mxfp8"):
    if data_hp.dtype not in (torch.bfloat16, torch.float):
        raise AssertionError(f"{data_hp.dtype} is not supported yet")
    if data_hp.shape[-1] % block_size != 0:
        raise AssertionError(
            f"the last dimension of shape {data_hp.shape} must be divisible by block_size {block_size}"
        )
    if not data_hp.is_contiguous():
        raise AssertionError("unsupported: data_hp must be contiguous")

    orig_shape = data_hp.shape
    data_hp = data_hp.reshape(*orig_shape[:-1], orig_shape[-1] // block_size, block_size)

    max_abs = torch.amax(torch.abs(data_hp), -1).unsqueeze(-1)

    data_hp = data_hp.to(torch.float32)
    max_abs = max_abs.to(torch.float32)

    if format == "mxfp8":
        F8E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
        max_pos = F8E4M3_MAX
    elif format == "mxfp4":
        F4E2M1_MAX = 6.0
        max_pos = F4E2M1_MAX

    # RCEIL
    def _to_mx_rceil(
        data_hp: torch.Tensor, max_abs: torch.Tensor, max_pos: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        E8M0_EXPONENT_BIAS = 127
        descale = max_abs / max_pos
        exponent = torch.where(
            torch.isnan(descale),
            0xFF,  # Handle biased exponent for nan
            (
                torch.clamp(
                    torch.ceil(torch.log2(descale)), min=-E8M0_EXPONENT_BIAS, max=E8M0_EXPONENT_BIAS
                )
                + E8M0_EXPONENT_BIAS
            ).to(torch.uint8),
        )

        descale_fp = torch.where(
            exponent == 0, 1.0, torch.exp2(E8M0_EXPONENT_BIAS - exponent.to(torch.float32))
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
        padded = torch.zeros(
            (padded_rows, padded_cols), device=input_matrix.device, dtype=input_matrix.dtype
        )
        padded[:rows, :cols] = input_matrix

    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)

    return rearranged.flatten()


# ──────────────────────────────────────────────────────────────────────
# mxfp8_quantize (Triton kernel)
# ──────────────────────────────────────────────────────────────────────


class TestMxfp8Quantize:

    @pytest.mark.parametrize(
        "M,K",
        [
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
            (128, 2688),  # nanov3 hidden_size
            (256, 1856),  # nanov3 moe_ffn_hidden_size
            (512, 2688),
        ],
    )
    def test_data_matches_reference(self, M, K):
        """Quantized FP8 data matches PyTorch reference."""
        torch.manual_seed(42)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

        triton_data, _ = mxfp8_quantize(x)
        _, ref_data = ref_to_mxfp(x)

        assert triton_data.shape == (M, K)
        assert triton_data.dtype == torch.float8_e4m3fn
        torch.testing.assert_close(
            triton_data.view(torch.uint8), ref_data.view(torch.uint8), atol=0, rtol=0
        )

    @pytest.mark.parametrize(
        "M,K",
        [
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
        ],
    )
    def test_scales_match_reference(self, M, K):
        """Swizzled scales match ref_to_mxfp scales passed through ref_swizzle."""
        torch.manual_seed(42)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

        _, triton_scales = mxfp8_quantize(x)
        ref_scales_2d, _ = ref_to_mxfp(x)  # [M, K//32] e8m0

        # Swizzle the reference scales
        ref_swizzled = ref_swizzle(ref_scales_2d)

        # Compare as uint8 since e8m0 is just exponent bytes
        torch.testing.assert_close(
            triton_scales.view(torch.uint8), ref_swizzled.view(torch.uint8), atol=0, rtol=0
        )

    @pytest.mark.parametrize("M,K", [(1, 32), (16, 128), (128, 2688)])
    def test_all_zeros_input(self, M, K):
        """All-zero input produces all-zero FP8 data and zero scales."""
        x = torch.zeros(M, K, device="cuda", dtype=torch.bfloat16)
        data, scales = mxfp8_quantize(x)
        assert (data.float() == 0).all()
        assert (scales.view(torch.uint8) == 0).all()

    @pytest.mark.parametrize("M,K", [(1, 32), (16, 128), (128, 256)])
    def test_constant_input(self, M, K):
        """Constant input: all elements in a group have the same value."""
        x = torch.full((M, K), 1.0, device="cuda", dtype=torch.bfloat16)
        data, _ = mxfp8_quantize(x)
        _, ref_data = ref_to_mxfp(x)
        torch.testing.assert_close(
            data.view(torch.uint8), ref_data.view(torch.uint8), atol=0, rtol=0
        )

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_input_dtypes(self, dtype):
        """Kernel accepts bf16, fp16, and fp32 inputs."""
        x = torch.randn(16, 128, device="cuda", dtype=dtype)
        data, _ = mxfp8_quantize(x)
        assert data.dtype == torch.float8_e4m3fn
        assert data.shape == (16, 128)

    @pytest.mark.parametrize("M", [1, 127, 128, 129, 255, 256, 257, 512])
    def test_various_row_counts(self, M):
        """Test row counts that are not multiples of 128 (macro tile boundary)."""
        K = 128
        torch.manual_seed(42)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        data, _ = mxfp8_quantize(x)
        _, ref_data = ref_to_mxfp(x)
        torch.testing.assert_close(
            data.view(torch.uint8), ref_data.view(torch.uint8), atol=0, rtol=0
        )

    @pytest.mark.parametrize("seed", [0, 7, 42, 123, 999])
    def test_reproducible(self, seed):
        """Same input always produces same output."""
        torch.manual_seed(seed)
        x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
        d1, s1 = mxfp8_quantize(x)
        d2, s2 = mxfp8_quantize(x)
        torch.testing.assert_close(d1.view(torch.uint8), d2.view(torch.uint8), atol=0, rtol=0)
        torch.testing.assert_close(s1.view(torch.uint8), s2.view(torch.uint8), atol=0, rtol=0)

    def test_quantize_into_reuses_storage_and_clears_padding(self):
        """Existing output buffers match allocating quantization byte-for-byte."""
        torch.manual_seed(42)
        x = torch.randn(129, 160, device="cuda", dtype=torch.bfloat16)
        expected_data, expected_scale = mxfp8_quantize(x)
        out_data = torch.empty_like(expected_data)
        out_scale = torch.full_like(expected_scale.view(torch.uint8), 0xFF).view(
            torch.float8_e8m0fnu
        )
        data_ptr = out_data.data_ptr()
        scale_ptr = out_scale.data_ptr()

        mxfp8_quantize_into(x, out_data, out_scale)

        assert out_data.data_ptr() == data_ptr
        assert out_scale.data_ptr() == scale_ptr
        assert torch.equal(out_data, expected_data)
        assert torch.equal(out_scale.view(torch.uint8), expected_scale.view(torch.uint8))


# ──────────────────────────────────────────────────────────────────────
# MXFP8Tensor
# ──────────────────────────────────────────────────────────────────────


class TestMXFP8Tensor:

    def test_restore_uint8_scale_dtype_after_resharding(self):
        from megatron.core.inference.quantization.mxfp8_tensor import ensure_mxfp8_scale_dtype

        scale_bytes = torch.arange(16, dtype=torch.uint8)
        scale = ensure_mxfp8_scale_dtype(scale_bytes)

        assert scale.dtype == torch.float8_e8m0fnu
        torch.testing.assert_close(scale.view(torch.uint8), scale_bytes)

    @pytest.mark.parametrize("backend", ["triton", "flashinfer"])
    def test_copy_preserves_storage_and_logical_metadata(self, backend):
        if backend == "flashinfer" and not HAVE_FLASHINFER:
            pytest.skip("FlashInfer not available")

        source = torch.randn(16, 128, device="cuda", dtype=torch.bfloat16)
        tensor = MXFP8Tensor.from_bf16(source, backend=backend)
        data_ptr = tensor.data.data_ptr()
        scale_ptr = tensor.scale.data_ptr()

        assert tensor.shape == source.shape
        assert tensor.dtype == source.dtype
        assert tensor.device == source.device

        updated = torch.randn_like(source)
        assert tensor.copy_(updated) is tensor
        expected = MXFP8Tensor.from_bf16(updated, backend=backend)

        assert tensor.data.data_ptr() == data_ptr
        assert tensor.scale.data_ptr() == scale_ptr
        assert torch.equal(tensor.data, expected.data)
        assert torch.equal(tensor.scale.view(torch.uint8), expected.scale.view(torch.uint8))

    def test_copy_rejects_shape_mismatch(self):
        tensor = MXFP8Tensor.from_bf16(
            torch.randn(16, 128, device="cuda", dtype=torch.bfloat16), backend="triton"
        )
        with pytest.raises(ValueError, match="shape mismatch"):
            tensor.copy_(torch.randn(32, 128, device="cuda", dtype=torch.bfloat16))

    def test_triton_copy_does_not_allocate_quantized_outputs(self, monkeypatch):
        source = torch.randn(129, 160, device="cuda", dtype=torch.bfloat16)
        tensor = MXFP8Tensor.from_bf16(source, backend="triton")
        updated = torch.randn_like(source)
        expected = MXFP8Tensor.from_bf16(updated, backend="triton")

        def fail_from_bf16(*args, **kwargs):
            raise AssertionError("Triton copy_ allocated temporary quantized outputs")

        monkeypatch.setattr(MXFP8Tensor, "from_bf16", fail_from_bf16)

        tensor.copy_(updated)

        assert torch.equal(tensor.data, expected.data)
        assert torch.equal(tensor.scale.view(torch.uint8), expected.scale.view(torch.uint8))

    def test_copy_requires_backend(self):
        source = torch.randn(16, 128, device="cuda", dtype=torch.bfloat16)
        quantized = MXFP8Tensor.from_bf16(source, backend="triton")
        backendless = MXFP8Tensor(data=quantized.data, scale=quantized.scale, dtype=source.dtype)
        with pytest.raises(ValueError, match="without a quantization backend"):
            backendless.copy_(source)

    def test_copy_preserves_source_dtype_for_legacy_constructor(self):
        initial = MXFP8Tensor.from_bf16(
            torch.randn(16, 128, device="cuda", dtype=torch.bfloat16), backend="triton"
        )
        tensor = MXFP8Tensor(initial.data.clone(), initial.scale.clone(), "triton")
        source = torch.randn(16, 128, device="cuda", dtype=torch.float16)
        expected = MXFP8Tensor.from_bf16(source, backend="triton")

        assert tensor.dtype is None
        tensor.copy_(source)
        assert tensor.dtype == source.dtype
        assert torch.equal(tensor.data, expected.data)
        assert torch.equal(tensor.scale, expected.scale)

    def test_failed_legacy_copy_keeps_dtype_unknown(self):
        initial = MXFP8Tensor.from_bf16(
            torch.randn(16, 128, device="cuda", dtype=torch.bfloat16), backend="triton"
        )
        tensor = MXFP8Tensor(
            initial.data.clone(), initial.scale.clone(), "unsupported"  # type: ignore[arg-type]
        )

        with pytest.raises(ValueError, match="Unknown MXFP8 quantization backend"):
            tensor.copy_(torch.randn(16, 128, device="cuda", dtype=torch.float16))
        assert tensor.dtype is None

    def test_copy_does_not_downcast_update_to_logical_dtype(self):
        source = torch.randn(16, 128, device="cuda", dtype=torch.float16)
        tensor = MXFP8Tensor.from_bf16(source, backend="triton")
        update = torch.linspace(
            -100_000, 100_000, steps=16 * 128, device="cuda", dtype=torch.float32
        ).reshape(16, 128)
        expected = MXFP8Tensor.from_bf16(update, backend="triton")

        assert tensor.dtype == torch.float16
        tensor.copy_(update)
        assert tensor.dtype == torch.float16
        assert torch.equal(tensor.data, expected.data)
        assert torch.equal(tensor.scale.view(torch.uint8), expected.scale.view(torch.uint8))

    def test_copy_normalizes_flashinfer_fp32_input_to_bf16(self, monkeypatch):
        tensor = MXFP8Tensor(
            data=torch.empty((16, 128), device="cuda", dtype=torch.float8_e4m3fn),
            scale=torch.empty(512, device="cuda", dtype=torch.uint8),
            backend="flashinfer",
        )
        update = torch.randn(16, 128, device="cuda", dtype=torch.float32)
        expected_data = torch.zeros_like(tensor.data)
        expected_scale = torch.zeros_like(tensor.scale)
        quantizer_input = {}

        def fake_from_bf16(cls, value, group_size=32, backend="flashinfer"):
            quantizer_input["dtype"] = value.dtype
            return cls(data=expected_data, scale=expected_scale, backend=backend, dtype=value.dtype)

        monkeypatch.setattr(MXFP8Tensor, "from_bf16", classmethod(fake_from_bf16))

        tensor.copy_(update)

        assert quantizer_input["dtype"] == torch.bfloat16
        assert tensor.dtype == torch.bfloat16
        assert torch.equal(tensor.data, expected_data)
        assert torch.equal(tensor.scale, expected_scale)

    def test_copy_rejects_stacked_storage(self):
        tensor = MXFP8Tensor.from_bf16(
            torch.randn(16, 128, device="cuda", dtype=torch.bfloat16), backend="triton"
        )
        stacked = MXFP8Tensor(
            data=torch.stack([tensor.data, tensor.data]),
            scale=torch.stack([tensor.scale, tensor.scale]),
            dtype=tensor.dtype,
            backend=tensor.backend,
        )

        with pytest.raises(ValueError, match="require 2D destination storage"):
            stacked.copy_(torch.randn(2, 16, 128, device="cuda", dtype=torch.bfloat16))

    def test_reject_invalid_scale_dtype(self):
        from megatron.core.inference.quantization.mxfp8_tensor import ensure_mxfp8_scale_dtype

        with pytest.raises(TypeError, match="MXFP8 scales must use"):
            ensure_mxfp8_scale_dtype(torch.ones(16, dtype=torch.bfloat16))

    def test_validate_unswizzled_2d_scale_geometry(self):
        from megatron.core.inference.quantization.mxfp8_tensor import (
            MXFP8Tensor,
            validate_mxfp8_tensor,
        )

        data = torch.empty((64, 128), dtype=torch.float8_e4m3fn, device="cuda")
        scale = torch.empty((64, 4), dtype=torch.uint8, device="cuda")

        validate_mxfp8_tensor(MXFP8Tensor(data, scale, dtype=torch.bfloat16, backend="flashinfer"))

        invalid = MXFP8Tensor(data, scale[:63], dtype=torch.bfloat16, backend="flashinfer")
        with pytest.raises(ValueError, match="2D scale has shape"):
            validate_mxfp8_tensor(invalid)

    def test_missing_backend_error_is_actionable(self):
        from megatron.core.inference.quantization.mxfp8_tensor import (
            MXFP8Tensor,
            validate_mxfp8_tensor,
        )

        data = torch.empty((128, 128), dtype=torch.float8_e4m3fn, device="cuda")
        scale = torch.empty(512, dtype=torch.uint8, device="cuda")

        with pytest.raises(ValueError, match="backend= explicitly"):
            validate_mxfp8_tensor(MXFP8Tensor(data, scale, dtype=torch.bfloat16))

    @pytest.mark.parametrize("M,K", [(16, 128), (64, 256), (128, 2688)])
    def test_from_bf16_triton(self, M, K):
        """The Triton backend produces correct data and scales."""
        from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

        torch.manual_seed(42)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

        tensor = MXFP8Tensor.from_bf16(x, backend="triton")
        _, ref_data = ref_to_mxfp(x)

        assert tensor.data.shape == (M, K)
        assert tensor.data.dtype == torch.float8_e4m3fn
        assert tensor.backend == "triton"
        torch.testing.assert_close(
            tensor.data.view(torch.uint8), ref_data.view(torch.uint8), atol=0, rtol=0
        )

    @pytest.mark.parametrize("M,K", [(16, 128), (64, 256), (128, 2688)])
    def test_from_bf16_flashinfer(self, M, K):
        """from_bf16 with flashinfer backend produces valid output."""
        from megatron.core.inference.quantization.mxfp8_tensor import HAVE_FLASHINFER, MXFP8Tensor

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

    @pytest.mark.parametrize("M,K", [(1, 32), (16, 128), (128, 2688), (256, 1856)])
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

    @pytest.mark.parametrize("M,K", [(16, 128), (128, 2688)])
    def test_scale_2d_idempotent(self, M, K):
        """Calling scale_2d twice returns the same result."""
        from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

        torch.manual_seed(42)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        tensor = MXFP8Tensor.from_bf16(x, backend="triton")

        s1 = tensor.scale_2d()
        s2 = tensor.scale_2d()
        torch.testing.assert_close(s1.view(torch.uint8), s2.view(torch.uint8), atol=0, rtol=0)

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


@pytest.mark.skipif(
    torch.cuda.get_device_capability()[0] < 10,
    reason="MXFP8 FlashInfer comparison requires Blackwell (SM 100+)",
)
class TestTritonVsFlashinfer:

    @pytest.mark.parametrize("M,K", [(1, 32), (16, 128), (64, 256), (128, 2688), (256, 1856)])
    def test_data_matches(self, M, K):
        """Triton and FlashInfer backends produce identical FP8 data."""
        from megatron.core.inference.quantization.mxfp8_tensor import HAVE_FLASHINFER, MXFP8Tensor

        if not HAVE_FLASHINFER:
            pytest.skip("FlashInfer not available")

        torch.manual_seed(42)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

        triton_tensor = MXFP8Tensor.from_bf16(x, backend="triton")
        flashinfer_tensor = MXFP8Tensor.from_bf16(x, backend="flashinfer")

        torch.testing.assert_close(
            triton_tensor.data.float(), flashinfer_tensor.data.float(), atol=0, rtol=0
        )

    @pytest.mark.parametrize("M,K", [(1, 32), (16, 128), (64, 256), (128, 2688), (256, 1856)])
    def test_scales_match(self, M, K):
        """Triton and FlashInfer backends produce identical swizzled scales."""
        from megatron.core.inference.quantization.mxfp8_tensor import HAVE_FLASHINFER, MXFP8Tensor

        if not HAVE_FLASHINFER:
            pytest.skip("FlashInfer not available")

        torch.manual_seed(42)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

        triton_tensor = MXFP8Tensor.from_bf16(x, backend="triton")
        flashinfer_tensor = MXFP8Tensor.from_bf16(x, backend="flashinfer")

        torch.testing.assert_close(
            triton_tensor.scale.view(torch.uint8),
            flashinfer_tensor.scale.view(torch.uint8),
            atol=0,
            rtol=0,
        )


def _make_permutation_map(M, num_padding=0):
    """Create a permutation_map with optional padding rows at the end."""
    real = torch.arange(M - num_padding, dtype=torch.int32, device="cuda")
    pad = torch.full((num_padding,), -1, dtype=torch.int32, device="cuda")
    return torch.cat([real, pad])


def _vt(n):
    return torch.tensor(n, dtype=torch.int32, device="cuda")


# ──────────────────────────────────────────────────────────────────────
# squared_relu_and_quantize_mxfp8 vs PyTorch reference
# ──────────────────────────────────────────────────────────────────────


class TestSquaredReluAndQuantizeMxfp8:
    """Compare fused squared_relu + mxfp8 quantize against PyTorch reference.

    Reference: torch.relu(x.float()).pow(2).to(bf16) -> ref_to_mxfp -> ref_swizzle.
    The fused kernel matches the BF16 materialization used by training and the
    unfused inference path before quantizing to MXFP8.
    """

    @pytest.mark.parametrize(
        "M,K",
        [
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
        ],
    )
    def test_data_matches_pytorch_ref(self, M, K):
        """Fused FP8 data matches PyTorch squared ReLU + ref_to_mxfp."""
        from megatron.core.inference.moe.activations import squared_relu_and_quantize_mxfp8

        torch.manual_seed(42)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        perm_map = _make_permutation_map(M, num_padding=0)

        # PyTorch reference: squared ReLU in fp32, then downcast to bf16, then quantize
        activated_ref = torch.relu(x.float()).pow(2).to(torch.bfloat16)
        _, ref_data = ref_to_mxfp(activated_ref)

        # Fused kernel
        fused_result = squared_relu_and_quantize_mxfp8(x, perm_map, _vt(M))

        torch.testing.assert_close(
            fused_result.data.view(torch.uint8), ref_data.view(torch.uint8), atol=0, rtol=0
        )

    @pytest.mark.parametrize("M,K", [(1, 32), (16, 128), (128, 128), (128, 2688), (256, 1856)])
    def test_scales_match_pytorch_ref(self, M, K):
        """Fused swizzled scales match PyTorch ref_to_mxfp + ref_swizzle."""
        from megatron.core.inference.moe.activations import squared_relu_and_quantize_mxfp8

        torch.manual_seed(42)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        perm_map = _make_permutation_map(M, num_padding=0)

        # PyTorch reference
        activated_ref = torch.relu(x.float()).pow(2).to(torch.bfloat16)
        ref_scales_2d, _ = ref_to_mxfp(activated_ref)
        ref_swizzled = ref_swizzle(ref_scales_2d)

        # Fused kernel
        fused_result = squared_relu_and_quantize_mxfp8(x, perm_map, _vt(M))

        torch.testing.assert_close(
            fused_result.scale.view(torch.uint8), ref_swizzled.view(torch.uint8), atol=0, rtol=0
        )

    @pytest.mark.parametrize(
        "M,K,num_padding",
        [(32, 128, 8), (64, 256, 16), (128, 128, 32), (128, 2688, 64), (256, 1856, 128)],
    )
    def test_real_rows_match_pytorch_ref_with_padding(self, M, K, num_padding):
        """Real rows match PyTorch reference even when padding rows are present."""
        from megatron.core.inference.moe.activations import squared_relu_and_quantize_mxfp8

        torch.manual_seed(42)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        perm_map = _make_permutation_map(M, num_padding=num_padding)

        # PyTorch reference (only real rows)
        real_rows = M - num_padding
        activated_ref = torch.relu(x[:real_rows].float()).pow(2).to(torch.bfloat16)
        _, ref_data = ref_to_mxfp(activated_ref)

        # Fused kernel
        fused_result = squared_relu_and_quantize_mxfp8(x, perm_map, _vt(M))

        torch.testing.assert_close(
            fused_result.data[:real_rows].view(torch.uint8),
            ref_data.view(torch.uint8),
            atol=0,
            rtol=0,
        )


CLAMP_SCALE = 16.0


def _ref_clamped_squared_relu(x, clamp_scale):
    """PyTorch model of the fused kernel's clamped activation, before quantization.

    Mirrors ``_clamped_relu`` + square: the tanh soft-clamped pre-activation stays in FP32
    (matching training's fused ``weighted_clamped_squared_relu``) and is rounded to BF16
    only after the square. This is the same BF16 tensor the separate-quantization route
    materializes with ``padded_squared_relu``, so quantizing it is what the fused kernel
    must reproduce.
    """
    relu = torch.clamp(x.float(), min=0.0)
    clamped = clamp_scale * torch.tanh(relu / clamp_scale)
    return (clamped**2).to(torch.bfloat16)


class TestSquaredReluAndQuantizeMxfp8Clamped:
    """The tanh soft clamp fused into squared_relu_and_quantize_mxfp8.

    config.activation_func_tanh_clamp_scale preconditions the pre-activation with
    ``s * tanh(x / s)``. In the fused kernel the clamp runs before the per-group-of-32
    amax, so it shapes the MXFP8 bins rather than being applied after quantization —
    these pin data and scales to the same PyTorch reference the unclamped tests use.

    Inputs are scaled past the clamp so the tanh saturates; inside the linear region a
    dropped clamp would be indistinguishable.
    """

    @staticmethod
    def _saturating_input(M, K, seed=42):
        torch.manual_seed(seed)
        return torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 50.0

    @pytest.mark.parametrize("M,K", [(1, 32), (16, 128), (128, 128), (128, 2688), (256, 1856)])
    def test_clamped_data_matches_pytorch_ref(self, M, K):
        """Fused FP8 data matches the clamped PyTorch activation quantized with ref_to_mxfp."""
        from megatron.core.inference.moe.activations import squared_relu_and_quantize_mxfp8

        x = self._saturating_input(M, K)
        perm_map = _make_permutation_map(M, num_padding=0)

        _, ref_data = ref_to_mxfp(_ref_clamped_squared_relu(x, CLAMP_SCALE))
        fused_result = squared_relu_and_quantize_mxfp8(x, perm_map, _vt(M), clamp_scale=CLAMP_SCALE)

        torch.testing.assert_close(
            fused_result.data.view(torch.uint8), ref_data.view(torch.uint8), atol=0, rtol=0
        )

    @pytest.mark.parametrize("M,K", [(1, 32), (16, 128), (128, 128), (128, 2688)])
    def test_clamped_scales_match_pytorch_ref(self, M, K):
        """Clamping before the amax gives the block scales of the clamped activation."""
        from megatron.core.inference.moe.activations import squared_relu_and_quantize_mxfp8

        x = self._saturating_input(M, K)
        perm_map = _make_permutation_map(M, num_padding=0)

        ref_scales_2d, _ = ref_to_mxfp(_ref_clamped_squared_relu(x, CLAMP_SCALE))
        ref_swizzled = ref_swizzle(ref_scales_2d)
        fused_result = squared_relu_and_quantize_mxfp8(x, perm_map, _vt(M), clamp_scale=CLAMP_SCALE)

        torch.testing.assert_close(
            fused_result.scale.view(torch.uint8), ref_swizzled.view(torch.uint8), atol=0, rtol=0
        )

    @pytest.mark.parametrize("M,K,num_padding", [(32, 128, 8), (128, 2688, 64)])
    def test_clamped_real_rows_match_pytorch_ref_with_padding(self, M, K, num_padding):
        """Alignment-padding rows stay skipped on the clamped path."""
        from megatron.core.inference.moe.activations import squared_relu_and_quantize_mxfp8

        x = self._saturating_input(M, K)
        perm_map = _make_permutation_map(M, num_padding=num_padding)

        real_rows = M - num_padding
        _, ref_data = ref_to_mxfp(_ref_clamped_squared_relu(x[:real_rows], CLAMP_SCALE))
        fused_result = squared_relu_and_quantize_mxfp8(x, perm_map, _vt(M), clamp_scale=CLAMP_SCALE)

        torch.testing.assert_close(
            fused_result.data[:real_rows].view(torch.uint8),
            ref_data.view(torch.uint8),
            atol=0,
            rtol=0,
        )

    def test_clamp_changes_the_result(self):
        """Guard against the clamp being silently dropped inside the fused quantize kernel."""
        from megatron.core.inference.moe.activations import squared_relu_and_quantize_mxfp8

        M, K = 128, 256
        x = self._saturating_input(M, K)
        perm_map = _make_permutation_map(M, num_padding=0)

        unclamped = squared_relu_and_quantize_mxfp8(x, perm_map, _vt(M))
        clamped = squared_relu_and_quantize_mxfp8(x, perm_map, _vt(M), clamp_scale=CLAMP_SCALE)

        assert not torch.equal(clamped.data.view(torch.uint8), unclamped.data.view(torch.uint8))
        # The clamp bounds the activation by s ** 2, so it must also lower the block scales.
        assert not torch.equal(clamped.scale.view(torch.uint8), unclamped.scale.view(torch.uint8))

    def test_matches_unfused_activation_then_quantize(self):
        """Fused clamp+quantize agrees with the disable_fused_quant_kernels route.

        mcore_fused_moe falls back to padded_squared_relu followed by
        MXFP8Tensor.from_bf16 when fused quant kernels are disabled; both routes must
        quantize the same clamped BF16 activation.
        """
        from megatron.core.inference.moe.activations import (
            padded_squared_relu,
            squared_relu_and_quantize_mxfp8,
        )

        M, K = 128, 256
        x = self._saturating_input(M, K)
        perm_map = _make_permutation_map(M, num_padding=0)

        fused = squared_relu_and_quantize_mxfp8(x, perm_map, _vt(M), clamp_scale=CLAMP_SCALE)
        unfused = MXFP8Tensor.from_bf16(
            padded_squared_relu(x, perm_map, _vt(M), clamp_scale=CLAMP_SCALE), backend="triton"
        )

        torch.testing.assert_close(
            fused.data.view(torch.uint8), unfused.data.view(torch.uint8), atol=0, rtol=0
        )
        torch.testing.assert_close(
            fused.scale.view(torch.uint8), unfused.scale.view(torch.uint8), atol=0, rtol=0
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

    @pytest.mark.parametrize(
        "num_tokens,K,topk,num_experts",
        [
            (4, 128, 2, 4),
            (16, 128, 2, 8),
            (32, 256, 4, 8),
            (64, 128, 6, 8),
            (64, 2688, 8, 128),
            (128, 1856, 4, 32),
        ],
    )
    def test_data_matches_pytorch_ref(self, num_tokens, K, topk, num_experts):
        """For each real row, fused FP8 data matches ref_to_mxfp of the source token."""
        from megatron.core.inference.moe.permute import permute_and_quantize_mxfp8

        hidden, probs, routing_map = self._make_inputs(num_tokens, K, topk, num_experts)

        fused_mxfp8, _, fused_perm_map, offs = permute_and_quantize_mxfp8(
            hidden, probs, routing_map, 0, num_experts, _vt(num_tokens), alignment=128
        )

        # For each real row, quantize the source token with PyTorch ref and compare
        for i in range(offs[-1].item()):
            src = fused_perm_map[i].item()
            if src < 0:
                continue
            _, ref_data = ref_to_mxfp(hidden[src].unsqueeze(0))
            torch.testing.assert_close(
                fused_mxfp8.data[i].view(torch.uint8),
                ref_data.squeeze(0).view(torch.uint8),
                atol=0,
                rtol=0,
                msg=f"Row {i} (src={src}) FP8 data mismatch vs PyTorch ref",
            )

    @pytest.mark.parametrize(
        "num_tokens,K,topk,num_experts", [(16, 128, 2, 8), (32, 256, 4, 8), (64, 2688, 8, 128)]
    )
    def test_batch_data_matches_pytorch_ref(self, num_tokens, K, topk, num_experts):
        """Batch comparison: gather all real rows, quantize as batch, compare."""
        from megatron.core.inference.moe.permute import permute_and_quantize_mxfp8

        hidden, probs, routing_map = self._make_inputs(num_tokens, K, topk, num_experts)

        fused_mxfp8, _, fused_perm_map, offs = permute_and_quantize_mxfp8(
            hidden, probs, routing_map, 0, num_experts, _vt(num_tokens), alignment=128
        )

        real_mask = fused_perm_map[: offs[-1].item()] >= 0
        real_indices = real_mask.nonzero(as_tuple=True)[0]
        if len(real_indices) == 0:
            return

        src_tokens = fused_perm_map[real_indices].long()
        permuted_bf16 = hidden[src_tokens]

        _, ref_data = ref_to_mxfp(permuted_bf16)

        torch.testing.assert_close(
            fused_mxfp8.data[real_indices].view(torch.uint8),
            ref_data.view(torch.uint8),
            atol=0,
            rtol=0,
        )

    @pytest.mark.parametrize(
        "num_tokens,K,topk,num_experts", [(16, 128, 2, 8), (32, 256, 4, 8), (64, 2688, 8, 128)]
    )
    def test_correct_token_count(self, num_tokens, K, topk, num_experts):
        """Number of real rows equals total (token, topk) pairs routed to local experts."""
        from megatron.core.inference.moe.permute import permute_and_quantize_mxfp8

        hidden, probs, routing_map = self._make_inputs(num_tokens, K, topk, num_experts)

        _, _, fused_perm_map, offs = permute_and_quantize_mxfp8(
            hidden, probs, routing_map, 0, num_experts, _vt(num_tokens), alignment=128
        )

        real_count = (fused_perm_map[: offs[-1].item()] >= 0).sum().item()
        # All experts are local, so every pair should appear
        assert real_count == num_tokens * topk

    @pytest.mark.parametrize(
        "num_tokens,K,topk,num_experts,local_start,num_local",
        [(64, 128, 4, 8, 2, 3), (64, 256, 4, 8, 0, 4), (128, 128, 8, 128, 96, 32)],
    )
    def test_expert_subset(self, num_tokens, K, topk, num_experts, local_start, num_local):
        """Fused kernel correctly handles local expert subsets."""
        from megatron.core.inference.moe.permute import permute_and_quantize_mxfp8

        hidden, probs, routing_map = self._make_inputs(num_tokens, K, topk, num_experts)

        _, _, fused_perm_map, offs = permute_and_quantize_mxfp8(
            hidden, probs, routing_map, local_start, num_local, _vt(num_tokens), alignment=128
        )

        real_count = (fused_perm_map[: offs[-1].item()] >= 0).sum().item()
        local_mask = (routing_map >= local_start) & (routing_map < local_start + num_local)
        expected_count = local_mask.sum().item()
        assert real_count == expected_count

    def test_returns_mxfp8_tensor(self):
        """Result is an MXFP8Tensor with correct backend."""
        from megatron.core.inference.moe.permute import permute_and_quantize_mxfp8
        from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

        hidden, probs, routing_map = self._make_inputs(16, 128, 2, 4)
        result, _, _, _ = permute_and_quantize_mxfp8(
            hidden, probs, routing_map, 0, 4, _vt(16), alignment=128
        )
        assert isinstance(result, MXFP8Tensor)
        assert result.backend == "triton"
        assert result.data.dtype == torch.float8_e4m3fn

    @pytest.mark.parametrize("alignment", [64, 128])
    def test_fixed_buffer_rows_match_swizzled_scale_padding(self, alignment):
        from megatron.core.inference.moe.permute import permute_and_quantize_mxfp8

        hidden, probs, routing_map = self._make_inputs(72, 2688, 6, 128)
        result, _, _, _ = permute_and_quantize_mxfp8(
            hidden, probs, routing_map, 0, 64, _vt(72), alignment=alignment
        )

        unaligned_rows = 72 * 6 + alignment * 64
        assert result.data.shape[0] == ceil_div(unaligned_rows, 128) * 128
        assert result.scale_2d().shape[0] == result.data.shape[0]

    def test_unfused_fixed_buffer_rows_match_swizzled_scale_padding(self):
        """The disable-fused-quant route preserves the same MXFP8 row invariant."""
        from megatron.core.inference.moe.permute import permute_tokens
        from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

        hidden, probs, routing_map = self._make_inputs(72, 2688, 6, 128)
        permuted, _, _, _ = permute_tokens(
            hidden, probs, routing_map, 0, 64, _vt(72), alignment=128, row_alignment=128
        )
        result = MXFP8Tensor.from_bf16(permuted, backend="triton")

        assert result.data.shape[0] == 8704
        assert result.scale_2d().shape[0] == result.data.shape[0]

    @pytest.mark.skipif(
        torch.cuda.get_device_capability()[0] < 10,
        reason="MXFP8 scaled_grouped_mm requires Blackwell (SM 100+)",
    )
    def test_unfused_moe_accepts_non_aligned_fixed_token_capacity(self):
        """The separate-quantization route feeds matching rows to scaled_grouped_mm."""
        from megatron.core.inference.moe.fused_moe import ActivationType, mcore_fused_moe
        from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

        num_tokens, hidden_size, topk, num_experts = 72, 128, 2, 4
        hidden, probs, routing_map = self._make_inputs(num_tokens, hidden_size, topk, num_experts)

        def stack_weights() -> MXFP8Tensor:
            per_expert = [
                MXFP8Tensor.from_bf16(
                    torch.randn(hidden_size, hidden_size, device="cuda", dtype=torch.bfloat16),
                    backend="triton",
                )
                for _ in range(num_experts)
            ]
            return MXFP8Tensor(
                data=torch.stack([weight.data for weight in per_expert]),
                scale=torch.stack([weight.scale for weight in per_expert]),
                dtype=torch.bfloat16,
                backend="triton",
            )

        output = mcore_fused_moe(
            hidden,
            probs,
            stack_weights(),
            stack_weights(),
            ActivationType.SQUARED_RELU,
            num_experts,
            0,
            _vt(num_tokens),
            routing_map,
            disable_fused_quant_kernels=True,
        )

        assert output.shape == hidden.shape
        assert torch.isfinite(output).all()

    @pytest.mark.parametrize("alignment", [128])
    def test_offsets_aligned(self, alignment):
        """Inclusive offsets are multiples of alignment."""
        from megatron.core.inference.moe.permute import permute_and_quantize_mxfp8

        hidden, probs, routing_map = self._make_inputs(64, 128, 4, 8)
        _, _, _, offs = permute_and_quantize_mxfp8(
            hidden, probs, routing_map, 0, 8, _vt(64), alignment=alignment
        )
        for i in range(offs.shape[0]):
            assert (
                offs[i].item() % alignment == 0
            ), f"Offset {i}={offs[i].item()} not aligned to {alignment}"


@pytest.mark.launch_on_gb200
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not HAVE_TE_GROUPED_MXFP8
    or torch.cuda.get_device_capability()[0] < 10,
    reason="Native TE MXFP8 grouped GEMM requires its device-metadata APIs and Blackwell",
)
class TestTENativeGroupedMxfp8:
    """Native TE MXFP8 grouped quantization/GEMM for inference-optimized MoE."""

    @staticmethod
    def _quantized_weights(num_experts, in_features, out_features=None, optimize_for_gemm=True):
        import transformer_engine_torch as tex
        from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

        quantizer = MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=False)
        quantizer.optimize_for_gemm = optimize_for_gemm
        if out_features is None:
            out_features = in_features
        bf16_weights = [
            torch.randn(out_features, in_features, device="cuda", dtype=torch.bfloat16) * 0.02
            for _ in range(num_experts)
        ]
        return bf16_weights, [quantizer(weight) for weight in bf16_weights]

    def test_moe_zero_pads_splits_to_256_and_matches_reference(self, monkeypatch):
        import megatron.core.inference.moe.fused_moe as fused_moe

        torch.manual_seed(11)
        num_tokens, hidden_size, num_experts, topk = 19, 128, 4, 2
        hidden = torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16) * 0.2
        probs = torch.rand(num_tokens, topk, device="cuda", dtype=torch.float32)
        routing_map = torch.tensor(
            [[token % 3, (token + 1) % 3] for token in range(num_tokens)],
            device="cuda",
            dtype=torch.int64,
        )
        fc1_bf16, fc1_mxfp8 = self._quantized_weights(num_experts, hidden_size)
        fc2_bf16, fc2_mxfp8 = self._quantized_weights(num_experts, hidden_size)

        captured_splits = []
        original_grouped_mm = fused_moe._te_mxfp8_grouped_mm

        def record_splits(x, weight, first_dims):
            captured_splits.append(first_dims.clone())
            return original_grouped_mm(x, weight, first_dims)

        monkeypatch.setattr(fused_moe, "_te_mxfp8_grouped_mm", record_splits)
        actual = fused_moe.mcore_fused_moe(
            hidden,
            probs,
            fc1_mxfp8,
            fc2_mxfp8,
            fused_moe.ActivationType.SQUARED_RELU,
            num_experts,
            0,
            _vt(num_tokens),
            routing_map,
        )

        reference = torch.zeros_like(hidden, dtype=torch.float32)
        for token in range(num_tokens):
            for route in range(topk):
                expert = routing_map[token, route].item()
                intermediate = torch.nn.functional.linear(
                    hidden[token : token + 1], fc1_bf16[expert]
                )
                intermediate = torch.relu(intermediate.float()).square().to(torch.bfloat16)
                expert_output = torch.nn.functional.linear(intermediate, fc2_bf16[expert])
                reference[token] += expert_output[0].float() * probs[token, route]

        assert len(captured_splits) == 2
        for splits in captured_splits:
            assert splits.tolist() == [256, 256, 256, 0]
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, reference, atol=5e-4, rtol=0.25)

    def test_swiglu_moe(self):
        from megatron.core.inference.moe.fused_moe import ActivationType, mcore_fused_moe

        torch.manual_seed(14)
        num_tokens, hidden_size, num_experts = 19, 128, 4
        hidden = torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
        probs = torch.rand(num_tokens, 2, device="cuda", dtype=torch.float32)
        routing_map = torch.tensor(
            [[token % 3, (token + 1) % 3] for token in range(num_tokens)],
            device="cuda",
            dtype=torch.int64,
        )
        _, fc1_mxfp8 = self._quantized_weights(
            num_experts, hidden_size, out_features=2 * hidden_size
        )
        _, fc2_mxfp8 = self._quantized_weights(num_experts, hidden_size)

        output = mcore_fused_moe(
            hidden,
            probs,
            fc1_mxfp8,
            fc2_mxfp8,
            ActivationType.SWIGLU,
            num_experts,
            0,
            _vt(num_tokens),
            routing_map,
        )

        assert output.shape == hidden.shape
        assert torch.isfinite(output).all()

    def test_model_conversion_preserves_te_experts(self):
        from megatron.core.inference.moe import InferenceGroupedGemmBackend
        from megatron.core.inference.quantization.utils import (
            get_te_grouped_moe_parameter_ids,
            quantize_model_to_mxfp8,
        )
        from megatron.core.transformer.moe.experts import InferenceGroupedMLP

        hidden_size = 128
        root = torch.nn.Module()
        root.dense = torch.nn.Module()
        _, dense_weights = self._quantized_weights(1, hidden_size, optimize_for_gemm=False)
        root.dense.weight = torch.nn.Parameter(dense_weights[0], requires_grad=False)

        root.experts = torch.nn.Module()
        root.experts.num_local_experts = 1
        root.experts.inference_grouped_gemm_backend = InferenceGroupedGemmBackend.TE
        root.experts.linear_fc1 = torch.nn.Module()
        root.experts.linear_fc2 = torch.nn.Module()
        _, fc1_weights = self._quantized_weights(1, hidden_size)
        _, fc2_weights = self._quantized_weights(1, hidden_size)
        root.experts.linear_fc1.weight0 = torch.nn.Parameter(fc1_weights[0], requires_grad=False)
        root.experts.linear_fc2.weight0 = torch.nn.Parameter(fc2_weights[0], requires_grad=False)

        excluded = get_te_grouped_moe_parameter_ids(root)
        quantize_model_to_mxfp8(root, backend="triton", excluded_parameter_ids=excluded)
        InferenceGroupedMLP._build_te_mxfp8_weights(root.experts)

        assert isinstance(root.dense.weight, MXFP8Tensor)
        assert not isinstance(root.experts.linear_fc1.weight0, MXFP8Tensor)
        assert not isinstance(root.experts.linear_fc2.weight0, MXFP8Tensor)
        assert root.experts._fc1_weight[0] is root.experts.linear_fc1.weight0
        assert root.experts._fc2_weight[0] is root.experts.linear_fc2.weight0

    def test_single_grouped_weight_representation(self, monkeypatch):
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import MXFP8BlockScaling

        from megatron.core.inference.moe.fused_moe import ActivationType, mcore_fused_moe
        from megatron.core.transformer.moe.experts import InferenceGroupedMLP

        # TE gates the experimental single-parameter representation behind this flag.
        # Without it, GroupedLinear silently falls back to weight0..weightN.
        monkeypatch.setenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "1")
        torch.manual_seed(12)
        num_tokens, hidden_size, num_experts = 19, 128, 4
        with te.fp8_model_init(enabled=True, recipe=MXFP8BlockScaling()):
            fc1 = te.GroupedLinear(
                num_experts,
                hidden_size,
                hidden_size,
                bias=False,
                params_dtype=torch.bfloat16,
                device="cuda",
                single_grouped_weight=True,
            )
            fc2 = te.GroupedLinear(
                num_experts,
                hidden_size,
                hidden_size,
                bias=False,
                params_dtype=torch.bfloat16,
                device="cuda",
                single_grouped_weight=True,
            )

        grouped_mlp = torch.nn.Module()
        grouped_mlp.num_local_experts = num_experts
        grouped_mlp.linear_fc1 = fc1
        grouped_mlp.linear_fc2 = fc2
        InferenceGroupedMLP._build_te_mxfp8_weights(grouped_mlp)

        hidden = torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
        probs = torch.rand(num_tokens, 2, device="cuda", dtype=torch.float32)
        routing_map = torch.tensor(
            [[token % 3, (token + 1) % 3] for token in range(num_tokens)],
            device="cuda",
            dtype=torch.int64,
        )
        output = mcore_fused_moe(
            hidden,
            probs,
            grouped_mlp._fc1_weight,
            grouped_mlp._fc2_weight,
            ActivationType.SQUARED_RELU,
            num_experts,
            0,
            _vt(num_tokens),
            routing_map,
        )

        assert grouped_mlp._fc1_weight is fc1.weight
        assert grouped_mlp._fc2_weight is fc2.weight
        assert output.shape == hidden.shape
        assert torch.isfinite(output).all()

    def test_cuda_graph_replays_with_new_device_splits(self):
        from megatron.core.inference.moe.fused_moe import ActivationType, mcore_fused_moe

        torch.manual_seed(13)
        num_tokens, hidden_size, num_experts = 17, 128, 4
        hidden = torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16) * 0.2
        probs = torch.rand(num_tokens, 1, device="cuda", dtype=torch.float32)
        routing_map = torch.zeros(num_tokens, 1, device="cuda", dtype=torch.int64)
        _, fc1_mxfp8 = self._quantized_weights(num_experts, hidden_size)
        _, fc2_mxfp8 = self._quantized_weights(num_experts, hidden_size)
        valid_tokens = _vt(num_tokens)

        def run_moe():
            return mcore_fused_moe(
                hidden,
                probs,
                fc1_mxfp8,
                fc2_mxfp8,
                ActivationType.SQUARED_RELU,
                num_experts,
                0,
                valid_tokens,
                routing_map,
            )

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                warmup_output = run_moe()
        torch.cuda.current_stream().wait_stream(warmup_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = run_moe()
        graph.replay()
        first_output = graph_output.clone()

        # The captured operations must consume this new device-resident routing state.
        routing_map.fill_(2)
        graph.replay()
        second_output = graph_output.clone()
        eager_output = run_moe()
        torch.cuda.synchronize()

        assert warmup_output is not None
        assert torch.isfinite(second_output).all()
        assert not torch.equal(first_output, second_output)
        torch.testing.assert_close(second_output, eager_output, atol=0, rtol=0)
