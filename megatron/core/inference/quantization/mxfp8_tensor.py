# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import dataclass, field
from typing import Literal, Optional

import torch

try:
    from flashinfer import mxfp8_quantize as flashinfer_mxfp8_quantize

    HAVE_FLASHINFER = True
except ImportError:
    HAVE_FLASHINFER = False

from megatron.core.inference.quantization.mxfp8_quantize import (
    MXFP8_BLOCK_SIZE,
    MXFP8_SCALE_COL_BLOCK,
    MXFP8_SCALE_ROW_BLOCK,
)
from megatron.core.inference.quantization.mxfp8_quantize import (
    mxfp8_quantize as mcore_mxfp8_quantize,
)
from megatron.core.inference.quantization.mxfp8_quantize import (
    mxfp8_quantize_into as mcore_mxfp8_quantize_into,
)


def _ceil_div(a, b):
    return (a + b - 1) // b


def ensure_mxfp8_scale_dtype(scale: torch.Tensor) -> torch.Tensor:
    """Return the E8M0 view required by PyTorch scaled GEMM APIs.

    FlashInfer exposes the same MXFP8 scale bytes as ``uint8``.
    """
    if scale.dtype == torch.uint8:
        return scale.view(torch.float8_e8m0fnu)
    if scale.dtype != torch.float8_e8m0fnu:
        raise TypeError(
            "MXFP8 scales must use torch.float8_e8m0fnu or its uint8 byte "
            f"representation, got {scale.dtype}."
        )
    return scale


MXFP8Backend = Literal["flashinfer", "triton"]
_MXFP8_SCALE_DTYPES: dict[MXFP8Backend, torch.dtype] = {
    "flashinfer": torch.uint8,
    "triton": torch.float8_e8m0fnu,
}


def validate_mxfp8_tensor(
    tensor: "MXFP8Tensor",
    *,
    expected_backend: Optional[MXFP8Backend] = None,
    tensor_name: str = "MXFP8 tensor",
) -> None:
    """Thorough MXFP8 validation for GEMM consumers, courtesy of Codex."""
    if tensor.data.ndim != 2:
        raise ValueError(
            f"{tensor_name} data must be 2D before grouped-weight stacking; "
            f"got shape {tuple(tensor.data.shape)}."
        )
    if tensor.data.dtype != torch.float8_e4m3fn:
        raise TypeError(
            f"{tensor_name} data must use torch.float8_e4m3fn, got {tensor.data.dtype}."
        )
    backend = expected_backend if expected_backend is not None else tensor.backend
    if expected_backend is not None and tensor.backend != expected_backend:
        raise ValueError(
            f"{tensor_name} backend is {tensor.backend!r}; expected "
            f"{expected_backend!r} for the configured GEMM consumer."
        )
    expected_scale_dtype = _MXFP8_SCALE_DTYPES.get(backend)
    if expected_scale_dtype is None:
        raise ValueError(
            f"{tensor_name} has no valid MXFP8 backend; got {backend!r}. "
            "Construct it via MXFP8Tensor.from_bf16(..., backend=...) or pass "
            "backend= explicitly to MXFP8Tensor."
        )
    if tensor.scale.dtype != expected_scale_dtype:
        raise TypeError(
            f"{tensor_name} scales for backend {backend!r} must use "
            f"{expected_scale_dtype}, got {tensor.scale.dtype}."
        )

    rows, cols = tensor.data.shape
    if cols % MXFP8_BLOCK_SIZE != 0:
        raise ValueError(
            f"{tensor_name} K dimension must be divisible by the MXFP8 block size "
            f"{MXFP8_BLOCK_SIZE}, "
            f"got {cols}."
        )
    scale_cols = cols // MXFP8_BLOCK_SIZE
    if tensor.scale.ndim == 2:
        expected_scale_shape = (rows, scale_cols)
        if tuple(tensor.scale.shape) != expected_scale_shape:
            raise ValueError(
                f"{tensor_name} 2D scale has shape {tuple(tensor.scale.shape)}; "
                f"expected {expected_scale_shape} for data shape {tuple(tensor.data.shape)}."
            )
    elif tensor.scale.ndim == 1:
        padded_rows = _ceil_div(rows, MXFP8_SCALE_ROW_BLOCK) * MXFP8_SCALE_ROW_BLOCK
        padded_scale_cols = _ceil_div(scale_cols, MXFP8_SCALE_COL_BLOCK) * MXFP8_SCALE_COL_BLOCK
        expected_scale_elements = padded_rows * padded_scale_cols
        if tensor.scale.numel() != expected_scale_elements:
            raise ValueError(
                f"{tensor_name} swizzled scale storage has {tensor.scale.numel()} elements; "
                f"expected {expected_scale_elements} for data shape "
                f"{tuple(tensor.data.shape)}."
            )
    else:
        raise ValueError(
            f"{tensor_name} scale must be a 1D swizzled tensor or a 2D unswizzled "
            f"tensor, got shape {tuple(tensor.scale.shape)}."
        )


@dataclass
class MXFP8Tensor:
    """MXFP8 tensor wrapper storing quantized data and E8M0 scale bytes."""

    data: torch.Tensor  # [M, K] fp8_e4m3fn
    scale: torch.Tensor  # 1D swizzled or [M, K // 32] unswizzled scales
    backend: Optional[MXFP8Backend] = None  # quantization and GEMM backend
    # Keyword-only preserves the pre-existing third positional ``backend`` argument.
    # This metadata supports tensor-like conversion consumers such as Megatron
    # Bridge; it does not constrain the precision accepted by ``quantize_``/``copy_``.
    # Legacy direct constructors have unknown logical dtype until their next
    # successful update.
    dtype: Optional[torch.dtype] = field(default=None, kw_only=True)

    @property
    def shape(self) -> torch.Size:
        """Shape of the quantized data storage."""
        return self.data.shape

    @property
    def device(self) -> torch.device:
        """Device holding the quantized tensor."""
        return self.data.device

    def size(self, idx: Optional[int] = None):
        """Wrapper for calling self.data.size()"""
        return self.data.size(idx)

    def scale_2d(self, K: Optional[int] = None) -> torch.Tensor:
        """Reshape 1D swizzled scale to 2D for scaled_grouped_mm / scaled_mm.

        Swizzle pads rows to multiples of 128 and cols to multiples of 4.
        Returns (padded_M, padded_cols) where padded_cols = ceil(K//32, 4) * 4.
        """
        if self.scale.dim() == 2:
            return self.scale
        if K is None:
            K = self.data.shape[-1]
        n_col_blocks = _ceil_div(K // MXFP8_BLOCK_SIZE, MXFP8_SCALE_COL_BLOCK)
        padded_cols = n_col_blocks * MXFP8_SCALE_COL_BLOCK
        return self.scale.reshape(-1, padded_cols)

    def quantize_(self, value: torch.Tensor) -> "MXFP8Tensor":
        """Quantize a logical tensor into existing storage without changing pointers.

        The source dtype is preserved when supported by the configured backend so
        the quantizer does not introduce an avoidable intermediate downcast.
        FlashInfer FP32 inputs are converted to BF16 because that backend accepts
        FP16/BF16 inputs only. For legacy instances with unknown logical dtype, the
        first successful update records the dtype actually passed to the quantizer.
        Shape broadcasting is unsupported because MXFP8 scale storage has fixed
        geometry.
        """
        if self.data.ndim != 2:
            raise ValueError(
                "In-place MXFP8 updates require 2D destination storage; "
                f"got shape {tuple(self.data.shape)}."
            )
        if value.shape != self.shape:
            raise ValueError(
                f"MXFP8 shape mismatch: expected {tuple(self.shape)}, got {tuple(value.shape)}."
            )
        if self.backend is None:
            raise ValueError("Cannot update an MXFP8Tensor without a quantization backend.")
        value = value.to(device=self.device)
        if self.backend == "flashinfer" and value.dtype == torch.float32:
            value = value.to(dtype=torch.bfloat16)
        value = value.contiguous()
        if (
            self.backend == "triton"
            and self.data.dtype == torch.float8_e4m3fn
            and self.data.is_contiguous()
            and self.scale.ndim == 1
            and self.scale.dtype in (torch.uint8, torch.float8_e8m0fnu)
            and self.scale.is_contiguous()
        ):
            mcore_mxfp8_quantize_into(value, self.data, self.scale)
        else:
            quantized = MXFP8Tensor.from_bf16(value, backend=self.backend)
            self.data.copy_(quantized.data)
            self.scale.view(torch.uint8).copy_(quantized.scale.view(torch.uint8))
        if self.dtype is None:
            self.dtype = value.dtype
        return self

    def copy_(self, value: torch.Tensor) -> "MXFP8Tensor":
        """Tensor-compatible alias for :meth:`quantize_` used by generic writeback.

        Unlike ``torch.Tensor.copy_``, this method requires an exact shape and
        does not accept ``non_blocking``; quantization and storage updates are
        ordered on the current CUDA stream. The alias lets external conversion
        integrations such as Megatron Bridge treat plain and MXFP8 destinations
        uniformly while ``quantize_`` remains available to existing callers.
        """
        return self.quantize_(value)

    @classmethod
    def from_bf16(cls, x: torch.Tensor, group_size: int = 32, backend: MXFP8Backend = "flashinfer"):
        """Quantize a floating-point CUDA tensor to MXFP8.

        The historical method name is retained for compatibility. The Triton
        backend accepts BF16, FP16, and FP32 inputs; FlashInfer accepts BF16 and
        FP16 inputs.

        Args:
            x: [M, K] floating-point tensor on CUDA.
            group_size: MXFP8 group size (default 32).
            backend: 'triton' (fused quantize + swizzle Triton kernel) or
                     'flashinfer' (single fused FlashInfer CUDA kernel).
        """
        assert x.is_cuda and x.dim() == 2
        assert x.shape[-1] % group_size == 0
        if backend == "flashinfer":
            assert HAVE_FLASHINFER, "FlashInfer not available"
            data, scale = flashinfer_mxfp8_quantize(x)
            if scale.dtype == torch.float8_e8m0fnu:
                scale = scale.view(torch.uint8)
            elif scale.dtype != torch.uint8:
                raise TypeError(f"FlashInfer MXFP8 scales must be uint8 bytes, got {scale.dtype}.")
        elif backend == "triton":
            data, scale = mcore_mxfp8_quantize(x)
            scale = ensure_mxfp8_scale_dtype(scale)
        else:
            raise ValueError(
                f"Unknown MXFP8 quantization backend: '{backend}'. "
                "Must be 'triton' or 'flashinfer'."
            )
        return cls(data=data, scale=scale, backend=backend, dtype=x.dtype)
