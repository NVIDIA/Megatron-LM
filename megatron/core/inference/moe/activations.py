# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Padding-aware activation kernels for fused MoE.

These kernels skip padding rows (where permutation_map == -1) to avoid
wasted computation on aligned-but-empty expert slots.
"""

from unittest.mock import MagicMock

import torch
from packaging import version

from megatron.core.utils import null_decorator

try:
    import triton
    import triton.language as tl

    if version.parse(triton.__version__) < version.parse("3.4.0") and not torch.cuda.is_available():
        HAVE_TRITON = False
    else:
        HAVE_TRITON = tl.constexpr(version.parse(triton.__version__) >= version.parse("2.0.0"))
except ImportError:
    HAVE_TRITON = False

if not HAVE_TRITON:
    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()


@triton.jit
def _squared_relu_kernel(
    input_ptr, output_ptr, src_idx_ptr, M, N,
    BLOCK_N: tl.constexpr,
):
    """Squared ReLU that skips padding rows (permutation_map == -1)."""
    row = tl.program_id(0)
    if tl.load(src_idx_ptr + row) < 0:
        return
    for n in tl.range(0, N, BLOCK_N):
        o = n + tl.arange(0, BLOCK_N)
        m = o < N
        x = tl.load(input_ptr + row * N + o, mask=m)
        r = tl.maximum(x, 0.0)
        tl.store(output_ptr + row * N + o, r * r, mask=m)


def padded_squared_relu(
    x: torch.Tensor, permutation_map: torch.Tensor
) -> torch.Tensor:
    """Squared ReLU activation that skips padding rows."""
    M, N = x.shape
    out = torch.zeros(M, N, dtype=x.dtype, device=x.device)
    BLOCK_N = min(triton.next_power_of_2(N), 1024)
    _squared_relu_kernel[(M,)](
        x, out, permutation_map, M, N, BLOCK_N=BLOCK_N,
    )
    return out


@triton.jit
def _swiglu_kernel(
    input_ptr, output_ptr, src_idx_ptr, M, N,
    BLOCK_N: tl.constexpr,
):
    """SwiGLU activation that skips padding rows (permutation_map == -1)."""
    row = tl.program_id(0)
    if tl.load(src_idx_ptr + row) < 0:
        return
    for n in tl.range(0, N, BLOCK_N):
        o = n + tl.arange(0, BLOCK_N)
        m = o < N
        gate = tl.load(input_ptr + row * 2 * N + o, mask=m)
        up = tl.load(input_ptr + row * 2 * N + N + o, mask=m)
        tl.store(output_ptr + row * N + o,
                 tl.sigmoid(gate.to(tl.float32)).to(gate.dtype) * gate * up, mask=m)


def padded_swiglu(
    x: torch.Tensor, permutation_map: torch.Tensor
) -> torch.Tensor:
    """SwiGLU activation that skips padding rows."""
    M = x.shape[0]
    N = x.shape[1] // 2
    out = torch.zeros(M, N, dtype=x.dtype, device=x.device)
    BLOCK_N = min(triton.next_power_of_2(N), 1024)
    _swiglu_kernel[(M,)](
        x, out, permutation_map, M, N, BLOCK_N=BLOCK_N,
    )
    return out
