# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Forked from `fla/ops/utils/cumsum.py` in flash-linear-attention v0.5.1
# (https://github.com/fla-org/flash-linear-attention).
#
# Licensed under the MIT license; see the LICENSE file in the repository root.

"""Within-chunk cumulative sum of the scalar log decays.

Only the scalar, non-reversed, variable-length path is provided -- the one the
Gated Delta Product prefill calls.
"""

import torch

from .common import HAVE_TRITON, prepare_chunk_indices, tl, triton


@triton.heuristics(
    {
        'HAS_SCALE': lambda args: args['scale'] is not None,
        'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8]],
    key=['B', 'H', 'BT', 'IS_VARLEN', 'REVERSE'],
)
@triton.jit(do_not_specialize=['T'])
def chunk_local_cumsum_scalar_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    """Cumulative sum within each chunk of a `[B, T, H]` scalar sequence."""
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if HEAD_FIRST:
        p_s = tl.make_block_ptr(s + bos * H + i_h * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + bos * H + i_h * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    else:
        p_s = tl.make_block_ptr(s + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    # [BT]
    b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0)
    if REVERSE:
        b_z = tl.sum(b_s, axis=0)
        b_o = -b_o + b_z[None] + b_s
    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


def chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
    chunk_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cumulative sum of `g` within each chunk of length `chunk_size`.

    Args:
        g: Scalar sequence `[B, T, H]` (or `[B, H, T]` when `head_first`).
        chunk_size: Chunk length; must be a power of two.
        reverse: Accumulate from the end of each chunk instead of the start.
        scale: Optional scale applied to the result.
        cu_seqlens: Sequence boundaries `[N+1]` for variable-length input.
        head_first: Whether `g` is laid out head-major.
        output_dtype: Result dtype; `None` keeps `g`'s dtype.
        chunk_indices: Precomputed chunk descriptors. Derived from `cu_seqlens`
            when omitted, which synchronizes on the device.

    Returns the cumulative sums, shaped like `g`.
    """
    assert HAVE_TRITON, "chunk_local_cumsum requires Triton"
    if head_first:
        B, H, T = g.shape
    else:
        B, T, H = g.shape
    assert chunk_size == 2 ** (chunk_size.bit_length() - 1), "chunk_size must be a power of 2"
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)
    chunk_local_cumsum_scalar_kernel[(NT, B * H)](
        s=g_org,
        o=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        B=B,
        H=H,
        BT=BT,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
    )
    return g
