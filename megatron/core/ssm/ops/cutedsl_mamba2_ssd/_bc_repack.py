# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Tiled B/C repack (token-major -> dense chunk-major) for the CuteDSL SSD
front-end.

The divisible path feeds the kernel dense chunk-major ``(1, G, N_pad, TC, L)``
B/C workspace buffers built from the token-packed ``(T, G, N)`` inputs. Doing
that with ``B_d[:, :, :N].copy_(B.as_strided(...))`` is an (n, l)-plane
TRANSPOSE per (group, chunk): torch's generic strided-copy kernel is coalesced
on only one side and runs at ~1 TB/s (~250 us for B+C at 32K tokens on GB200 —
comparable to the SSD kernel itself). This kernel stages (BLOCK_L, BLOCK_N)
tiles through registers/smem (``tl.trans``) so BOTH the token-major loads
(contiguous along n) and the chunk-major stores (contiguous along l) are
coalesced, and handles B and C in one launch (grid axis 1).

Rows ``n >= N`` of the ``N_pad``-padded destination are never written: the
workspace buffers are zero-initialized once and the padding must stay zero,
exactly like the ``[:, :, :N]`` slice of the copy_ it replaces.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _bc_repack_kernel(
    b_src_ptr,
    c_src_ptr,
    b_dst_ptr,
    c_dst_ptr,
    N,
    TC,
    stride_src_token,
    stride_src_g,
    stride_dst_g,
    stride_dst_n,
    stride_dst_c,
    L: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Grid: axis 0 = (g, chunk, l-tile), axis 1 = 0 for B / 1 for C.
    pid = tl.program_id(0)
    num_l_tiles: tl.constexpr = L // BLOCK_L
    l_tile = pid % num_l_tiles
    tmp = pid // num_l_tiles
    c = tmp % TC
    g = tmp // TC
    if tl.program_id(1) == 0:
        src_ptr = b_src_ptr
        dst_ptr = b_dst_ptr
    else:
        src_ptr = c_src_ptr
        dst_ptr = c_dst_ptr

    offs_l = l_tile * BLOCK_L + tl.arange(0, BLOCK_L)  # L % BLOCK_L == 0: no l mask
    offs_n = tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    # Load (BLOCK_L, BLOCK_N): contiguous along n (stride 1) -> coalesced.
    token = c * L + offs_l
    src_off = token[:, None] * stride_src_token + g * stride_src_g + offs_n[None, :]
    tile = tl.load(src_ptr + src_off, mask=n_mask[None, :], other=0.0)

    # Store transposed (BLOCK_N, BLOCK_L): contiguous along l -> coalesced.
    dst_off = g * stride_dst_g + offs_n[:, None] * stride_dst_n + c * stride_dst_c + offs_l[None, :]
    tl.store(dst_ptr + dst_off, tl.trans(tile), mask=n_mask[:, None])


def repack_bc_chunk_major(
    B: torch.Tensor,
    C: torch.Tensor,
    B_dst: torch.Tensor,
    C_dst: torch.Tensor,
    N: int,
    total_chunks: int,
    kernel_chunk_size: int,
) -> None:
    """Repack token-major B and C into the dense chunk-major workspace buffers.

    Equivalent to (for each of B/C)::

        dst[:, :, :N].copy_(src.as_strided((1, G, N, TC, L), (0, N, 1, L*G*N, G*N)))

    but with coalesced loads AND stores, and both tensors in one launch.

    Args:
        B: Token-packed ``(T, G, N)`` input, innermost dim contiguous.
        C: Same shape/layout as ``B``.
        B_dst: Dense ``(1, G, N_pad, TC, L)`` workspace buffer, ``L`` contiguous.
        C_dst: Same shape/layout as ``B_dst``.
        N: Real state dim (``N_pad`` rows beyond it are left untouched).
        total_chunks: ``TC`` (== real tokens / kernel_chunk_size).
        kernel_chunk_size: The kernel chunk size ``L`` (compile-time constant).
    """
    G = B.shape[1]
    L = kernel_chunk_size
    assert B.stride(2) == 1 and C.stride(2) == 1, "B/C must be n-contiguous"
    assert B_dst.stride(-1) == 1 and C_dst.stride(-1) == 1, "dst must be L-contiguous"
    assert B.stride(1) == C.stride(1) and B.stride(0) == C.stride(0), "B/C layouts must match"
    assert B_dst.stride(1) == C_dst.stride(1), "dst layouts must match"

    BLOCK_L = 64 if L % 64 == 0 else L
    BLOCK_N = max(16, triton.next_power_of_2(N))
    grid = (G * total_chunks * (L // BLOCK_L), 2)
    _bc_repack_kernel[grid](
        B,
        C,
        B_dst,
        C_dst,
        N,
        total_chunks,
        B.stride(0),
        B.stride(1),
        B_dst.stride(1),
        B_dst.stride(2),
        B_dst.stride(3),
        L=L,
        BLOCK_L=BLOCK_L,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )
