# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
    n_valid_tokens,
    chunk_token_base_ptr,
    chunk_valid_start_ptr,
    chunk_valid_end_ptr,
    stride_src_token,
    stride_src_g,
    stride_dst_g,
    stride_dst_n,
    stride_dst_c,
    RAGGED: tl.constexpr,
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
    # Lanes past the last real token (the pad tail of a ragged final chunk) read
    # as 0: they contribute nothing anyway (their delta is 0), and writing zeros
    # keeps the reused workspace free of stale values.
    if RAGGED:
        # General ragged: per-workspace-chunk token base + real-token window.
        token = tl.load(chunk_token_base_ptr + c) + offs_l
        tok_mask = (token >= tl.load(chunk_valid_start_ptr + c)) & (
            token < tl.load(chunk_valid_end_ptr + c)
        )
    else:
        token = c * L + offs_l
        tok_mask = token < n_valid_tokens
    src_off = token[:, None] * stride_src_token + g * stride_src_g + offs_n[None, :]
    tile = tl.load(src_ptr + src_off, mask=n_mask[None, :] & tok_mask[:, None], other=0.0)

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
    n_valid_tokens: int | None = None,
    ragged_chunks: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
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
        n_valid_tokens: Number of REAL tokens; rows at or beyond it (the pad
            tail of a ragged final chunk) are written as zeros instead of being
            read. Defaults to ``total_chunks * kernel_chunk_size``.
    """
    G = B.shape[1]
    L = kernel_chunk_size
    assert B.stride(2) == 1 and C.stride(2) == 1, "B/C must be n-contiguous"
    assert B_dst.stride(-1) == 1 and C_dst.stride(-1) == 1, "dst must be L-contiguous"
    assert B.stride(1) == C.stride(1) and B.stride(0) == C.stride(0), "B/C layouts must match"
    assert B_dst.stride(1) == C_dst.stride(1), "dst layouts must match"

    if n_valid_tokens is None:
        n_valid_tokens = total_chunks * L
    ragged = ragged_chunks is not None
    base_p, lo_p, hi_p = ragged_chunks if ragged else (B, B, B)
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
        n_valid_tokens,
        base_p,
        lo_p,
        hi_p,
        B.stride(0),
        B.stride(1),
        B_dst.stride(1),
        B_dst.stride(2),
        B_dst.stride(3),
        RAGGED=ragged,
        L=L,
        BLOCK_L=BLOCK_L,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )
