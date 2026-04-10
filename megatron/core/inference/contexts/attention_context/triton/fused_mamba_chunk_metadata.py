# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Fused Triton kernels for Mamba chunk metadata computation.

Replaces the CPU-synchronizing ``.tolist()`` + Python loop + ``torch.tensor().copy_()``
pattern in ``MambaMetadata.update()`` with GPU-only kernels.

The original code (lines 258-299 of mamba_metadata.py) does:
  1. ``.tolist()`` on cu_seqlens — forces GPU→CPU sync
  2. Python loop computing chunk boundaries, seq_idx, last_chunk_indices
  3. ``torch.tensor(list).copy_()`` — CPU→GPU copies

This module provides two Triton kernels that do the same work entirely on GPU:
  - Kernel 1: Compute per-sequence n_chunks, cumulative offsets, last_chunk_indices
  - Kernel 2: Scatter-write cu_chunk_seqlens and seq_idx_for_varlen per sequence
"""

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    from unittest.mock import MagicMock

    from megatron.core.utils import null_decorator

    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()
    HAVE_TRITON = False


@triton.jit
def _compute_chunk_offsets_kernel(
    # Inputs
    CU_SEQLENS_PTR,
    # Outputs
    CUM_CHUNKS_BUF_PTR,
    LAST_CHUNK_INDICES_PTR,
    CU_CHUNK_SEQLENS_PTR,
    # Scalars
    padded_prefill_count,
    chunk_size,
    # Constexpr
    BLOCK_SIZE: tl.constexpr,
):
    """Phase 1: compute per-sequence chunk counts, cumulative offsets, and last_chunk_indices.

    Single program (grid=(1,)). Vectorized over sequences.

    Outputs:
        cum_chunks_buf: ``[padded_prefill+1]`` — ``cum_chunks[0]=0``, then cumsum of n_chunks.
            Used by kernel 2 to locate each sequence's output range.
        last_chunk_indices: ``[padded_prefill]`` — global index of last chunk per sequence.
        cu_chunk_seqlens[0] = 0.
    """
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < padded_prefill_count

    cu_start = tl.load(CU_SEQLENS_PTR + offsets, mask=mask, other=0)
    cu_end = tl.load(CU_SEQLENS_PTR + offsets + 1, mask=mask, other=0)
    seq_lens = cu_end - cu_start

    # n_chunks = max(1, ceil(seq_len / chunk_size)) for real seqs, 0 for padding
    n_chunks_raw = (seq_lens + chunk_size - 1) // chunk_size
    n_chunks = tl.where(mask, tl.maximum(n_chunks_raw, 1), 0)

    # Cumulative chunk counts → global offsets
    cum_chunks = tl.cumsum(n_chunks, axis=0)

    # Write cum_chunks buffer: [0, cumsum_0, cumsum_1, ...]
    tl.store(CUM_CHUNKS_BUF_PTR, 0)
    tl.store(CUM_CHUNKS_BUF_PTR + offsets + 1, cum_chunks, mask=mask)

    # last_chunk_indices[i] = cum_chunks[i] - 1
    tl.store(LAST_CHUNK_INDICES_PTR + offsets, cum_chunks - 1, mask=mask)

    # cu_chunk_seqlens[0] = 0
    tl.store(CU_CHUNK_SEQLENS_PTR, 0)


@triton.jit
def _scatter_chunk_boundaries_kernel(
    # Inputs
    CU_SEQLENS_PTR,
    CUM_CHUNKS_BUF_PTR,
    # Outputs
    CU_CHUNK_SEQLENS_PTR,
    SEQ_IDX_FOR_VARLEN_PTR,
    # Scalars
    chunk_size,
    # Constexpr
    MAX_CHUNKS_PER_SEQ: tl.constexpr,
):
    """Phase 2: scatter-write chunk boundaries and seq_idx for one sequence.

    Grid: ``(padded_prefill_count,)`` — one program per sequence.

    For sequence ``i`` with token range ``[start, end)`` and ``n_chunks`` chunks,
    writes:
        cu_chunk_seqlens[global_offset + k + 1] = min(start + (k+1)*chunk_size, end)
        seq_idx_for_varlen[global_offset + k] = i

    where ``global_offset = cum_chunks[i]`` and ``k`` ranges over ``[0, n_chunks)``.
    """
    seq_i = tl.program_id(0)

    start = tl.load(CU_SEQLENS_PTR + seq_i)
    end = tl.load(CU_SEQLENS_PTR + seq_i + 1)
    global_offset = tl.load(CUM_CHUNKS_BUF_PTR + seq_i)
    global_end = tl.load(CUM_CHUNKS_BUF_PTR + seq_i + 1)
    n_chunks_for_seq = global_end - global_offset

    k = tl.arange(0, MAX_CHUNKS_PER_SEQ)
    chunk_mask = k < n_chunks_for_seq

    boundary = tl.minimum(start + (k + 1) * chunk_size, end)

    tl.store(CU_CHUNK_SEQLENS_PTR + global_offset + k + 1, boundary, mask=chunk_mask)
    tl.store(SEQ_IDX_FOR_VARLEN_PTR + global_offset + k, seq_i, mask=chunk_mask)


@triton.jit
def _scatter_conv_metadata_kernel(
    # Inputs
    CU_SEQLENS_PTR,
    # Outputs
    CONV_SEQ_IDX_PTR,
    CONV_SEQ_START_PTR,
    # Constexpr
    MAX_TOKENS_PER_SEQ: tl.constexpr,
):
    """Write conv1d per-token metadata for one sequence.

    Grid: ``(real_prefill_count,)`` — one program per real prefill sequence.

    For sequence ``i`` with token range ``[start, end)``, writes:
        conv_seq_idx[start:end] = i
        conv_seq_start[start:end] = start

    Replaces two ``torch.repeat_interleave`` calls (each does cumsum + scatter
    internally) with a single kernel that writes both buffers.
    """
    seq_i = tl.program_id(0)
    start = tl.load(CU_SEQLENS_PTR + seq_i)
    end = tl.load(CU_SEQLENS_PTR + seq_i + 1)
    seq_len = end - start

    k = tl.arange(0, MAX_TOKENS_PER_SEQ)
    mask = k < seq_len
    idx = start + k

    tl.store(CONV_SEQ_IDX_PTR + idx, seq_i, mask=mask)
    tl.store(CONV_SEQ_START_PTR + idx, start, mask=mask)


def fused_conv_metadata(
    cu_seqlens_buf: torch.Tensor,
    conv_seq_idx_buf: torch.Tensor,
    conv_seq_start_buf: torch.Tensor,
    real_prefill_count: int,
    padded_token_count: int,
) -> None:
    """Compute conv1d per-token metadata on GPU, replacing repeat_interleave.

    Args:
        cu_seqlens_buf: ``[>=real_prefill_count+1]`` int32 — cumulative sequence lengths.
        conv_seq_idx_buf: ``[>=padded_token_count]`` int32 — output: per-token sequence ID.
        conv_seq_start_buf: ``[>=padded_token_count]`` int32 — output: per-token sequence start.
        real_prefill_count: Number of real prefill sequences.
        padded_token_count: Total padded token count.
    """
    # Pre-fill with padding value (0) — kernel overwrites real positions.
    conv_seq_idx_buf[:padded_token_count] = 0
    conv_seq_start_buf[:padded_token_count] = 0

    if real_prefill_count == 0:
        return

    # Conservative upper bound on tokens per sequence.
    max_tokens_per_seq = padded_token_count
    MAX_TOKENS_PER_SEQ = triton.next_power_of_2(max(max_tokens_per_seq, 1))

    _scatter_conv_metadata_kernel[(real_prefill_count,)](
        cu_seqlens_buf,
        conv_seq_idx_buf,
        conv_seq_start_buf,
        MAX_TOKENS_PER_SEQ=MAX_TOKENS_PER_SEQ,
    )


def fused_mamba_chunk_metadata(
    cu_seqlens_buf: torch.Tensor,
    cum_chunks_buf: torch.Tensor,
    cu_chunk_seqlens_buf: torch.Tensor,
    last_chunk_indices_buf: torch.Tensor,
    seq_idx_for_varlen_buf: torch.Tensor,
    padded_prefill_count: int,
    chunk_size: int,
    padded_max_chunks: int,
    padded_token_count: int,
) -> None:
    """Compute mamba chunk metadata entirely on GPU.

    Replaces the ``.tolist()`` + Python loop + ``torch.tensor().copy_()`` pattern.

    Args:
        cu_seqlens_buf: ``[>=padded_prefill_count+1]`` int32 — cumulative sequence lengths.
        cum_chunks_buf: ``[>=padded_prefill_count+1]`` int32 — scratch buffer for cumulative
            chunk counts (written by kernel 1, read by kernel 2).
        cu_chunk_seqlens_buf: ``[>=padded_max_chunks+1]`` int32 — output chunk boundaries.
        last_chunk_indices_buf: ``[>=padded_prefill_count]`` int32 — output last chunk index
            per sequence.
        seq_idx_for_varlen_buf: ``[>=padded_max_chunks]`` int32 — output sequence ID per chunk.
        padded_prefill_count: Number of prefill sequences (real + padding).
        chunk_size: Mamba chunk size (e.g. 128).
        padded_max_chunks: Total padded chunk count for CUDA graph compatibility.
        padded_token_count: Total padded token count.
    """
    if padded_prefill_count == 0:
        cu_chunk_seqlens_buf[0] = 0
        return

    # Pre-fill output buffers with padding values.
    # Kernel 2 overwrites real positions; remaining positions keep these defaults.
    # cu_chunk_seqlens padding = last boundary = cu_seqlens[padded_prefill_count]
    # Use slice assignment (not .fill_/.item()) to avoid a CPU-GPU sync.
    cu_chunk_seqlens_buf[1 : padded_max_chunks + 1] = cu_seqlens_buf[padded_prefill_count]
    seq_idx_for_varlen_buf[:padded_max_chunks] = 0

    # Kernel 1: compute cumulative chunk offsets and last_chunk_indices
    BLOCK_SEQ = triton.next_power_of_2(padded_prefill_count)
    _compute_chunk_offsets_kernel[(1,)](
        cu_seqlens_buf,
        cum_chunks_buf,
        last_chunk_indices_buf,
        cu_chunk_seqlens_buf,
        padded_prefill_count,
        chunk_size,
        BLOCK_SIZE=BLOCK_SEQ,
    )

    # Kernel 2: scatter chunk boundaries and seq_idx per sequence
    max_chunks_per_seq = padded_token_count // chunk_size + 1
    MAX_CHUNKS_PER_SEQ = triton.next_power_of_2(max(max_chunks_per_seq, 1))
    _scatter_chunk_boundaries_kernel[(padded_prefill_count,)](
        cu_seqlens_buf,
        cum_chunks_buf,
        cu_chunk_seqlens_buf,
        seq_idx_for_varlen_buf,
        chunk_size,
        MAX_CHUNKS_PER_SEQ=MAX_CHUNKS_PER_SEQ,
    )
