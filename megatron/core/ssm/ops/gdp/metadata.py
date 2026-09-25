# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Chunk descriptors consumed by the forked Gated Delta Product prefill kernels.

Upstream FLA derives these inside the op with `prepare_chunk_indices`, which
synchronizes on the device (`.tolist()`) and returns a data-dependent shape.
Both are fatal under CUDA-graph capture, so they are built here instead --
once per step, on the host, outside the graph -- and padded to a size that
depends only on the padded batch shape. The kernels then read fixed-size,
fixed-address buffers.

Two chunkings are needed. `chunk_indices` describes the token stream as the
queries see it; `chunk_indices_dp` describes the Householder-expanded stream
that the keys, values and betas live on, whose sequences are `M` times longer.
The second is not a rescaling of the first: `ceil(L*M/64) != M*ceil(L/64)`
whenever `L` is not a multiple of the chunk size.
"""

from typing import List, Tuple

from .common import CHUNK_SIZE


def max_gdp_chunk_counts(
    max_tokens: int, max_requests: int, num_householder: int
) -> Tuple[int, int]:
    """Worst-case chunk counts, used to size the persistent buffers.

    Returns `(max_chunks, max_chunks_dp)` for the unexpanded and the
    Householder-expanded stream respectively. The `+ max_requests` term covers
    the partial trailing chunk each sequence may own.
    """
    max_chunks = max_tokens // CHUNK_SIZE + max_requests
    max_chunks_dp = (max_tokens * num_householder) // CHUNK_SIZE + max_requests
    return max_chunks, max_chunks_dp


def build_gdp_chunk_descriptors(
    cu_seqlens: List[int], prefill_count: int, num_householder: int, padded_token_count: int
) -> Tuple[List[int], List[int], List[int], int, int]:
    """Build the padded chunk descriptors for one prefill step.

    Args:
        cu_seqlens: `prefill_count + 1` cumulative sequence lengths, covering
            real and padding requests (padding requests are zero-length).
        prefill_count: Padded number of prefill requests.
        num_householder: Number of Householder copies `M`.
        padded_token_count: Padded token count, which fixes the buffer sizes.

    Returns:
        `(chunk_indices, chunk_indices_dp, chunk_offsets, num_chunks,
        num_chunks_dp)`. The two `chunk_indices` lists are flattened
        `(sequence, chunk-within-sequence)` pairs padded to their fixed
        lengths; `chunk_offsets` is the per-sequence prefix sum of unexpanded
        chunk counts.
    """
    chunk_indices: List[int] = []
    chunk_indices_dp: List[int] = []
    chunk_offsets: List[int] = [0]

    for i in range(prefill_count):
        seq_len = cu_seqlens[i + 1] - cu_seqlens[i]
        n_chunks = (seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE
        for j in range(n_chunks):
            chunk_indices.extend((i, j))
        chunk_offsets.append(chunk_offsets[-1] + n_chunks)

        n_chunks_dp = (seq_len * num_householder + CHUNK_SIZE - 1) // CHUNK_SIZE
        for j in range(n_chunks_dp):
            chunk_indices_dp.extend((i, j))

    num_chunks, num_chunks_dp = max_gdp_chunk_counts(
        padded_token_count, prefill_count, num_householder
    )
    assert len(chunk_indices) // 2 <= num_chunks, "GDP chunk descriptor overflow"
    assert len(chunk_indices_dp) // 2 <= num_chunks_dp, "GDP expanded descriptor overflow"

    # Pad with chunks that are addressed past the end of sequence 0. Every kernel
    # reads and writes through block pointers with boundary checks, so an
    # out-of-range chunk loads zeros and stores nothing: padding programs run but
    # have no effect, which is what keeps the grid size constant.
    pad_chunk = padded_token_count // CHUNK_SIZE + 1
    while len(chunk_indices) // 2 < num_chunks:
        chunk_indices.extend((0, pad_chunk))
    while len(chunk_indices_dp) // 2 < num_chunks_dp:
        chunk_indices_dp.extend((0, pad_chunk * num_householder + 1))

    return chunk_indices, chunk_indices_dp, chunk_offsets, num_chunks, num_chunks_dp
