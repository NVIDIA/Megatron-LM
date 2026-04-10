# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
def _fused_mha_metadata_update_kernel(
    # Inputs (1D, int32)
    QUERY_LENGTHS_PTR,
    KV_LENGTH_OFFSETS_PTR,
    # Outputs (1D, int32)
    QUERY_LENGTHS_BUF_PTR,
    CU_QUERY_SEQ_LENGTHS_BUF_PTR,
    KV_SEQ_LENGTHS_BUF_PTR,
    CU_KV_SEQ_LENGTHS_BUF_PTR,
    # Scalar outputs (max values, written as single-element tensors)
    MAX_SEQLEN_Q_PTR,
    MAX_SEQLEN_K_PTR,
    # Runtime dimensions
    real_bs,
    padded_bs,
    # Compile-time block size (next power of 2 of padded_bs)
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel computing copy+pad, add, cumsum, and max for MHA metadata buffers.

    Runs as a single program (grid=(1,)). The entire batch fits in one block
    since inference batch sizes are typically ≤ a few thousand.

    The zero-padding trick: loading beyond ``real_bs`` with ``other=0`` means
    ``tl.cumsum`` naturally propagates the last real prefix-sum value through
    the padded positions — exactly matching the cumulative-pad semantics of
    ``tensor_copy_and_pad(..., is_cumulative_tensor=True)``.
    """
    offsets = tl.arange(0, BLOCK_SIZE)
    real_mask = offsets < real_bs
    padded_mask = offsets < padded_bs

    # --- Load inputs (zero-padded beyond real_bs) ---
    query_lengths = tl.load(QUERY_LENGTHS_PTR + offsets, mask=real_mask, other=0)
    kv_offsets = tl.load(KV_LENGTH_OFFSETS_PTR + offsets, mask=real_mask, other=0)

    # --- Derived values ---
    kv_seq_lengths = kv_offsets + query_lengths

    # --- Prefix sums (zeros beyond real_bs propagate last value) ---
    cu_query = tl.cumsum(query_lengths, axis=0)
    cu_kv = tl.cumsum(kv_seq_lengths, axis=0)

    # --- Max values over real entries (for NonGraphedMHAMetadata) ---
    max_q = tl.max(query_lengths, axis=0)
    max_k = tl.max(kv_seq_lengths, axis=0)
    tl.store(MAX_SEQLEN_Q_PTR, max_q)
    tl.store(MAX_SEQLEN_K_PTR, max_k)

    # --- Write query_lengths_buf  [0:padded_bs], pad=0 ---
    tl.store(QUERY_LENGTHS_BUF_PTR + offsets, query_lengths, mask=padded_mask)

    # --- Write kv_seq_lengths_buf [0:padded_bs], pad=0 ---
    tl.store(KV_SEQ_LENGTHS_BUF_PTR + offsets, kv_seq_lengths, mask=padded_mask)

    # --- Write cu_query_seq_lengths_buf [0]=0, [1:padded_bs+1]=cumsum ---
    tl.store(CU_QUERY_SEQ_LENGTHS_BUF_PTR, 0)
    tl.store(CU_QUERY_SEQ_LENGTHS_BUF_PTR + offsets + 1, cu_query, mask=padded_mask)

    # --- Write cu_kv_seq_lengths_buf [0]=0, [1:padded_bs+1]=cumsum ---
    tl.store(CU_KV_SEQ_LENGTHS_BUF_PTR, 0)
    tl.store(CU_KV_SEQ_LENGTHS_BUF_PTR + offsets + 1, cu_kv, mask=padded_mask)


def fused_mha_metadata_update(
    query_lengths: torch.Tensor,
    kv_length_offsets: torch.Tensor,
    query_lengths_buf: torch.Tensor,
    cu_query_seq_lengths_buf: torch.Tensor,
    kv_seq_lengths_buf: torch.Tensor,
    cu_kv_seq_lengths_buf: torch.Tensor,
    max_seqlen_q_buf: torch.Tensor,
    max_seqlen_k_buf: torch.Tensor,
    real_batch_size: int,
    padded_batch_size: int,
) -> None:
    """Launch the fused MHA metadata update kernel.

    Args:
        query_lengths: ``[>=real_batch_size]`` int32 - per-request query lengths.
        kv_length_offsets: ``[>=real_batch_size]`` int32 - per-request KV offsets.
        query_lengths_buf: ``[>=padded_batch_size]`` int32 - output buffer.
        cu_query_seq_lengths_buf: ``[>=padded_batch_size+1]`` int32 - output buffer.
        kv_seq_lengths_buf: ``[>=padded_batch_size]`` int32 - output buffer.
        cu_kv_seq_lengths_buf: ``[>=padded_batch_size+1]`` int32 - output buffer.
        max_seqlen_q_buf: ``[1]`` int32 - output: max query length across real requests.
        max_seqlen_k_buf: ``[1]`` int32 - output: max kv length across real requests.
        real_batch_size: Number of real requests.
        padded_batch_size: Padded request count (≥ real_batch_size).
    """
    if padded_batch_size == 0:
        cu_query_seq_lengths_buf[0] = 0
        cu_kv_seq_lengths_buf[0] = 0
        max_seqlen_q_buf[0] = 0
        max_seqlen_k_buf[0] = 0
        return

    BLOCK_SIZE = triton.next_power_of_2(padded_batch_size)

    _fused_mha_metadata_update_kernel[(1,)](
        query_lengths,
        kv_length_offsets,
        query_lengths_buf,
        cu_query_seq_lengths_buf,
        kv_seq_lengths_buf,
        cu_kv_seq_lengths_buf,
        max_seqlen_q_buf,
        max_seqlen_k_buf,
        real_batch_size,
        padded_batch_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
