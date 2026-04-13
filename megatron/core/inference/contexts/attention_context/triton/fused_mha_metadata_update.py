# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import torch

# HAVE_TRITON gates whether mha_metadata.py uses the optimized fused path
# or falls back to the original tensor_copy_and_pad approach.
try:
    import triton  # noqa: F401

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False


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
    max_batch_size: int = 0,
    compute_max: bool = True,
) -> None:
    """Compute all MHA metadata buffers using pure PyTorch ops (fully async, no CPU-GPU syncs).

    Uses zero-padded cumsum trick: writing real values into a zero buffer then
    calling cumsum naturally propagates the last prefix-sum value through
    the padded positions — matching the cumulative-pad semantics of
    ``tensor_copy_and_pad(..., is_cumulative_tensor=True)`` without the
    CPU-GPU sync that ``tensor_copy_and_pad`` requires to read the last value.

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
        max_batch_size: Unused (kept for API compat).
        compute_max: If True, compute max_seqlen values into the output buffers.
            GraphedMHAMetadata overrides these values, so it passes False to skip.
    """
    if padded_batch_size == 0:
        cu_query_seq_lengths_buf[0] = 0
        cu_kv_seq_lengths_buf[0] = 0
        max_seqlen_q_buf[0] = 0
        max_seqlen_k_buf[0] = 0
        return

    rbs = real_batch_size
    pbs = padded_batch_size

    # --- query_lengths_buf: copy real, zero-pad rest ---
    query_lengths_buf[:rbs] = query_lengths[:rbs]
    if pbs > rbs:
        query_lengths_buf[rbs:pbs] = 0

    # --- kv_seq_lengths = kv_offsets + query_lengths, zero-padded ---
    kv_seq_lengths_buf[:rbs] = kv_length_offsets[:rbs] + query_lengths[:rbs]
    if pbs > rbs:
        kv_seq_lengths_buf[rbs:pbs] = 0

    # --- cumsum on the padded buffer: zeros propagate last real value ---
    cu_query_seq_lengths_buf[0] = 0
    torch.cumsum(query_lengths_buf[:pbs], dim=0, out=cu_query_seq_lengths_buf[1 : pbs + 1])

    cu_kv_seq_lengths_buf[0] = 0
    torch.cumsum(kv_seq_lengths_buf[:pbs], dim=0, out=cu_kv_seq_lengths_buf[1 : pbs + 1])

    # --- max values (GPU-only, no .item() sync) ---
    # Only needed for NonGraphedMHAMetadata; GraphedMHAMetadata overrides these.
    if compute_max:
        max_seqlen_q_buf[0] = query_lengths_buf[:rbs].max()
        max_seqlen_k_buf[0] = kv_seq_lengths_buf[:rbs].max()
