# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import Dict, Optional

import torch
from torch import Tensor

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
def _move_or_swap_int32(ptr, src, dst, IS_SWAP: tl.constexpr):
    """Move or swap a single int32 element between src and dst indices."""
    src_val = tl.load(ptr + src)
    if IS_SWAP:
        dst_val = tl.load(ptr + dst)
        tl.store(ptr + src, dst_val)
    tl.store(ptr + dst, src_val)


@triton.jit
def _move_or_swap_int64(ptr, src, dst, IS_SWAP: tl.constexpr):
    """Move or swap a single int64 element between src and dst indices."""
    src_val = tl.load(ptr + src).to(tl.int64)
    if IS_SWAP:
        dst_val = tl.load(ptr + dst).to(tl.int64)
        tl.store(ptr + src, dst_val)
    tl.store(ptr + dst, src_val)


@triton.jit
def _move_or_swap_uint8(ptr, src, dst, IS_SWAP: tl.constexpr):
    """Move or swap a single uint8 element between src and dst indices."""
    src_val = tl.load(ptr + src).to(tl.uint8)
    if IS_SWAP:
        dst_val = tl.load(ptr + dst).to(tl.uint8)
        tl.store(ptr + src, dst_val)
    tl.store(ptr + dst, src_val)


@triton.jit
def _reorder_bookkeeping_kernel(
    src_idxs_ptr,
    dst_idxs_ptr,
    # 8 core 1D int32 request tensors
    kv_length_offsets_ptr,
    prefill_status_ptr,
    query_lengths_ptr,
    output_lengths_ptr,
    request_ids_ptr,
    kv_block_counts_ptr,
    last_kv_block_id_ptr,
    last_kv_block_offset_ptr,
    # 2D int32 request tensor
    kv_block_ids_ptr,
    kv_block_ids_stride: tl.int32,
    # Token tensors
    next_tokens_ptr,
    spec_tokens_ptr,
    spec_tokens_stride: tl.int32,
    # 7 metadata tensors (explicit pointers)
    # 4-byte metadata (float32 loaded as int32, int32 native)
    meta_temperature_ptr,
    meta_top_k_ptr,
    meta_top_p_ptr,
    meta_top_n_logprobs_ptr,
    # 8-byte metadata
    meta_termination_id_ptr,
    # 1-byte metadata (bool loaded as uint8)
    meta_return_log_probs_ptr,
    meta_skip_prompt_log_probs_ptr,
    # Mamba
    mamba_state_idx_ptr,
    # Guard pointer (GPU scalar count for sync-free launches)
    num_valid_ptr,
    # Runtime scalar
    max_kv_blocks: tl.int32,
    # Compile-time constants
    IS_SWAP: tl.constexpr,
    HAS_NEXT_TOKENS: tl.constexpr,
    HAS_SPEC_TOKENS: tl.constexpr,
    IS_HYBRID: tl.constexpr,
    NUM_SPEC: tl.constexpr,
    MAX_KV_BLOCKS: tl.constexpr,
    KV_TILE_SIZE: tl.constexpr,
    USE_GUARD: tl.constexpr,
):
    """Fused move/swap kernel for all bookkeeping tensors.

    Grid: (num_indices,) or (max_grid,) when USE_GUARD=True.
    Each program handles one src/dst index pair across ALL tensors.
    """
    idx = tl.program_id(0)
    if USE_GUARD:
        num_valid = tl.load(num_valid_ptr)
        if idx >= num_valid:
            return
    src = tl.load(src_idxs_ptr + idx).to(tl.int32)
    dst = tl.load(dst_idxs_ptr + idx).to(tl.int32)

    # --- 8 core 1D int32 tensors ---
    _move_or_swap_int32(kv_length_offsets_ptr, src, dst, IS_SWAP)
    _move_or_swap_int32(prefill_status_ptr, src, dst, IS_SWAP)
    _move_or_swap_int32(query_lengths_ptr, src, dst, IS_SWAP)
    _move_or_swap_int32(output_lengths_ptr, src, dst, IS_SWAP)
    _move_or_swap_int32(request_ids_ptr, src, dst, IS_SWAP)
    _move_or_swap_int32(kv_block_counts_ptr, src, dst, IS_SWAP)
    _move_or_swap_int32(last_kv_block_id_ptr, src, dst, IS_SWAP)
    _move_or_swap_int32(last_kv_block_offset_ptr, src, dst, IS_SWAP)

    # --- 4-byte metadata (float32 reinterpreted as int32 for bitwise copy) ---
    _move_or_swap_int32(meta_temperature_ptr, src, dst, IS_SWAP)
    _move_or_swap_int32(meta_top_k_ptr, src, dst, IS_SWAP)
    _move_or_swap_int32(meta_top_p_ptr, src, dst, IS_SWAP)
    _move_or_swap_int32(meta_top_n_logprobs_ptr, src, dst, IS_SWAP)

    # --- 8-byte metadata (int64) ---
    _move_or_swap_int64(meta_termination_id_ptr, src, dst, IS_SWAP)

    # --- 1-byte metadata (bool as uint8) ---
    _move_or_swap_uint8(meta_return_log_probs_ptr, src, dst, IS_SWAP)
    _move_or_swap_uint8(meta_skip_prompt_log_probs_ptr, src, dst, IS_SWAP)

    # --- Mamba state idx (conditional) ---
    if IS_HYBRID:
        _move_or_swap_int32(mamba_state_idx_ptr, src, dst, IS_SWAP)

    # --- next_tokens (int64, conditional) ---
    if HAS_NEXT_TOKENS:
        _move_or_swap_int64(next_tokens_ptr, src, dst, IS_SWAP)

    # --- new_speculative_tokens (int64, 2D, conditional) ---
    if HAS_SPEC_TOKENS:
        for d in tl.static_range(NUM_SPEC):
            base = d * spec_tokens_stride
            _move_or_swap_int64(spec_tokens_ptr + base, src, dst, IS_SWAP)

    # --- 2D request_to_kv_block_ids (tiled loop) ---
    src_row = src * kv_block_ids_stride
    dst_row = dst * kv_block_ids_stride
    for offset in tl.static_range(0, MAX_KV_BLOCKS, KV_TILE_SIZE):
        cols = tl.arange(0, KV_TILE_SIZE)
        mask = (offset + cols) < max_kv_blocks
        src_data = tl.load(kv_block_ids_ptr + src_row + offset + cols, mask=mask)
        if IS_SWAP:
            dst_data = tl.load(kv_block_ids_ptr + dst_row + offset + cols, mask=mask)
            tl.store(kv_block_ids_ptr + src_row + offset + cols, dst_data, mask=mask)
        tl.store(kv_block_ids_ptr + dst_row + offset + cols, src_data, mask=mask)


def _launch_reorder_kernel(
    is_swap: bool,
    src_idxs: Tensor,
    dst_idxs: Tensor,
    # Core request tensors
    request_kv_length_offsets: Tensor,
    request_in_prefill_status_tensor: Tensor,
    request_query_lengths: Tensor,
    request_output_lengths: Tensor,
    request_ids: Tensor,
    request_kv_block_counts: Tensor,
    request_last_kv_block_id: Tensor,
    request_last_kv_block_offset: Tensor,
    request_to_kv_block_ids: Tensor,
    # Token tensors
    next_tokens: Optional[Tensor],
    new_speculative_tokens: Optional[Tensor],
    # Metadata dict
    request_metadata: Dict[str, Tensor],
    # Mamba
    mamba_state_idx: Optional[Tensor],
    is_hybrid_model: bool,
    # Config
    num_speculative_tokens: int,
    max_kv_block_count: int,
    # Sync-free guard: GPU scalar count pointer + max grid size
    num_valid_ptr: Optional[Tensor] = None,
    max_grid: int = 0,
) -> None:
    # Ensure index tensors are on GPU (callers sometimes pass CPU tensors)
    if src_idxs.device.type != "cuda":
        src_idxs = src_idxs.cuda()
    if dst_idxs.device.type != "cuda":
        dst_idxs = dst_idxs.cuda()

    use_guard = num_valid_ptr is not None
    if use_guard:
        num_indices = max_grid
    else:
        num_indices = src_idxs.shape[0]
    if num_indices == 0:
        return

    has_next_tokens = next_tokens is not None
    has_spec_tokens = new_speculative_tokens is not None
    num_spec = num_speculative_tokens if has_spec_tokens else 1

    # Dummy pointers for disabled paths
    dummy_int64 = src_idxs
    dummy_int32 = request_ids

    next_tokens_ptr = next_tokens if has_next_tokens else dummy_int64
    spec_tokens_ptr = new_speculative_tokens if has_spec_tokens else dummy_int64
    spec_stride = new_speculative_tokens.stride(0) if has_spec_tokens else 0
    mamba_ptr = mamba_state_idx if is_hybrid_model else dummy_int32

    # Round up max_kv_block_count to next multiple of tile size for constexpr loop
    kv_tile_size = 128
    max_kv_blocks_rounded = ((max_kv_block_count + kv_tile_size - 1) // kv_tile_size) * kv_tile_size

    grid = (num_indices,)
    guard_ptr = num_valid_ptr if use_guard else src_idxs  # dummy when not guarded

    _reorder_bookkeeping_kernel[grid](
        src_idxs,
        dst_idxs,
        # 8 core tensors
        request_kv_length_offsets,
        request_in_prefill_status_tensor,
        request_query_lengths,
        request_output_lengths,
        request_ids,
        request_kv_block_counts,
        request_last_kv_block_id,
        request_last_kv_block_offset,
        # 2D tensor
        request_to_kv_block_ids,
        request_to_kv_block_ids.stride(0),
        # Tokens
        next_tokens_ptr,
        spec_tokens_ptr,
        spec_stride,
        # 7 metadata (explicit pointers, bitwise reinterpret for float32/bool)
        request_metadata["temperature"].view(torch.int32),
        request_metadata["top_k"],
        request_metadata["top_p"].view(torch.int32),
        request_metadata["top_n_logprobs"],
        request_metadata["termination_id"],
        request_metadata["return_log_probs"].view(torch.uint8),
        request_metadata["skip_prompt_log_probs"].view(torch.uint8),
        # Mamba
        mamba_ptr,
        # Guard
        guard_ptr,
        # Runtime scalar
        max_kv_block_count,
        # Constexprs
        IS_SWAP=is_swap,
        HAS_NEXT_TOKENS=has_next_tokens,
        HAS_SPEC_TOKENS=has_spec_tokens,
        IS_HYBRID=is_hybrid_model,
        NUM_SPEC=num_spec,
        MAX_KV_BLOCKS=max_kv_blocks_rounded,
        KV_TILE_SIZE=kv_tile_size,
        USE_GUARD=use_guard,
    )


def triton_move_bookkeeping(
    src_idxs: Tensor,
    dst_idxs: Tensor,
    request_kv_length_offsets: Tensor,
    request_in_prefill_status_tensor: Tensor,
    request_query_lengths: Tensor,
    request_output_lengths: Tensor,
    request_ids: Tensor,
    request_kv_block_counts: Tensor,
    request_last_kv_block_id: Tensor,
    request_last_kv_block_offset: Tensor,
    request_to_kv_block_ids: Tensor,
    next_tokens: Tensor,
    new_speculative_tokens: Optional[Tensor],
    request_metadata: Dict[str, Tensor],
    mamba_state_idx: Optional[Tensor],
    is_hybrid_model: bool,
    num_speculative_tokens: int,
    max_kv_block_count: int,
    num_valid_ptr: Optional[Tensor] = None,
    max_grid: int = 0,
) -> None:
    """Fused move: tensor[dst_idxs] = tensor[src_idxs] for all bookkeeping tensors."""
    _launch_reorder_kernel(
        is_swap=False,
        src_idxs=src_idxs,
        dst_idxs=dst_idxs,
        request_kv_length_offsets=request_kv_length_offsets,
        request_in_prefill_status_tensor=request_in_prefill_status_tensor,
        request_query_lengths=request_query_lengths,
        request_output_lengths=request_output_lengths,
        request_ids=request_ids,
        request_kv_block_counts=request_kv_block_counts,
        request_last_kv_block_id=request_last_kv_block_id,
        request_last_kv_block_offset=request_last_kv_block_offset,
        request_to_kv_block_ids=request_to_kv_block_ids,
        next_tokens=next_tokens,
        new_speculative_tokens=new_speculative_tokens,
        request_metadata=request_metadata,
        mamba_state_idx=mamba_state_idx,
        is_hybrid_model=is_hybrid_model,
        num_speculative_tokens=num_speculative_tokens,
        max_kv_block_count=max_kv_block_count,
        num_valid_ptr=num_valid_ptr,
        max_grid=max_grid,
    )


def triton_swap_bookkeeping(
    src_idxs: Tensor,
    dst_idxs: Tensor,
    request_kv_length_offsets: Tensor,
    request_in_prefill_status_tensor: Tensor,
    request_query_lengths: Tensor,
    request_output_lengths: Tensor,
    request_ids: Tensor,
    request_kv_block_counts: Tensor,
    request_last_kv_block_id: Tensor,
    request_last_kv_block_offset: Tensor,
    request_to_kv_block_ids: Tensor,
    next_tokens: Optional[Tensor],
    new_speculative_tokens: Optional[Tensor],
    request_metadata: Dict[str, Tensor],
    mamba_state_idx: Optional[Tensor],
    is_hybrid_model: bool,
    num_speculative_tokens: int,
    max_kv_block_count: int,
) -> None:
    """Fused swap: exchange tensor[src_idxs] and tensor[dst_idxs] for all bookkeeping tensors."""
    _launch_reorder_kernel(
        is_swap=True,
        src_idxs=src_idxs,
        dst_idxs=dst_idxs,
        request_kv_length_offsets=request_kv_length_offsets,
        request_in_prefill_status_tensor=request_in_prefill_status_tensor,
        request_query_lengths=request_query_lengths,
        request_output_lengths=request_output_lengths,
        request_ids=request_ids,
        request_kv_block_counts=request_kv_block_counts,
        request_last_kv_block_id=request_last_kv_block_id,
        request_last_kv_block_offset=request_last_kv_block_offset,
        request_to_kv_block_ids=request_to_kv_block_ids,
        next_tokens=next_tokens,
        new_speculative_tokens=new_speculative_tokens,
        request_metadata=request_metadata,
        mamba_state_idx=mamba_state_idx,
        is_hybrid_model=is_hybrid_model,
        num_speculative_tokens=num_speculative_tokens,
        max_kv_block_count=max_kv_block_count,
    )
