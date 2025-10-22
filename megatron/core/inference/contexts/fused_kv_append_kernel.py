# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import Optional, Tuple

import triton
import triton.language as tl
from torch import Tensor


# ============================================================================
# TRITON KERNELS
# ============================================================================


@triton.jit
def _append_kv_merged_kernel(
    # --- Pointers to Tensors ---
    key_ptr,
    value_ptr,
    cache_ptr,
    block_idx_ptr,
    local_kv_seq_idx_ptr,
    # --- Strides for Tensor Memory Layout ---
    stride_key_token,
    stride_key_head,
    stride_key_hdim,
    stride_value_token,
    stride_value_head,
    stride_value_hdim,
    stride_cache_kv,
    stride_cache_block,
    stride_cache_pos,
    stride_cache_head,
    stride_cache_hdim,
    # --- Other Parameters ---
    n_tokens: tl.int32,
    num_heads: tl.int32,
    H_DIM: tl.int32,
    # --- Compile-Time Constants ---
    BLOCK_SIZE_H: tl.constexpr,
):
    """
    Triton kernel for merged KV cache layouts (M_2NCHD, M_N2HCD, M_N2CHD).
    Handles caches where K and V are stored in a single tensor with a KV dimension.

    Each program instance handles one head of one token. The grid is 2D: (n_tokens, num_heads).
    """
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    if token_idx >= n_tokens or head_idx >= num_heads:
        return

    # --- Load destination indices for the current token ---
    block_idx = tl.load(block_idx_ptr + token_idx)
    local_pos = tl.load(local_kv_seq_idx_ptr + token_idx)

    # --- Load the key and value data for the current head of the current token ---
    offs_h = tl.arange(0, BLOCK_SIZE_H)
    mask_h = offs_h < H_DIM

    key_head_ptr = key_ptr + token_idx * stride_key_token + head_idx * stride_key_head
    value_head_ptr = value_ptr + token_idx * stride_value_token + head_idx * stride_value_head

    key_to_write = tl.load(key_head_ptr + offs_h * stride_key_hdim, mask=mask_h, other=0.0)
    value_to_write = tl.load(value_head_ptr + offs_h * stride_value_hdim, mask=mask_h, other=0.0)

    # --- Calculate destination pointers in the merged cache ---
    # The stride_cache_kv allows us to select between K (0) and V (1)
    base_offset = (
        block_idx * stride_cache_block
        + local_pos * stride_cache_pos
        + head_idx * stride_cache_head
    )

    key_dest_ptr = cache_ptr + 0 * stride_cache_kv + base_offset
    value_dest_ptr = cache_ptr + 1 * stride_cache_kv + base_offset

    # --- Store the head data into the cache ---
    tl.store(key_dest_ptr + offs_h * stride_cache_hdim, key_to_write, mask=mask_h)
    tl.store(value_dest_ptr + offs_h * stride_cache_hdim, value_to_write, mask=mask_h)


@triton.jit
def _append_kv_separate_kernel(
    # --- Pointers to Tensors ---
    key_ptr,
    value_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_idx_ptr,
    local_kv_seq_idx_ptr,
    # --- Strides for Tensor Memory Layout ---
    stride_key_token,
    stride_key_head,
    stride_key_hdim,
    stride_value_token,
    stride_value_head,
    stride_value_hdim,
    stride_cache_block,
    stride_cache_pos,
    stride_cache_head,
    stride_cache_hdim,
    # --- Other Parameters ---
    n_tokens: tl.int32,
    num_heads: tl.int32,
    H_DIM: tl.int32,
    # --- Compile-Time Constants ---
    BLOCK_SIZE_H: tl.constexpr,
):
    """
    Triton kernel for separate K/V cache layouts (S_NCHD, S_NHCD).
    Handles caches where K and V are stored in separate tensors.

    Each program instance handles one head of one token. The grid is 2D: (n_tokens, num_heads).
    """
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    if token_idx >= n_tokens or head_idx >= num_heads:
        return

    # --- Load destination indices for the current token ---
    block_idx = tl.load(block_idx_ptr + token_idx)
    local_pos = tl.load(local_kv_seq_idx_ptr + token_idx)

    # --- Load the key and value data for the current head of the current token ---
    offs_h = tl.arange(0, BLOCK_SIZE_H)
    mask_h = offs_h < H_DIM

    key_head_ptr = key_ptr + token_idx * stride_key_token + head_idx * stride_key_head
    value_head_ptr = value_ptr + token_idx * stride_value_token + head_idx * stride_value_head

    key_to_write = tl.load(key_head_ptr + offs_h * stride_key_hdim, mask=mask_h, other=0.0)
    value_to_write = tl.load(value_head_ptr + offs_h * stride_value_hdim, mask=mask_h, other=0.0)

    # --- Calculate destination pointers in the separate caches ---
    dest_offset = (
        block_idx * stride_cache_block + local_pos * stride_cache_pos + head_idx * stride_cache_head
    )

    key_dest_ptr = key_cache_ptr + dest_offset
    value_dest_ptr = value_cache_ptr + dest_offset

    # --- Store the head data into the cache ---
    tl.store(key_dest_ptr + offs_h * stride_cache_hdim, key_to_write, mask=mask_h)
    tl.store(value_dest_ptr + offs_h * stride_cache_hdim, value_to_write, mask=mask_h)


@triton.jit
def _append_mla_kernel(
    # --- Pointers to Tensors ---
    kv_concat_ptr,
    cache_ptr,
    block_idx_ptr,
    local_kv_seq_idx_ptr,
    # --- Strides for Tensor Memory Layout ---
    stride_kv_token,
    stride_kv_dim,
    stride_cache_block,
    stride_cache_pos,
    stride_cache_dim,
    # --- Other Parameters ---
    n_tokens: tl.int32,
    LATENT_DIM: tl.int32,
    # --- Compile-Time Constants ---
    BLOCK_SIZE_D: tl.constexpr,
):
    """
    Triton kernel for MLA cache layout.
    Handles compressed latent representation (no K/V split, no heads).

    Each program instance handles one token. The grid is 1D: (n_tokens,).
    """
    token_idx = tl.program_id(0)

    if token_idx >= n_tokens:
        return

    # --- Load destination indices for the current token ---
    block_idx = tl.load(block_idx_ptr + token_idx)
    local_pos = tl.load(local_kv_seq_idx_ptr + token_idx)

    # --- Load the latent representation for the current token ---
    offs_d = tl.arange(0, BLOCK_SIZE_D)
    mask_d = offs_d < LATENT_DIM

    kv_ptr = kv_concat_ptr + token_idx * stride_kv_token
    kv_to_write = tl.load(kv_ptr + offs_d * stride_kv_dim, mask=mask_d, other=0.0)

    # --- Calculate destination pointer in the MLA cache ---
    dest_offset = block_idx * stride_cache_block + local_pos * stride_cache_pos

    dest_ptr = cache_ptr + dest_offset

    # --- Store the latent data into the cache ---
    tl.store(dest_ptr + offs_d * stride_cache_dim, kv_to_write, mask=mask_d)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _validate_and_prepare_tensors(
    key: Tensor, value: Optional[Tensor], n_tokens: int
) -> Tuple[Tensor, Optional[Tensor], int, int, int]:
    """
    Validate input tensors and extract common dimensions.

    Args:
        key: Key tensor of shape (batch_size, 1, num_heads, h_dim) or (batch_size, 1, latent_dim)
        value: Value tensor of shape (batch_size, 1, num_heads, h_dim) or None for MLA
        n_tokens: Number of tokens to process

    Returns:
        Tuple of (squeezed_key, squeezed_value, num_heads, h_dim, n_tokens)
    """
    assert key.device.type == 'cuda', "All tensors must be on CUDA devices."
    if value is not None:
        assert value.device.type == 'cuda', "All tensors must be on CUDA devices."

    assert key.size(1) == 1, "Key should have a sequence length of 1."
    key = key.squeeze(1)

    if value is not None:
        assert value.size(1) == 1, "Value should have a sequence length of 1."
        value = value.squeeze(1)

    if n_tokens == 0:
        return key, value, 0, 0, 0

    # Extract dimensions
    if key.dim() == 3:  # [batch, heads, dim]
        _, num_heads, h_dim = key.shape
    elif key.dim() == 2:  # [batch, dim] for MLA
        num_heads = 0
        h_dim = key.size(-1)
    else:
        raise ValueError(f"Unexpected key shape: {key.shape}")

    return key, value, num_heads, h_dim, n_tokens


# ============================================================================
# GENERIC WRAPPERS
# ============================================================================


def _append_merged_cache(
    key: Tensor,
    value: Tensor,
    cache: Tensor,
    n_tokens: int,
    num_heads: int,
    h_dim: int,
    block_idx_active: Tensor,
    local_kv_seq_idx_active: Tensor,
    kv_dim_idx: int,
    block_dim_idx: int,
    pos_dim_idx: int,
    head_dim_idx: int,
) -> None:
    """
    Generic wrapper for merged cache layouts.

    Args:
        kv_dim_idx: Index of the KV dimension (2) in the cache shape
        block_dim_idx: Index of the block/chunk dimension (N) in the cache shape
        pos_dim_idx: Index of the position dimension (C) in the cache shape
        head_dim_idx: Index of the head dimension (H) in the cache shape
    """
    grid = (n_tokens, num_heads)
    BLOCK_SIZE_H = triton.next_power_of_2(h_dim)

    cache_strides = cache.stride()
    stride_cache_kv = cache_strides[kv_dim_idx]
    stride_cache_block = cache_strides[block_dim_idx]
    stride_cache_pos = cache_strides[pos_dim_idx]
    stride_cache_head = cache_strides[head_dim_idx]
    stride_cache_hdim = cache_strides[-1]  # Last dimension is always head_dim

    _append_kv_merged_kernel[grid](
        key,
        value,
        cache,
        block_idx_active,
        local_kv_seq_idx_active,
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        stride_cache_kv,
        stride_cache_block,
        stride_cache_pos,
        stride_cache_head,
        stride_cache_hdim,
        n_tokens=n_tokens,
        num_heads=num_heads,
        H_DIM=h_dim,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
    )


def _append_separate_cache(
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    n_tokens: int,
    num_heads: int,
    h_dim: int,
    block_idx_active: Tensor,
    local_kv_seq_idx_active: Tensor,
    block_dim_idx: int,
    pos_dim_idx: int,
    head_dim_idx: int,
) -> None:
    """
    Generic wrapper for separate cache layouts.

    Args:
        block_dim_idx: Index of the block/chunk dimension (N) in the cache shape
        pos_dim_idx: Index of the position dimension (C) in the cache shape
        head_dim_idx: Index of the head dimension (H) in the cache shape
    """
    grid = (n_tokens, num_heads)
    BLOCK_SIZE_H = triton.next_power_of_2(h_dim)

    cache_strides = key_cache.stride()
    stride_cache_block = cache_strides[block_dim_idx]
    stride_cache_pos = cache_strides[pos_dim_idx]
    stride_cache_head = cache_strides[head_dim_idx]
    stride_cache_hdim = cache_strides[-1]  # Last dimension is always head_dim

    _append_kv_separate_kernel[grid](
        key,
        value,
        key_cache,
        value_cache,
        block_idx_active,
        local_kv_seq_idx_active,
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        stride_cache_block,
        stride_cache_pos,
        stride_cache_head,
        stride_cache_hdim,
        n_tokens=n_tokens,
        num_heads=num_heads,
        H_DIM=h_dim,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
    )


def _append_mla_cache(
    kv_concat: Tensor,
    cache: Tensor,
    n_tokens: int,
    latent_dim: int,
    block_idx_active: Tensor,
    local_kv_seq_idx_active: Tensor,
) -> None:
    """
    Wrapper for MLA cache layout: [N, C, D]
    """
    grid = (n_tokens,)
    BLOCK_SIZE_D = triton.next_power_of_2(latent_dim)

    cache_strides = cache.stride()

    _append_mla_kernel[grid](
        kv_concat,
        cache,
        block_idx_active,
        local_kv_seq_idx_active,
        kv_concat.stride(0),
        kv_concat.stride(1),
        cache_strides[0],  # stride_cache_block
        cache_strides[1],  # stride_cache_pos
        cache_strides[2],  # stride_cache_dim
        n_tokens=n_tokens,
        LATENT_DIM=latent_dim,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )


# ============================================================================
# MAIN DISPATCHER
# ============================================================================


def triton_append_key_value_cache(
    key: Tensor,
    value: Optional[Tensor],
    cache,  # KVCacheBase instance
    padded_active_token_count: int,
    token_to_block_idx: Tensor,
    token_to_local_position_within_kv_block: Tensor,
) -> None:
    """
    Append to KV cache using high-performance Triton kernels.

    This function supports all cache layouts:
    - M_2NCHD: Merged [2, N, C, H, D]
    - M_N2CHD: Merged [N, 2, C, H, D]
    - M_N2HCD: Merged [N, 2, H, C, D]
    - S_NCHD: Separate [N, C, H, D]
    - S_NHCD: Separate [N, H, C, D]
    - MLA: [N, C, D]

    Args:
        key (Tensor): Key tensor of shape (batch_size, 1, num_heads, h_dim) or
                      (batch_size, 1, latent_dim) for MLA.
        value (Optional[Tensor]): Value tensor of shape (batch_size, 1, num_heads, h_dim)
                                  or None for MLA.
        cache: KVCacheBase instance (must support Triton).
        padded_active_token_count (int): The number of active tokens to process.
        token_to_block_idx (Tensor): Tensor mapping token index to its block index in
        the cache.
        token_to_local_position_within_kv_block (Tensor): Tensor mapping token index
        to its position within a block.
    """
    # Import cache classes (avoid circular imports by importing locally)
    from megatron.core.inference.kv_cache import (
        KVCacheM2NCHD,
        KVCacheMN2CHD,
        KVCacheMN2HCD,
        KVCacheSNCHD,
        KVCacheSNHCD,
        MLACache,
    )

    # --- Input Validation and Preparation ---
    key, value, num_heads, h_dim, n_tokens = _validate_and_prepare_tensors(
        key, value, padded_active_token_count
    )

    if n_tokens == 0:
        return

    # Get active slices
    key_to_cache = key[:n_tokens]
    value_to_cache = value[:n_tokens] if value is not None else None
    block_idx_active = token_to_block_idx[:n_tokens].contiguous()
    local_kv_seq_idx_active = token_to_local_position_within_kv_block[:n_tokens].contiguous()

    # Get cache tensors from the cache object
    cache_content = cache.get_content()

    # Dispatch based on cache type
    if isinstance(cache, MLACache):
        # MLA cache: [N, C, D]
        _append_mla_cache(
            kv_concat=key_to_cache,
            cache=cache_content,
            n_tokens=n_tokens,
            latent_dim=h_dim,
            block_idx_active=block_idx_active,
            local_kv_seq_idx_active=local_kv_seq_idx_active,
        )
    elif isinstance(cache, KVCacheM2NCHD):
        # M_2NCHD: [2, N, C, H, D] - KV at dim 0, block at 1, pos at 2, head at 3
        _append_merged_cache(
            key=key_to_cache,
            value=value_to_cache,
            cache=cache_content,
            n_tokens=n_tokens,
            num_heads=num_heads,
            h_dim=h_dim,
            block_idx_active=block_idx_active,
            local_kv_seq_idx_active=local_kv_seq_idx_active,
            kv_dim_idx=0,
            block_dim_idx=1,
            pos_dim_idx=2,
            head_dim_idx=3,
        )
    elif isinstance(cache, KVCacheMN2CHD):
        # M_N2CHD: [N, 2, C, H, D] - block at 0, KV at 1, pos at 2, head at 3
        _append_merged_cache(
            key=key_to_cache,
            value=value_to_cache,
            cache=cache_content,
            n_tokens=n_tokens,
            num_heads=num_heads,
            h_dim=h_dim,
            block_idx_active=block_idx_active,
            local_kv_seq_idx_active=local_kv_seq_idx_active,
            kv_dim_idx=1,
            block_dim_idx=0,
            pos_dim_idx=2,
            head_dim_idx=3,
        )
    elif isinstance(cache, KVCacheMN2HCD):
        # M_N2HCD: [N, 2, H, C, D] - block at 0, KV at 1, head at 2, pos at 3
        _append_merged_cache(
            key=key_to_cache,
            value=value_to_cache,
            cache=cache_content,
            n_tokens=n_tokens,
            num_heads=num_heads,
            h_dim=h_dim,
            block_idx_active=block_idx_active,
            local_kv_seq_idx_active=local_kv_seq_idx_active,
            kv_dim_idx=1,
            block_dim_idx=0,
            pos_dim_idx=3,
            head_dim_idx=2,
        )
    elif isinstance(cache, KVCacheSNCHD):
        # S_NCHD: [N, C, H, D] - block at 0, pos at 1, head at 2
        key_cache, value_cache = cache_content
        _append_separate_cache(
            key=key_to_cache,
            value=value_to_cache,
            key_cache=key_cache,
            value_cache=value_cache,
            n_tokens=n_tokens,
            num_heads=num_heads,
            h_dim=h_dim,
            block_idx_active=block_idx_active,
            local_kv_seq_idx_active=local_kv_seq_idx_active,
            block_dim_idx=0,
            pos_dim_idx=1,
            head_dim_idx=2,
        )
    elif isinstance(cache, KVCacheSNHCD):
        # S_NHCD: [N, H, C, D] - block at 0, head at 1, pos at 2
        key_cache, value_cache = cache_content
        _append_separate_cache(
            key=key_to_cache,
            value=value_to_cache,
            key_cache=key_cache,
            value_cache=value_cache,
            n_tokens=n_tokens,
            num_heads=num_heads,
            h_dim=h_dim,
            block_idx_active=block_idx_active,
            local_kv_seq_idx_active=local_kv_seq_idx_active,
            block_dim_idx=0,
            pos_dim_idx=2,
            head_dim_idx=1,
        )
    else:
        raise TypeError(
            f"Unsupported cache type: {type(cache).__name__}. "
            f"Triton kernel only supports M_2NCHD, M_N2CHD, M_N2HCD, S_NCHD, S_NHCD, and MLA layouts."
        )
