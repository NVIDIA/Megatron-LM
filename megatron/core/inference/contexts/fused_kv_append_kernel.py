# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _append_kv_cache_kernel(
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
    Triton kernel to append key and value vectors to pre-sliced paged KV cache tensors.

    Each program instance handles one head of one token. The grid is 2D: (n_tokens, num_heads).

    1. It identifies which token and head it is responsible for using `tl.program_id`.
    2. It loads the `block_idx` and `local_pos` for that token.
    3. It loads the `h_dim` vector for its assigned key/value head.
    4. It calculates the destination address in the 4D cache slices.
    5. It writes (scatters) the head vector to its destination in the cache.
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

    # --- Calculate destination pointers in the 4D KV cache slices ---
    dest_offset = (
        block_idx * stride_cache_block + local_pos * stride_cache_pos + head_idx * stride_cache_head
    )

    key_dest_ptr = key_cache_ptr + dest_offset
    value_dest_ptr = value_cache_ptr + dest_offset

    # --- Store the head data into the cache ---
    tl.store(key_dest_ptr + offs_h * stride_cache_hdim, key_to_write, mask=mask_h)
    tl.store(value_dest_ptr + offs_h * stride_cache_hdim, value_to_write, mask=mask_h)


def triton_append_key_value_cache(
    layer_number: int,
    key: Tensor,
    value: Tensor,
    memory_buffer: Tensor,
    padded_active_token_count: int,
    token_to_block_idx: Tensor,
    token_to_local_position_within_kv_block: Tensor,
) -> None:
    """
    Append to KV cache using a high-performance, standalone Triton kernel.

    Args:
        layer_number (int): Layer number (1-based).
        key (Tensor): Key tensor of shape (batch_size, 1, num_heads, h_dim).
        value (Tensor): Value tensor of shape (batch_size, 1, num_heads, h_dim).
        memory_buffer (Tensor): The 6D KV cache tensor to write to.
        padded_active_token_count (int): The number of active tokens to process.
        token_to_block_idx (Tensor): Tensor mapping token index to its block index in
        the cache.
        token_to_local_position_within_kv_block (Tensor): Tensor mapping token index
        to its position within a block.
    """
    # --- Input Validation and Preparation ---
    assert (
        key.device.type == 'cuda'
        and value.device.type == 'cuda'
        and memory_buffer.device.type == 'cuda'
    ), "All tensors must be on CUDA devices."

    assert (
        key.size(1) == 1 and value.size(1) == 1
    ), "Key and Value should have a sequence length of 1."
    key = key.squeeze(1)
    value = value.squeeze(1)

    n_tokens = padded_active_token_count
    if n_tokens == 0:
        return

    _, num_heads, h_dim = key.shape

    key_cache = memory_buffer[0, layer_number]
    value_cache = memory_buffer[1, layer_number]

    key_to_cache = key[:n_tokens]
    value_to_cache = value[:n_tokens]
    block_idx_active = token_to_block_idx[:n_tokens]
    local_kv_seq_idx_active = token_to_local_position_within_kv_block[:n_tokens]

    assert (
        key_cache.dim() == 4 and value_cache.dim() == 4
    ), f"Sliced key_cache and value_cache should be 4D"
    assert (
        num_heads == key_cache.shape[-2]
    ), f"Head count mismatch. Key/Value has {num_heads} but cache expects {key_cache.shape[-2]}."
    assert (
        h_dim == key_cache.shape[-1]
    ), f"Head dimension mismatch. Key/Value has {h_dim} but cache expects {key_cache.shape[-1]}."

    block_idx_active = block_idx_active.contiguous()
    local_kv_seq_idx_active = local_kv_seq_idx_active.contiguous()

    grid = (n_tokens, num_heads)
    BLOCK_SIZE_H = triton.next_power_of_2(h_dim)

    cache_strides = key_cache.stride()

    _append_kv_cache_kernel[grid](
        # Pointers
        key_to_cache,
        value_to_cache,
        key_cache,
        value_cache,
        block_idx_active,
        local_kv_seq_idx_active,
        # Strides for 3D key/value tensors
        key_to_cache.stride(0),
        key_to_cache.stride(1),
        key_to_cache.stride(2),
        value_to_cache.stride(0),
        value_to_cache.stride(1),
        value_to_cache.stride(2),
        # Strides for the 4D sliced cache
        cache_strides[0],
        cache_strides[1],
        cache_strides[2],
        cache_strides[3],
        # Other parameters
        n_tokens=n_tokens,
        num_heads=num_heads,
        H_DIM=h_dim,
        # Compile-time constant
        BLOCK_SIZE_H=BLOCK_SIZE_H,
    )
