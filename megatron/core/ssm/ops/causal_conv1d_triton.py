# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

# Some of this code was adopted from https://github.com/Dao-AILab/causal-conv1d/
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch
import triton
import triton.language as tl


@triton.jit
def _roll_circular_buffer_kernel(
    in_ptr,
    out_ptr,
    cu_seqlens_ptr,
    B: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    stride_in_b,
    stride_in_d,
    stride_in_w,
    stride_out_b,
    stride_out_d,
    stride_out_w,
    BLOCK_W: tl.constexpr,
):
    # We map a 1D grid over B * D
    pid = tl.program_id(0)
    b = pid // D
    d = pid % D

    # 1. Load sequence lengths to calculate shift
    seqlen_start = tl.load(cu_seqlens_ptr + b)
    seqlen_end = tl.load(cu_seqlens_ptr + b + 1)
    seqlen = seqlen_end - seqlen_start

    shift = seqlen % W

    # 2. Setup standard W offsets
    w_offsets = tl.arange(0, BLOCK_W)
    mask = w_offsets < W

    # 3. Calculate gathered indices
    # NOTE: Triton/C++ modulo operator truncates towards zero for negative numbers.
    # Because shift < W, (w_offsets - shift) is at least -W + 1.
    # Adding W ensures the dividend is strictly positive, giving the correct wrapping behavior.
    src_w_offsets = (w_offsets - shift + W) % W

    # 4. Compute memory pointers
    in_offsets = in_ptr + (b * stride_in_b) + (d * stride_in_d) + (src_w_offsets * stride_in_w)
    out_offsets = out_ptr + (b * stride_out_b) + (d * stride_out_d) + (w_offsets * stride_out_w)

    # 5. Load and Store
    vals = tl.load(in_offsets, mask=mask)
    tl.store(out_offsets, vals, mask=mask)


def roll_conv_varlen_states(
    conv_varlen_states: torch.Tensor, cu_seqlens: torch.Tensor
) -> torch.Tensor:
    """
    Rolls the convolution states into a circular buffer layout based on sequence lengths.
    """
    B, D, W = conv_varlen_states.shape
    out = torch.empty_like(conv_varlen_states)

    # Next power of 2 for block size (e.g. W=4 -> BLOCK_W=4)
    BLOCK_W = triton.next_power_of_2(W)

    # Grid of size B * D
    grid = lambda meta: (B * D,)

    _roll_circular_buffer_kernel[grid](
        conv_varlen_states,
        out,
        cu_seqlens,
        B,
        D,
        W,
        conv_varlen_states.stride(0),
        conv_varlen_states.stride(1),
        conv_varlen_states.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_W=BLOCK_W,
    )

    return out


@triton.jit
def _gather_conv_state_kernel(
    conv_state_ptr,
    batch_indices_ptr,
    cache_seqlens_ptr,
    out_ptr,
    stride_cs_b,
    stride_cs_d,
    stride_cs_w,
    stride_out_b,
    stride_out_d,
    stride_out_w,
    D: tl.constexpr,
    state_len: tl.constexpr,
    d_conv: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // D
    d = pid % D

    # Load batch map
    req_idx = tl.load(batch_indices_ptr + b)

    # Check for padding/invalid batch index
    if req_idx < 0:
        w_offsets = tl.arange(0, BLOCK_W)
        mask = w_offsets < (d_conv - 1)
        out_offsets = out_ptr + (b * stride_out_b) + (d * stride_out_d) + (w_offsets * stride_out_w)
        # Store 0.0 to prevent NaNs/garbage data from propagating
        tl.store(out_offsets, 0.0, mask=mask)
        return

    # Load sequence length
    seq_len = tl.load(cache_seqlens_ptr + b)

    w_offsets = tl.arange(0, BLOCK_W)
    mask = w_offsets < (d_conv - 1)

    # Calculate circular buffer index.
    # We add state_len before modulo to prevent negative values in C++ modulo
    # when seq_len < d_conv - 1.
    val = seq_len - d_conv + 1 + w_offsets
    gather_indices = (val + state_len) % state_len

    # Calculate memory offsets
    cs_offsets = (
        conv_state_ptr
        + (req_idx * stride_cs_b)
        + (d * stride_cs_d)
        + (gather_indices * stride_cs_w)
    )
    out_offsets = out_ptr + (b * stride_out_b) + (d * stride_out_d) + (w_offsets * stride_out_w)

    data = tl.load(cs_offsets, mask=mask)
    tl.store(out_offsets, data, mask=mask)


def gather_conv_state(
    conv_state: torch.Tensor, batch_indices: torch.Tensor, cache_seqlens: torch.Tensor, d_conv: int
) -> torch.Tensor:
    """Reads the last (d_conv - 1) tokens from the circular convolution state."""
    B = batch_indices.shape[0]
    D = conv_state.shape[1]
    state_len = conv_state.shape[2]

    out = torch.empty((B, D, d_conv - 1), device=conv_state.device, dtype=conv_state.dtype)
    BLOCK_W = triton.next_power_of_2(d_conv - 1)

    grid = lambda meta: (B * D,)
    _gather_conv_state_kernel[grid](
        conv_state,
        batch_indices,
        cache_seqlens,
        out,
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        D,
        state_len,
        d_conv,
        BLOCK_W=BLOCK_W,
    )
    return out


@triton.jit
def _scatter_conv_state_kernel(
    conv_state_ptr,
    batch_indices_ptr,
    cache_seqlens_ptr,
    xbc_tail_ptr,
    stride_cs_b,
    stride_cs_d,
    stride_cs_w,
    stride_xbc_b,
    stride_xbc_d,
    stride_xbc_w,
    D: tl.constexpr,
    state_len: tl.constexpr,
    chunk_len: tl.constexpr,
    copy_len: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // D
    d = pid % D

    # Load batch map
    req_idx = tl.load(batch_indices_ptr + b)

    # Check for padding/invalid batch index and safely return
    if req_idx < 0:
        return

    # Load sequence length
    seq_len = tl.load(cache_seqlens_ptr + b)

    w_offsets = tl.arange(0, BLOCK_W)
    mask = w_offsets < copy_len

    # seq_len >= 0 and chunk_len >= copy_len, so this is guaranteed to be >= 0.
    update_indices = (seq_len + chunk_len - copy_len + w_offsets) % state_len

    # Calculate memory offsets
    xbc_offsets = (
        xbc_tail_ptr + (b * stride_xbc_b) + (d * stride_xbc_d) + (w_offsets * stride_xbc_w)
    )
    cs_offsets = (
        conv_state_ptr
        + (req_idx * stride_cs_b)
        + (d * stride_cs_d)
        + (update_indices * stride_cs_w)
    )

    data = tl.load(xbc_offsets, mask=mask)
    tl.store(cs_offsets, data, mask=mask)


def scatter_conv_state(
    conv_state: torch.Tensor,
    xbc_tail: torch.Tensor,
    batch_indices: torch.Tensor,
    cache_seqlens: torch.Tensor,
    chunk_len: int,
    copy_len: int,
):
    """Writes the newest chunk of tokens into the circular convolution state."""
    B, D, _ = xbc_tail.shape
    state_len = conv_state.shape[2]
    BLOCK_W = triton.next_power_of_2(copy_len)

    grid = lambda meta: (B * D,)
    _scatter_conv_state_kernel[grid](
        conv_state,
        batch_indices,
        cache_seqlens,
        xbc_tail,
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        xbc_tail.stride(0),
        xbc_tail.stride(1),
        xbc_tail.stride(2),
        D,
        state_len,
        chunk_len,
        copy_len,
        BLOCK_W=BLOCK_W,
    )


@triton.jit
def causal_conv1d_update_kernel(
    x_ptr,
    x_b_stride,
    x_s_stride,
    x_c_stride,
    conv_state_ptr,
    conv_state_b_stride,
    conv_state_c_stride,
    conv_state_l_stride,
    int_state_ptr,
    int_state_b_stride,
    int_state_s_stride,
    int_state_c_stride,
    int_state_l_stride,
    weight_ptr,
    weight_c_stride,
    weight_width_stride,
    bias_ptr,
    bias_stride,
    out_ptr,
    out_b_stride,
    out_s_stride,
    out_c_stride,
    conv_state_indices_ptr,
    cache_seqlens_ptr,
    batch,
    seq_len,
    dim,
    state_len,
    WIDTH: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    IS_CIRCULAR: tl.constexpr,
    HAS_STATE_INDICES: tl.constexpr,
    HAS_INT_STATE: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
):
    """Triton implementation of causal_conv1d_update (kernel)."""
    batch_id = tl.program_id(0)
    channel_block_id = tl.program_id(1)

    channel_offsets = channel_block_id * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    mask = channel_offsets < dim

    # State batch coordinate mapping
    if HAS_STATE_INDICES:
        state_batch_coord = tl.load(conv_state_indices_ptr + batch_id)
    else:
        state_batch_coord = batch_id

    # Base Pointers
    conv_state_ptrs = (
        conv_state_ptr
        + state_batch_coord * conv_state_b_stride
        + channel_offsets * conv_state_c_stride
    )
    weight_ptrs = weight_ptr + channel_offsets * weight_c_stride

    # Skip padding tokens (block-level uniform condition)
    if state_batch_coord < 0:
        for s in range(seq_len):
            out_ptrs = (
                out_ptr
                + batch_id * out_b_stride
                + s * out_s_stride
                + channel_offsets * out_c_stride
            )
            tl.store(out_ptrs, 0.0, mask=mask)
        return

    # Load Bias
    if HAS_BIAS:
        bias_val = tl.load(bias_ptr + channel_offsets * bias_stride, mask=mask).to(tl.float32)
    else:
        bias_val = tl.zeros([BLOCK_DIM], dtype=tl.float32)

    # Load Weights
    if WIDTH == 2:
        w0 = tl.load(weight_ptrs + 0 * weight_width_stride, mask=mask).to(tl.float32)
        w1 = tl.load(weight_ptrs + 1 * weight_width_stride, mask=mask).to(tl.float32)
    elif WIDTH == 3:
        w0 = tl.load(weight_ptrs + 0 * weight_width_stride, mask=mask).to(tl.float32)
        w1 = tl.load(weight_ptrs + 1 * weight_width_stride, mask=mask).to(tl.float32)
        w2 = tl.load(weight_ptrs + 2 * weight_width_stride, mask=mask).to(tl.float32)
    elif WIDTH == 4:
        w0 = tl.load(weight_ptrs + 0 * weight_width_stride, mask=mask).to(tl.float32)
        w1 = tl.load(weight_ptrs + 1 * weight_width_stride, mask=mask).to(tl.float32)
        w2 = tl.load(weight_ptrs + 2 * weight_width_stride, mask=mask).to(tl.float32)
        w3 = tl.load(weight_ptrs + 3 * weight_width_stride, mask=mask).to(tl.float32)

    # Initialize independent x_vals to match unrolled float array
    x_val_0 = tl.zeros([BLOCK_DIM], dtype=tl.float32)
    x_val_1 = tl.zeros([BLOCK_DIM], dtype=tl.float32)
    x_val_2 = tl.zeros([BLOCK_DIM], dtype=tl.float32)
    x_val_3 = tl.zeros([BLOCK_DIM], dtype=tl.float32)

    # If circular, we only need to read the base cache sequence length once
    if IS_CIRCULAR:
        base_cache_seqlen = tl.load(cache_seqlens_ptr + batch_id)
    else:
        base_cache_seqlen = None

    # Loop over the sequence dimension (e.g., speculative tokens)
    for s in range(seq_len):
        x_ptrs = x_ptr + batch_id * x_b_stride + s * x_s_stride + channel_offsets * x_c_stride
        out_ptrs = (
            out_ptr + batch_id * out_b_stride + s * out_s_stride + channel_offsets * out_c_stride
        )

        if not IS_CIRCULAR:
            # Load the last (WIDTH - 1) elements to use them BEFORE they are overwritten
            # by the shift
            if WIDTH >= 2:
                x_val_0 = tl.load(
                    conv_state_ptrs + (state_len - WIDTH + 1) * conv_state_l_stride, mask=mask
                ).to(tl.float32)
            if WIDTH >= 3:
                x_val_1 = tl.load(
                    conv_state_ptrs + (state_len - WIDTH + 2) * conv_state_l_stride, mask=mask
                ).to(tl.float32)
            if WIDTH >= 4:
                x_val_2 = tl.load(
                    conv_state_ptrs + (state_len - WIDTH + 3) * conv_state_l_stride, mask=mask
                ).to(tl.float32)

            # Shift the linear state buffer left by 1
            i = 0
            while i < state_len - 1:
                val = tl.load(conv_state_ptrs + (i + 1) * conv_state_l_stride, mask=mask)
                tl.store(conv_state_ptrs + i * conv_state_l_stride, val, mask=mask)
                i += 1
        else:
            cache_seqlen = base_cache_seqlen + s
            update_idx = cache_seqlen % state_len
            read_idx = update_idx - (WIDTH - 1)
            read_idx = tl.where(read_idx < 0, read_idx + state_len, read_idx)

            if WIDTH >= 2:
                state_val = tl.load(conv_state_ptrs + read_idx * conv_state_l_stride, mask=mask)
                x_val_0 = state_val.to(tl.float32)
                read_idx = tl.where(
                    read_idx + 1 >= state_len, read_idx + 1 - state_len, read_idx + 1
                )
            if WIDTH >= 3:
                state_val = tl.load(conv_state_ptrs + read_idx * conv_state_l_stride, mask=mask)
                x_val_1 = state_val.to(tl.float32)
                read_idx = tl.where(
                    read_idx + 1 >= state_len, read_idx + 1 - state_len, read_idx + 1
                )
            if WIDTH >= 4:
                state_val = tl.load(conv_state_ptrs + read_idx * conv_state_l_stride, mask=mask)
                x_val_2 = state_val.to(tl.float32)

        # Process the single token for the current sequence step
        x_val = tl.load(x_ptrs, mask=mask)

        # Store the new token in the state buffer
        if not IS_CIRCULAR:
            tl.store(conv_state_ptrs + (state_len - 1) * conv_state_l_stride, x_val, mask=mask)
        else:
            cache_seqlen = base_cache_seqlen + s
            update_idx = cache_seqlen % state_len
            tl.store(conv_state_ptrs + update_idx * conv_state_l_stride, x_val, mask=mask)

        # Write out to the intermediate state buffer if requested
        if HAS_INT_STATE:
            i = 0
            while i < state_len:
                val = tl.load(conv_state_ptrs + i * conv_state_l_stride, mask=mask)
                int_ptr = (
                    int_state_ptr
                    + state_batch_coord * int_state_b_stride
                    + s * int_state_s_stride
                    + channel_offsets * int_state_c_stride
                    + i * int_state_l_stride
                )
                tl.store(int_ptr, val, mask=mask)
                i += 1

        # Advance registers for calculation
        x_val_f32 = x_val.to(tl.float32)
        if WIDTH == 2:
            x_val_1 = x_val_f32
        elif WIDTH == 3:
            x_val_2 = x_val_f32
        elif WIDTH == 4:
            x_val_3 = x_val_f32

        # Compute output
        out_val = bias_val
        if WIDTH == 2:
            out_val += w0 * x_val_0 + w1 * x_val_1
        elif WIDTH == 3:
            out_val += w0 * x_val_0 + w1 * x_val_1 + w2 * x_val_2
        elif WIDTH == 4:
            out_val += w0 * x_val_0 + w1 * x_val_1 + w2 * x_val_2 + w3 * x_val_3

        if SILU_ACTIVATION:
            out_val = out_val * tl.sigmoid(out_val)

        tl.store(out_ptrs, out_val.to(out_ptrs.dtype.element_ty), mask=mask)


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    silu_activation: bool,
    cache_seqlens: torch.Tensor | None,
    conv_state_indices: torch.Tensor | None,
    intermediate_conv_states: torch.Tensor | None = None,
) -> torch.Tensor:
    """Triton implementation of causal_conv1d_update (entrypoint)."""

    # Check if input is 2D, temporarily treat as 3D for uniform processing
    is_2d = x.dim() == 2
    if is_2d:
        x = x.unsqueeze(1)

    batch, seq_len, dim = x.shape
    out = torch.empty_like(x)
    state_len = conv_state.shape[-1]
    width = weight.shape[-1]

    if bias is not None:
        bias_stride = bias.stride(0)
        has_bias = True
    else:
        bias = x  # Dummy pointer
        bias_stride = 0
        has_bias = False

    if cache_seqlens is not None:
        is_circular = True
    else:
        cache_seqlens = x  # Dummy pointer
        is_circular = False

    if conv_state_indices is not None:
        has_state_indices = True
    else:
        conv_state_indices = x  # Dummy pointer
        has_state_indices = False

    # Extract intermediate state strides if provided
    if intermediate_conv_states is not None:
        has_int_state = True
        int_state_ptr = intermediate_conv_states
        int_state_b_stride = intermediate_conv_states.stride(0)
        int_state_s_stride = intermediate_conv_states.stride(1)
        int_state_c_stride = intermediate_conv_states.stride(2)
        int_state_l_stride = intermediate_conv_states.stride(3)
    else:
        has_int_state = False
        int_state_ptr = x  # Dummy pointer
        int_state_b_stride = 0
        int_state_s_stride = 0
        int_state_c_stride = 0
        int_state_l_stride = 0

    BLOCK_DIM = 64
    grid = (batch, triton.cdiv(dim, BLOCK_DIM))

    causal_conv1d_update_kernel[grid](
        x,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        conv_state,
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        int_state_ptr,
        int_state_b_stride,
        int_state_s_stride,
        int_state_c_stride,
        int_state_l_stride,
        weight,
        weight.stride(0),
        weight.stride(1),
        bias,
        bias_stride,
        out,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        conv_state_indices,
        cache_seqlens,
        batch,
        seq_len,
        dim,
        state_len,
        WIDTH=width,
        BLOCK_DIM=BLOCK_DIM,
        HAS_BIAS=has_bias,
        IS_CIRCULAR=is_circular,
        HAS_STATE_INDICES=has_state_indices,
        HAS_INT_STATE=has_int_state,
        SILU_ACTIVATION=silu_activation == "silu",
    )

    if is_2d:
        out = out.squeeze(1)

    return out
