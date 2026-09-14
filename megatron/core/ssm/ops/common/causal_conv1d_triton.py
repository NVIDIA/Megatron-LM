# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

# Some of this code was adopted from https://github.com/Dao-AILab/causal-conv1d/
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

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
    batch,
    seq_len,
    dim,
    state_len,
    WIDTH: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    HAS_BIAS: tl.constexpr,
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

    # Loop over the sequence dimension (e.g., speculative tokens)
    for s in range(seq_len):
        x_ptrs = x_ptr + batch_id * x_b_stride + s * x_s_stride + channel_offsets * x_c_stride
        out_ptrs = (
            out_ptr + batch_id * out_b_stride + s * out_s_stride + channel_offsets * out_c_stride
        )

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

        # Process the single token for the current sequence step
        x_val = tl.load(x_ptrs, mask=mask)

        # Shift the linear state buffer left by 1. When state_len == WIDTH (the
        # common case: conv_state dim == conv kernel width) the shifted values
        # are already resident in the x_val_* registers from the loads above, so
        # we can write them back without a second HBM read per position. For
        # state_len > WIDTH the leading positions are untouched by compute; fall
        # through to the explicit load+store shift.
        if state_len == WIDTH:
            out_dtype = conv_state_ptrs.dtype.element_ty
            if WIDTH >= 2:
                tl.store(
                    conv_state_ptrs + 0 * conv_state_l_stride, x_val_0.to(out_dtype), mask=mask
                )
            if WIDTH >= 3:
                tl.store(
                    conv_state_ptrs + 1 * conv_state_l_stride, x_val_1.to(out_dtype), mask=mask
                )
            if WIDTH >= 4:
                tl.store(
                    conv_state_ptrs + 2 * conv_state_l_stride, x_val_2.to(out_dtype), mask=mask
                )
        else:
            i = 0
            while i < state_len - 1:
                val = tl.load(conv_state_ptrs + (i + 1) * conv_state_l_stride, mask=mask)
                tl.store(conv_state_ptrs + i * conv_state_l_stride, val, mask=mask)
                i += 1

        # Store the new token at the end of the linear state buffer
        tl.store(conv_state_ptrs + (state_len - 1) * conv_state_l_stride, x_val, mask=mask)

        # Write out to the intermediate state buffer if requested. Reuse the
        # register values from the shift above (and x_val for the new tail
        # position) when state_len == WIDTH, instead of re-reading from HBM.
        if HAS_INT_STATE:
            if state_len == WIDTH:
                int_base = (
                    int_state_ptr
                    + state_batch_coord * int_state_b_stride
                    + s * int_state_s_stride
                    + channel_offsets * int_state_c_stride
                )
                out_dtype = int_base.dtype.element_ty
                if WIDTH >= 2:
                    tl.store(int_base + 0 * int_state_l_stride, x_val_0.to(out_dtype), mask=mask)
                if WIDTH >= 3:
                    tl.store(int_base + 1 * int_state_l_stride, x_val_1.to(out_dtype), mask=mask)
                if WIDTH >= 4:
                    tl.store(int_base + 2 * int_state_l_stride, x_val_2.to(out_dtype), mask=mask)
                tl.store(int_base + (state_len - 1) * int_state_l_stride, x_val, mask=mask)
            else:
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
        batch,
        seq_len,
        dim,
        state_len,
        WIDTH=width,
        BLOCK_DIM=BLOCK_DIM,
        HAS_BIAS=has_bias,
        HAS_STATE_INDICES=has_state_indices,
        HAS_INT_STATE=has_int_state,
        SILU_ACTIVATION=silu_activation == "silu",
    )

    if is_2d:
        out = out.squeeze(1)

    return out
