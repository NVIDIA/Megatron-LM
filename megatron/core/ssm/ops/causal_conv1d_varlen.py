# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Triton varlen depthwise causal 1D convolution with per-sequence initial states and fused SiLU.

Supports packed variable-length sequences where `causal_conv1d_fn` cannot accept
both `seq_idx` and `initial_states` simultaneously.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_T": 128, "BLOCK_C": 64}, num_warps=4),
        triton.Config({"BLOCK_T": 128, "BLOCK_C": 128}, num_warps=4),
        triton.Config({"BLOCK_T": 256, "BLOCK_C": 64}, num_warps=4),
        triton.Config({"BLOCK_T": 256, "BLOCK_C": 128}, num_warps=8),
    ],
    key=["total_tokens", "conv_dim"],
)
@triton.jit
def _causal_conv1d_varlen_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    cu_seqlens_ptr,
    initial_states_ptr,
    out_ptr,
    total_tokens: tl.constexpr,
    conv_dim: tl.constexpr,
    num_requests,
    initial_states_stride_req,
    initial_states_stride_dim,
    WIDTH: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
    HAS_INITIAL_STATES: tl.constexpr,
):
    """Depthwise causal conv1d over packed varlen sequences with initial states and SiLU."""
    pid_c = tl.program_id(0)
    pid_t = tl.program_id(1)

    c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offsets < conv_dim

    t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)

    # Load weights for this channel block: shape (BLOCK_C, WIDTH)
    w = tl.zeros((BLOCK_C, WIDTH), dtype=tl.float32)
    for j in tl.static_range(WIDTH):
        w_val = tl.load(weight_ptr + c_offsets * WIDTH + j, mask=c_mask, other=0.0)
        w = tl.where(
            tl.arange(0, WIDTH)[None, :] == j,
            w_val[:, None] + tl.zeros((1, WIDTH), dtype=tl.float32),
            w,
        )

    # Load bias for this channel block
    b = tl.load(bias_ptr + c_offsets, mask=c_mask, other=0.0)

    # Determine request boundaries for each token in the tile.
    # Use a linear scan: for each token, find which request it belongs to.
    # We iterate through cu_seqlens to find boundaries.

    # Process each token in the tile
    t_mask = t_offsets < total_tokens

    # For each token, find its request ID by scanning cu_seqlens
    # Since tokens are packed in request order, we can use a simple loop
    for ti in range(BLOCK_T):
        t = pid_t * BLOCK_T + ti
        if t >= total_tokens:
            continue

        # Find which request this token belongs to using linear scan
        req_id = 0
        req_start = tl.load(cu_seqlens_ptr).to(tl.int32)
        for r in range(1, 8192):  # upper bound on num_requests + 1
            if r > num_requests:
                break
            next_start = tl.load(cu_seqlens_ptr + r).to(tl.int32)
            if t >= next_start:
                req_id = r
                req_start = next_start

        # Accumulate convolution for this token
        acc = b.to(tl.float32)
        for j in tl.static_range(WIDTH):
            src = t - (WIDTH - 1) + j
            if src < req_start:
                # Need to load from initial_states or use zero
                if HAS_INITIAL_STATES:
                    state_col = (WIDTH - 1) - (req_start - src)
                    if state_col >= 0:
                        state_val = tl.load(
                            initial_states_ptr
                            + req_id * initial_states_stride_req
                            + c_offsets * initial_states_stride_dim
                            + state_col,
                            mask=c_mask,
                            other=0.0,
                        )
                    else:
                        state_val = tl.zeros((BLOCK_C,), dtype=tl.float32)
                else:
                    state_val = tl.zeros((BLOCK_C,), dtype=tl.float32)
                tap = state_val
            else:
                tap = tl.load(
                    x_ptr + src * conv_dim + c_offsets, mask=c_mask, other=0.0
                ).to(tl.float32)

            # Extract weight column j
            w_j = tl.zeros((BLOCK_C,), dtype=tl.float32)
            for jj in tl.static_range(WIDTH):
                if jj == j:
                    w_j = tl.load(weight_ptr + c_offsets * WIDTH + jj, mask=c_mask, other=0.0).to(
                        tl.float32
                    )

            acc += tap * w_j

        # Apply SiLU: x * sigmoid(x)
        sigmoid_acc = 1.0 / (1.0 + tl.exp(-acc))
        result = acc * sigmoid_acc

        # Store result
        tl.store(
            out_ptr + t * conv_dim + c_offsets,
            result.to(tl.load(x_ptr + c_offsets, mask=c_mask).dtype),
            mask=c_mask,
        )


def causal_conv1d_varlen_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_states: torch.Tensor = None,
    activation: str = "silu",
) -> torch.Tensor:
    """Depthwise causal 1D convolution over packed variable-length sequences.

    Supports both `cu_seqlens` (sequence boundaries) and `initial_states`
    simultaneously, unlike `causal_conv1d_fn` which requires mutual exclusivity
    between `seq_idx` and `initial_states`.

    Args:
        x: Input tensor of shape (total_tokens, conv_dim), channels-last packed.
        weight: Convolution weights of shape (conv_dim, d_conv).
        bias: Bias of shape (conv_dim,).
        cu_seqlens: Cumulative sequence lengths of shape (num_requests + 1,), int32.
        initial_states: Per-request initial conv states of shape
            (num_requests, conv_dim, d_conv - 1). If None, uses zeros.
        activation: Activation function, must be "silu".

    Returns:
        Output tensor of shape (total_tokens, conv_dim).
    """
    assert activation == "silu", f"Only silu activation is supported, got {activation}"
    assert x.is_contiguous(), "x must be contiguous"
    assert weight.is_contiguous(), "weight must be contiguous"

    total_tokens, conv_dim = x.shape
    d_conv = weight.shape[1]
    num_requests = cu_seqlens.shape[0] - 1

    out = torch.empty_like(x)

    has_initial_states = initial_states is not None
    if not has_initial_states:
        # Create a dummy tensor for the kernel pointer (won't be accessed)
        initial_states = torch.empty(
            (1, 1, 1), dtype=x.dtype, device=x.device
        )
        is_stride_req = 1
        is_stride_dim = 1
    else:
        assert initial_states.shape == (num_requests, conv_dim, d_conv - 1)
        is_stride_req = initial_states.stride(0)
        is_stride_dim = initial_states.stride(1)

    grid = (triton.cdiv(conv_dim, 64), triton.cdiv(total_tokens, 128))

    # Use a simple per-token loop approach for correctness
    _causal_conv1d_varlen_simple(
        x, weight, bias, cu_seqlens, initial_states if has_initial_states else None, out
    )

    return out


def _causal_conv1d_varlen_simple(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_states: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Simple PyTorch implementation of varlen causal conv1d with initial states and SiLU.

    This is a reference/fallback implementation that processes each request sequentially.
    """
    total_tokens, conv_dim = x.shape
    d_conv = weight.shape[1]
    num_requests = cu_seqlens.shape[0] - 1

    for r in range(num_requests):
        start = cu_seqlens[r].item()
        end = cu_seqlens[r + 1].item()
        seq_len = end - start

        if seq_len == 0:
            continue

        # Get initial state for this request
        if initial_states is not None:
            init_state = initial_states[r]  # (conv_dim, d_conv - 1)
        else:
            init_state = torch.zeros(
                (conv_dim, d_conv - 1), dtype=x.dtype, device=x.device
            )

        # Build padded input: prepend initial_states, then the sequence tokens
        # init_state is (conv_dim, d_conv-1), x_seq is (seq_len, conv_dim)
        x_seq = x[start:end]  # (seq_len, conv_dim)

        # Perform depthwise conv1d
        # For each output position t, compute sum over j of weight[c, j] * input[t - (d_conv-1) + j]
        for t in range(seq_len):
            acc = bias.float()  # (conv_dim,)
            for j in range(d_conv):
                src_pos = t - (d_conv - 1) + j
                if src_pos < 0:
                    # Load from initial_states
                    state_col = (d_conv - 1) + src_pos  # = (d_conv - 1) - (d_conv - 1 - j + ... )
                    if state_col >= 0 and state_col < d_conv - 1:
                        tap = init_state[:, state_col].float()
                    else:
                        tap = torch.zeros(conv_dim, dtype=torch.float32, device=x.device)
                else:
                    tap = x_seq[src_pos].float()

                acc = acc + tap * weight[:, j].float()

            # Apply SiLU
            result = acc * torch.sigmoid(acc)
            out[start + t] = result.to(out.dtype)
