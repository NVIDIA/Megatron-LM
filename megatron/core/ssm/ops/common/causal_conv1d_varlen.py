# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Triton varlen depthwise causal 1D convolution with per-sequence initial states and fused SiLU.

Supports packed variable-length sequences where `causal_conv1d_fn` cannot accept
both `seq_idx` and `initial_states` simultaneously.
"""

import torch
import triton
import triton.language as tl

from megatron.core.ssm.ops.common.determinism import autotune_configs


# Two block-dim regimes:
#   1. vLLM-style: small BLOCK_T, large BLOCK_C, pipelined. Many small programs maximize
#      GPU occupancy, BLOCK_C=256 fully fills HBM transactions, sequence-pure programs
#      avoid in-block boundary branching. Usually wins at moderate-to-large conv_dim.
#   2. Large-block fallback: bigger tiles, fewer programs. Can win for small conv_dim
#      where vLLM's regime over-parallelizes, or when launch overhead dominates.
@triton.autotune(
    configs=autotune_configs(
        [
            triton.Config({"BLOCK_T": 8, "BLOCK_C": 256}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_T": 128, "BLOCK_C": 128}, num_warps=4),
        ]
    ),
    key=["conv_dim"],
)
@triton.jit
def _causal_conv1d_varlen_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    seq_idx_ptr,
    seq_start_ptr,
    initial_states_ptr,
    out_ptr,
    total_tokens,
    conv_dim: tl.constexpr,
    initial_states_stride_req,
    initial_states_stride_dim,
    WIDTH: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
    HAS_INITIAL_STATES: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Depthwise causal conv1d over packed varlen sequences with initial states and SiLU.

    Fully vectorized over BLOCK_T tokens x BLOCK_C channels per thread block.
    """
    pid_c = tl.program_id(0)
    pid_t = tl.program_id(1)

    c_off = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)  # (BLOCK_C,)
    c_mask = c_off < conv_dim
    t_off = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)  # (BLOCK_T,)
    t_mask = t_off < total_tokens

    # Load bias: (BLOCK_C,) broadcast to (BLOCK_T, BLOCK_C)
    acc = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + c_off, mask=c_mask, other=0.0).to(tl.float32)
        acc += bias[None, :]

    # Load per-token request ID and request start position
    req_id = tl.load(seq_idx_ptr + t_off, mask=t_mask, other=0)  # (BLOCK_T,)
    req_start = tl.load(seq_start_ptr + t_off, mask=t_mask, other=0)  # (BLOCK_T,)

    # Unrolled convolution over WIDTH taps (typically 4)
    for j in tl.static_range(WIDTH):
        # Load weight column j: (BLOCK_C,)
        w_j = tl.load(weight_ptr + c_off * WIDTH + j, mask=c_mask, other=0.0).to(tl.float32)

        # Source position for this tap
        src = t_off - (WIDTH - 1) + j  # (BLOCK_T,)
        in_seq = src >= req_start  # (BLOCK_T,) — True if source is within the sequence

        # Load from x for in-sequence positions (mask out out-of-bounds)
        src_safe = tl.maximum(src, 0)
        x_val = tl.load(
            x_ptr + src_safe[:, None] * conv_dim + c_off[None, :],
            mask=t_mask[:, None] & c_mask[None, :] & in_seq[:, None],
            other=0.0,
        ).to(
            tl.float32
        )  # (BLOCK_T, BLOCK_C)

        if HAS_INITIAL_STATES:
            # For tokens where src < req_start, load from initial_states
            state_col = (WIDTH - 1) - (req_start - src)  # (BLOCK_T,)
            valid_state = (~in_seq) & (state_col >= 0)  # (BLOCK_T,)
            state_col_safe = tl.maximum(state_col, 0)

            state_val = tl.load(
                initial_states_ptr
                + req_id[:, None] * initial_states_stride_req
                + c_off[None, :] * initial_states_stride_dim
                + state_col_safe[:, None],
                mask=t_mask[:, None] & c_mask[None, :] & valid_state[:, None],
                other=0.0,
            ).to(
                tl.float32
            )  # (BLOCK_T, BLOCK_C)

            tap = tl.where(in_seq[:, None], x_val, state_val)
        else:
            tap = x_val

        acc += tap * w_j[None, :]

    # SiLU activation: x * sigmoid(x)
    sigmoid_acc = 1.0 / (1.0 + tl.exp(-acc))
    result = acc * sigmoid_acc

    # Store output (cast back to input dtype)
    tl.store(
        out_ptr + t_off[:, None] * conv_dim + c_off[None, :],
        result,
        mask=t_mask[:, None] & c_mask[None, :],
    )


def causal_conv1d_varlen_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_states: torch.Tensor = None,
    activation: str = "silu",
    precomputed_seq_idx: torch.Tensor = None,
    precomputed_seq_start: torch.Tensor = None,
) -> torch.Tensor:
    """Depthwise causal 1D convolution over packed variable-length sequences.

    Supports both `cu_seqlens` (sequence boundaries) and `initial_states`
    simultaneously, unlike `causal_conv1d_fn` which requires mutual exclusivity
    between `seq_idx` and `initial_states`.

    Args:
        x: Input tensor of shape (total_tokens, conv_dim), channels-last packed.
        weight: Convolution weights of shape (conv_dim, d_conv).
        bias: Bias of shape (conv_dim,), or None for a bias-free convolution
            (Gated Delta Product layers default to ``conv_bias=False``).
        cu_seqlens: Cumulative sequence lengths of shape (num_requests + 1,), int32.
        initial_states: Per-request initial conv states of shape
            (num_requests, conv_dim, d_conv - 1). If None, uses zeros.
        activation: Activation function, must be "silu".
        precomputed_seq_idx: Precomputed per-token request ID of shape
            (total_tokens,). If provided, skips repeat_interleave (CUDA
            graph compatible). Padding tokens should use 0 as sentinel.
        precomputed_seq_start: Precomputed per-token request start position
            of shape (total_tokens,). Must be provided together with
            precomputed_seq_idx.

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

    # Use precomputed per-token metadata if provided (CUDA graph compatible),
    # otherwise compute from cu_seqlens via repeat_interleave.
    if precomputed_seq_idx is not None:
        assert precomputed_seq_start is not None
        assert (
            precomputed_seq_idx.shape[0] >= total_tokens
            and precomputed_seq_start.shape[0] >= total_tokens
        ), "precomputed conv metadata must cover every token in x"
        seq_idx = precomputed_seq_idx
        seq_start = precomputed_seq_start
    else:
        seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        seq_idx = torch.repeat_interleave(
            torch.arange(num_requests, device=x.device, dtype=torch.int32), seq_lengths
        )
        seq_start = torch.repeat_interleave(cu_seqlens[:-1], seq_lengths).to(torch.int32)

    has_bias = bias is not None
    if not has_bias:
        bias = x  # Dummy pointer; HAS_BIAS gates every read of it.

    has_initial_states = initial_states is not None
    if not has_initial_states:
        initial_states = torch.empty((1, 1, 1), dtype=x.dtype, device=x.device)
        is_stride_req = 1
        is_stride_dim = 1
    else:
        assert initial_states.shape == (num_requests, conv_dim, d_conv - 1)
        is_stride_req = initial_states.stride(0)
        is_stride_dim = initial_states.stride(1)

    grid = lambda meta: (
        triton.cdiv(conv_dim, meta["BLOCK_C"]),
        triton.cdiv(total_tokens, meta["BLOCK_T"]),
    )

    _causal_conv1d_varlen_kernel[grid](
        x,
        weight,
        bias,
        seq_idx,
        seq_start,
        initial_states,
        out,
        total_tokens,
        conv_dim,
        is_stride_req,
        is_stride_dim,
        WIDTH=d_conv,
        HAS_INITIAL_STATES=has_initial_states,
        HAS_BIAS=has_bias,
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

    This is a reference implementation for testing. Processes each request and token
    sequentially.
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

        if initial_states is not None:
            init_state = initial_states[r]  # (conv_dim, d_conv - 1)
        else:
            init_state = torch.zeros((conv_dim, d_conv - 1), dtype=x.dtype, device=x.device)

        x_seq = x[start:end]  # (seq_len, conv_dim)

        for t in range(seq_len):
            if bias is not None:
                acc = bias.float()  # (conv_dim,)
            else:
                acc = torch.zeros(conv_dim, dtype=torch.float32, device=x.device)
            for j in range(d_conv):
                src_pos = t - (d_conv - 1) + j
                if src_pos < 0:
                    state_col = (d_conv - 1) + src_pos
                    if state_col >= 0 and state_col < d_conv - 1:
                        tap = init_state[:, state_col].float()
                    else:
                        tap = torch.zeros(conv_dim, dtype=torch.float32, device=x.device)
                else:
                    tap = x_seq[src_pos].float()

                acc = acc + tap * weight[:, j].float()

            result = acc * torch.sigmoid(acc)
            out[start + t] = result.to(out.dtype)


def causal_conv1d_varlen_carry_states(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    previous_states: torch.Tensor,
    *,
    token_indices: torch.Tensor = None,
    prev_columns: torch.Tensor = None,
    from_slice: torch.Tensor = None,
) -> torch.Tensor:
    """Per-request conv state after appending a slice, carrying prior history.

    A request's conv state holds the last `d_conv` tokens it has seen. It exists
    so that the next call can convolve across the boundary: the causal
    convolution at token `t` reads tokens `t - d_conv + 1 .. t`, so continuing a
    sequence requires the tokens that came before the current input.

    When a prompt is processed one slice at a time, the new state is generally a
    mix of both sources -- the tail of the slice, and whatever the request was
    already carrying. With `d_conv = 4`, a request whose incoming state is
    `[a b c d]` and whose slice this call is `[e f]`:

        incoming state   a b c d
        slice                    e f
        new state            c d e f

    Two columns survive from the incoming state, two come from the slice. The
    same request handed a slice of `[e f g h]` or longer takes all four columns
    from the slice, and the incoming state does not contribute at all:

        incoming state   a b c d
        slice                    e f g h
        new state                e f g h

    So the incoming state matters exactly when a slice is shorter than `d_conv`.
    That is not an edge case to design around -- a prompt of any length can end
    in a short slice, since the split points are chosen by a token budget that
    knows nothing about `d_conv`. Reading only the slice in the first example
    would produce `[0 0 e f]`, and the next call would convolve `e` and `f`
    against zeros instead of against `c` and `d`.

    Zero-length (padding) requests take every column from the incoming state and
    so reproduce it unchanged.

    Every shape here is fixed by the padded request count and `d_conv`, and no
    value crosses to the host, so this is safe to capture in a CUDA graph.

    Args:
        x: Packed slice tokens `(total_tokens, conv_dim)`.
        cu_seqlens: Slice boundaries `(num_requests + 1,)`. Only read when the
            gather plan below is not supplied.
        previous_states: Incoming per-request states `(num_requests, conv_dim,
            d_conv)`, in the same request order as `cu_seqlens`.
        token_indices: Optional precomputed `(num_requests, d_conv)` token index
            per state column. The plan depends only on `cu_seqlens` and `d_conv`,
            so dynamic batching builds it once per step in `MambaMetadata` rather
            than paying the index arithmetic once per SSM layer. Must be passed
            together with `prev_columns` and `from_slice`.
        prev_columns: Optional precomputed `(num_requests, d_conv)` column of
            `previous_states` per state column.
        from_slice: Optional precomputed `(num_requests, d_conv)` mask selecting
            `token_indices` over `prev_columns`.

    Returns:
        The updated states, `(num_requests, conv_dim, d_conv)`.
    """
    num_requests, conv_dim, d_conv = previous_states.shape

    have_plan = token_indices is not None
    assert (prev_columns is not None) == have_plan and (from_slice is not None) == have_plan, (
        "the precomputed carry plan must be passed in full or not at all; got "
        f"token_indices={token_indices is not None}, "
        f"prev_columns={prev_columns is not None}, from_slice={from_slice is not None}"
    )

    total_tokens = x.shape[0]
    if total_tokens == 0:
        # Nothing to append, so every state is unchanged. Branching on a shape
        # (not a value) keeps this safe to capture: a graph is captured at a
        # fixed padded token count.
        return previous_states.clone()

    if token_indices is None:
        assert cu_seqlens.shape[0] == num_requests + 1, (
            f"cu_seqlens describes {cu_seqlens.shape[0] - 1} requests but "
            f"previous_states has {num_requests}"
        )
        starts = cu_seqlens[:-1].to(torch.long)
        lengths = cu_seqlens[1:].to(torch.long) - starts
        columns = torch.arange(d_conv, device=x.device, dtype=torch.long)

        # Offset of each state column from the slice start; negative means the
        # token predates this slice and must come from the previous state.
        offsets = lengths[:, None] - d_conv + columns[None, :]  # [N, d_conv]
        from_slice = offsets >= 0
        # offsets >= -d_conv always, so both clamps only guard the entries that
        # `from_slice` discards.
        token_indices = (starts[:, None] + offsets).clamp_(min=0)
        prev_columns = (offsets + d_conv).clamp_(0, d_conv - 1)
    else:
        expected = (num_requests, d_conv)
        for name, plan in (
            ("token_indices", token_indices),
            ("prev_columns", prev_columns),
            ("from_slice", from_slice),
        ):
            assert plan.shape == expected, (
                f"carry plan {name} describes {tuple(plan.shape)} but previous_states "
                f"has {expected}"
            )

    # Columns that `from_slice` discards may name a token outside `x` -- the plan
    # resolves them to whatever the clamped arithmetic produced, and `torch.where`
    # below throws the value away. Rewrite those lanes to 0 so the gather stays in
    # bounds. A live (`from_slice=True`) column that lands out of range is a caller
    # bug, not something to absorb: sending it to token 0 rather than clamping it
    # to the last token keeps it from quietly folding another request's final token
    # into this request's state. The check stays on device, so no host sync.
    in_range = token_indices < total_tokens
    token_indices = torch.where(in_range, token_indices, torch.zeros_like(token_indices))
    slice_values = x[token_indices].transpose(1, 2)  # [N, conv_dim, d_conv]

    previous_values = previous_states.gather(
        2, prev_columns[:, None, :].expand(num_requests, conv_dim, d_conv)
    )

    return torch.where(from_slice[:, None, :], slice_values, previous_values)
