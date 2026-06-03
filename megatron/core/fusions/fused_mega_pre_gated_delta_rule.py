# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Mega fused pre-gated-delta-rule kernels.

This is the "mega" sibling of :mod:`fused_pre_gated_delta_rule`. The streamed
path splits the pre-gated-delta-rule front-end into four separate Triton launch
scopes (QK / V / Z / g-beta) plus an external conv backward, optimized for
per-kernel quality and overlap under CUDA-graph capture. The mega path instead
folds **all forward tasks into a single Triton launch** using a flat logical
task space, trading a little per-kernel efficiency for far fewer host-side
launches. It is the right choice for non-CUDA-graph recipes where launch
overhead is visible in the trace.

Public contract is identical across unfused / streamed / mega:
``(query, key, value, gate, beta, g)``.

Design notes:

* The forward kernel maps ``program_id(0)`` onto a flat row space partitioned
  into QK, V, Z, and g/beta ranges; ``program_id(1)`` tiles the sequence axis.
  Each program inspects its row id and runs exactly one task body. This keeps
  every sub-computation in one launch while still letting Triton schedule
  memory-bound (Z copy) and compute-bound (QK/V conv) tiles concurrently on the
  SMs.
* Numerics mirror the streamed/unfused reference **bit-for-bit within the unit
  test tolerance**: the conv accumulator is rounded through the activation
  dtype before SiLU; the SiLU output is rounded again before the L2-norm
  reduction; ``g`` uses an fp32 ``log(1+exp(...))`` softplus; ``beta`` uses an
  fp32 sigmoid. The QK ``silu(conv(x))`` intermediate is persisted channel-last
  exactly as the streamed path saves it, so the backward can be shared.
* The kernel assumes ``key_head_dim == value_head_dim`` so a single
  ``HEAD_DIM`` constexpr drives the QK/V/Z channel tiles (true for the GDN
  production shapes and the unit tests). This is asserted at the Python entry.

The backward mirrors the forward: a single fused Triton kernel folds the four
streamed branch backward scopes (QK l2norm/repeat, V layout, Z layout, g/beta
chain rule) into one flat-task launch, then the depthwise conv input/weight
gradients are delegated to the same external ``causal_conv1d_bwd_function`` the
streamed path uses (its hand-tuned C++ remains the conv-backward anchor). That
is two launches total (one fused branch kernel + one external conv backward),
down from the streamed path's five, while staying numerically bit-identical to
the streamed branch kernels.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor

# Reuse the streamed module's validated constants and the external conv backward
# binding. Importing is not a modification of that module.
from megatron.core.fusions.fused_pre_gated_delta_rule import (
    _L2NORM_EPS,
    _causal_conv1d_bwd_function,
    _is_power_of_two,
    _resolve_packed_seq_idx,
)


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------


def _mega_autotune_configs():
    return [
        triton.Config({"BLOCK_S": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_S": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_S": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_S": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_S": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_S": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_S": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_S": 256}, num_warps=8, num_stages=2),
    ]


@triton.jit
def _mega_seq_bounds(cu_seqlens_ptr, token_offsets, total_tokens, num_packed_seqs):
    """Lane-wise packed-sequence [start, end) bounds for flattened THD tokens.

    Local copy of the streamed helper so this kernel never depends on
    cross-module ``@triton.jit`` symbol resolution.
    """

    safe_tokens = tl.minimum(token_offsets, total_tokens - 1)
    seq_start = token_offsets * 0
    seq_end = token_offsets * 0 + total_tokens

    seq_id = 0
    while seq_id < num_packed_seqs:
        start = tl.load(cu_seqlens_ptr + seq_id)
        end = tl.load(cu_seqlens_ptr + seq_id + 1)
        in_seq = (safe_tokens >= start) & (safe_tokens < end)
        seq_start = tl.where(in_seq, start, seq_start)
        seq_end = tl.where(in_seq, end, seq_end)
        seq_id += 1

    return seq_start, seq_end


@triton.autotune(
    configs=_mega_autotune_configs(),
    key=["seq_len", "HEAD_DIM", "K_W", "num_key_heads", "num_value_heads", "REPEAT", "HAS_THD"],
)
@triton.jit
def _mega_forward_kernel(
    qkvzba_ptr,
    weight_ptr,
    A_log_ptr,
    dt_bias_ptr,
    qk_out_ptr,
    value_ptr,
    gate_ptr,
    g_ptr,
    beta_ptr,
    silu_save_ptr,
    cu_seqlens_ptr,
    seq_len,
    num_packed_seqs,
    num_key_heads,
    num_value_heads,
    qk_channels,
    v_channels,
    R_qk,
    R_v,
    R_z,
    qkvzba_s_stride,
    qkvzba_b_stride,
    qkvzba_c_stride,
    weight_c_stride,
    weight_w_stride,
    qk_g_stride,
    qk_b_stride,
    qk_s_stride,
    qk_h_stride,
    v_b_stride,
    v_s_stride,
    v_h_stride,
    z_b_stride,
    z_s_stride,
    z_h_stride,
    g_b_stride,
    g_s_stride,
    g_h_stride,
    beta_b_stride,
    beta_s_stride,
    beta_h_stride,
    silu_b_stride,
    silu_c_stride,
    silu_s_stride,
    eps,
    HEAD_DIM: tl.constexpr,
    K_W: tl.constexpr,
    REPEAT: tl.constexpr,
    HAS_THD: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """All-in-one forward for the pre-gated-delta-rule front-end.

    Flat task space on ``program_id(0)``:
        rows [0, R_qk)            -> QK conv+silu+l2norm+repeat
        rows [R_qk, R_qk+R_v)     -> V conv+silu
        rows [R_qk+R_v, +R_z)     -> Z copy
        rows [..,  end)           -> g/beta
    ``program_id(1)`` tiles the (flattened, for THD) sequence axis.
    """

    pid_row = tl.program_id(0)
    pid_s = tl.program_id(1)

    # Common rounding dtype (activation dtype, e.g. bf16). All q/k/v/gate/beta
    # outputs share this; g is fp32.
    out_ty = qk_out_ptr.dtype.element_ty

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < seq_len
    chan_off = tl.arange(0, HEAD_DIM)

    R_qkv = R_qk + R_v
    R_qkvz = R_qkv + R_z

    if pid_row < R_qk:
        # ---- QK: depthwise causal conv + silu + l2norm + head repeat ----
        local = pid_row
        heads_per_batch = 2 * num_key_heads
        batch_id = local // heads_per_batch
        lb = local - batch_id * heads_per_batch
        group_id = lb // num_key_heads  # 0 -> Q, 1 -> K
        head_id = lb - group_id * num_key_heads
        chan = group_id * qk_channels + head_id * HEAD_DIM + chan_off

        if HAS_THD:
            seq_start, seq_end = _mega_seq_bounds(
                cu_seqlens_ptr, s_offs, seq_len, num_packed_seqs
            )

        acc = tl.zeros([BLOCK_S, HEAD_DIM], dtype=tl.float32)
        for i in tl.static_range(K_W):
            x_s = s_offs - (K_W - 1) + i
            if HAS_THD:
                x_mask = s_mask & (x_s >= seq_start) & (x_s < seq_end)
                safe_x_s = tl.minimum(tl.maximum(x_s, 0), seq_len - 1)
            else:
                x_mask = (x_s >= 0) & (x_s < seq_len)
                safe_x_s = x_s
            x_ptr = (
                qkvzba_ptr
                + safe_x_s[:, None] * qkvzba_s_stride
                + batch_id * qkvzba_b_stride
                + chan[None, :] * qkvzba_c_stride
            )
            x_val = tl.load(x_ptr, mask=x_mask[:, None], other=0.0).to(tl.float32)
            w_tap = tl.load(
                weight_ptr + chan * weight_c_stride + i * weight_w_stride
            ).to(tl.float32)
            acc += w_tap[None, :] * x_val

        acc = acc.to(out_ty).to(tl.float32)  # F.conv1d rounding
        silu_out = acc * tl.sigmoid(acc)
        silu_out = silu_out.to(out_ty).to(tl.float32)  # round before l2norm

        # Persist silu(conv(x)) for the QK channels, channel-last (b, 2*qk, s).
        silu_chan = group_id * qk_channels + head_id * HEAD_DIM + chan_off
        silu_ptrs = (
            silu_save_ptr
            + batch_id * silu_b_stride
            + silu_chan[None, :] * silu_c_stride
            + s_offs[:, None] * silu_s_stride
        )
        tl.store(
            silu_ptrs,
            silu_out.to(silu_save_ptr.dtype.element_ty),
            mask=s_mask[:, None],
        )

        norm_sq = tl.sum(silu_out * silu_out, axis=1)
        rstd = 1.0 / tl.sqrt(norm_sq + eps)
        out_typed = (silu_out * rstd[:, None]).to(out_ty)

        for r in tl.static_range(REPEAT):
            v_head = head_id * REPEAT + r
            write_ptr = (
                qk_out_ptr
                + group_id * qk_g_stride
                + batch_id * qk_b_stride
                + s_offs[:, None] * qk_s_stride
                + v_head * qk_h_stride
                + chan_off[None, :]
            )
            tl.store(write_ptr, out_typed, mask=s_mask[:, None])

    elif pid_row < R_qkv:
        # ---- V: depthwise causal conv + silu (no l2norm, no repeat) ----
        local = pid_row - R_qk
        batch_id = local // num_value_heads
        head_id = local - batch_id * num_value_heads
        chan = 2 * qk_channels + head_id * HEAD_DIM + chan_off

        if HAS_THD:
            seq_start, seq_end = _mega_seq_bounds(
                cu_seqlens_ptr, s_offs, seq_len, num_packed_seqs
            )

        acc = tl.zeros([BLOCK_S, HEAD_DIM], dtype=tl.float32)
        for i in tl.static_range(K_W):
            x_s = s_offs - (K_W - 1) + i
            if HAS_THD:
                x_mask = s_mask & (x_s >= seq_start) & (x_s < seq_end)
                safe_x_s = tl.minimum(tl.maximum(x_s, 0), seq_len - 1)
            else:
                x_mask = (x_s >= 0) & (x_s < seq_len)
                safe_x_s = x_s
            x_ptr = (
                qkvzba_ptr
                + safe_x_s[:, None] * qkvzba_s_stride
                + batch_id * qkvzba_b_stride
                + chan[None, :] * qkvzba_c_stride
            )
            x_val = tl.load(x_ptr, mask=x_mask[:, None], other=0.0).to(tl.float32)
            w_tap = tl.load(
                weight_ptr + chan * weight_c_stride + i * weight_w_stride
            ).to(tl.float32)
            acc += w_tap[None, :] * x_val

        acc = acc.to(out_ty).to(tl.float32)
        silu_out = acc * tl.sigmoid(acc)
        out_typed = silu_out.to(out_ty)
        write_ptr = (
            value_ptr
            + batch_id * v_b_stride
            + s_offs[:, None] * v_s_stride
            + head_id * v_h_stride
            + chan_off[None, :]
        )
        tl.store(write_ptr, out_typed, mask=s_mask[:, None])

    elif pid_row < R_qkvz:
        # ---- Z: copy qkvzba z slice into the final gate layout ----
        local = pid_row - R_qkv
        batch_id = local // num_value_heads
        head_id = local - batch_id * num_value_heads
        z_chan = 2 * qk_channels + v_channels + head_id * HEAD_DIM + chan_off
        src_ptr = (
            qkvzba_ptr
            + s_offs[:, None] * qkvzba_s_stride
            + batch_id * qkvzba_b_stride
            + z_chan[None, :] * qkvzba_c_stride
        )
        z_val = tl.load(src_ptr, mask=s_mask[:, None])
        write_ptr = (
            gate_ptr
            + batch_id * z_b_stride
            + s_offs[:, None] * z_s_stride
            + head_id * z_h_stride
            + chan_off[None, :]
        )
        tl.store(write_ptr, z_val, mask=s_mask[:, None])

    else:
        # ---- g/beta: -exp(A_log)*softplus(alpha+dt_bias) and sigmoid(beta) ----
        local = pid_row - R_qkvz
        batch_id = local // num_value_heads
        head_id = local - batch_id * num_value_heads
        beta_chan = 2 * qk_channels + 2 * v_channels + head_id
        alpha_chan = beta_chan + num_value_heads

        alpha_ptr = (
            qkvzba_ptr
            + s_offs * qkvzba_s_stride
            + batch_id * qkvzba_b_stride
            + alpha_chan * qkvzba_c_stride
        )
        beta_raw_ptr = (
            qkvzba_ptr
            + s_offs * qkvzba_s_stride
            + batch_id * qkvzba_b_stride
            + beta_chan * qkvzba_c_stride
        )
        alpha = tl.load(alpha_ptr, mask=s_mask, other=0.0).to(tl.float32)
        beta_raw = tl.load(beta_raw_ptr, mask=s_mask, other=0.0).to(tl.float32)
        A_log = tl.load(A_log_ptr + head_id).to(tl.float32)
        dt_bias = tl.load(dt_bias_ptr + head_id).to(tl.float32)

        pre = alpha + dt_bias
        softplus_val = tl.log(1.0 + tl.exp(pre))
        g = -tl.exp(A_log) * softplus_val
        beta_sig = tl.sigmoid(beta_raw)

        g_store_ptr = (
            g_ptr + batch_id * g_b_stride + s_offs * g_s_stride + head_id * g_h_stride
        )
        beta_store_ptr = (
            beta_ptr
            + batch_id * beta_b_stride
            + s_offs * beta_s_stride
            + head_id * beta_h_stride
        )
        tl.store(g_store_ptr, g.to(g_ptr.dtype.element_ty), mask=s_mask)
        tl.store(beta_store_ptr, beta_sig.to(beta_ptr.dtype.element_ty), mask=s_mask)


# ---------------------------------------------------------------------------
# Forward orchestration
# ---------------------------------------------------------------------------


def _mega_pre_gated_delta_rule_forward(
    qkvzba: Tensor,
    conv1d_weight: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
    *,
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    cu_seqlens: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Single-launch mega forward.

    Returns ``(query, key, value, gate, beta, g, silu_qk_save)``; the last
    element is the bf16-rounded QK ``silu(conv(x))`` laid out channel-last,
    matching the streamed forward so the shared backward can consume it.
    """

    seq_len, batch, total_channels = qkvzba.shape
    is_packed_thd = cu_seqlens is not None
    num_packed_seqs = (cu_seqlens.shape[0] - 1) if is_packed_thd else 0

    assert key_head_dim == value_head_dim, (
        "fused_mega_pre_gated_delta_rule currently requires "
        f"key_head_dim == value_head_dim; got {key_head_dim=} {value_head_dim=}."
    )
    assert _is_power_of_two(key_head_dim), (
        f"Mega kernel expects key_head_dim to be a power of two; got {key_head_dim=}."
    )
    head_dim = key_head_dim

    qk_channels = num_key_heads * key_head_dim
    v_channels = num_value_heads * value_head_dim
    repeat_factor = num_value_heads // num_key_heads
    k_w = conv1d_weight.shape[-1]

    expected_channels = 2 * qk_channels + 2 * v_channels + 2 * num_value_heads
    assert total_channels == expected_channels, (
        f"qkvzba last-dim mismatch: got {total_channels}, expected {expected_channels}."
    )

    out_dtype = qkvzba.dtype
    device = qkvzba.device

    # Output buffers (identical layouts to the streamed path).
    qk_out = torch.empty(
        2, batch, seq_len, num_value_heads, key_head_dim, dtype=out_dtype, device=device
    )
    query = qk_out[0]
    key = qk_out[1]
    value = torch.empty(
        batch, seq_len, num_value_heads, value_head_dim, dtype=out_dtype, device=device
    )
    gate = torch.empty(
        batch, seq_len, num_value_heads, value_head_dim, dtype=out_dtype, device=device
    )
    g = torch.empty(batch, seq_len, num_value_heads, dtype=torch.float32, device=device)
    beta = torch.empty(batch, seq_len, num_value_heads, dtype=out_dtype, device=device)

    # QK silu(conv(x)) persisted channel-last: (b, 2*qk_channels, s), stride(1)==1.
    silu_qk_save = torch.empty(
        (batch, seq_len, 2 * qk_channels), dtype=out_dtype, device=device
    ).permute(0, 2, 1)

    weight_2d = conv1d_weight.view(conv1d_weight.shape[0], k_w)

    # Flat task-space row partition.
    R_qk = batch * 2 * num_key_heads
    R_v = batch * num_value_heads
    R_z = batch * num_value_heads
    R_gb = batch * num_value_heads
    num_rows = R_qk + R_v + R_z + R_gb

    cu_seqlens_arg = cu_seqlens if is_packed_thd else qkvzba  # dummy when dense

    grid = lambda meta: (num_rows, triton.cdiv(seq_len, meta["BLOCK_S"]))
    _mega_forward_kernel[grid](
        qkvzba,
        weight_2d,
        A_log,
        dt_bias,
        qk_out,
        value,
        gate,
        g,
        beta,
        silu_qk_save,
        cu_seqlens_arg,
        seq_len,
        num_packed_seqs,
        num_key_heads,
        num_value_heads,
        qk_channels,
        v_channels,
        R_qk,
        R_v,
        R_z,
        qkvzba.stride(0),
        qkvzba.stride(1),
        qkvzba.stride(2),
        weight_2d.stride(0),
        weight_2d.stride(1),
        qk_out.stride(0),
        qk_out.stride(1),
        qk_out.stride(2),
        qk_out.stride(3),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        gate.stride(0),
        gate.stride(1),
        gate.stride(2),
        g.stride(0),
        g.stride(1),
        g.stride(2),
        beta.stride(0),
        beta.stride(1),
        beta.stride(2),
        silu_qk_save.stride(0),
        silu_qk_save.stride(1),
        silu_qk_save.stride(2),
        _L2NORM_EPS,
        HEAD_DIM=head_dim,
        K_W=k_w,
        REPEAT=repeat_factor,
        HAS_THD=is_packed_thd,
    )

    return query, key, value, gate, beta, g, silu_qk_save


# ---------------------------------------------------------------------------
# Backward kernel
# ---------------------------------------------------------------------------


def _mega_backward_autotune_configs():
    return [
        triton.Config({"BLOCK_S": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_S": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_S": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_S": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_S": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_S": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_S": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_S": 256}, num_warps=8, num_stages=2),
    ]


@triton.autotune(
    configs=_mega_backward_autotune_configs(),
    key=["seq_len", "HEAD_DIM", "REPEAT", "num_key_heads", "num_value_heads"],
    # The g/beta task atomic-adds per-head partials into these accumulators.
    # reset_to_zero clears them before each autotune trial so trials don't stack.
    reset_to_zero=["d_A_log_ptr", "d_dt_bias_ptr"],
)
@triton.jit
def _mega_backward_kernel(
    # inputs
    dq_ptr,
    dk_ptr,
    dv_ptr,
    dgate_ptr,
    dg_ptr,
    dbeta_ptr,
    silu_save_ptr,
    qkvzba_ptr,
    A_log_ptr,
    dt_bias_ptr,
    # outputs
    d_silu_conv_ptr,
    d_qkvzba_ptr,
    d_A_log_ptr,
    d_dt_bias_ptr,
    # sizes / layout
    seq_len,
    num_key_heads,
    num_value_heads,
    qk_channels,
    v_channels,
    R_qk,
    R_v,
    R_z,
    eps,
    # dq / dk strides (b, s, h, d)
    dq_b_stride,
    dq_s_stride,
    dq_h_stride,
    dk_b_stride,
    dk_s_stride,
    dk_h_stride,
    # dv strides (b, s, h, d)
    dv_b_stride,
    dv_s_stride,
    dv_h_stride,
    # dgate strides (b, s, h, d)
    dgate_b_stride,
    dgate_s_stride,
    dgate_h_stride,
    # dg / dbeta strides (b, s, h)
    dg_b_stride,
    dg_s_stride,
    dg_h_stride,
    dbeta_b_stride,
    dbeta_s_stride,
    dbeta_h_stride,
    # silu_save strides (b, 2*qk, s)
    silu_b_stride,
    silu_c_stride,
    silu_s_stride,
    # qkvzba / d_qkvzba strides (s, b, C)
    qkvzba_s_stride,
    qkvzba_b_stride,
    qkvzba_c_stride,
    # d_silu_conv strides (b, conv_dim, s)
    dsc_b_stride,
    dsc_c_stride,
    dsc_s_stride,
    HEAD_DIM: tl.constexpr,
    REPEAT: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """All-in-one backward for the QK / V / Z / g-beta branches.

    Mirrors the four streamed branch kernels in one flat task space. Conv input
    and weight gradients are NOT produced here; the caller feeds ``d_silu_conv``
    into the external ``causal_conv1d_bwd_function`` exactly as the streamed
    path does. Flat task space on ``program_id(0)``:
        rows [0, R_qk)        -> QK l2norm + repeat backward -> d_silu_conv[Q/K]
        rows [R_qk, +R_v)     -> V layout copy             -> d_silu_conv[V]
        rows [.., +R_z)       -> Z layout copy             -> d_qkvzba[z]
        rows [.., end)        -> g/beta chain rule         -> d_qkvzba[alpha,beta],
                                                              atomic d_A_log/d_dt_bias
    """

    pid_row = tl.program_id(0)
    pid_s = tl.program_id(1)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < seq_len
    chan_off = tl.arange(0, HEAD_DIM)

    R_qkv = R_qk + R_v
    R_qkvz = R_qkv + R_z

    if pid_row < R_qk:
        # ---- QK: repeat-reduce + l2norm backward -> d_silu_conv[Q/K] ----
        local = pid_row
        heads_per_batch = 2 * num_key_heads
        batch_id = local // heads_per_batch
        lb = local - batch_id * heads_per_batch
        group_id = lb // num_key_heads
        head_id = lb - group_id * num_key_heads
        is_query = group_id == 0
        is_key = group_id == 1
        chan = group_id * qk_channels + head_id * HEAD_DIM + chan_off

        d_normed = tl.zeros([BLOCK_S, HEAD_DIM], dtype=tl.float32)
        for r in tl.static_range(REPEAT):
            v_head = head_id * REPEAT + r
            dq_ptrs = (
                dq_ptr
                + batch_id * dq_b_stride
                + s_offs[:, None] * dq_s_stride
                + v_head * dq_h_stride
                + chan_off[None, :]
            )
            dk_ptrs = (
                dk_ptr
                + batch_id * dk_b_stride
                + s_offs[:, None] * dk_s_stride
                + v_head * dk_h_stride
                + chan_off[None, :]
            )
            d_normed += tl.load(dq_ptrs, mask=s_mask[:, None] & is_query, other=0.0).to(tl.float32)
            d_normed += tl.load(dk_ptrs, mask=s_mask[:, None] & is_key, other=0.0).to(tl.float32)

        silu_ptrs = (
            silu_save_ptr
            + batch_id * silu_b_stride
            + chan[None, :] * silu_c_stride
            + s_offs[:, None] * silu_s_stride
        )
        silu_bf16 = tl.load(silu_ptrs, mask=s_mask[:, None], other=0.0).to(tl.float32)

        norm_sq = tl.sum(silu_bf16 * silu_bf16, axis=1)
        rstd = 1.0 / tl.sqrt(norm_sq + eps)
        s_row = tl.sum(d_normed * silu_bf16, axis=1)
        rstd3 = rstd * rstd * rstd
        d_silu = rstd[:, None] * d_normed - rstd3[:, None] * silu_bf16 * s_row[:, None]

        dsc_ptrs = (
            d_silu_conv_ptr
            + batch_id * dsc_b_stride
            + chan[None, :] * dsc_c_stride
            + s_offs[:, None] * dsc_s_stride
        )
        tl.store(dsc_ptrs, d_silu.to(d_silu_conv_ptr.dtype.element_ty), mask=s_mask[:, None])

    elif pid_row < R_qkv:
        # ---- V: relayout dv -> d_silu_conv[V] ----
        local = pid_row - R_qk
        batch_id = local // num_value_heads
        head_id = local - batch_id * num_value_heads
        dv_ptrs = (
            dv_ptr
            + batch_id * dv_b_stride
            + s_offs[:, None] * dv_s_stride
            + head_id * dv_h_stride
            + chan_off[None, :]
        )
        dv_val = tl.load(dv_ptrs, mask=s_mask[:, None], other=0.0)
        dsc_chan = 2 * qk_channels + head_id * HEAD_DIM + chan_off
        dsc_ptrs = (
            d_silu_conv_ptr
            + batch_id * dsc_b_stride
            + dsc_chan[None, :] * dsc_c_stride
            + s_offs[:, None] * dsc_s_stride
        )
        tl.store(dsc_ptrs, dv_val, mask=s_mask[:, None])

    elif pid_row < R_qkvz:
        # ---- Z: relayout dgate -> d_qkvzba[z] ----
        local = pid_row - R_qkv
        batch_id = local // num_value_heads
        head_id = local - batch_id * num_value_heads
        dgate_ptrs = (
            dgate_ptr
            + batch_id * dgate_b_stride
            + s_offs[:, None] * dgate_s_stride
            + head_id * dgate_h_stride
            + chan_off[None, :]
        )
        dgate_val = tl.load(dgate_ptrs, mask=s_mask[:, None], other=0.0)
        dz_chan = 2 * qk_channels + v_channels + head_id * HEAD_DIM + chan_off
        dz_ptrs = (
            d_qkvzba_ptr
            + s_offs[:, None] * qkvzba_s_stride
            + batch_id * qkvzba_b_stride
            + dz_chan[None, :] * qkvzba_c_stride
        )
        tl.store(dz_ptrs, dgate_val, mask=s_mask[:, None])

    else:
        # ---- g/beta: chain rule -> d_qkvzba[alpha,beta] + atomic d_A_log/d_dt_bias ----
        local = pid_row - R_qkvz
        batch_id = local // num_value_heads
        head_id = local - batch_id * num_value_heads
        beta_chan = 2 * qk_channels + 2 * v_channels + head_id
        alpha_chan = beta_chan + num_value_heads

        alpha_ptrs = (
            qkvzba_ptr
            + s_offs * qkvzba_s_stride
            + batch_id * qkvzba_b_stride
            + alpha_chan * qkvzba_c_stride
        )
        beta_ptrs = (
            qkvzba_ptr
            + s_offs * qkvzba_s_stride
            + batch_id * qkvzba_b_stride
            + beta_chan * qkvzba_c_stride
        )
        alpha = tl.load(alpha_ptrs, mask=s_mask, other=0.0).to(tl.float32)
        beta_raw = tl.load(beta_ptrs, mask=s_mask, other=0.0).to(tl.float32)
        A_log = tl.load(A_log_ptr + head_id).to(tl.float32)
        dt_bias = tl.load(dt_bias_ptr + head_id).to(tl.float32)

        pre = alpha + dt_bias
        sigmoid_pre = tl.sigmoid(pre)
        softplus_pre = tl.log(1.0 + tl.exp(pre))
        exp_A = tl.exp(A_log)
        g = -exp_A * softplus_pre
        beta_sig = tl.sigmoid(beta_raw)

        dg_ptrs = (
            dg_ptr + batch_id * dg_b_stride + s_offs * dg_s_stride + head_id * dg_h_stride
        )
        dbeta_ptrs = (
            dbeta_ptr
            + batch_id * dbeta_b_stride
            + s_offs * dbeta_s_stride
            + head_id * dbeta_h_stride
        )
        d_g = tl.load(dg_ptrs, mask=s_mask, other=0.0).to(tl.float32)
        d_beta_out = tl.load(dbeta_ptrs, mask=s_mask, other=0.0).to(tl.float32)

        d_alpha = d_g * (-exp_A * sigmoid_pre)
        d_beta_raw = d_beta_out * beta_sig * (1.0 - beta_sig)

        d_alpha_ptrs = (
            d_qkvzba_ptr
            + s_offs * qkvzba_s_stride
            + batch_id * qkvzba_b_stride
            + alpha_chan * qkvzba_c_stride
        )
        d_beta_ptrs = (
            d_qkvzba_ptr
            + s_offs * qkvzba_s_stride
            + batch_id * qkvzba_b_stride
            + beta_chan * qkvzba_c_stride
        )
        tl.store(d_alpha_ptrs, d_alpha.to(d_qkvzba_ptr.dtype.element_ty), mask=s_mask)
        tl.store(d_beta_ptrs, d_beta_raw.to(d_qkvzba_ptr.dtype.element_ty), mask=s_mask)

        d_g_masked = tl.where(s_mask, d_g, 0.0)
        d_alpha_masked = tl.where(s_mask, d_alpha, 0.0)
        d_A_log_partial = tl.sum(d_g_masked * g)
        d_dt_bias_partial = tl.sum(d_alpha_masked)
        tl.atomic_add(d_A_log_ptr + head_id, d_A_log_partial)
        tl.atomic_add(d_dt_bias_ptr + head_id, d_dt_bias_partial)


# ---------------------------------------------------------------------------
# Backward orchestration
# ---------------------------------------------------------------------------


def _mega_pre_gated_delta_rule_backward(
    qkvzba: Tensor,
    conv1d_weight: Tensor,
    silu_qk_save: Tensor,
    dq: Tensor,
    dk: Tensor,
    dv: Tensor,
    dgate: Tensor,
    dbeta: Tensor,
    dg: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
    *,
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    seq_idx: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Two-launch mega backward: one fused branch kernel + external conv bwd.

    Collapses the four streamed branch kernels (QK l2norm/repeat, V layout, Z
    layout, g/beta chain rule) into a single Triton launch, then delegates the
    depthwise conv input/weight gradients to ``causal_conv1d_bwd_function`` as
    the streamed path does. Returns ``(d_qkvzba, d_weight, d_A_log, d_dt_bias)``.
    """

    seq_len, batch, _ = qkvzba.shape
    qk_channels = num_key_heads * key_head_dim
    v_channels = num_value_heads * value_head_dim
    conv_dim = 2 * qk_channels + v_channels
    k_w = conv1d_weight.shape[-1]
    device = qkvzba.device
    head_dim = key_head_dim

    weight_2d = conv1d_weight.view(conv1d_weight.shape[0], k_w)
    # Channel-last conv input view (stride(1)==1) — no copy.
    qkvzba_conv = qkvzba[:, :, :conv_dim].permute(1, 2, 0)

    # d_silu_conv channel-last (b, conv_dim, s), stride(1)==1.
    d_silu_conv = torch.empty(
        (batch, seq_len, conv_dim), dtype=qkvzba.dtype, device=device
    ).permute(0, 2, 1)
    d_qkvzba = torch.empty_like(qkvzba)
    d_A_log_fp32 = torch.zeros(num_value_heads, dtype=torch.float32, device=device)
    d_dt_bias_fp32 = torch.zeros(num_value_heads, dtype=torch.float32, device=device)

    R_qk = batch * 2 * num_key_heads
    R_v = batch * num_value_heads
    R_z = batch * num_value_heads
    R_gb = batch * num_value_heads
    num_rows = R_qk + R_v + R_z + R_gb

    grid = lambda meta: (num_rows, triton.cdiv(seq_len, meta["BLOCK_S"]))
    _mega_backward_kernel[grid](
        dq,
        dk,
        dv,
        dgate,
        dg,
        dbeta,
        silu_qk_save,
        qkvzba,
        A_log,
        dt_bias,
        d_silu_conv,
        d_qkvzba,
        d_A_log_fp32,
        d_dt_bias_fp32,
        seq_len,
        num_key_heads,
        num_value_heads,
        qk_channels,
        v_channels,
        R_qk,
        R_v,
        R_z,
        _L2NORM_EPS,
        dq.stride(0),
        dq.stride(1),
        dq.stride(2),
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),
        dv.stride(0),
        dv.stride(1),
        dv.stride(2),
        dgate.stride(0),
        dgate.stride(1),
        dgate.stride(2),
        dg.stride(0),
        dg.stride(1),
        dg.stride(2),
        dbeta.stride(0),
        dbeta.stride(1),
        dbeta.stride(2),
        silu_qk_save.stride(0),
        silu_qk_save.stride(1),
        silu_qk_save.stride(2),
        qkvzba.stride(0),
        qkvzba.stride(1),
        qkvzba.stride(2),
        d_silu_conv.stride(0),
        d_silu_conv.stride(1),
        d_silu_conv.stride(2),
        HEAD_DIM=head_dim,
        REPEAT=num_value_heads // num_key_heads,
    )

    # External conv backward: writes d_x into d_qkvzba's conv slice (strided
    # view, no copy) and returns d_weight. Same call shape as the streamed path.
    seq_stride = qkvzba.stride(0)
    batch_stride = qkvzba.stride(1)
    d_x_conv_view = d_qkvzba.as_strided(
        (batch, conv_dim, seq_len),
        (batch_stride, 1, seq_stride),
    )
    if _causal_conv1d_bwd_function is None:
        raise RuntimeError(
            "Fused pre-gated-delta-rule backward requires the 'causal_conv1d' package. "
            "Install it, or use pre_gated_delta_rule_impl='unfused'."
        )
    _, d_weight_fp32, _, _ = _causal_conv1d_bwd_function(
        qkvzba_conv,
        weight_2d,
        None,  # no bias
        d_silu_conv,
        seq_idx,
        None,  # initial_states
        None,  # dfinal_states
        d_x_conv_view,  # dx pre-allocated into d_qkvzba's conv slice
        False,  # return_dinitial_states
        True,  # activation (silu folded into conv bwd)
    )

    d_weight = d_weight_fp32.view(*conv1d_weight.shape).to(conv1d_weight.dtype)
    d_A_log = d_A_log_fp32.to(A_log.dtype)
    d_dt_bias = d_dt_bias_fp32.to(dt_bias.dtype)
    return d_qkvzba, d_weight, d_A_log, d_dt_bias


# ---------------------------------------------------------------------------
# Autograd wiring
# ---------------------------------------------------------------------------


class _FusedMegaPreGatedDeltaRuleFunction(torch.autograd.Function):
    """Autograd entry point for the mega path.

    Forward dispatches to the single-launch mega forward. Backward currently
    reuses the streamed conv-backend-delegated backward (which consumes the
    same saved ``silu_qk_save`` layout); a dedicated mega backward is layered
    in behind this same entry point.
    """

    @staticmethod
    def forward(
        ctx,
        qkvzba,
        conv1d_weight,
        A_log,
        dt_bias,
        cu_seqlens,
        seq_idx,
        num_key_heads,
        num_value_heads,
        key_head_dim,
        value_head_dim,
    ):
        ctx.num_key_heads = num_key_heads
        ctx.num_value_heads = num_value_heads
        ctx.key_head_dim = key_head_dim
        ctx.value_head_dim = value_head_dim
        query, key, value, gate, beta, g, silu_qk_save = (
            _mega_pre_gated_delta_rule_forward(
                qkvzba,
                conv1d_weight,
                A_log,
                dt_bias,
                num_key_heads=num_key_heads,
                num_value_heads=num_value_heads,
                key_head_dim=key_head_dim,
                value_head_dim=value_head_dim,
                cu_seqlens=cu_seqlens,
            )
        )
        ctx.has_seq_idx = seq_idx is not None
        if ctx.has_seq_idx:
            ctx.save_for_backward(qkvzba, conv1d_weight, A_log, dt_bias, silu_qk_save, seq_idx)
        else:
            ctx.save_for_backward(qkvzba, conv1d_weight, A_log, dt_bias, silu_qk_save)
        return query, key, value, gate, beta, g

    @staticmethod
    def backward(ctx, dq, dk, dv, dgate, dbeta, dg):
        if ctx.has_seq_idx:
            qkvzba, conv1d_weight, A_log, dt_bias, silu_qk_save, seq_idx = ctx.saved_tensors
        else:
            qkvzba, conv1d_weight, A_log, dt_bias, silu_qk_save = ctx.saved_tensors
            seq_idx = None
        d_qkvzba, d_weight, d_A_log, d_dt_bias = _mega_pre_gated_delta_rule_backward(
            qkvzba,
            conv1d_weight,
            silu_qk_save,
            dq,
            dk,
            dv,
            dgate,
            dbeta,
            dg,
            A_log,
            dt_bias,
            num_key_heads=ctx.num_key_heads,
            num_value_heads=ctx.num_value_heads,
            key_head_dim=ctx.key_head_dim,
            value_head_dim=ctx.value_head_dim,
            seq_idx=seq_idx,
        )
        return (d_qkvzba, d_weight, d_A_log, d_dt_bias, None, None, None, None, None, None)


def fused_mega_pre_gated_delta_rule(
    qkvzba: Tensor,
    conv1d_weight: Tensor,
    conv1d_bias: Optional[Tensor],
    A_log: Tensor,
    dt_bias: Tensor,
    *,
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    use_qk_l2norm: bool = True,
    cu_seqlens: Optional[Tensor] = None,
    seq_idx: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Mega fused pre-gated-delta-rule entry point.

    Args:
        qkvzba: ``[seq_len, batch, in_proj_dim]`` projection output.
        conv1d_weight: ``[conv_dim, 1, k_w]`` depthwise conv weight.
        conv1d_bias: Must be ``None`` in the mega path.
        A_log: ``[num_value_heads]`` raw decay parameter.
        dt_bias: ``[num_value_heads]`` time-step bias.
        num_key_heads / num_value_heads / key_head_dim / value_head_dim: GDN
            architecture parameters. ``num_value_heads`` must be a multiple of
            ``num_key_heads`` and ``key_head_dim == value_head_dim``.
        use_qk_l2norm: Must be ``True`` for parity with the streamed path.
        cu_seqlens: Optional packed THD cumulative sequence lengths.
        seq_idx: Optional precomputed token-to-sequence map for packed THD mode.

    Returns:
        ``(query, key, value, gate, beta, g)`` matching the unfused and streamed
        fused pre-GDR APIs.
    """

    assert qkvzba.is_cuda, (
        "fused_mega_pre_gated_delta_rule requires CUDA inputs; "
        f"got qkvzba.device={qkvzba.device}."
    )
    assert conv1d_bias is None, (
        "Conv bias is not supported by fused_mega_pre_gated_delta_rule "
        "(production GDN config has none)."
    )
    assert use_qk_l2norm, (
        "use_qk_l2norm=False is not supported by fused_mega_pre_gated_delta_rule "
        "(the backward closes over the l2norm path)."
    )
    assert num_value_heads % num_key_heads == 0, (
        f"{num_value_heads=} must be a multiple of {num_key_heads=}."
    )
    assert key_head_dim == value_head_dim, (
        "fused_mega_pre_gated_delta_rule currently requires "
        f"key_head_dim == value_head_dim; got {key_head_dim=} {value_head_dim=}."
    )
    if cu_seqlens is not None:
        assert cu_seqlens.is_cuda, (
            "Packed fused_mega_pre_gated_delta_rule requires CUDA cu_seqlens; "
            f"got cu_seqlens.device={cu_seqlens.device}."
        )
        assert cu_seqlens.dtype == torch.int32, (
            "Packed fused_mega_pre_gated_delta_rule requires int32 cu_seqlens; "
            f"got {cu_seqlens.dtype=}."
        )
        assert cu_seqlens.dim() == 1, (
            "Packed fused_mega_pre_gated_delta_rule expects 1-D cu_seqlens; "
            f"got {cu_seqlens.shape=}."
        )
        assert qkvzba.shape[1] == 1, (
            "Packed THD fused_mega_pre_gated_delta_rule expects batch dimension 1; "
            f"got qkvzba.shape={qkvzba.shape}."
        )
        assert cu_seqlens.shape[0] >= 2, (
            "Packed fused_mega_pre_gated_delta_rule requires at least one packed sequence; "
            f"got {cu_seqlens.shape=}."
        )
        assert cu_seqlens[0].item() == 0, (
            "Packed fused_mega_pre_gated_delta_rule requires cu_seqlens[0] == 0, "
            f"got {cu_seqlens[0].item()}."
        )
        assert cu_seqlens[-1].item() == qkvzba.shape[0], (
            "Packed fused_mega_pre_gated_delta_rule requires cu_seqlens[-1] to match "
            f"seq_len, got {cu_seqlens[-1].item()} vs {qkvzba.shape[0]}."
        )
        cu_seqlens = cu_seqlens.contiguous()
        seq_idx = _resolve_packed_seq_idx(cu_seqlens, seq_idx, qkvzba.shape[0])
    else:
        assert seq_idx is None, "seq_idx requires cu_seqlens for packed THD mode."

    return _FusedMegaPreGatedDeltaRuleFunction.apply(
        qkvzba,
        conv1d_weight,
        A_log,
        dt_bias,
        cu_seqlens,
        seq_idx,
        num_key_heads,
        num_value_heads,
        key_head_dim,
        value_head_dim,
    )
