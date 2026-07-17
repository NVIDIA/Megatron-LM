# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fused pre-gated-delta-rule projection kernels.

The public entry point consumes the dense ``qkvzba`` projection and returns
``query``, ``key``, ``value``, ``gate``, ``beta``, and ``g`` in the layouts
expected by the gated delta rule. The forward path keeps QK, V, Z, and
G/Beta as separate streamed scopes. The backward mirrors those scopes for
layout/l2norm/g-beta work, then delegates depthwise conv gradients to the
``causal_conv1d`` backend.

Unsupported cases are rejected at the Python entry point: CPU tensors,
conv bias, and ``use_qk_l2norm=False``. Packed THD sequences use separate
QK/V causal-conv kernels so the dense BSHD kernels stay free of packed
metadata and runtime branches.
"""

from contextlib import nullcontext
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor

try:
    from causal_conv1d.cpp_functions import causal_conv1d_bwd_function
except ImportError:
    causal_conv1d_bwd_function = None

_L2NORM_EPS = 1e-6

_QK_STREAM_SLOT = 0
_V_STREAM_SLOT = 2
_G_BETA_STREAM_SLOT = 3
_Z_STREAM_SLOT = 4

_LAYOUT_BLOCK_S = 64

# ---------------------------------------------------------------------------
# Forward kernels
# ---------------------------------------------------------------------------


def _conv_autotune_configs():
    return [
        triton.Config({"BLOCK_S": 16}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_S": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_S": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_S": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_S": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_S": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_S": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_S": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_S": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_S": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_S": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_S": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_S": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_S": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_S": 256}, num_warps=8, num_stages=2),
    ]


def _g_beta_autotune_configs():
    return [
        triton.Config({"BLOCK_S": 32, "BLOCK_H": 16}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_S": 64, "BLOCK_H": 16}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_S": 64, "BLOCK_H": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_S": 128, "BLOCK_H": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_S": 128, "BLOCK_H": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_S": 128, "BLOCK_H": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_S": 256, "BLOCK_H": 16}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_S": 256, "BLOCK_H": 32}, num_warps=8, num_stages=2),
    ]


@triton.jit
def _softplus_with_torch_threshold(x):
    """Match torch.nn.functional.softplus default threshold without exp overflow."""

    exp_input = tl.minimum(x, 20.0)
    return tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(exp_input)))


@triton.autotune(
    configs=_conv_autotune_configs(),
    key=["seq_len", "HEAD_DIM", "K_W", "APPLY_L2", "REPEAT", "NUM_GROUPS", "HAS_LEFT_BOUNDARY"],
)
@triton.jit
def _conv_silu_project_kernel(
    qkvzba_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    silu_save_ptr,
    left_boundary_ptr,
    seq_len,
    num_in_heads,
    in_channel_offset,
    in_group_stride,
    silu_save_chan_offset,
    silu_save_group_stride,
    qkvzba_s_stride,
    qkvzba_b_stride,
    qkvzba_c_stride,
    weight_c_stride,
    weight_w_stride,
    bias_stride,
    out_group_dim_stride,
    out_b_stride,
    out_s_stride,
    out_h_stride,
    silu_save_b_stride,
    silu_save_c_stride,
    silu_save_s_stride,
    left_boundary_s_stride,
    left_boundary_b_stride,
    left_boundary_c_stride,
    eps,
    HEAD_DIM: tl.constexpr,
    K_W: tl.constexpr,
    REPEAT: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_LEFT_BOUNDARY: tl.constexpr,
    APPLY_L2: tl.constexpr,
    SAVE_SILU: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Depthwise conv1d + silu + (optional l2norm) + (optional head repeat).

    Grid layout (program_id):
        0: batch * NUM_GROUPS * num_in_heads (flat)
        1: num_seq_blocks

    Args:
        in_channel_offset: starting channel index of the first group inside
            ``qkvzba``. 0 for QK, ``v_channel_offset`` for V.
        in_group_stride: channel distance between logical groups. For QK this
            is ``qk_channels`` so group 0 is Q and group 1 is K. For V this is
            0 because ``NUM_GROUPS == 1``.
        out_group_dim_stride: output-storage distance between logical groups. QK
            passes a grouped output buffer and V passes 0.
    """

    pid_bgh = tl.program_id(0)
    pid_s = tl.program_id(1)

    heads_per_batch = num_in_heads * NUM_GROUPS
    batch_id = pid_bgh // heads_per_batch
    local_bgh = pid_bgh - batch_id * heads_per_batch
    group_id = local_bgh // num_in_heads
    head_id = local_bgh - group_id * num_in_heads

    chan_off = tl.arange(0, HEAD_DIM)
    group_channel_offset = in_channel_offset + group_id * in_group_stride
    chan = group_channel_offset + head_id * HEAD_DIM + chan_off

    if HAS_BIAS:
        bias = tl.load(bias_ptr + chan * bias_stride).to(tl.float32)
    else:
        bias = tl.zeros([HEAD_DIM], dtype=tl.float32)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < seq_len

    acc = tl.zeros([BLOCK_S, HEAD_DIM], dtype=tl.float32)
    for i in tl.static_range(K_W):
        x_s = s_offs - (K_W - 1) + i
        x_mask = (x_s >= 0) & (x_s < seq_len)
        safe_x_s = tl.minimum(tl.maximum(x_s, 0), seq_len - 1)
        x_ptr = (
            qkvzba_ptr
            + safe_x_s[:, None] * qkvzba_s_stride
            + batch_id * qkvzba_b_stride
            + chan[None, :] * qkvzba_c_stride
        )
        x_val = tl.load(x_ptr, mask=x_mask[:, None], other=0.0).to(tl.float32)
        if HAS_LEFT_BOUNDARY:
            boundary_s = x_s + (K_W - 1)
            boundary_mask = (x_s < 0) & (boundary_s >= 0)
            safe_boundary_s = tl.maximum(boundary_s, 0)
            boundary_ptr = (
                left_boundary_ptr
                + safe_boundary_s[:, None] * left_boundary_s_stride
                + batch_id * left_boundary_b_stride
                + chan[None, :] * left_boundary_c_stride
            )
            boundary_val = tl.load(boundary_ptr, mask=boundary_mask[:, None], other=0.0).to(tl.float32)
            x_val = tl.where(x_mask[:, None], x_val, boundary_val)
        w_tap = tl.load(weight_ptr + chan * weight_c_stride + i * weight_w_stride).to(tl.float32)
        acc += w_tap[None, :] * x_val

    acc += bias[None, :]
    # Mimic the unfused F.conv1d rounding: the reference path stores the conv
    # output in the input dtype (bf16) before silu, so do the same here. This
    # keeps the fused output bit-aligned with the reference within one ULP.
    acc = acc.to(out_ptr.dtype.element_ty).to(tl.float32)
    silu_out = acc * tl.sigmoid(acc)

    if APPLY_L2:
        # F.silu rounds to the input dtype before l2norm reads it. Round-trip
        # via bf16 to match that precision.
        silu_out = silu_out.to(out_ptr.dtype.element_ty).to(tl.float32)
        if SAVE_SILU:
            # Persist only the QK silu output in the channel-last layout
            # consumed by the QK l2norm backward.
            silu_save_chan = (
                silu_save_chan_offset
                + group_id * silu_save_group_stride
                + head_id * HEAD_DIM
                + chan_off
            )
            silu_save_ptrs = (
                silu_save_ptr
                + batch_id * silu_save_b_stride
                + silu_save_chan[None, :] * silu_save_c_stride
                + s_offs[:, None] * silu_save_s_stride
            )
            tl.store(
                silu_save_ptrs, silu_out.to(silu_save_ptr.dtype.element_ty), mask=s_mask[:, None]
            )
        norm_sq = tl.sum(silu_out * silu_out, axis=1)
        rstd = 1.0 / tl.sqrt(norm_sq + eps)
        out = silu_out * rstd[:, None]
    else:
        # No l2norm follows. The final store→bf16 already does the rounding;
        # an intermediate bf16 round-trip would be redundant.
        out = silu_out

    out_typed = out.to(out_ptr.dtype.element_ty)

    # Write the same data to ``REPEAT`` adjacent value heads. ``REPEAT == 1``
    # is the no-repeat case (V branch is handled by a separate kernel that
    # always has REPEAT == 1, but using the same code here is convenient).
    for r in tl.static_range(REPEAT):
        v_head = head_id * REPEAT + r
        write_ptr = (
            out_ptr
            + group_id * out_group_dim_stride
            + batch_id * out_b_stride
            + s_offs[:, None] * out_s_stride
            + v_head * out_h_stride
            + chan_off[None, :]
        )
        tl.store(write_ptr, out_typed, mask=s_mask[:, None])


@triton.jit
def _thd_seq_bounds(cu_seqlens_ptr, token_offsets, total_tokens, num_packed_seqs):
    """Return lane-wise packed sequence bounds for flattened THD tokens."""

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
    configs=_conv_autotune_configs(),
    key=["seq_len", "HEAD_DIM", "K_W", "APPLY_L2", "REPEAT", "NUM_GROUPS", "HAS_LEFT_BOUNDARY"],
)
@triton.jit
def _conv_silu_project_thd_kernel(
    qkvzba_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    silu_save_ptr,
    left_boundary_ptr,
    cu_seqlens_ptr,
    seq_len,
    global_token_offset,
    global_seq_len,
    num_packed_seqs,
    num_in_heads,
    in_channel_offset,
    in_group_stride,
    silu_save_chan_offset,
    silu_save_group_stride,
    qkvzba_s_stride,
    qkvzba_b_stride,
    qkvzba_c_stride,
    weight_c_stride,
    weight_w_stride,
    bias_stride,
    out_group_dim_stride,
    out_b_stride,
    out_s_stride,
    out_h_stride,
    silu_save_b_stride,
    silu_save_c_stride,
    silu_save_s_stride,
    left_boundary_s_stride,
    left_boundary_b_stride,
    left_boundary_c_stride,
    eps,
    HEAD_DIM: tl.constexpr,
    K_W: tl.constexpr,
    REPEAT: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_LEFT_BOUNDARY: tl.constexpr,
    APPLY_L2: tl.constexpr,
    SAVE_SILU: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """THD depthwise conv1d + silu + optional l2norm/repeat.

    This is intentionally separate from ``_conv_silu_project_kernel`` so
    packed sequence boundary metadata never enters the dense BSHD hot path.
    Only the causal-conv loads use ``cu_seqlens``; the following per-token
    transforms and stores are identical to the dense path.
    """

    pid_bgh = tl.program_id(0)
    pid_s = tl.program_id(1)

    heads_per_batch = num_in_heads * NUM_GROUPS
    batch_id = pid_bgh // heads_per_batch
    local_bgh = pid_bgh - batch_id * heads_per_batch
    group_id = local_bgh // num_in_heads
    head_id = local_bgh - group_id * num_in_heads

    chan_off = tl.arange(0, HEAD_DIM)
    group_channel_offset = in_channel_offset + group_id * in_group_stride
    chan = group_channel_offset + head_id * HEAD_DIM + chan_off

    if HAS_BIAS:
        bias = tl.load(bias_ptr + chan * bias_stride).to(tl.float32)
    else:
        bias = tl.zeros([HEAD_DIM], dtype=tl.float32)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < seq_len
    global_s = global_token_offset + s_offs
    seq_start, seq_end = _thd_seq_bounds(
        cu_seqlens_ptr, global_s, global_seq_len, num_packed_seqs
    )

    acc = tl.zeros([BLOCK_S, HEAD_DIM], dtype=tl.float32)
    for i in tl.static_range(K_W):
        x_s = s_offs - (K_W - 1) + i
        global_x_s = global_token_offset + x_s
        x_mask = s_mask & (x_s >= 0) & (global_x_s >= seq_start) & (global_x_s < seq_end)
        safe_x_s = tl.minimum(tl.maximum(x_s, 0), seq_len - 1)
        x_ptr = (
            qkvzba_ptr
            + safe_x_s[:, None] * qkvzba_s_stride
            + batch_id * qkvzba_b_stride
            + chan[None, :] * qkvzba_c_stride
        )
        x_val = tl.load(x_ptr, mask=x_mask[:, None], other=0.0).to(tl.float32)
        if HAS_LEFT_BOUNDARY:
            boundary_s = x_s + (K_W - 1)
            boundary_mask = (
                s_mask
                & (x_s < 0)
                & (boundary_s >= 0)
                & (global_x_s >= seq_start)
                & (global_x_s < seq_end)
            )
            safe_boundary_s = tl.maximum(boundary_s, 0)
            boundary_ptr = (
                left_boundary_ptr
                + safe_boundary_s[:, None] * left_boundary_s_stride
                + batch_id * left_boundary_b_stride
                + chan[None, :] * left_boundary_c_stride
            )
            boundary_val = tl.load(boundary_ptr, mask=boundary_mask[:, None], other=0.0).to(tl.float32)
            x_val = tl.where(x_mask[:, None], x_val, boundary_val)
        w_tap = tl.load(weight_ptr + chan * weight_c_stride + i * weight_w_stride).to(tl.float32)
        acc += w_tap[None, :] * x_val

    acc += bias[None, :]
    acc = acc.to(out_ptr.dtype.element_ty).to(tl.float32)
    silu_out = acc * tl.sigmoid(acc)

    if APPLY_L2:
        silu_out = silu_out.to(out_ptr.dtype.element_ty).to(tl.float32)
        if SAVE_SILU:
            silu_save_chan = (
                silu_save_chan_offset
                + group_id * silu_save_group_stride
                + head_id * HEAD_DIM
                + chan_off
            )
            silu_save_ptrs = (
                silu_save_ptr
                + batch_id * silu_save_b_stride
                + silu_save_chan[None, :] * silu_save_c_stride
                + s_offs[:, None] * silu_save_s_stride
            )
            tl.store(
                silu_save_ptrs, silu_out.to(silu_save_ptr.dtype.element_ty), mask=s_mask[:, None]
            )
        norm_sq = tl.sum(silu_out * silu_out, axis=1)
        rstd = 1.0 / tl.sqrt(norm_sq + eps)
        out = silu_out * rstd[:, None]
    else:
        out = silu_out

    out_typed = out.to(out_ptr.dtype.element_ty)

    for r in tl.static_range(REPEAT):
        v_head = head_id * REPEAT + r
        write_ptr = (
            out_ptr
            + group_id * out_group_dim_stride
            + batch_id * out_b_stride
            + s_offs[:, None] * out_s_stride
            + v_head * out_h_stride
            + chan_off[None, :]
        )
        tl.store(write_ptr, out_typed, mask=s_mask[:, None])


@triton.jit
def _copy_z_kernel(
    qkvzba_ptr,
    gate_ptr,
    seq_len,
    num_v_heads,
    z_channel_offset,
    qkvzba_s_stride,
    qkvzba_b_stride,
    qkvzba_c_stride,
    gate_b_stride,
    gate_s_stride,
    gate_h_stride,
    HEAD_DIM: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Copy the z slice from qkvzba into the final gate layout."""

    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)

    batch_id = pid_bh // num_v_heads
    head_id = pid_bh - batch_id * num_v_heads

    chan_off = tl.arange(0, HEAD_DIM)
    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < seq_len

    z_chan = z_channel_offset + head_id * HEAD_DIM + chan_off
    z_src_ptr = (
        qkvzba_ptr
        + s_offs[:, None] * qkvzba_s_stride
        + batch_id * qkvzba_b_stride
        + z_chan[None, :] * qkvzba_c_stride
    )
    z_val = tl.load(z_src_ptr, mask=s_mask[:, None])
    z_write_ptr = (
        gate_ptr
        + batch_id * gate_b_stride
        + s_offs[:, None] * gate_s_stride
        + head_id * gate_h_stride
        + chan_off[None, :]
    )
    tl.store(z_write_ptr, z_val, mask=s_mask[:, None])


@triton.autotune(configs=_g_beta_autotune_configs(), key=["seq_len", "num_v_heads"])
@triton.jit
def _compute_g_and_beta_kernel(
    qkvzba_ptr,
    A_log_ptr,
    dt_bias_ptr,
    g_out_ptr,
    beta_out_ptr,
    seq_len,
    num_v_heads,
    beta_channel_offset,
    alpha_channel_offset,
    qkvzba_s_stride,
    qkvzba_b_stride,
    qkvzba_c_stride,
    g_b_stride,
    g_s_stride,
    g_h_stride,
    beta_b_stride,
    beta_s_stride,
    beta_h_stride,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Compute ``g = -exp(A_log) * softplus(alpha + dt_bias)`` and ``sigmoid(beta)``."""

    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    s_mask = s_offs < seq_len
    h_mask = h_offs < num_v_heads
    mask = s_mask[:, None] & h_mask[None, :]

    alpha_ptr = (
        qkvzba_ptr
        + s_offs[:, None] * qkvzba_s_stride
        + pid_b * qkvzba_b_stride
        + (alpha_channel_offset + h_offs[None, :]) * qkvzba_c_stride
    )
    beta_ptr = (
        qkvzba_ptr
        + s_offs[:, None] * qkvzba_s_stride
        + pid_b * qkvzba_b_stride
        + (beta_channel_offset + h_offs[None, :]) * qkvzba_c_stride
    )

    alpha = tl.load(alpha_ptr, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr, mask=mask, other=0.0).to(tl.float32)

    A_log = tl.load(A_log_ptr + h_offs, mask=h_mask, other=0.0).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + h_offs, mask=h_mask, other=0.0).to(tl.float32)

    pre = alpha + dt_bias[None, :]
    softplus_val = _softplus_with_torch_threshold(pre)
    g = -tl.exp(A_log)[None, :] * softplus_val
    beta_sig = tl.sigmoid(beta)

    g_ptr = (
        g_out_ptr + pid_b * g_b_stride + s_offs[:, None] * g_s_stride + h_offs[None, :] * g_h_stride
    )
    beta_out_ptr_calc = (
        beta_out_ptr
        + pid_b * beta_b_stride
        + s_offs[:, None] * beta_s_stride
        + h_offs[None, :] * beta_h_stride
    )
    tl.store(g_ptr, g.to(g_out_ptr.dtype.element_ty), mask=mask)
    tl.store(beta_out_ptr_calc, beta_sig.to(beta_out_ptr.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# Backward kernels
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_S": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_S": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_S": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_S": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_S": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_S": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_S": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_S": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_S": 256}, num_warps=8, num_stages=2),
    ],
    key=["seq_len", "HEAD_DIM", "REPEAT"],
)
@triton.jit
def _qk_l2norm_repeat_backward_kernel(
    dq_ptr,
    dk_ptr,
    silu_bf16_ptr,
    d_silu_bf16_ptr,
    seq_len,
    num_qk_heads,
    qk_channels,
    eps,
    dq_b_stride,
    dq_s_stride,
    dq_h_stride,
    dk_b_stride,
    dk_s_stride,
    dk_h_stride,
    silu_b_stride,
    silu_c_stride,
    silu_s_stride,
    d_silu_b_stride,
    d_silu_c_stride,
    d_silu_s_stride,
    HEAD_DIM: tl.constexpr,
    REPEAT: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Merged Q/K l2norm + REPEAT-way head broadcast backward."""

    pid_bgh = tl.program_id(0)
    pid_s = tl.program_id(1)

    heads_per_batch = num_qk_heads * 2
    batch_id = pid_bgh // heads_per_batch
    local_bgh = pid_bgh - batch_id * heads_per_batch
    group_id = local_bgh // num_qk_heads
    head_id = local_bgh - group_id * num_qk_heads
    is_query = group_id == 0
    is_key = group_id == 1

    chan_off = tl.arange(0, HEAD_DIM)
    chan = group_id * qk_channels + head_id * HEAD_DIM + chan_off

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < seq_len

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
        silu_bf16_ptr
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

    d_silu_ptrs = (
        d_silu_bf16_ptr
        + batch_id * d_silu_b_stride
        + chan[None, :] * d_silu_c_stride
        + s_offs[:, None] * d_silu_s_stride
    )
    tl.store(d_silu_ptrs, d_silu.to(d_silu_bf16_ptr.dtype.element_ty), mask=s_mask[:, None])


@triton.jit
def _v_layout_to_conv_kernel(
    dv_ptr,  # (b, s, num_v_heads, value_head_dim)
    d_silu_conv_ptr,  # (b, conv_dim, s) — write into V channel slice
    seq_len,
    num_v_heads,
    v_channel_offset,  # = 2 * qk_channels
    dv_b_stride,
    dv_s_stride,
    dv_h_stride,
    d_silu_b_stride,
    d_silu_c_stride,
    d_silu_s_stride,
    HEAD_DIM: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Write V-branch gradients into the conv-backward layout.

    ``dv`` is the gradient of ``value`` (forward layout
    ``(b, s, num_v_heads, value_head_dim)``). The conv backward needs
    ``d_silu_conv`` in layout ``(b, conv_dim, s)`` for the V channel
    slice.
    """

    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)

    batch_id = pid_bh // num_v_heads
    head_id = pid_bh - batch_id * num_v_heads

    chan_off = tl.arange(0, HEAD_DIM)
    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < seq_len

    # Read dv at (batch, s, head, chan).
    dv_ptrs = (
        dv_ptr
        + batch_id * dv_b_stride
        + s_offs[:, None] * dv_s_stride
        + head_id * dv_h_stride
        + chan_off[None, :]
    )
    dv_val = tl.load(dv_ptrs, mask=s_mask[:, None], other=0.0)

    # Write to d_silu_conv at (batch, v_channel_offset + head*HEAD_DIM + chan, s).
    d_silu_chan = v_channel_offset + head_id * HEAD_DIM + chan_off
    d_silu_ptrs = (
        d_silu_conv_ptr
        + batch_id * d_silu_b_stride
        + d_silu_chan[None, :] * d_silu_c_stride
        + s_offs[:, None] * d_silu_s_stride
    )
    tl.store(d_silu_ptrs, dv_val, mask=s_mask[:, None])


@triton.jit
def _z_layout_to_qkvzba_kernel(
    dgate_ptr,  # (b, s, num_v_heads, value_head_dim)
    d_qkvzba_ptr,  # (s, b, total_channels) — write into z channel slice
    seq_len,
    num_v_heads,
    z_channel_offset,  # = 2 * qk_channels + v_channels
    dgate_b_stride,
    dgate_s_stride,
    dgate_h_stride,
    d_qkvzba_s_stride,
    d_qkvzba_b_stride,
    d_qkvzba_c_stride,
    HEAD_DIM: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Write gate gradients into the z slice of ``d_qkvzba``.

    ``dgate`` is the autograd-supplied gradient of ``gate`` (= the z
    slice of qkvzba in forward) with layout
    ``(b, s, num_v_heads, value_head_dim)``. We need to write it into
    ``d_qkvzba``'s z slice — layout ``(s, b, total_channels)`` with
    channels in ``[z_channel_offset, z_channel_offset + v_channels)``.
    """

    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)

    batch_id = pid_bh // num_v_heads
    head_id = pid_bh - batch_id * num_v_heads

    chan_off = tl.arange(0, HEAD_DIM)
    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < seq_len

    # Read dgate at (batch, s, head, chan).
    dgate_ptrs = (
        dgate_ptr
        + batch_id * dgate_b_stride
        + s_offs[:, None] * dgate_s_stride
        + head_id * dgate_h_stride
        + chan_off[None, :]
    )
    dgate_val = tl.load(dgate_ptrs, mask=s_mask[:, None], other=0.0)

    # Write to d_qkvzba at (s, batch, z_channel_offset + head*HEAD_DIM + chan).
    d_qkvzba_chan = z_channel_offset + head_id * HEAD_DIM + chan_off
    d_qkvzba_ptrs = (
        d_qkvzba_ptr
        + s_offs[:, None] * d_qkvzba_s_stride
        + batch_id * d_qkvzba_b_stride
        + d_qkvzba_chan[None, :] * d_qkvzba_c_stride
    )
    tl.store(d_qkvzba_ptrs, dgate_val, mask=s_mask[:, None])


@triton.jit
def _load_virtual_conv_input(
    qkvzba_ptr,
    left_boundary_ptr,
    src,
    chan,
    chan_mask,
    batch_id,
    qkvzba_s_stride,
    qkvzba_b_stride,
    qkvzba_c_stride,
    left_boundary_s_stride,
    left_boundary_b_stride,
    left_boundary_c_stride,
    seq_len,
    valid_src,
    HAS_LEFT_BOUNDARY: tl.constexpr,
    BOUNDARY: tl.constexpr,
):
    local_mask = valid_src & (src >= 0) & (src < seq_len)
    safe_src = tl.minimum(tl.maximum(src, 0), seq_len - 1)
    local_ptr = (
        qkvzba_ptr
        + safe_src[:, None] * qkvzba_s_stride
        + batch_id * qkvzba_b_stride
        + chan[None, :] * qkvzba_c_stride
    )
    local_val = tl.load(
        local_ptr, mask=local_mask[:, None] & chan_mask[None, :], other=0.0
    ).to(tl.float32)
    if HAS_LEFT_BOUNDARY:
        boundary_s = src + BOUNDARY
        boundary_mask = valid_src & (src < 0) & (boundary_s >= 0)
        safe_boundary_s = tl.maximum(boundary_s, 0)
        boundary_ptr = (
            left_boundary_ptr
            + safe_boundary_s[:, None] * left_boundary_s_stride
            + batch_id * left_boundary_b_stride
            + chan[None, :] * left_boundary_c_stride
        )
        boundary_val = tl.load(
            boundary_ptr, mask=boundary_mask[:, None] & chan_mask[None, :], other=0.0
        ).to(tl.float32)
        return tl.where(local_mask[:, None], local_val, boundary_val)
    return local_val


@triton.jit
def _conv_silu_boundary_backward_kernel(
    qkvzba_ptr,
    weight_ptr,
    d_silu_ptr,
    d_qkvzba_ptr,
    d_weight_accum_ptr,
    left_boundary_ptr,
    d_left_boundary_ptr,
    cu_seqlens_ptr,
    seq_len,
    source_len,
    num_packed_seqs,
    global_token_offset,
    global_seq_len,
    conv_dim,
    qkvzba_s_stride,
    qkvzba_b_stride,
    qkvzba_c_stride,
    weight_c_stride,
    weight_w_stride,
    d_weight_accum_c_stride,
    d_weight_accum_w_stride,
    d_silu_b_stride,
    d_silu_c_stride,
    d_silu_s_stride,
    d_qkvzba_s_stride,
    d_qkvzba_b_stride,
    d_qkvzba_c_stride,
    left_boundary_s_stride,
    left_boundary_b_stride,
    left_boundary_c_stride,
    d_left_boundary_s_stride,
    d_left_boundary_b_stride,
    d_left_boundary_c_stride,
    BOUNDARY: tl.constexpr,
    K_W: tl.constexpr,
    IS_PACKED: tl.constexpr,
    APPLY_MAIN_CORRECTION: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Compute CP left-boundary grad and optional packed boundary correction."""

    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    src_offs = tl.arange(0, BLOCK_H)
    source_mask = src_offs < source_len
    src = src_offs - BOUNDARY
    boundary_s = src + BOUNDARY
    is_boundary_source = src < 0
    is_local_source = src >= 0
    global_src = global_token_offset + src
    if IS_PACKED:
        local_seq_start, local_seq_end = _thd_seq_bounds(
            cu_seqlens_ptr, global_token_offset, global_seq_len, num_packed_seqs
        )
        valid_boundary_source = (
            is_boundary_source & (global_src >= local_seq_start) & (global_src < local_seq_end)
        )
    else:
        valid_boundary_source = is_boundary_source
    valid_local_source = APPLY_MAIN_CORRECTION & is_local_source & (src < seq_len)
    source_valid = source_mask & (valid_boundary_source | valid_local_source)

    chan = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    chan_mask = chan < conv_dim
    d_src = tl.zeros((BLOCK_H, BLOCK_C), dtype=tl.float32)

    for i in tl.static_range(K_W):
        t = src + (K_W - 1) - i
        t_mask = source_valid & (t >= 0) & (t < seq_len)
        if APPLY_MAIN_CORRECTION:
            # Local-source corrections are only needed for outputs whose conv
            # window could have consumed the virtual left boundary.
            t_mask = t_mask & (is_boundary_source | (t < BOUNDARY))
        global_t = global_token_offset + t
        if IS_PACKED:
            seq_start, seq_end = _thd_seq_bounds(
                cu_seqlens_ptr, global_t, global_seq_len, num_packed_seqs
            )
            same_sequence = (global_src >= seq_start) & (global_src < seq_end)
        else:
            same_sequence = t_mask
        source_contributes = t_mask & same_sequence

        acc = tl.zeros((BLOCK_H, BLOCK_C), dtype=tl.float32)
        base_acc = tl.zeros((BLOCK_H, BLOCK_C), dtype=tl.float32)
        for j in tl.static_range(K_W):
            tap_src = t - (K_W - 1) + j
            global_tap_src = global_token_offset + tap_src
            if IS_PACKED:
                tap_valid = (
                    t_mask
                    & (global_tap_src >= seq_start)
                    & (global_tap_src < seq_end)
                    & (tap_src < seq_len)
                )
            else:
                tap_valid = t_mask & (tap_src < seq_len)
            x_tap = _load_virtual_conv_input(
                qkvzba_ptr,
                left_boundary_ptr,
                tap_src,
                chan,
                chan_mask,
                pid_b,
                qkvzba_s_stride,
                qkvzba_b_stride,
                qkvzba_c_stride,
                left_boundary_s_stride,
                left_boundary_b_stride,
                left_boundary_c_stride,
                seq_len,
                tap_valid,
                True,
                BOUNDARY,
            )
            w_tap = tl.load(
                weight_ptr + chan * weight_c_stride + j * weight_w_stride,
                mask=chan_mask,
                other=0.0,
            ).to(tl.float32)
            acc += x_tap * w_tap[None, :]
            if APPLY_MAIN_CORRECTION:
                local_tap_mask = tap_valid & (tap_src >= 0)
                safe_tap_src = tl.minimum(tl.maximum(tap_src, 0), seq_len - 1)
                x_base_ptrs = (
                    qkvzba_ptr
                    + safe_tap_src[:, None] * qkvzba_s_stride
                    + pid_b * qkvzba_b_stride
                    + chan[None, :] * qkvzba_c_stride
                )
                x_base = tl.load(
                    x_base_ptrs, mask=local_tap_mask[:, None] & chan_mask[None, :], other=0.0
                ).to(tl.float32)
                base_acc += x_base * w_tap[None, :]

        sig = tl.sigmoid(acc)
        silu_grad = sig * (1.0 + acc * (1.0 - sig))
        if APPLY_MAIN_CORRECTION:
            base_sig = tl.sigmoid(base_acc)
            base_silu_grad = base_sig * (1.0 + base_acc * (1.0 - base_sig))
        d_silu_ptrs = (
            d_silu_ptr
            + pid_b * d_silu_b_stride
            + chan[None, :] * d_silu_c_stride
            + tl.minimum(tl.maximum(t, 0), seq_len - 1)[:, None] * d_silu_s_stride
        )
        d_silu = tl.load(
            d_silu_ptrs, mask=source_contributes[:, None] & chan_mask[None, :], other=0.0
        ).to(tl.float32)
        d_acc = d_silu * silu_grad
        if APPLY_MAIN_CORRECTION:
            d_base_acc = d_silu * base_silu_grad
            d_acc = tl.where(is_boundary_source[:, None], d_acc, d_acc - d_base_acc)
        w_i = tl.load(
            weight_ptr + chan * weight_c_stride + i * weight_w_stride,
            mask=chan_mask,
            other=0.0,
        ).to(tl.float32)
        d_src += d_acc * w_i[None, :]
        if APPLY_MAIN_CORRECTION:
            x_src = _load_virtual_conv_input(
                qkvzba_ptr,
                left_boundary_ptr,
                src,
                chan,
                chan_mask,
                pid_b,
                qkvzba_s_stride,
                qkvzba_b_stride,
                qkvzba_c_stride,
                left_boundary_s_stride,
                left_boundary_b_stride,
                left_boundary_c_stride,
                seq_len,
                source_contributes,
                True,
                BOUNDARY,
            )
            d_w = tl.sum(d_acc * x_src, axis=0)
            tl.atomic_add(
                d_weight_accum_ptr
                + chan * d_weight_accum_c_stride
                + i * d_weight_accum_w_stride,
                d_w,
                mask=chan_mask,
            )

    d_boundary_ptrs = (
        d_left_boundary_ptr
        + boundary_s[:, None] * d_left_boundary_s_stride
        + pid_b * d_left_boundary_b_stride
        + chan[None, :] * d_left_boundary_c_stride
    )
    tl.store(
        d_boundary_ptrs,
        d_src.to(d_left_boundary_ptr.dtype.element_ty),
        mask=source_mask[:, None] & is_boundary_source[:, None] & chan_mask[None, :],
    )
    if APPLY_MAIN_CORRECTION:
        d_qkvzba_ptrs = (
            d_qkvzba_ptr
            + src[:, None] * d_qkvzba_s_stride
            + pid_b * d_qkvzba_b_stride
            + chan[None, :] * d_qkvzba_c_stride
        )
        local_store_mask = source_mask & is_local_source & (src < seq_len)
        d_qkvzba_current = tl.load(
            d_qkvzba_ptrs, mask=local_store_mask[:, None] & chan_mask[None, :], other=0.0
        )
        tl.store(
            d_qkvzba_ptrs,
            d_qkvzba_current + d_src.to(d_qkvzba_ptr.dtype.element_ty),
            mask=local_store_mask[:, None] & chan_mask[None, :],
        )


@triton.autotune(
    configs=_g_beta_autotune_configs(),
    key=["seq_len", "num_v_heads"],
    # Each autotune trial atomic-adds partial sums into these accumulators.
    # Without reset_to_zero the trials would stack on top of one another and
    # produce values that are ``num_trials`` × the correct result.
    reset_to_zero=["d_A_log_ptr", "d_dt_bias_ptr"],
)
@triton.jit
def _g_beta_backward_kernel(
    qkvzba_ptr,
    A_log_ptr,
    dt_bias_ptr,
    d_g_ptr,
    d_beta_out_ptr,
    d_qkvzba_ptr,
    d_A_log_ptr,
    d_dt_bias_ptr,
    seq_len,
    num_v_heads,
    beta_channel_offset,
    alpha_channel_offset,
    qkvzba_s_stride,
    qkvzba_b_stride,
    qkvzba_c_stride,
    d_g_b_stride,
    d_g_s_stride,
    d_g_h_stride,
    d_beta_b_stride,
    d_beta_s_stride,
    d_beta_h_stride,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Backward for ``_compute_g_and_beta_kernel``.

    Forward:
        pre = alpha + dt_bias                       # fp32
        softplus_pre = softplus(pre)                # torch default threshold
        g       = -exp(A_log) * softplus_pre
        beta_sig = sigmoid(beta_raw)

    Backward (given d_g and d_beta_out):
        d_alpha     = d_g * (-exp(A_log) * sigmoid(pre))
        d_beta_raw  = d_beta_out * beta_sig * (1 - beta_sig)
        d_dt_bias[h] = Σ_{b,s} d_alpha[b,s,h]
        d_A_log[h]   = Σ_{b,s} d_g[b,s,h] * g[b,s,h]

    ``d_alpha`` and ``d_beta_raw`` are written into the matching channel slices
    of ``d_qkvzba``. ``d_A_log`` and ``d_dt_bias`` are reduced via per-element
    atomic_add to fp32 buffers; the caller casts those to the parameter dtype.
    """

    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    s_mask = s_offs < seq_len
    h_mask = h_offs < num_v_heads
    mask = s_mask[:, None] & h_mask[None, :]

    # ----- Forward recompute -----
    alpha_ptr = (
        qkvzba_ptr
        + s_offs[:, None] * qkvzba_s_stride
        + pid_b * qkvzba_b_stride
        + (alpha_channel_offset + h_offs[None, :]) * qkvzba_c_stride
    )
    beta_ptr = (
        qkvzba_ptr
        + s_offs[:, None] * qkvzba_s_stride
        + pid_b * qkvzba_b_stride
        + (beta_channel_offset + h_offs[None, :]) * qkvzba_c_stride
    )
    alpha = tl.load(alpha_ptr, mask=mask, other=0.0).to(tl.float32)
    beta_raw = tl.load(beta_ptr, mask=mask, other=0.0).to(tl.float32)
    A_log = tl.load(A_log_ptr + h_offs, mask=h_mask, other=0.0).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + h_offs, mask=h_mask, other=0.0).to(tl.float32)

    pre = alpha + dt_bias[None, :]
    sigmoid_pre = tl.where(pre > 20.0, 1.0, tl.sigmoid(pre))
    softplus_pre = _softplus_with_torch_threshold(pre)
    exp_A = tl.exp(A_log)[None, :]
    g = -exp_A * softplus_pre
    beta_sig = tl.sigmoid(beta_raw)

    # ----- Load upstream gradients -----
    d_g_ptrs = (
        d_g_ptr
        + pid_b * d_g_b_stride
        + s_offs[:, None] * d_g_s_stride
        + h_offs[None, :] * d_g_h_stride
    )
    d_beta_out_ptrs = (
        d_beta_out_ptr
        + pid_b * d_beta_b_stride
        + s_offs[:, None] * d_beta_s_stride
        + h_offs[None, :] * d_beta_h_stride
    )
    d_g = tl.load(d_g_ptrs, mask=mask, other=0.0).to(tl.float32)
    d_beta_out = tl.load(d_beta_out_ptrs, mask=mask, other=0.0).to(tl.float32)

    # ----- Per-element gradients -----
    d_alpha = d_g * (-exp_A * sigmoid_pre)
    d_beta_raw = d_beta_out * beta_sig * (1.0 - beta_sig)

    # ----- (b, s) → h reductions -----
    d_g_masked = tl.where(mask, d_g, 0.0)
    d_alpha_masked = tl.where(mask, d_alpha, 0.0)
    d_A_log_partial = tl.sum(d_g_masked * g, axis=0)
    d_dt_bias_partial = tl.sum(d_alpha_masked, axis=0)

    # ----- Store per-element grads back to d_qkvzba -----
    d_alpha_ptrs = (
        d_qkvzba_ptr
        + s_offs[:, None] * qkvzba_s_stride
        + pid_b * qkvzba_b_stride
        + (alpha_channel_offset + h_offs[None, :]) * qkvzba_c_stride
    )
    d_beta_ptrs = (
        d_qkvzba_ptr
        + s_offs[:, None] * qkvzba_s_stride
        + pid_b * qkvzba_b_stride
        + (beta_channel_offset + h_offs[None, :]) * qkvzba_c_stride
    )
    tl.store(d_alpha_ptrs, d_alpha.to(d_qkvzba_ptr.dtype.element_ty), mask=mask)
    tl.store(d_beta_ptrs, d_beta_raw.to(d_qkvzba_ptr.dtype.element_ty), mask=mask)

    # ----- Atomic-add (b, s) partials into per-head accumulators -----
    tl.atomic_add(d_A_log_ptr + h_offs, d_A_log_partial, mask=h_mask)
    tl.atomic_add(d_dt_bias_ptr + h_offs, d_dt_bias_partial, mask=h_mask)


@triton.jit
def _finalize_pre_gdr_backward_kernel(
    d_weight_accum_ptr,
    d_weight_ptr,
    d_A_log_fp32_ptr,
    d_A_log_ptr,
    d_dt_bias_fp32_ptr,
    d_dt_bias_ptr,
    d_qkvzba_ptr,
    d_right_boundary_ptr,
    d_weight_numel,
    d_param_numel,
    right_boundary_numel,
    seq_len,
    weight_width,
    boundary,
    batch,
    conv_dim,
    d_weight_accum_c_stride,
    d_weight_accum_w_stride,
    d_weight_c_stride,
    d_weight_w_stride,
    d_A_log_fp32_stride,
    d_A_log_stride,
    d_dt_bias_fp32_stride,
    d_dt_bias_stride,
    d_qkvzba_s_stride,
    d_qkvzba_b_stride,
    d_qkvzba_c_stride,
    d_right_boundary_s_stride,
    d_right_boundary_b_stride,
    d_right_boundary_c_stride,
    CAST_WEIGHT: tl.constexpr,
    CAST_A_LOG: tl.constexpr,
    CAST_DT_BIAS: tl.constexpr,
    HAS_RIGHT_BOUNDARY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Finalize small fused pre-GDR backward casts and optional boundary add."""

    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)

    if CAST_WEIGHT:
        weight_mask = offs < d_weight_numel
        tap = offs % weight_width
        chan = offs // weight_width
        d_weight_val = tl.load(
            d_weight_accum_ptr
            + chan * d_weight_accum_c_stride
            + tap * d_weight_accum_w_stride,
            mask=weight_mask,
            other=0.0,
        )
        tl.store(
            d_weight_ptr + chan * d_weight_c_stride + tap * d_weight_w_stride,
            d_weight_val,
            mask=weight_mask,
        )

    if HAS_RIGHT_BOUNDARY:
        boundary_mask = offs < right_boundary_numel
        chan = offs % conv_dim
        batch_id = (offs // conv_dim) % batch
        boundary_s = offs // (batch * conv_dim)
        qkvzba_s = seq_len - boundary + boundary_s
        d_qkvzba_ptrs = (
            d_qkvzba_ptr
            + qkvzba_s * d_qkvzba_s_stride
            + batch_id * d_qkvzba_b_stride
            + chan * d_qkvzba_c_stride
        )
        d_right_boundary_ptrs = (
            d_right_boundary_ptr
            + boundary_s * d_right_boundary_s_stride
            + batch_id * d_right_boundary_b_stride
            + chan * d_right_boundary_c_stride
        )
        d_qkvzba_val = tl.load(d_qkvzba_ptrs, mask=boundary_mask, other=0.0).to(tl.float32)
        d_right_boundary_val = tl.load(
            d_right_boundary_ptrs, mask=boundary_mask, other=0.0
        ).to(tl.float32)
        tl.store(d_qkvzba_ptrs, d_qkvzba_val + d_right_boundary_val, mask=boundary_mask)

    if CAST_A_LOG:
        param_mask = offs < d_param_numel
        d_A_log_val = tl.load(
            d_A_log_fp32_ptr + offs * d_A_log_fp32_stride, mask=param_mask, other=0.0
        )
        tl.store(d_A_log_ptr + offs * d_A_log_stride, d_A_log_val, mask=param_mask)

    if CAST_DT_BIAS:
        param_mask = offs < d_param_numel
        d_dt_bias_val = tl.load(
            d_dt_bias_fp32_ptr + offs * d_dt_bias_fp32_stride, mask=param_mask, other=0.0
        )
        tl.store(d_dt_bias_ptr + offs * d_dt_bias_stride, d_dt_bias_val, mask=param_mask)


# ---------------------------------------------------------------------------
# Python entry points
# ---------------------------------------------------------------------------


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


_SIDE_STREAMS: dict = {}


def _get_side_stream(device: torch.device, slot: int) -> "torch.cuda.Stream":
    """Lazily allocate and cache CUDA streams keyed by ``(device, slot)``.

    Reusing streams across calls keeps launches free of stream-creation
    overhead, which would otherwise dominate the small kernels.
    """

    key = (device.index if device.index is not None else torch.cuda.current_device(), slot)
    stream = _SIDE_STREAMS.get(key)
    if stream is None:
        stream = torch.cuda.Stream(device=device)
        _SIDE_STREAMS[key] = stream
    return stream


def _triton_qk_l2norm_repeat_backward(
    dq: Tensor,
    dk: Tensor,
    silu_bf16: Tensor,
    d_silu_bf16: Tensor,
    *,
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    eps: float = 1e-6,
    stream: Optional["torch.cuda.Stream"] = None,
) -> Tensor:
    """Merged Q/K l2norm + REPEAT backward launch."""

    batch = dq.shape[0]
    seq_len = dq.shape[1]
    qk_channels = num_key_heads * key_head_dim
    repeat = num_value_heads // num_key_heads
    device = dq.device

    grid = lambda meta: (batch * 2 * num_key_heads, triton.cdiv(seq_len, meta["BLOCK_S"]))

    with _launch_context(device, stream):
        _qk_l2norm_repeat_backward_kernel[grid](
            dq,
            dk,
            silu_bf16,
            d_silu_bf16,
            seq_len,
            num_key_heads,
            qk_channels,
            eps,
            dq.stride(0),
            dq.stride(1),
            dq.stride(2),
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            silu_bf16.stride(0),
            silu_bf16.stride(1),
            silu_bf16.stride(2),
            d_silu_bf16.stride(0),
            d_silu_bf16.stride(1),
            d_silu_bf16.stride(2),
            HEAD_DIM=key_head_dim,
            REPEAT=repeat,
        )

    return d_silu_bf16


def _triton_v_layout_to_conv(
    dv: Tensor,
    d_silu_conv: Tensor,
    *,
    v_channel_offset: int,
    num_value_heads: int,
    value_head_dim: int,
    stream: Optional["torch.cuda.Stream"] = None,
) -> None:
    """Write ``dv`` into ``d_silu_conv``'s V channel slice."""

    batch, seq_len, _, _ = dv.shape
    device = dv.device

    BLOCK_S = _LAYOUT_BLOCK_S
    num_seq_blocks = triton.cdiv(seq_len, BLOCK_S)
    grid = (batch * num_value_heads, num_seq_blocks)

    with _launch_context(device, stream):
        _v_layout_to_conv_kernel[grid](
            dv,
            d_silu_conv,
            seq_len,
            num_value_heads,
            v_channel_offset,
            dv.stride(0),
            dv.stride(1),
            dv.stride(2),
            d_silu_conv.stride(0),
            d_silu_conv.stride(1),
            d_silu_conv.stride(2),
            HEAD_DIM=value_head_dim,
            BLOCK_S=BLOCK_S,
            num_warps=4,
            num_stages=2,
        )


def _triton_z_layout_to_qkvzba(
    dgate: Tensor,
    d_qkvzba: Tensor,
    *,
    z_channel_offset: int,
    num_value_heads: int,
    value_head_dim: int,
    stream: Optional["torch.cuda.Stream"] = None,
) -> None:
    """Write ``dgate`` into ``d_qkvzba``'s z channel slice."""

    batch, seq_len, _, _ = dgate.shape
    device = dgate.device

    BLOCK_S = _LAYOUT_BLOCK_S
    num_seq_blocks = triton.cdiv(seq_len, BLOCK_S)
    grid = (batch * num_value_heads, num_seq_blocks)

    with _launch_context(device, stream):
        _z_layout_to_qkvzba_kernel[grid](
            dgate,
            d_qkvzba,
            seq_len,
            num_value_heads,
            z_channel_offset,
            dgate.stride(0),
            dgate.stride(1),
            dgate.stride(2),
            d_qkvzba.stride(0),
            d_qkvzba.stride(1),
            d_qkvzba.stride(2),
            HEAD_DIM=value_head_dim,
            BLOCK_S=BLOCK_S,
            num_warps=4,
            num_stages=2,
        )


def _triton_conv_silu_boundary_backward(
    qkvzba: Tensor,
    conv1d_weight: Tensor,
    d_weight_accum: Tensor,
    d_silu_conv: Tensor,
    d_qkvzba: Tensor,
    left_boundary: Tensor,
    *,
    cu_seqlens: Optional[Tensor],
    global_token_offset: int,
    global_seq_len: int,
    apply_main_correction: bool,
) -> Tensor:
    """Compute chunkwise-CP boundary grad and optional packed main correction."""

    seq_len, batch, _ = qkvzba.shape
    conv_dim = conv1d_weight.shape[0]
    k_w = conv1d_weight.shape[-1]
    boundary = k_w - 1
    weight_2d = conv1d_weight.view(conv1d_weight.shape[0], k_w)
    d_left_boundary = torch.empty_like(left_boundary)
    # Packed boundary correction is accumulated directly into causal-conv's
    # returned d_weight buffer, avoiding a separate zero + add correction tensor.
    d_weight_accum_arg = d_weight_accum if apply_main_correction else qkvzba
    d_weight_accum_c_stride = d_weight_accum.stride(0) if apply_main_correction else 0
    d_weight_accum_w_stride = d_weight_accum.stride(1) if apply_main_correction else 0
    num_packed_seqs = 0 if cu_seqlens is None else cu_seqlens.shape[0] - 1
    is_packed = cu_seqlens is not None
    if cu_seqlens is None:
        cu_seqlens = qkvzba

    source_len = 2 * boundary if apply_main_correction else boundary
    BLOCK_H = 1 << (source_len - 1).bit_length()
    BLOCK_C = 128
    grid = (batch, triton.cdiv(conv_dim, BLOCK_C))
    _conv_silu_boundary_backward_kernel[grid](
        qkvzba,
        weight_2d,
        d_silu_conv,
        d_qkvzba,
        d_weight_accum_arg,
        left_boundary,
        d_left_boundary,
        cu_seqlens,
        seq_len,
        source_len,
        num_packed_seqs,
        global_token_offset,
        global_seq_len,
        conv_dim,
        qkvzba.stride(0),
        qkvzba.stride(1),
        qkvzba.stride(2),
        weight_2d.stride(0),
        weight_2d.stride(1),
        d_weight_accum_c_stride,
        d_weight_accum_w_stride,
        d_silu_conv.stride(0),
        d_silu_conv.stride(1),
        d_silu_conv.stride(2),
        d_qkvzba.stride(0),
        d_qkvzba.stride(1),
        d_qkvzba.stride(2),
        left_boundary.stride(0),
        left_boundary.stride(1),
        left_boundary.stride(2),
        d_left_boundary.stride(0),
        d_left_boundary.stride(1),
        d_left_boundary.stride(2),
        BOUNDARY=boundary,
        K_W=k_w,
        IS_PACKED=is_packed,
        APPLY_MAIN_CORRECTION=apply_main_correction,
        BLOCK_H=BLOCK_H,
        BLOCK_C=BLOCK_C,
        num_warps=4,
        num_stages=2,
    )
    return d_left_boundary


def _triton_g_beta_backward(
    qkvzba: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
    d_g: Tensor,
    d_beta_out: Tensor,
    *,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    num_key_heads: int,
    d_qkvzba_out: Optional[Tensor] = None,
    stream: Optional["torch.cuda.Stream"] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Launch ``_g_beta_backward_kernel`` and return its outputs.

    Returns:
        ``(d_qkvzba_out, d_A_log, d_dt_bias)``. ``d_qkvzba_out`` only has its
        alpha and beta slices filled in; the caller is expected to allocate
        the buffer while the other backward kernels fill the rest.
        ``d_A_log`` and ``d_dt_bias`` are fp32 and need to be cast back to
        the parameter dtype by the caller.
    """

    seq_len, batch, total_channels = qkvzba.shape
    qk_channels = num_key_heads * key_head_dim
    v_channels = num_value_heads * value_head_dim
    beta_channel_offset = 2 * qk_channels + 2 * v_channels
    alpha_channel_offset = beta_channel_offset + num_value_heads

    if d_qkvzba_out is None:
        d_qkvzba_out = torch.zeros_like(qkvzba)

    device = qkvzba.device

    g_beta_grid = lambda meta: (
        batch,
        triton.cdiv(seq_len, meta["BLOCK_S"]),
        triton.cdiv(num_value_heads, meta["BLOCK_H"]),
    )
    with _launch_context(device, stream):
        d_param_grads = torch.empty((2, num_value_heads), dtype=torch.float32, device=device)
        d_param_grads.zero_()
        d_A_log = d_param_grads[0]
        d_dt_bias = d_param_grads[1]
        _g_beta_backward_kernel[g_beta_grid](
            qkvzba,
            A_log,
            dt_bias,
            d_g,
            d_beta_out,
            d_qkvzba_out,
            d_A_log,
            d_dt_bias,
            seq_len,
            num_value_heads,
            beta_channel_offset,
            alpha_channel_offset,
            qkvzba.stride(0),
            qkvzba.stride(1),
            qkvzba.stride(2),
            d_g.stride(0),
            d_g.stride(1),
            d_g.stride(2),
            d_beta_out.stride(0),
            d_beta_out.stride(1),
            d_beta_out.stride(2),
        )
    return d_qkvzba_out, d_A_log, d_dt_bias


def _triton_finalize_pre_gdr_backward(
    d_weight_accum: Tensor,
    conv1d_weight: Tensor,
    d_A_log_fp32: Tensor,
    A_log: Tensor,
    d_dt_bias_fp32: Tensor,
    dt_bias: Tensor,
    d_qkvzba: Tensor,
    d_right_boundary: Optional[Tensor],
    *,
    conv_dim: int,
    boundary: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Finalize small fused pre-GDR backward casts and optional CP boundary add."""

    cast_weight = d_weight_accum.dtype != conv1d_weight.dtype
    cast_A_log = d_A_log_fp32.dtype != A_log.dtype
    cast_dt_bias = d_dt_bias_fp32.dtype != dt_bias.dtype
    has_right_boundary = d_right_boundary is not None

    if cast_weight:
        d_weight = torch.empty_like(conv1d_weight)
    else:
        d_weight = d_weight_accum.view(*conv1d_weight.shape)
    d_A_log = torch.empty_like(A_log) if cast_A_log else d_A_log_fp32
    d_dt_bias = torch.empty_like(dt_bias) if cast_dt_bias else d_dt_bias_fp32

    d_weight_numel = d_weight_accum.numel() if cast_weight else 0
    d_param_numel = d_A_log_fp32.numel() if (cast_A_log or cast_dt_bias) else 0
    right_boundary_numel = d_right_boundary.numel() if has_right_boundary else 0
    total_numel = max(d_weight_numel, d_param_numel, right_boundary_numel)
    if total_numel == 0:
        return d_weight, d_A_log, d_dt_bias

    dummy = d_qkvzba
    d_right_boundary_arg = d_right_boundary if has_right_boundary else dummy
    BLOCK = 256
    grid = (triton.cdiv(total_numel, BLOCK),)
    _finalize_pre_gdr_backward_kernel[grid](
        d_weight_accum,
        d_weight,
        d_A_log_fp32,
        d_A_log,
        d_dt_bias_fp32,
        d_dt_bias,
        d_qkvzba,
        d_right_boundary_arg,
        d_weight_numel,
        d_param_numel,
        right_boundary_numel,
        d_qkvzba.shape[0],
        conv1d_weight.shape[-1],
        boundary,
        d_qkvzba.shape[1],
        conv_dim,
        d_weight_accum.stride(0),
        d_weight_accum.stride(1),
        d_weight.stride(0),
        d_weight.stride(2),
        d_A_log_fp32.stride(0),
        d_A_log.stride(0),
        d_dt_bias_fp32.stride(0),
        d_dt_bias.stride(0),
        d_qkvzba.stride(0),
        d_qkvzba.stride(1),
        d_qkvzba.stride(2),
        d_right_boundary_arg.stride(0),
        d_right_boundary_arg.stride(1),
        d_right_boundary_arg.stride(2),
        CAST_WEIGHT=cast_weight,
        CAST_A_LOG=cast_A_log,
        CAST_DT_BIAS=cast_dt_bias,
        HAS_RIGHT_BOUNDARY=has_right_boundary,
        BLOCK=BLOCK,
        num_warps=4,
        num_stages=2,
    )
    return d_weight, d_A_log, d_dt_bias


def _launch_context(device: torch.device, stream: Optional["torch.cuda.Stream"]):
    """Return a CUDA launch context after wiring the optional side stream."""

    if stream is None:
        return nullcontext()
    stream.wait_stream(torch.cuda.current_stream(device))
    return torch.cuda.stream(stream)


def _wait_for_streams(dst_stream: "torch.cuda.Stream", *src_streams: "torch.cuda.Stream") -> None:
    for stream in src_streams:
        dst_stream.wait_stream(stream)


@triton.jit
def _chunkwise_cp_seq_idx_kernel(
    cu_seqlens_ptr,
    seq_idx_ptr,
    local_start,
    local_seq_len,
    BLOCK_T: tl.constexpr,
):
    """Fill local chunkwise-CP packed sequence ids without searchsorted."""

    pid_seq = tl.program_id(0)
    pid_block = tl.program_id(1)
    local_offsets = pid_block * BLOCK_T + tl.arange(0, BLOCK_T)
    global_offsets = local_start + local_offsets
    seq_start = tl.load(cu_seqlens_ptr + pid_seq)
    seq_end = tl.load(cu_seqlens_ptr + pid_seq + 1)
    mask = (
        (local_offsets < local_seq_len)
        & (global_offsets >= seq_start)
        & (global_offsets < seq_end)
    )
    tl.store(seq_idx_ptr + local_offsets, pid_seq, mask=mask)


def _resolve_packed_seq_idx(
    cu_seqlens: Optional[Tensor], seq_idx: Optional[Tensor], total_tokens: int
) -> Optional[Tensor]:
    """Return the token-level sequence-id buffer for causal-conv backward."""

    if cu_seqlens is None:
        assert seq_idx is None, "seq_idx requires cu_seqlens for packed THD mode."
        return None

    if seq_idx is None:
        seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        seq_idx = torch.repeat_interleave(
            torch.arange(seq_lengths.numel(), device=cu_seqlens.device, dtype=torch.int32),
            seq_lengths,
        )
        seq_idx = seq_idx.unsqueeze(0)
    elif seq_idx.dim() == 1:
        seq_idx = seq_idx.unsqueeze(0)

    assert seq_idx.is_cuda, f"Packed seq_idx must be CUDA, got {seq_idx.device}."
    assert seq_idx.dtype == torch.int32, f"Packed seq_idx must be int32, got {seq_idx.dtype}."
    assert seq_idx.shape == (1, total_tokens), (
        "Packed seq_idx must have shape [1, total_tokens], "
        f"got {seq_idx.shape=} and {total_tokens=}."
    )
    return seq_idx.contiguous()


def _resolve_chunkwise_cp_packed_seq_idx(cu_seqlens: Tensor, local_seq_len: int, cp_rank: int):
    """Build local packed sequence ids for a contiguous chunkwise-CP rank interval."""

    local_start = cp_rank * local_seq_len
    seq_idx = torch.empty((1, local_seq_len), device=cu_seqlens.device, dtype=torch.int32)
    block_t = 512
    grid = (cu_seqlens.shape[0] - 1, triton.cdiv(local_seq_len, block_t))
    _chunkwise_cp_seq_idx_kernel[grid](
        cu_seqlens,
        seq_idx,
        local_start,
        local_seq_len,
        BLOCK_T=block_t,
        num_warps=8,
        num_stages=2,
    )
    return seq_idx


def _cp_neighbor_global_ranks(cp_group) -> Tuple[int, int, Optional[int], Optional[int]]:
    """Return ``(cp_size, cp_rank, prev_global_rank, next_global_rank)``."""

    cp_size = cp_group.size()
    cp_rank = cp_group.rank()
    prev_rank = (
        torch.distributed.get_global_rank(cp_group, cp_rank - 1) if cp_rank > 0 else None
    )
    next_rank = (
        torch.distributed.get_global_rank(cp_group, cp_rank + 1)
        if cp_rank < cp_size - 1
        else None
    )
    return cp_size, cp_rank, prev_rank, next_rank


def _wait_distributed_ops(ops) -> None:
    """Wait for a small list of async distributed work handles."""

    if ops is None:
        return
    for op in ops:
        op.wait()


def _split_batched_recv_send_works(works, op_roles) -> Tuple[Tuple, Tuple]:
    """Split batched recv/send work handles when the backend exposes them separately.

    Some NCCL/PyTorch combinations return one grouped ``Work`` for the whole
    ``batch_isend_irecv`` launch, while others return one ``Work`` per P2P op.
    If the backend returns a single grouped handle, waiting for it at the recv
    dependency point also completes the send.
    """

    works = tuple(works)
    op_roles = tuple(op_roles)
    if len(works) == 1:
        if "recv" in op_roles:
            return works, ()
        return (), works
    if len(works) == len(op_roles):
        recv_ops = tuple(work for work, role in zip(works, op_roles) if role == "recv")
        send_ops = tuple(work for work, role in zip(works, op_roles) if role == "send")
        return recv_ops, send_ops
    raise RuntimeError(
        "Expected batch_isend_irecv to return one grouped work handle or one "
        "work handle per P2P op for chunkwise CP boundary exchange; "
        f"got {len(works)} work handles for {len(op_roles)} ops."
    )


def _start_left_boundary_exchange(
    qkvzba: Tensor, *, conv_dim: int, boundary: int, cp_group
) -> Tuple[Optional[Tensor], Tuple, Tuple, Optional[Tensor]]:
    """Start chunkwise-CP left-boundary exchange without waiting for completion."""

    _, _, prev_rank, next_rank = _cp_neighbor_global_ranks(cp_group)
    left_boundary = None
    send_buf = None
    recv_ops = []
    send_ops = []
    p2p_ops = []
    op_roles = []
    if prev_rank is not None:
        left_boundary = qkvzba.new_empty((boundary, qkvzba.shape[1], conv_dim))
        p2p_ops.append(
            torch.distributed.P2POp(torch.distributed.irecv, left_boundary, prev_rank, cp_group)
        )
        op_roles.append("recv")
    if next_rank is not None:
        send_buf = qkvzba[-boundary:, :, :conv_dim].contiguous()
        p2p_ops.append(
            torch.distributed.P2POp(torch.distributed.isend, send_buf, next_rank, cp_group)
        )
        op_roles.append("send")
    if p2p_ops:
        recv_ops, send_ops = _split_batched_recv_send_works(
            torch.distributed.batch_isend_irecv(p2p_ops), op_roles
        )

    return left_boundary, tuple(recv_ops), tuple(send_ops), send_buf


def _start_boundary_grad_exchange(
    qkvzba: Tensor, d_left_boundary: Optional[Tensor], *, conv_dim: int, boundary: int, cp_group
) -> Tuple[Optional[Tensor], Tuple, Tuple, Optional[Tensor]]:
    """Start chunkwise-CP boundary-gradient exchange without waiting for completion."""

    _, _, prev_rank, next_rank = _cp_neighbor_global_ranks(cp_group)
    d_right_boundary = None
    send_buf = None
    recv_ops = []
    send_ops = []
    p2p_ops = []
    op_roles = []
    if next_rank is not None:
        d_right_boundary = qkvzba.new_empty((boundary, qkvzba.shape[1], conv_dim))
        p2p_ops.append(
            torch.distributed.P2POp(torch.distributed.irecv, d_right_boundary, next_rank, cp_group)
        )
        op_roles.append("recv")
    if prev_rank is not None:
        if d_left_boundary is None:
            raise RuntimeError("Chunkwise CP backward requires a left-boundary gradient to send.")
        send_buf = d_left_boundary.contiguous()
        p2p_ops.append(
            torch.distributed.P2POp(torch.distributed.isend, send_buf, prev_rank, cp_group)
        )
        op_roles.append("send")
    if p2p_ops:
        recv_ops, send_ops = _split_batched_recv_send_works(
            torch.distributed.batch_isend_irecv(p2p_ops), op_roles
        )

    return d_right_boundary, tuple(recv_ops), tuple(send_ops), send_buf


def _triton_pre_gated_delta_rule_forward(
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
    cp_group=None,
    cp_size: int = 1,
) -> Tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Optional[Tensor],
    int,
    int,
]:
    """Triton-backed forward for the pre-gated-delta-rule front-end.

    Returns ``(query, key, value, gate, beta, g, silu_qk_save, left_boundary,
    global_token_offset, global_seq_len)``. ``silu_qk_save`` is the bf16-rounded
    ``silu(conv(x))`` for the QK channel range laid out channel-last for the QK
    l2norm/repeat backward scope. ``left_boundary`` is returned so the autograd
    wrapper can save it for backward.
    """

    seq_len, batch, total_channels = qkvzba.shape
    is_packed_thd = cu_seqlens is not None
    if is_packed_thd:
        assert batch == 1, (
            "Packed THD fused_pre_gated_delta_rule expects batch dimension 1; " f"got {batch=}."
        )
        num_packed_seqs = cu_seqlens.shape[0] - 1
    else:
        num_packed_seqs = 0

    qk_channels = num_key_heads * key_head_dim
    v_channels = num_value_heads * value_head_dim
    conv_dim = 2 * qk_channels + v_channels
    repeat_factor = num_value_heads // num_key_heads
    k_w = conv1d_weight.shape[-1]
    boundary = k_w - 1
    cp_active = cp_group is not None and cp_size > 1 and boundary > 0
    cp_rank = cp_group.rank() if cp_active else 0
    left_boundary = None
    left_boundary_recv_ops = None
    left_boundary_send_ops = None
    _left_boundary_send_buf = None
    global_token_offset = 0
    global_seq_len = seq_len
    if cp_active:
        (
            left_boundary,
            left_boundary_recv_ops,
            left_boundary_send_ops,
            _left_boundary_send_buf,
        ) = _start_left_boundary_exchange(qkvzba, conv_dim=conv_dim, boundary=boundary, cp_group=cp_group)
        if is_packed_thd:
            global_token_offset = cp_rank * seq_len
            global_seq_len = seq_len * cp_size
    has_left_boundary = left_boundary is not None
    if left_boundary is None:
        left_boundary = qkvzba
    assert _is_power_of_two(key_head_dim), (
        "Triton kernel currently expects key_head_dim to be a power of two; "
        f"got {key_head_dim=}."
    )
    assert _is_power_of_two(value_head_dim), (
        "Triton kernel currently expects value_head_dim to be a power of two; "
        f"got {value_head_dim=}."
    )

    expected_channels = 2 * qk_channels + 2 * v_channels + 2 * num_value_heads
    assert (
        total_channels == expected_channels
    ), f"qkvzba last-dim mismatch: got {total_channels}, expected {expected_channels}."

    out_dtype = qkvzba.dtype
    device = qkvzba.device

    # Output buffers: contiguous (b, s, h, d) for q/k/v and (b, s, h) for g/beta.
    # Q and K share one allocation so the fused-streamed QK kernel can select
    # the logical group by pointer stride instead of branching between two
    # unrelated base pointers inside Triton.
    qk_out = torch.empty(
        2, batch, seq_len, num_value_heads, key_head_dim, dtype=out_dtype, device=device
    )
    query = qk_out[0]
    key = qk_out[1]
    value = torch.empty(
        batch, seq_len, num_value_heads, value_head_dim, dtype=out_dtype, device=device
    )
    g = torch.empty(batch, seq_len, num_value_heads, dtype=torch.float32, device=device)
    beta = torch.empty(batch, seq_len, num_value_heads, dtype=out_dtype, device=device)

    # Conv weight is (conv_dim, 1, K_W); we treat it as (conv_dim, K_W).
    weight_2d = conv1d_weight.view(conv1d_weight.shape[0], k_w)

    # No conv bias support: the entry point asserts this. We still pass a
    # dummy ``bias_tensor`` to the kernel so the launch signature stays
    # stable; ``HAS_BIAS=False`` ensures the kernel never reads it.
    bias_tensor = qkvzba
    bias_stride = 0

    # Allocate the gate (z) output buffer that the independent Z kernel will
    # populate. Keeping Z separate makes the forward scopes QK / V / Z /
    # G-Beta explicit.
    gate = torch.empty(
        batch, seq_len, num_value_heads, value_head_dim, dtype=out_dtype, device=device
    )

    # Persist the QK silu(conv(x)) intermediate in channel-last layout so the
    # backward can feed it directly into the l2norm backward.
    silu_qk_save = torch.empty(
        (batch, seq_len, 2 * qk_channels), dtype=out_dtype, device=device
    ).permute(
        0, 2, 1
    )  # → (b, 2*qk_c, s) with stride(1)==1
    silu_save_b_stride = silu_qk_save.stride(0)
    silu_save_c_stride = silu_qk_save.stride(1)
    silu_save_s_stride = silu_qk_save.stride(2)

    overlap_boundary_exchange = bool(left_boundary_recv_ops or left_boundary_send_ops)

    # Stream setup. Each side stream handles one of the four sub-computations
    # (QK conv+l2norm, V conv, Z copy, g/beta). When chunkwise CP boundary exchange
    # is active, QK/V consume the received boundary and stay on the caller stream.
    # Z and g/beta do not depend on the boundary, so they can still overlap with
    # the boundary exchange on side streams.
    main_stream = torch.cuda.current_stream(device=device)
    if overlap_boundary_exchange:
        qk_stream = main_stream
        v_stream = main_stream
        g_beta_stream = _get_side_stream(device, slot=_G_BETA_STREAM_SLOT)
        z_stream = _get_side_stream(device, slot=_Z_STREAM_SLOT)
    else:
        qk_stream = _get_side_stream(device, slot=_QK_STREAM_SLOT)
        v_stream = _get_side_stream(device, slot=_V_STREAM_SLOT)
        g_beta_stream = _get_side_stream(device, slot=_G_BETA_STREAM_SLOT)
        z_stream = _get_side_stream(device, slot=_Z_STREAM_SLOT)
    for stream in (qk_stream, v_stream, g_beta_stream, z_stream):
        stream.wait_stream(main_stream)

    v_channel_offset = 2 * qk_channels
    z_channel_offset = 2 * qk_channels + v_channels
    beta_channel_offset = 2 * qk_channels + 2 * v_channels
    alpha_channel_offset = beta_channel_offset + num_value_heads

    def _launch_z_copy() -> None:
        BLOCK_Z_S = _LAYOUT_BLOCK_S
        z_grid = (batch * num_value_heads, triton.cdiv(seq_len, BLOCK_Z_S))
        with torch.cuda.stream(z_stream):
            _copy_z_kernel[z_grid](
                qkvzba,
                gate,
                seq_len,
                num_value_heads,
                z_channel_offset,
                qkvzba.stride(0),
                qkvzba.stride(1),
                qkvzba.stride(2),
                gate.stride(0),
                gate.stride(1),
                gate.stride(2),
                HEAD_DIM=value_head_dim,
                BLOCK_S=BLOCK_Z_S,
                num_warps=4,
                num_stages=2,
            )

    def _launch_g_beta() -> None:
        g_beta_grid = lambda meta: (
            batch,
            triton.cdiv(seq_len, meta["BLOCK_S"]),
            triton.cdiv(num_value_heads, meta["BLOCK_H"]),
        )
        with torch.cuda.stream(g_beta_stream):
            _compute_g_and_beta_kernel[g_beta_grid](
                qkvzba,
                A_log,
                dt_bias,
                g,
                beta,
                seq_len,
                num_value_heads,
                beta_channel_offset,
                alpha_channel_offset,
                qkvzba.stride(0),
                qkvzba.stride(1),
                qkvzba.stride(2),
                g.stride(0),
                g.stride(1),
                g.stride(2),
                beta.stride(0),
                beta.stride(1),
                beta.stride(2),
            )

    if overlap_boundary_exchange:
        # Z and g/beta do not consume the CP left boundary, so issue them while
        # the boundary recv is in flight. QK/V wait below right before they need it.
        _launch_z_copy()
        _launch_g_beta()
    _wait_distributed_ops(left_boundary_recv_ops)

    # --- QK conv + silu + l2norm + repeat ---
    qk_grid = lambda meta: (batch * 2 * num_key_heads, triton.cdiv(seq_len, meta["BLOCK_S"]))
    with torch.cuda.stream(qk_stream):
        if is_packed_thd:
            _conv_silu_project_thd_kernel[qk_grid](
                qkvzba,
                weight_2d,
                bias_tensor,
                qk_out,
                silu_qk_save,
                left_boundary,
                cu_seqlens,
                seq_len,
                global_token_offset,
                global_seq_len,
                num_packed_seqs,
                num_key_heads,
                0,  # QK starts at channel 0; group 1 starts at +qk_channels.
                qk_channels,
                0,  # silu_save_chan_offset
                qk_channels,
                qkvzba.stride(0),
                qkvzba.stride(1),
                qkvzba.stride(2),
                weight_2d.stride(0),
                weight_2d.stride(1),
                bias_stride,
                qk_out.stride(0),
                qk_out.stride(1),
                qk_out.stride(2),
                qk_out.stride(3),
                silu_save_b_stride,
                silu_save_c_stride,
                silu_save_s_stride,
                left_boundary.stride(0),
                left_boundary.stride(1),
                left_boundary.stride(2),
                _L2NORM_EPS,
                HEAD_DIM=key_head_dim,
                K_W=k_w,
                REPEAT=repeat_factor,
                NUM_GROUPS=2,
                HAS_BIAS=False,
                HAS_LEFT_BOUNDARY=has_left_boundary,
                SAVE_SILU=True,
                APPLY_L2=True,
            )
        else:
            _conv_silu_project_kernel[qk_grid](
                qkvzba,
                weight_2d,
                bias_tensor,
                qk_out,
                silu_qk_save,
                left_boundary,
                seq_len,
                num_key_heads,
                0,  # QK starts at channel 0; group 1 starts at +qk_channels.
                qk_channels,
                0,  # silu_save_chan_offset
                qk_channels,
                qkvzba.stride(0),
                qkvzba.stride(1),
                qkvzba.stride(2),
                weight_2d.stride(0),
                weight_2d.stride(1),
                bias_stride,
                qk_out.stride(0),
                qk_out.stride(1),
                qk_out.stride(2),
                qk_out.stride(3),
                silu_save_b_stride,
                silu_save_c_stride,
                silu_save_s_stride,
                left_boundary.stride(0),
                left_boundary.stride(1),
                left_boundary.stride(2),
                _L2NORM_EPS,
                HEAD_DIM=key_head_dim,
                K_W=k_w,
                REPEAT=repeat_factor,
                NUM_GROUPS=2,
                HAS_BIAS=False,
                HAS_LEFT_BOUNDARY=has_left_boundary,
                SAVE_SILU=True,
                APPLY_L2=True,
            )

    # --- V conv + silu (no l2norm, no repeat) ---
    v_grid = lambda meta: (batch * num_value_heads, triton.cdiv(seq_len, meta["BLOCK_S"]))
    with torch.cuda.stream(v_stream):
        if is_packed_thd:
            _conv_silu_project_thd_kernel[v_grid](
                qkvzba,
                weight_2d,
                bias_tensor,
                value,
                qkvzba,  # silu_save unused (SAVE_SILU=False)
                left_boundary,
                cu_seqlens,
                seq_len,
                global_token_offset,
                global_seq_len,
                num_packed_seqs,
                num_value_heads,
                v_channel_offset,
                0,  # in_group_stride unused for NUM_GROUPS=1
                0,  # silu_save_chan_offset unused
                0,  # silu_save_group_stride unused
                qkvzba.stride(0),
                qkvzba.stride(1),
                qkvzba.stride(2),
                weight_2d.stride(0),
                weight_2d.stride(1),
                bias_stride,
                0,  # out_group_dim_stride unused for NUM_GROUPS=1
                value.stride(0),
                value.stride(1),
                value.stride(2),
                0,  # silu_save strides unused
                0,
                0,
                left_boundary.stride(0),
                left_boundary.stride(1),
                left_boundary.stride(2),
                _L2NORM_EPS,
                HEAD_DIM=value_head_dim,
                K_W=k_w,
                REPEAT=1,
                NUM_GROUPS=1,
                HAS_BIAS=False,
                HAS_LEFT_BOUNDARY=has_left_boundary,
                SAVE_SILU=False,
                APPLY_L2=False,
            )
        else:
            _conv_silu_project_kernel[v_grid](
                qkvzba,
                weight_2d,
                bias_tensor,
                value,
                qkvzba,  # silu_save unused (SAVE_SILU=False)
                left_boundary,
                seq_len,
                num_value_heads,
                v_channel_offset,
                0,  # in_group_stride unused for NUM_GROUPS=1
                0,  # silu_save_chan_offset unused
                0,  # silu_save_group_stride unused
                qkvzba.stride(0),
                qkvzba.stride(1),
                qkvzba.stride(2),
                weight_2d.stride(0),
                weight_2d.stride(1),
                bias_stride,
                0,  # out_group_dim_stride unused for NUM_GROUPS=1
                value.stride(0),
                value.stride(1),
                value.stride(2),
                0,  # silu_save strides unused
                0,
                0,
                left_boundary.stride(0),
                left_boundary.stride(1),
                left_boundary.stride(2),
                _L2NORM_EPS,
                HEAD_DIM=value_head_dim,
                K_W=k_w,
                REPEAT=1,
                NUM_GROUPS=1,
                HAS_BIAS=False,
                HAS_LEFT_BOUNDARY=has_left_boundary,
                SAVE_SILU=False,
                APPLY_L2=False,
            )

    if not overlap_boundary_exchange:
        # --- Z copy ---
        _launch_z_copy()

        # --- g and beta ---
        _launch_g_beta()

    # Re-join the side streams so the caller's stream observes the writes.
    _wait_for_streams(main_stream, qk_stream, v_stream, z_stream, g_beta_stream)
    # The boundary send uses an owned contiguous buffer that no downstream
    # kernel reads, so defer this wait to overlap the send with QK/V compute.
    _wait_distributed_ops(left_boundary_send_ops)
    _ = _left_boundary_send_buf

    return (
        query,
        key,
        value,
        gate,
        beta,
        g,
        silu_qk_save,
        left_boundary if has_left_boundary else None,
        global_token_offset,
        global_seq_len,
    )


def _triton_pre_gated_delta_rule_backward(
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
    left_boundary: Optional[Tensor] = None,
    cu_seqlens: Optional[Tensor] = None,
    global_token_offset: int = 0,
    global_seq_len: Optional[int] = None,
    cp_group=None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    """Triton-backed backward for the pre-gated-delta-rule front-end.

    Mirror of :func:`_triton_pre_gated_delta_rule_forward`. Takes upstream
    gradients (``dq``/``dk``/``dv``/``dgate``/``dbeta``/``dg``) plus the
    saved forward intermediates and returns input/parameter gradients
    ``(d_qkvzba, d_weight, d_A_log, d_dt_bias, d_left_boundary)``. The
    last element is ``None`` unless chunkwise CP supplied a left boundary.

    The hand-tuned C++ ``causal_conv1d_bwd_function`` computes the main
    conv+SiLU input and weight gradients. Chunkwise CP passes the left boundary as
    causal-conv initial states for the main backward and uses a tiny Triton
    kernel only for the left-boundary gradient that must be sent to the previous
    CP rank.
    """

    seq_len, batch, _ = qkvzba.shape
    qk_channels = num_key_heads * key_head_dim
    v_channels = num_value_heads * value_head_dim
    conv_dim = 2 * qk_channels + v_channels
    z_offset = 2 * qk_channels + v_channels
    k_w = conv1d_weight.shape[-1]
    device = qkvzba.device

    # Rebuild the conv input as a NON-contiguous (b, c, s) view of qkvzba.
    # ``causal_conv1d_fn`` / ``_bwd_function`` accept inputs where either
    # ``stride(1) == 1`` or ``stride(2) == 1``; the permuted view of qkvzba
    # satisfies the former (channel stride is 1 in the original (s, b, c)
    # layout), so we can skip a 256 MB ``.contiguous()`` copy.
    qkvzba_conv = qkvzba[:, :, :conv_dim].permute(1, 2, 0)
    weight_2d = conv1d_weight.view(conv1d_weight.shape[0], k_w)
    initial_states = None if left_boundary is None else left_boundary.permute(1, 2, 0)
    if global_seq_len is None:
        global_seq_len = seq_len

    # ``silu_qk_save`` is the (b, 2*qk_channels, s) bf16 buffer the
    # forward wrote ``silu(conv(x))`` into for QK. Reuse it directly as
    # the silu input to the l2norm backward.
    silu_conv = silu_qk_save

    # Allocate d_silu_conv channel-last (stride(1)==1) — that's what
    # ``causal_conv1d_channellast_bwd_kernel`` consumes natively.
    d_silu_conv = torch.empty(
        (batch, seq_len, conv_dim), dtype=qkvzba.dtype, device=device
    ).permute(0, 2, 1)

    # Use the same stream slots as the forward for the matching scopes.
    qk_stream = _get_side_stream(device, slot=_QK_STREAM_SLOT)
    v_stream = _get_side_stream(device, slot=_V_STREAM_SLOT)
    g_beta_stream = _get_side_stream(device, slot=_G_BETA_STREAM_SLOT)
    z_stream = _get_side_stream(device, slot=_Z_STREAM_SLOT)

    # Q + K: l2norm + REPEAT backward writes into d_silu_conv's Q/K slices.
    _triton_qk_l2norm_repeat_backward(
        dq,
        dk,
        silu_conv,
        d_silu_conv,
        num_key_heads=num_key_heads,
        num_value_heads=num_value_heads,
        key_head_dim=key_head_dim,
        stream=qk_stream,
    )

    # V: no l2norm and no REPEAT in forward, so d_silu_conv's V slice is
    # just dv re-laid-out from (b, s, num_v_heads, value_head_dim) to
    # (b, v_channels, s).
    _triton_v_layout_to_conv(
        dv,
        d_silu_conv,
        v_channel_offset=2 * qk_channels,
        num_value_heads=num_value_heads,
        value_head_dim=value_head_dim,
        stream=v_stream,
    )

    d_qkvzba = torch.empty_like(qkvzba)
    overlap_boundary_grad_exchange = cp_group is not None and k_w > 1

    def _launch_g_beta_and_z_backward() -> Tuple[Tensor, Tensor]:
        # g + beta backward fully stores d_qkvzba's alpha + beta slices, plus
        # per-head d_A_log / d_dt_bias. Conv and z slices are filled by the
        # causal-conv and z kernels, so d_qkvzba does not need a pre-zero.
        _, d_A_log_fp32, d_dt_bias_fp32 = _triton_g_beta_backward(
            qkvzba,
            A_log,
            dt_bias,
            dg,
            dbeta,
            num_value_heads=num_value_heads,
            key_head_dim=key_head_dim,
            value_head_dim=value_head_dim,
            num_key_heads=num_key_heads,
            d_qkvzba_out=d_qkvzba,
            stream=g_beta_stream,
        )

        # Z slice gradient: stream dgate into d_qkvzba's z slice.
        _triton_z_layout_to_qkvzba(
            dgate,
            d_qkvzba,
            z_channel_offset=z_offset,
            num_value_heads=num_value_heads,
            value_head_dim=value_head_dim,
            stream=z_stream,
        )
        return d_A_log_fp32, d_dt_bias_fp32

    if not overlap_boundary_grad_exchange:
        d_A_log_fp32, d_dt_bias_fp32 = _launch_g_beta_and_z_backward()

    # Join only streams that wrote into d_silu_conv before conv backward.
    # g/beta and z write disjoint outputs: non-CP launches them above to overlap
    # with conv backward, while chunkwise CP delays them below to overlap with
    # boundary-gradient P2P.
    default_stream = torch.cuda.current_stream(device)
    _wait_for_streams(default_stream, qk_stream, v_stream)

    # Pre-allocate d_x_conv as a strided view INTO d_qkvzba's conv slice.
    # d_qkvzba memory layout is (s, b, total_channels) contiguous, so
    # element [s, b, c] sits at offset s*b_stride + b*c_stride + c.
    # Re-interpreting that storage as (b, conv_dim, s) lets causal_conv1d
    # backward write d_x directly into the right cells.
    seq_stride = qkvzba.stride(0)
    batch_stride = qkvzba.stride(1)
    d_x_conv_view = d_qkvzba.as_strided((batch, conv_dim, seq_len), (batch_stride, 1, seq_stride))

    apply_boundary_correction = left_boundary is not None and seq_idx is not None
    causal_initial_states = None if apply_boundary_correction else initial_states

    # Hand-tuned C++ conv backward. Internally folds the silu' factor and
    # computes both d_x and d_w in fp32; writes d_x directly into the view above.
    # The public causal-conv1d wrapper casts d_weight back to weight.dtype before
    # returning, leaving one small vectorized dtype-conversion kernel here. We use
    # the public API for stability; to remove that launch, this call would need to
    # switch to causal-conv1d's private _causal_conv1d_bwd_cpp entry with a
    # caller-owned fp32 d_weight accumulator.
    assert causal_conv1d_bwd_function is not None
    _, d_weight_accum, _, _ = causal_conv1d_bwd_function(
        qkvzba_conv,
        weight_2d,
        None,  # no bias
        d_silu_conv,
        seq_idx,
        causal_initial_states,
        None,  # dfinal_states
        d_x_conv_view,  # dx pre-allocated into d_qkvzba's conv slice
        False,  # return_dinitial_states
        True,  # activation (silu)
    )
    d_weight_accum = d_weight_accum.view(conv1d_weight.shape[0], conv1d_weight.shape[-1])
    if left_boundary is None:
        d_left_boundary = None
    else:
        d_left_boundary = _triton_conv_silu_boundary_backward(
            qkvzba,
            conv1d_weight,
            d_weight_accum,
            d_silu_conv,
            d_qkvzba,
            left_boundary,
            cu_seqlens=cu_seqlens,
            global_token_offset=global_token_offset,
            global_seq_len=global_seq_len,
            apply_main_correction=apply_boundary_correction,
        )
    d_right_boundary = None
    right_boundary_recv_ops = None
    left_boundary_send_ops = None
    _left_boundary_send_buf = None
    if overlap_boundary_grad_exchange:
        # Exchange boundary gradients as soon as conv backward has produced the
        # local left-boundary gradient.
        # The remaining g/beta and z work writes disjoint d_qkvzba slices and
        # can continue overlapping with this P2P exchange.
        (
            d_right_boundary,
            right_boundary_recv_ops,
            left_boundary_send_ops,
            _left_boundary_send_buf,
        ) = _start_boundary_grad_exchange(
            qkvzba,
            d_left_boundary,
            conv_dim=conv_dim,
            boundary=k_w - 1,
            cp_group=cp_group,
        )
        d_A_log_fp32, d_dt_bias_fp32 = _launch_g_beta_and_z_backward()
        _wait_distributed_ops(right_boundary_recv_ops)
    default_stream.wait_stream(g_beta_stream)
    d_weight, d_A_log, d_dt_bias = _triton_finalize_pre_gdr_backward(
        d_weight_accum,
        conv1d_weight,
        d_A_log_fp32,
        A_log,
        d_dt_bias_fp32,
        dt_bias,
        d_qkvzba,
        d_right_boundary,
        conv_dim=conv_dim,
        boundary=k_w - 1,
    )
    default_stream.wait_stream(z_stream)
    _wait_distributed_ops(left_boundary_send_ops)
    _ = _left_boundary_send_buf

    return d_qkvzba, d_weight, d_A_log, d_dt_bias, d_left_boundary


class FusedPreGatedDeltaRuleFunction(torch.autograd.Function):
    """Thin :class:`torch.autograd.Function` wrapper around the fused path.

    Stashes the forward inputs + the saved ``silu_qk_save`` intermediate
    in ``ctx`` and dispatches to :func:`_triton_pre_gated_delta_rule_forward`
    / :func:`_triton_pre_gated_delta_rule_backward`. The actual kernel
    logic lives in those two free functions so it's easy to read and
    reuse outside the autograd machinery.
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
        cp_group,
        cp_size,
        num_key_heads,
        num_value_heads,
        key_head_dim,
        value_head_dim,
    ):
        """Run the fused pre-GDR forward path and stash tensors for backward."""

        ctx.num_key_heads = num_key_heads
        ctx.num_value_heads = num_value_heads
        ctx.key_head_dim = key_head_dim
        ctx.value_head_dim = value_head_dim
        boundary = conv1d_weight.shape[-1] - 1
        cp_active = cp_group is not None and cp_size > 1 and boundary > 0
        ctx.cp_active = cp_active
        ctx.cp_group = cp_group
        ctx.has_cu_seqlens = False
        ctx.has_left_boundary = False
        ctx.global_token_offset = 0
        ctx.global_seq_len = qkvzba.shape[0]
        cp_rank = cp_group.rank() if cp_active else 0

        seq_idx_for_backward = seq_idx
        seq_idx_ready_event = None
        if cp_active and cu_seqlens is not None:
            ctx.has_cu_seqlens = True
            seq_idx_for_backward = _resolve_chunkwise_cp_packed_seq_idx(
                cu_seqlens, qkvzba.shape[0], cp_rank
            )
            seq_idx_ready_event = torch.cuda.Event()
            seq_idx_ready_event.record(torch.cuda.current_stream(qkvzba.device))
        ctx.seq_idx_ready_event = seq_idx_ready_event

        (
            query,
            key,
            value,
            gate,
            beta,
            g,
            silu_qk_save,
            left_boundary,
            global_token_offset,
            global_seq_len,
        ) = _triton_pre_gated_delta_rule_forward(
            qkvzba,
            conv1d_weight,
            A_log,
            dt_bias,
            num_key_heads=num_key_heads,
            num_value_heads=num_value_heads,
            key_head_dim=key_head_dim,
            value_head_dim=value_head_dim,
            cu_seqlens=cu_seqlens,
            cp_group=cp_group,
            cp_size=cp_size,
        )
        ctx.has_left_boundary = left_boundary is not None
        ctx.global_token_offset = global_token_offset
        ctx.global_seq_len = global_seq_len
        ctx.has_seq_idx = seq_idx_for_backward is not None
        saved_tensors = [qkvzba, conv1d_weight, A_log, dt_bias, silu_qk_save]
        if ctx.has_seq_idx:
            saved_tensors.append(seq_idx_for_backward)
        if ctx.has_cu_seqlens:
            saved_tensors.append(cu_seqlens)
        if ctx.has_left_boundary:
            saved_tensors.append(left_boundary)
        ctx.save_for_backward(*saved_tensors)
        return query, key, value, gate, beta, g

    @staticmethod
    def backward(ctx, dq, dk, dv, dgate, dbeta, dg):
        """Run the fused pre-GDR backward path from saved forward tensors."""

        saved_idx = 0
        qkvzba_for_backward = ctx.saved_tensors[saved_idx]
        saved_idx += 1
        conv1d_weight = ctx.saved_tensors[saved_idx]
        saved_idx += 1
        A_log = ctx.saved_tensors[saved_idx]
        saved_idx += 1
        dt_bias = ctx.saved_tensors[saved_idx]
        saved_idx += 1
        silu_qk_save = ctx.saved_tensors[saved_idx]
        saved_idx += 1
        if ctx.has_seq_idx:
            seq_idx = ctx.saved_tensors[saved_idx]
            saved_idx += 1
            if ctx.seq_idx_ready_event is not None:
                torch.cuda.current_stream(qkvzba_for_backward.device).wait_event(
                    ctx.seq_idx_ready_event
                )
        else:
            seq_idx = None
        if ctx.has_cu_seqlens:
            cu_seqlens = ctx.saved_tensors[saved_idx]
            saved_idx += 1
        else:
            cu_seqlens = None
        if ctx.has_left_boundary:
            left_boundary = ctx.saved_tensors[saved_idx]
            saved_idx += 1
        else:
            left_boundary = None
        d_qkvzba, d_weight, d_A_log, d_dt_bias, d_left_boundary = _triton_pre_gated_delta_rule_backward(
            qkvzba_for_backward,
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
            left_boundary=left_boundary,
            cu_seqlens=cu_seqlens,
            global_token_offset=ctx.global_token_offset,
            global_seq_len=ctx.global_seq_len,
            cp_group=ctx.cp_group if ctx.cp_active else None,
        )
        # Match forward inputs: (qkvzba, conv1d_weight, A_log, dt_bias,
        # cu_seqlens, seq_idx, cp_group, cp_size, num_key_heads, num_value_heads,
        # key_head_dim, value_head_dim).
        # Non-tensor args get None.
        return (
            d_qkvzba,
            d_weight,
            d_A_log,
            d_dt_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def fused_streamed_pre_gated_delta_rule(
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
    cp_group=None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Streamed fused pre-gated-delta-rule entry point.

    Args:
        qkvzba: ``[seq_len, batch, in_proj_dim]`` projection output. Must be
            on CUDA.
        conv1d_weight: ``[conv_dim, 1, k_w]`` depthwise conv weight.
        conv1d_bias: Must be ``None`` (conv bias is not supported).
        A_log: ``[num_value_heads]`` raw decay parameter.
        dt_bias: ``[num_value_heads]`` time-step bias.
        num_key_heads / num_value_heads / key_head_dim / value_head_dim: GDN
            architecture parameters. ``num_value_heads`` must be an integer
            multiple of ``num_key_heads``.
        use_qk_l2norm: Must be ``True``; the fused backward closes over the
            l2norm path.
        cu_seqlens: Optional packed THD cumulative sequence lengths. When CP
            is inactive, ``cu_seqlens[-1]`` must equal local ``seq_len``. When
            chunkwise CP is active, pass global packed boundaries so the fused
            path can mask virtual boundary taps at packed sequence boundaries.
        seq_idx: Optional precomputed token-to-sequence map with shape
            ``[1, seq_len]``. Used by causal-conv backward in packed THD mode.
        cp_group: Optional chunkwise-CP process group. When it has size > 1,
            the fused path prepends a previous-rank conv boundary internally.

    Returns:
        ``(query, key, value, gate, beta, g)`` matching the unfused
        :meth:`GatedDeltaNet.pre_gated_delta_rule` API.
    """

    if causal_conv1d_bwd_function is None:
        raise ImportError(
            "gdn_pre_gated_delta_rule_fusion requires causal-conv1d. "
            "Install causal-conv1d~=1.6 in environments that enable this fusion."
        )

    assert qkvzba.is_cuda, (
        "fused_pre_gated_delta_rule requires CUDA inputs; " f"got qkvzba.device={qkvzba.device}."
    )
    assert conv1d_bias is None, (
        "Conv bias is not supported by fused_pre_gated_delta_rule "
        "(production GDN config has none)."
    )
    assert use_qk_l2norm, (
        "use_qk_l2norm=False is not supported by fused_pre_gated_delta_rule "
        "(the backward closes over the l2norm path)."
    )
    assert (
        num_value_heads % num_key_heads == 0
    ), f"{num_value_heads=} must be a multiple of {num_key_heads=}."
    cp_size = cp_group.size() if cp_group is not None else 1
    if cp_size > 1:
        if qkvzba.shape[1] != 1:
            raise ValueError(
                "GDN chunkwise CP with SBHD inputs currently requires micro_batch_size == 1 "
                f"for fused_pre_gated_delta_rule; got batch={qkvzba.shape[1]}."
            )
        boundary = conv1d_weight.shape[-1] - 1
        if boundary > 0 and qkvzba.shape[0] < boundary:
            raise ValueError(
                "fused_pre_gated_delta_rule chunkwise CP requires local chunk length "
                f"({qkvzba.shape[0]}) >= conv_kernel_dim - 1 ({boundary})."
            )
        if seq_idx is not None:
            raise ValueError(
                "fused_pre_gated_delta_rule derives packed seq_idx internally when "
                "chunkwise CP is active."
            )
    if cu_seqlens is not None:
        assert cu_seqlens.is_cuda, (
            "Packed fused_pre_gated_delta_rule requires CUDA cu_seqlens; "
            f"got cu_seqlens.device={cu_seqlens.device}."
        )
        assert cu_seqlens.dtype == torch.int32, (
            "Packed fused_pre_gated_delta_rule requires int32 cu_seqlens; "
            f"got {cu_seqlens.dtype=}."
        )
        assert cu_seqlens.dim() == 1, (
            "Packed fused_pre_gated_delta_rule expects 1-D cu_seqlens; " f"got {cu_seqlens.shape=}."
        )
        assert qkvzba.shape[1] == 1, (
            "Packed THD fused_pre_gated_delta_rule expects batch dimension 1; "
            f"got qkvzba.shape={qkvzba.shape}."
        )
        assert cu_seqlens.shape[0] >= 2, (
            "Packed fused_pre_gated_delta_rule requires at least one packed sequence; "
            f"got {cu_seqlens.shape=}."
        )
        # Caller contract: packed boundaries start at zero, are monotonically
        # non-decreasing, and end at the local token count for CP1 or the global
        # token count for chunkwise CP. Checking those values here requires GPU
        # reductions and D2H synchronizations, so the fused hot path trusts the
        # packed-sequence scheduler to provide valid cu_seqlens.
        cu_seqlens = cu_seqlens.contiguous()
        if cp_size == 1:
            seq_idx = _resolve_packed_seq_idx(cu_seqlens, seq_idx, qkvzba.shape[0])
    else:
        assert seq_idx is None, "seq_idx requires cu_seqlens for packed THD mode."

    return FusedPreGatedDeltaRuleFunction.apply(
        qkvzba,
        conv1d_weight,
        A_log,
        dt_bias,
        cu_seqlens,
        seq_idx,
        cp_group,
        cp_size,
        num_key_heads,
        num_value_heads,
        key_head_dim,
        value_head_dim,
    )
