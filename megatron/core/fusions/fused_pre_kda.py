# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fused preprocessing for Kimi Delta Attention.

This module reuses the QKV convolution, SiLU, L2-normalization, packed-sequence,
and chunkwise-CP kernels from :mod:`fused_pre_gated_delta_rule`. KDA keeps its
decay feature, output gate, and beta as explicit inputs because those tensors
may come from independent full-rank or low-rank projections.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor

from megatron.core.fusions.fused_pre_gated_delta_rule import (
    _G_BETA_STREAM_SLOT as _TAIL_STREAM_SLOT,
)
from megatron.core.fusions.fused_pre_gated_delta_rule import (
    _L2NORM_EPS,
    _QK_STREAM_SLOT,
    _V_STREAM_SLOT,
    _conv_autotune_configs,
    _conv_silu_project_kernel,
    _conv_silu_project_thd_kernel,
    _get_side_stream,
    _is_power_of_two,
    _launch_context,
    _resolve_chunkwise_cp_packed_seq_idx,
    _resolve_packed_seq_idx,
    _start_boundary_grad_exchange,
    _start_left_boundary_exchange,
    _triton_conv_silu_boundary_backward,
    _triton_qk_l2norm_repeat_backward,
    _triton_v_layout_to_conv,
    _wait_distributed_ops,
    _wait_for_streams,
    causal_conv1d_bwd_function,
)


@triton.autotune(configs=_conv_autotune_configs(), key=["seq_len", "HEAD_DIM"])
@triton.jit
def _prepare_kda_tail_kernel(
    raw_g_ptr,
    gate_ptr,
    beta_ptr,
    raw_g_out_ptr,
    gate_out_ptr,
    beta_out_ptr,
    seq_len,
    num_heads,
    raw_g_s_stride,
    raw_g_b_stride,
    raw_g_c_stride,
    gate_s_stride,
    gate_b_stride,
    gate_c_stride,
    beta_s_stride,
    beta_b_stride,
    beta_h_stride,
    raw_g_out_b_stride,
    raw_g_out_s_stride,
    raw_g_out_h_stride,
    gate_out_b_stride,
    gate_out_s_stride,
    gate_out_h_stride,
    beta_out_b_stride,
    beta_out_s_stride,
    beta_out_h_stride,
    HEAD_DIM: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Lay out raw decay/output-gate tensors and apply beta sigmoid."""

    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    batch_id = pid_bh // num_heads
    head_id = pid_bh - batch_id * num_heads

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    d_offs = tl.arange(0, HEAD_DIM)
    s_mask = s_offs < seq_len
    channel = head_id * HEAD_DIM + d_offs
    value_mask = s_mask[:, None]

    raw_g = tl.load(
        raw_g_ptr
        + s_offs[:, None] * raw_g_s_stride
        + batch_id * raw_g_b_stride
        + channel[None, :] * raw_g_c_stride,
        mask=value_mask,
        other=0.0,
    )
    gate = tl.load(
        gate_ptr
        + s_offs[:, None] * gate_s_stride
        + batch_id * gate_b_stride
        + channel[None, :] * gate_c_stride,
        mask=value_mask,
        other=0.0,
    )
    tl.store(
        raw_g_out_ptr
        + batch_id * raw_g_out_b_stride
        + s_offs[:, None] * raw_g_out_s_stride
        + head_id * raw_g_out_h_stride
        + d_offs[None, :],
        raw_g,
        mask=value_mask,
    )
    tl.store(
        gate_out_ptr
        + batch_id * gate_out_b_stride
        + s_offs[:, None] * gate_out_s_stride
        + head_id * gate_out_h_stride
        + d_offs[None, :],
        gate,
        mask=value_mask,
    )

    beta = tl.load(
        beta_ptr + s_offs * beta_s_stride + batch_id * beta_b_stride + head_id * beta_h_stride,
        mask=s_mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        beta_out_ptr
        + batch_id * beta_out_b_stride
        + s_offs * beta_out_s_stride
        + head_id * beta_out_h_stride,
        tl.sigmoid(beta),
        mask=s_mask,
    )


@triton.autotune(configs=_conv_autotune_configs(), key=["seq_len", "HEAD_DIM"])
@triton.jit
def _prepare_kda_tail_backward_kernel(
    beta_ptr,
    d_raw_g_out_ptr,
    d_gate_out_ptr,
    d_beta_out_ptr,
    d_raw_g_ptr,
    d_gate_ptr,
    d_beta_ptr,
    seq_len,
    num_heads,
    beta_s_stride,
    beta_b_stride,
    beta_h_stride,
    d_raw_g_out_b_stride,
    d_raw_g_out_s_stride,
    d_raw_g_out_h_stride,
    d_gate_out_b_stride,
    d_gate_out_s_stride,
    d_gate_out_h_stride,
    d_beta_out_b_stride,
    d_beta_out_s_stride,
    d_beta_out_h_stride,
    d_raw_g_s_stride,
    d_raw_g_b_stride,
    d_raw_g_c_stride,
    d_gate_s_stride,
    d_gate_b_stride,
    d_gate_c_stride,
    d_beta_s_stride,
    d_beta_b_stride,
    d_beta_h_stride,
    HEAD_DIM: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Backward for :func:`_prepare_kda_tail_kernel`."""

    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    batch_id = pid_bh // num_heads
    head_id = pid_bh - batch_id * num_heads

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    d_offs = tl.arange(0, HEAD_DIM)
    s_mask = s_offs < seq_len
    channel = head_id * HEAD_DIM + d_offs
    value_mask = s_mask[:, None]

    d_raw_g = tl.load(
        d_raw_g_out_ptr
        + batch_id * d_raw_g_out_b_stride
        + s_offs[:, None] * d_raw_g_out_s_stride
        + head_id * d_raw_g_out_h_stride
        + d_offs[None, :],
        mask=value_mask,
        other=0.0,
    )
    d_gate = tl.load(
        d_gate_out_ptr
        + batch_id * d_gate_out_b_stride
        + s_offs[:, None] * d_gate_out_s_stride
        + head_id * d_gate_out_h_stride
        + d_offs[None, :],
        mask=value_mask,
        other=0.0,
    )
    tl.store(
        d_raw_g_ptr
        + s_offs[:, None] * d_raw_g_s_stride
        + batch_id * d_raw_g_b_stride
        + channel[None, :] * d_raw_g_c_stride,
        d_raw_g,
        mask=value_mask,
    )
    tl.store(
        d_gate_ptr
        + s_offs[:, None] * d_gate_s_stride
        + batch_id * d_gate_b_stride
        + channel[None, :] * d_gate_c_stride,
        d_gate,
        mask=value_mask,
    )

    beta = tl.load(
        beta_ptr + s_offs * beta_s_stride + batch_id * beta_b_stride + head_id * beta_h_stride,
        mask=s_mask,
        other=0.0,
    ).to(tl.float32)
    d_beta_out = tl.load(
        d_beta_out_ptr
        + batch_id * d_beta_out_b_stride
        + s_offs * d_beta_out_s_stride
        + head_id * d_beta_out_h_stride,
        mask=s_mask,
        other=0.0,
    ).to(tl.float32)
    beta_sigmoid = tl.sigmoid(beta)
    d_beta = d_beta_out * beta_sigmoid * (1.0 - beta_sigmoid)
    tl.store(
        d_beta_ptr
        + s_offs * d_beta_s_stride
        + batch_id * d_beta_b_stride
        + head_id * d_beta_h_stride,
        d_beta,
        mask=s_mask,
    )


@triton.jit
def _finalize_kda_backward_kernel(
    d_weight_accum_ptr,
    d_weight_ptr,
    d_qkv_ptr,
    d_right_boundary_ptr,
    d_weight_numel,
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
    d_qkv_s_stride,
    d_qkv_b_stride,
    d_qkv_c_stride,
    d_right_boundary_s_stride,
    d_right_boundary_b_stride,
    d_right_boundary_c_stride,
    CAST_WEIGHT: tl.constexpr,
    HAS_RIGHT_BOUNDARY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Cast the conv-weight gradient and merge a received CP boundary gradient."""

    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)

    if CAST_WEIGHT:
        weight_mask = offs < d_weight_numel
        tap = offs % weight_width
        channel = offs // weight_width
        d_weight = tl.load(
            d_weight_accum_ptr + channel * d_weight_accum_c_stride + tap * d_weight_accum_w_stride,
            mask=weight_mask,
            other=0.0,
        )
        tl.store(
            d_weight_ptr + channel * d_weight_c_stride + tap * d_weight_w_stride,
            d_weight,
            mask=weight_mask,
        )

    if HAS_RIGHT_BOUNDARY:
        boundary_mask = offs < right_boundary_numel
        channel = offs % conv_dim
        batch_id = (offs // conv_dim) % batch
        boundary_s = offs // (batch * conv_dim)
        qkv_s = seq_len - boundary + boundary_s
        d_qkv_ptrs = (
            d_qkv_ptr
            + qkv_s * d_qkv_s_stride
            + batch_id * d_qkv_b_stride
            + channel * d_qkv_c_stride
        )
        d_right_boundary_ptrs = (
            d_right_boundary_ptr
            + boundary_s * d_right_boundary_s_stride
            + batch_id * d_right_boundary_b_stride
            + channel * d_right_boundary_c_stride
        )
        d_qkv_value = tl.load(d_qkv_ptrs, mask=boundary_mask, other=0.0).to(tl.float32)
        d_boundary_value = tl.load(d_right_boundary_ptrs, mask=boundary_mask, other=0.0).to(
            tl.float32
        )
        tl.store(d_qkv_ptrs, d_qkv_value + d_boundary_value, mask=boundary_mask)


def _launch_conv_silu_project(
    qkv: Tensor,
    weight_2d: Tensor,
    out: Tensor,
    silu_save: Tensor,
    left_boundary: Tensor,
    *,
    cu_seqlens: Optional[Tensor],
    global_token_offset: int,
    global_seq_len: int,
    num_packed_seqs: int,
    num_heads: int,
    in_channel_offset: int,
    in_group_stride: int,
    silu_save_group_stride: int,
    head_dim: int,
    repeat: int,
    num_groups: int,
    has_left_boundary: bool,
    apply_l2: bool,
    save_silu: bool,
) -> None:
    """Launch one QK or V convolution branch using the shared GDN kernel."""

    seq_len, batch, _ = qkv.shape
    grid = lambda meta: (batch * num_groups * num_heads, triton.cdiv(seq_len, meta["BLOCK_S"]))
    if num_groups == 2:
        out_group_stride = out.stride(0)
        out_b_stride = out.stride(1)
        out_s_stride = out.stride(2)
        out_h_stride = out.stride(3)
    else:
        out_group_stride = 0
        out_b_stride = out.stride(0)
        out_s_stride = out.stride(1)
        out_h_stride = out.stride(2)

    common_args = (qkv, weight_2d, qkv, out, silu_save, left_boundary)
    common_tail_args = (
        seq_len,
        num_heads,
        in_channel_offset,
        in_group_stride,
        0,
        silu_save_group_stride,
        qkv.stride(0),
        qkv.stride(1),
        qkv.stride(2),
        weight_2d.stride(0),
        weight_2d.stride(1),
        0,
        out_group_stride,
        out_b_stride,
        out_s_stride,
        out_h_stride,
        silu_save.stride(0) if save_silu else 0,
        silu_save.stride(1) if save_silu else 0,
        silu_save.stride(2) if save_silu else 0,
        left_boundary.stride(0),
        left_boundary.stride(1),
        left_boundary.stride(2),
        _L2NORM_EPS,
    )
    constexpr_args = dict(
        HEAD_DIM=head_dim,
        K_W=weight_2d.shape[-1],
        REPEAT=repeat,
        NUM_GROUPS=num_groups,
        HAS_BIAS=False,
        HAS_LEFT_BOUNDARY=has_left_boundary,
        APPLY_L2=apply_l2,
        SAVE_SILU=save_silu,
    )
    if cu_seqlens is None:
        _conv_silu_project_kernel[grid](*common_args, *common_tail_args, **constexpr_args)
    else:
        _conv_silu_project_thd_kernel[grid](
            *common_args,
            cu_seqlens,
            *common_tail_args[:1],
            global_token_offset,
            global_seq_len,
            num_packed_seqs,
            *common_tail_args[1:],
            **constexpr_args,
        )


def _triton_prepare_kda_tail(
    raw_g: Tensor,
    gate: Tensor,
    beta: Tensor,
    *,
    num_heads: int,
    head_dim: int,
    stream: Optional["torch.cuda.Stream"] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Prepare explicit KDA F-decay, output-gate, and beta tensors."""

    seq_len, batch, _ = raw_g.shape
    device = raw_g.device
    raw_g_out = torch.empty((batch, seq_len, num_heads, head_dim), dtype=raw_g.dtype, device=device)
    gate_out = torch.empty_like(raw_g_out, dtype=gate.dtype)
    beta_out = torch.empty((batch, seq_len, num_heads), dtype=torch.float32, device=device)
    grid = lambda meta: (batch * num_heads, triton.cdiv(seq_len, meta["BLOCK_S"]))
    with _launch_context(device, stream):
        _prepare_kda_tail_kernel[grid](
            raw_g,
            gate,
            beta,
            raw_g_out,
            gate_out,
            beta_out,
            seq_len,
            num_heads,
            raw_g.stride(0),
            raw_g.stride(1),
            raw_g.stride(2),
            gate.stride(0),
            gate.stride(1),
            gate.stride(2),
            beta.stride(0),
            beta.stride(1),
            beta.stride(2),
            raw_g_out.stride(0),
            raw_g_out.stride(1),
            raw_g_out.stride(2),
            gate_out.stride(0),
            gate_out.stride(1),
            gate_out.stride(2),
            beta_out.stride(0),
            beta_out.stride(1),
            beta_out.stride(2),
            HEAD_DIM=head_dim,
        )
    return raw_g_out, gate_out, beta_out


def _triton_prepare_kda_tail_backward(
    beta: Tensor,
    d_raw_g_out: Tensor,
    d_gate_out: Tensor,
    d_beta_out: Tensor,
    *,
    raw_g_dtype: torch.dtype,
    gate_dtype: torch.dtype,
    num_heads: int,
    head_dim: int,
    stream: Optional["torch.cuda.Stream"] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Return gradients in the original explicit projection layouts."""

    batch, seq_len, _, _ = d_raw_g_out.shape
    device = beta.device
    input_shape = (seq_len, batch, num_heads * head_dim)
    d_raw_g = torch.empty(input_shape, dtype=raw_g_dtype, device=device)
    d_gate = torch.empty(input_shape, dtype=gate_dtype, device=device)
    d_beta = torch.empty_like(beta)
    grid = lambda meta: (batch * num_heads, triton.cdiv(seq_len, meta["BLOCK_S"]))
    with _launch_context(device, stream):
        _prepare_kda_tail_backward_kernel[grid](
            beta,
            d_raw_g_out,
            d_gate_out,
            d_beta_out,
            d_raw_g,
            d_gate,
            d_beta,
            seq_len,
            num_heads,
            beta.stride(0),
            beta.stride(1),
            beta.stride(2),
            d_raw_g_out.stride(0),
            d_raw_g_out.stride(1),
            d_raw_g_out.stride(2),
            d_gate_out.stride(0),
            d_gate_out.stride(1),
            d_gate_out.stride(2),
            d_beta_out.stride(0),
            d_beta_out.stride(1),
            d_beta_out.stride(2),
            d_raw_g.stride(0),
            d_raw_g.stride(1),
            d_raw_g.stride(2),
            d_gate.stride(0),
            d_gate.stride(1),
            d_gate.stride(2),
            d_beta.stride(0),
            d_beta.stride(1),
            d_beta.stride(2),
            HEAD_DIM=head_dim,
        )
    return d_raw_g, d_gate, d_beta


def _triton_finalize_kda_backward(
    d_weight_accum: Tensor,
    conv1d_weight: Tensor,
    d_qkv: Tensor,
    d_right_boundary: Optional[Tensor],
    *,
    conv_dim: int,
    boundary: int,
) -> Tensor:
    """Finalize the conv-weight gradient and optional CP boundary add."""

    cast_weight = d_weight_accum.dtype != conv1d_weight.dtype
    has_right_boundary = d_right_boundary is not None
    d_weight = (
        torch.empty_like(conv1d_weight) if cast_weight else d_weight_accum.view_as(conv1d_weight)
    )
    d_weight_numel = d_weight_accum.numel() if cast_weight else 0
    right_boundary_numel = d_right_boundary.numel() if has_right_boundary else 0
    total_numel = max(d_weight_numel, right_boundary_numel)
    if total_numel == 0:
        return d_weight

    d_right_boundary_arg = d_right_boundary if has_right_boundary else d_qkv
    block = 256
    grid = (triton.cdiv(total_numel, block),)
    _finalize_kda_backward_kernel[grid](
        d_weight_accum,
        d_weight,
        d_qkv,
        d_right_boundary_arg,
        d_weight_numel,
        right_boundary_numel,
        d_qkv.shape[0],
        conv1d_weight.shape[-1],
        boundary,
        d_qkv.shape[1],
        conv_dim,
        d_weight_accum.stride(0),
        d_weight_accum.stride(1),
        d_weight.stride(0),
        d_weight.stride(2),
        d_qkv.stride(0),
        d_qkv.stride(1),
        d_qkv.stride(2),
        d_right_boundary_arg.stride(0),
        d_right_boundary_arg.stride(1),
        d_right_boundary_arg.stride(2),
        CAST_WEIGHT=cast_weight,
        HAS_RIGHT_BOUNDARY=has_right_boundary,
        BLOCK=block,
        num_warps=4,
        num_stages=2,
    )
    return d_weight


def _triton_pre_kda_forward(
    qkv: Tensor,
    raw_g: Tensor,
    gate: Tensor,
    beta: Tensor,
    conv1d_weight: Tensor,
    *,
    num_heads: int,
    head_dim: int,
    save_silu: bool = True,
    cu_seqlens: Optional[Tensor] = None,
    cp_group=None,
    cp_size: int = 1,
) -> Tuple[
    Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor], int, int
]:
    """Run the shared QKV kernels and the KDA-specific tail kernel."""

    seq_len, batch, total_channels = qkv.shape
    is_packed_thd = cu_seqlens is not None
    num_packed_seqs = cu_seqlens.shape[0] - 1 if is_packed_thd else 0
    qk_channels = num_heads * head_dim
    conv_dim = 3 * qk_channels
    assert (
        total_channels == conv_dim
    ), f"KDA qkv last-dim mismatch: got {total_channels}, expected {conv_dim}."
    assert _is_power_of_two(head_dim), (
        "Triton KDA fusion expects head_dim to be a power of two; " f"got {head_dim}."
    )

    boundary = conv1d_weight.shape[-1] - 1
    cp_active = cp_group is not None and cp_size > 1 and boundary > 0
    cp_rank = cp_group.rank() if cp_active else 0
    left_boundary = None
    left_boundary_recv_ops = None
    left_boundary_send_ops = None
    left_boundary_send_buf = None
    global_token_offset = 0
    global_seq_len = seq_len
    if cp_active:
        left_boundary, left_boundary_recv_ops, left_boundary_send_ops, left_boundary_send_buf = (
            _start_left_boundary_exchange(
                qkv, conv_dim=conv_dim, boundary=boundary, cp_group=cp_group
            )
        )
        if is_packed_thd:
            global_token_offset = cp_rank * seq_len
            global_seq_len = seq_len * cp_size
    has_left_boundary = left_boundary is not None
    if left_boundary is None:
        left_boundary = qkv

    device = qkv.device
    qk_out = torch.empty((2, batch, seq_len, num_heads, head_dim), dtype=qkv.dtype, device=device)
    query = qk_out[0]
    key = qk_out[1]
    value = torch.empty((batch, seq_len, num_heads, head_dim), dtype=qkv.dtype, device=device)
    if save_silu:
        silu_qk_save: Optional[Tensor] = torch.empty(
            (batch, seq_len, 2 * qk_channels), dtype=qkv.dtype, device=device
        ).permute(0, 2, 1)
    else:
        silu_qk_save = None
    silu_qk_buffer = qkv if silu_qk_save is None else silu_qk_save
    weight_2d = conv1d_weight.view(conv1d_weight.shape[0], conv1d_weight.shape[-1])

    main_stream = torch.cuda.current_stream(device=device)
    overlap_boundary_exchange = bool(left_boundary_recv_ops or left_boundary_send_ops)
    if overlap_boundary_exchange:
        qk_stream = main_stream
        v_stream = main_stream
    else:
        qk_stream = _get_side_stream(device, slot=_QK_STREAM_SLOT)
        v_stream = _get_side_stream(device, slot=_V_STREAM_SLOT)
    tail_stream = _get_side_stream(device, slot=_TAIL_STREAM_SLOT)
    for stream in (qk_stream, v_stream, tail_stream):
        stream.wait_stream(main_stream)

    raw_g_out, gate_out, beta_out = _triton_prepare_kda_tail(
        raw_g, gate, beta, num_heads=num_heads, head_dim=head_dim, stream=tail_stream
    )
    _wait_distributed_ops(left_boundary_recv_ops)

    with torch.cuda.stream(qk_stream):
        _launch_conv_silu_project(
            qkv,
            weight_2d,
            qk_out,
            silu_qk_buffer,
            left_boundary,
            cu_seqlens=cu_seqlens,
            global_token_offset=global_token_offset,
            global_seq_len=global_seq_len,
            num_packed_seqs=num_packed_seqs,
            num_heads=num_heads,
            in_channel_offset=0,
            in_group_stride=qk_channels,
            silu_save_group_stride=qk_channels,
            head_dim=head_dim,
            repeat=1,
            num_groups=2,
            has_left_boundary=has_left_boundary,
            apply_l2=True,
            save_silu=save_silu,
        )
    with torch.cuda.stream(v_stream):
        _launch_conv_silu_project(
            qkv,
            weight_2d,
            value,
            qkv,
            left_boundary,
            cu_seqlens=cu_seqlens,
            global_token_offset=global_token_offset,
            global_seq_len=global_seq_len,
            num_packed_seqs=num_packed_seqs,
            num_heads=num_heads,
            in_channel_offset=2 * qk_channels,
            in_group_stride=0,
            silu_save_group_stride=0,
            head_dim=head_dim,
            repeat=1,
            num_groups=1,
            has_left_boundary=has_left_boundary,
            apply_l2=False,
            save_silu=False,
        )

    _wait_for_streams(main_stream, qk_stream, v_stream, tail_stream)
    _wait_distributed_ops(left_boundary_send_ops)
    _ = left_boundary_send_buf
    return (
        query,
        key,
        value,
        gate_out,
        beta_out,
        raw_g_out,
        silu_qk_save,
        left_boundary if has_left_boundary else None,
        global_token_offset,
        global_seq_len,
    )


def _triton_pre_kda_backward(
    qkv: Tensor,
    beta: Tensor,
    conv1d_weight: Tensor,
    silu_qk_save: Tensor,
    dq: Tensor,
    dk: Tensor,
    dv: Tensor,
    dgate: Tensor,
    dbeta: Tensor,
    d_raw_g_out: Tensor,
    *,
    raw_g_dtype: torch.dtype,
    gate_dtype: torch.dtype,
    num_heads: int,
    head_dim: int,
    seq_idx: Optional[Tensor] = None,
    left_boundary: Optional[Tensor] = None,
    cu_seqlens: Optional[Tensor] = None,
    global_token_offset: int = 0,
    global_seq_len: Optional[int] = None,
    cp_group=None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Run the fused KDA preprocessing backward path."""

    seq_len, batch, conv_dim = qkv.shape
    qk_channels = num_heads * head_dim
    device = qkv.device
    kernel_width = conv1d_weight.shape[-1]
    if global_seq_len is None:
        global_seq_len = seq_len

    qkv_conv = qkv.permute(1, 2, 0)
    weight_2d = conv1d_weight.view(conv1d_weight.shape[0], kernel_width)
    initial_states = None if left_boundary is None else left_boundary.permute(1, 2, 0)
    d_silu_conv = torch.empty((batch, seq_len, conv_dim), dtype=qkv.dtype, device=device).permute(
        0, 2, 1
    )

    qk_stream = _get_side_stream(device, slot=_QK_STREAM_SLOT)
    v_stream = _get_side_stream(device, slot=_V_STREAM_SLOT)
    tail_stream = _get_side_stream(device, slot=_TAIL_STREAM_SLOT)
    _triton_qk_l2norm_repeat_backward(
        dq,
        dk,
        silu_qk_save,
        d_silu_conv,
        num_key_heads=num_heads,
        num_value_heads=num_heads,
        key_head_dim=head_dim,
        stream=qk_stream,
    )
    _triton_v_layout_to_conv(
        dv,
        d_silu_conv,
        v_channel_offset=2 * qk_channels,
        num_value_heads=num_heads,
        value_head_dim=head_dim,
        stream=v_stream,
    )

    d_qkv = torch.empty_like(qkv)
    overlap_boundary_grad_exchange = cp_group is not None and kernel_width > 1
    tail_grads = None

    def _launch_tail_backward() -> Tuple[Tensor, Tensor, Tensor]:
        return _triton_prepare_kda_tail_backward(
            beta,
            d_raw_g_out,
            dgate,
            dbeta,
            raw_g_dtype=raw_g_dtype,
            gate_dtype=gate_dtype,
            num_heads=num_heads,
            head_dim=head_dim,
            stream=tail_stream,
        )

    if not overlap_boundary_grad_exchange:
        tail_grads = _launch_tail_backward()

    default_stream = torch.cuda.current_stream(device)
    _wait_for_streams(default_stream, qk_stream, v_stream)
    d_x_conv_view = d_qkv.as_strided(
        (batch, conv_dim, seq_len), (d_qkv.stride(1), 1, d_qkv.stride(0))
    )
    apply_boundary_correction = left_boundary is not None and seq_idx is not None
    causal_initial_states = None if apply_boundary_correction else initial_states
    assert causal_conv1d_bwd_function is not None
    _, d_weight_accum, _, _ = causal_conv1d_bwd_function(
        qkv_conv,
        weight_2d,
        None,
        d_silu_conv,
        seq_idx,
        causal_initial_states,
        None,
        d_x_conv_view,
        False,
        True,
    )
    d_weight_accum = d_weight_accum.view(conv1d_weight.shape[0], kernel_width)

    if left_boundary is None:
        d_left_boundary = None
    else:
        d_left_boundary = _triton_conv_silu_boundary_backward(
            qkv,
            conv1d_weight,
            d_weight_accum,
            d_silu_conv,
            d_qkv,
            left_boundary,
            cu_seqlens=cu_seqlens,
            global_token_offset=global_token_offset,
            global_seq_len=global_seq_len,
            apply_main_correction=apply_boundary_correction,
        )

    d_right_boundary = None
    right_boundary_recv_ops = None
    left_boundary_send_ops = None
    left_boundary_send_buf = None
    if overlap_boundary_grad_exchange:
        (
            d_right_boundary,
            right_boundary_recv_ops,
            left_boundary_send_ops,
            left_boundary_send_buf,
        ) = _start_boundary_grad_exchange(
            qkv, d_left_boundary, conv_dim=conv_dim, boundary=kernel_width - 1, cp_group=cp_group
        )
        tail_grads = _launch_tail_backward()
        _wait_distributed_ops(right_boundary_recv_ops)

    assert tail_grads is not None
    d_weight = _triton_finalize_kda_backward(
        d_weight_accum,
        conv1d_weight,
        d_qkv,
        d_right_boundary,
        conv_dim=conv_dim,
        boundary=kernel_width - 1,
    )
    default_stream.wait_stream(tail_stream)
    _wait_distributed_ops(left_boundary_send_ops)
    _ = left_boundary_send_buf
    d_raw_g, d_gate, d_beta = tail_grads
    return d_qkv, d_raw_g, d_gate, d_beta, d_weight


class FusedPreKDAFunction(torch.autograd.Function):
    """Autograd wrapper for the KDA-specific streamed preprocessing path."""

    @staticmethod
    def forward(
        ctx,
        qkv,
        raw_g,
        gate,
        beta,
        conv1d_weight,
        cu_seqlens,
        seq_idx,
        cp_group,
        cp_size,
        num_heads,
        head_dim,
    ):
        """Run fused KDA preprocessing and save the minimum backward state."""

        ctx.num_heads = num_heads
        ctx.head_dim = head_dim
        ctx.raw_g_dtype = raw_g.dtype
        ctx.gate_dtype = gate.dtype
        boundary = conv1d_weight.shape[-1] - 1
        cp_active = cp_group is not None and cp_size > 1 and boundary > 0
        ctx.cp_active = cp_active
        ctx.cp_group = cp_group
        ctx.has_cu_seqlens = False
        ctx.has_left_boundary = False
        ctx.global_token_offset = 0
        ctx.global_seq_len = qkv.shape[0]
        cp_rank = cp_group.rank() if cp_active else 0

        seq_idx_for_backward = seq_idx
        seq_idx_ready_event = None
        if cp_active and cu_seqlens is not None:
            ctx.has_cu_seqlens = True
            seq_idx_for_backward = _resolve_chunkwise_cp_packed_seq_idx(
                cu_seqlens, qkv.shape[0], cp_rank
            )
            seq_idx_ready_event = torch.cuda.Event()
            seq_idx_ready_event.record(torch.cuda.current_stream(qkv.device))
        ctx.seq_idx_ready_event = seq_idx_ready_event

        (
            query,
            key,
            value,
            gate_out,
            beta_out,
            raw_g_out,
            silu_qk_save,
            left_boundary,
            global_token_offset,
            global_seq_len,
        ) = _triton_pre_kda_forward(
            qkv,
            raw_g,
            gate,
            beta,
            conv1d_weight,
            num_heads=num_heads,
            head_dim=head_dim,
            save_silu=True,
            cu_seqlens=cu_seqlens,
            cp_group=cp_group,
            cp_size=cp_size,
        )
        assert silu_qk_save is not None
        ctx.has_left_boundary = left_boundary is not None
        ctx.global_token_offset = global_token_offset
        ctx.global_seq_len = global_seq_len
        ctx.has_seq_idx = seq_idx_for_backward is not None
        saved_tensors = [qkv, beta, conv1d_weight, silu_qk_save]
        if ctx.has_seq_idx:
            saved_tensors.append(seq_idx_for_backward)
        if ctx.has_cu_seqlens:
            saved_tensors.append(cu_seqlens)
        if ctx.has_left_boundary:
            saved_tensors.append(left_boundary)
        ctx.save_for_backward(*saved_tensors)
        return query, key, value, gate_out, beta_out, raw_g_out

    @staticmethod
    def backward(ctx, dq, dk, dv, dgate, dbeta, d_raw_g_out):
        """Run the fused KDA preprocessing backward path."""

        saved_idx = 0
        qkv = ctx.saved_tensors[saved_idx]
        saved_idx += 1
        beta = ctx.saved_tensors[saved_idx]
        saved_idx += 1
        conv1d_weight = ctx.saved_tensors[saved_idx]
        saved_idx += 1
        silu_qk_save = ctx.saved_tensors[saved_idx]
        saved_idx += 1
        if ctx.has_seq_idx:
            seq_idx = ctx.saved_tensors[saved_idx]
            saved_idx += 1
            if ctx.seq_idx_ready_event is not None:
                torch.cuda.current_stream(qkv.device).wait_event(ctx.seq_idx_ready_event)
        else:
            seq_idx = None
        if ctx.has_cu_seqlens:
            cu_seqlens = ctx.saved_tensors[saved_idx]
            saved_idx += 1
        else:
            cu_seqlens = None
        if ctx.has_left_boundary:
            left_boundary = ctx.saved_tensors[saved_idx]
        else:
            left_boundary = None

        d_qkv, d_raw_g, d_gate, d_beta, d_weight = _triton_pre_kda_backward(
            qkv,
            beta,
            conv1d_weight,
            silu_qk_save,
            dq,
            dk,
            dv,
            dgate,
            dbeta,
            d_raw_g_out,
            raw_g_dtype=ctx.raw_g_dtype,
            gate_dtype=ctx.gate_dtype,
            num_heads=ctx.num_heads,
            head_dim=ctx.head_dim,
            seq_idx=seq_idx,
            left_boundary=left_boundary,
            cu_seqlens=cu_seqlens,
            global_token_offset=ctx.global_token_offset,
            global_seq_len=ctx.global_seq_len,
            cp_group=ctx.cp_group if ctx.cp_active else None,
        )
        return (d_qkv, d_raw_g, d_gate, d_beta, d_weight, None, None, None, None, None, None)


def _validate_fused_streamed_pre_kda_inputs(
    qkv: Tensor,
    raw_g: Tensor,
    gate: Tensor,
    beta: Tensor,
    conv1d_weight: Tensor,
    conv1d_bias: Optional[Tensor],
    *,
    num_heads: int,
    head_dim: int,
    use_qk_l2norm: bool,
    cu_seqlens: Optional[Tensor],
    seq_idx: Optional[Tensor],
    cp_size: int,
) -> None:
    """Validate the public contract for streamed fused KDA preprocessing."""

    if causal_conv1d_bwd_function is None:
        raise ImportError(
            "gdn_pre_gated_delta_rule_fusion requires causal-conv1d. "
            "Install causal-conv1d~=1.6 in environments that enable this fusion."
        )
    for name, tensor in (("qkv", qkv), ("raw_g", raw_g), ("gate", gate), ("beta", beta)):
        assert tensor.is_cuda, f"fused_streamed_pre_kda requires CUDA {name}; got {tensor.device}."
        assert (
            tensor.dim() == 3
        ), f"fused_streamed_pre_kda expects 3-D {name}; got shape {tuple(tensor.shape)}."
        assert tensor.shape[:2] == qkv.shape[:2], (
            f"KDA {name} leading dimensions {tuple(tensor.shape[:2])} do not match "
            f"qkv {tuple(qkv.shape[:2])}."
        )
        assert (
            tensor.device == qkv.device
        ), f"KDA {name} must be on {qkv.device}; got {tensor.device}."

    channels = num_heads * head_dim
    assert qkv.shape[-1] == 3 * channels, (
        f"KDA qkv width must be 3 * num_heads * head_dim ({3 * channels}); " f"got {qkv.shape[-1]}."
    )
    assert (
        raw_g.shape[-1] == channels
    ), f"KDA raw_g width must be num_heads * head_dim ({channels}); got {raw_g.shape[-1]}."
    assert (
        gate.shape[-1] == channels
    ), f"KDA gate width must be num_heads * head_dim ({channels}); got {gate.shape[-1]}."
    assert (
        beta.shape[-1] == num_heads
    ), f"KDA beta width must be num_heads ({num_heads}); got {beta.shape[-1]}."
    assert conv1d_weight.shape[:2] == (3 * channels, 1), (
        "KDA conv weight must have shape "
        f"[{3 * channels}, 1, kernel_width]; got {tuple(conv1d_weight.shape)}."
    )
    assert conv1d_bias is None, "Conv bias is not supported by fused_streamed_pre_kda."
    assert use_qk_l2norm, "use_qk_l2norm=False is not supported by fused_streamed_pre_kda."

    if cp_size > 1:
        if qkv.shape[1] != 1:
            raise ValueError(
                "KDA chunkwise CP with SBHD inputs currently requires micro_batch_size == 1 "
                f"for fused_streamed_pre_kda; got batch={qkv.shape[1]}."
            )
        boundary = conv1d_weight.shape[-1] - 1
        if boundary > 0 and qkv.shape[0] < boundary:
            raise ValueError(
                "fused_streamed_pre_kda chunkwise CP requires local chunk length "
                f"({qkv.shape[0]}) >= conv_kernel_dim - 1 ({boundary})."
            )
        if seq_idx is not None:
            raise ValueError(
                "fused_streamed_pre_kda derives packed seq_idx internally when "
                "chunkwise CP is active."
            )

    if cu_seqlens is not None:
        assert cu_seqlens.is_cuda, (
            "Packed fused_streamed_pre_kda requires CUDA cu_seqlens; " f"got {cu_seqlens.device}."
        )
        assert cu_seqlens.dtype == torch.int32, (
            "Packed fused_streamed_pre_kda requires int32 cu_seqlens; " f"got {cu_seqlens.dtype}."
        )
        assert cu_seqlens.dim() == 1 and cu_seqlens.shape[0] >= 2, (
            "Packed fused_streamed_pre_kda requires 1-D cu_seqlens containing "
            f"at least one sequence; got shape {tuple(cu_seqlens.shape)}."
        )
        assert qkv.shape[1] == 1, (
            "Packed THD fused_streamed_pre_kda expects batch dimension 1; "
            f"got qkv shape {tuple(qkv.shape)}."
        )
    else:
        assert seq_idx is None, "seq_idx requires cu_seqlens for packed THD mode."


def fused_streamed_pre_kda(
    qkv: Tensor,
    raw_g: Tensor,
    gate: Tensor,
    beta: Tensor,
    conv1d_weight: Tensor,
    conv1d_bias: Optional[Tensor],
    *,
    num_heads: int,
    head_dim: int,
    use_qk_l2norm: bool = True,
    cu_seqlens: Optional[Tensor] = None,
    seq_idx: Optional[Tensor] = None,
    cp_group=None,
    strict_runtime_validation: bool = True,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Run streamed fused preprocessing for explicit KDA projection tensors.

    ``qkv``, ``raw_g``, and ``gate`` may be independent projection outputs or
    views into KDA's legacy fused projection. Returning one gradient per input
    lets autograd merge the view gradients for the legacy checkpoint layout
    without coupling the fused kernel to either projection scheme.
    """

    cp_size = cp_group.size() if cp_group is not None else 1
    if strict_runtime_validation:
        _validate_fused_streamed_pre_kda_inputs(
            qkv,
            raw_g,
            gate,
            beta,
            conv1d_weight,
            conv1d_bias,
            num_heads=num_heads,
            head_dim=head_dim,
            use_qk_l2norm=use_qk_l2norm,
            cu_seqlens=cu_seqlens,
            seq_idx=seq_idx,
            cp_size=cp_size,
        )

    needs_backward = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (qkv, raw_g, gate, beta, conv1d_weight)
    )

    if cu_seqlens is not None:
        cu_seqlens = cu_seqlens.contiguous()
        if cp_size == 1 and needs_backward:
            seq_idx = _resolve_packed_seq_idx(cu_seqlens, seq_idx, qkv.shape[0])

    if not needs_backward:
        query, key, value, gate_out, beta_out, raw_g_out, _, _, _, _ = _triton_pre_kda_forward(
            qkv,
            raw_g,
            gate,
            beta,
            conv1d_weight,
            num_heads=num_heads,
            head_dim=head_dim,
            save_silu=False,
            cu_seqlens=cu_seqlens,
            cp_group=cp_group,
            cp_size=cp_size,
        )
        return query, key, value, gate_out, beta_out, raw_g_out

    return FusedPreKDAFunction.apply(
        qkv,
        raw_g,
        gate,
        beta,
        conv1d_weight,
        cu_seqlens,
        seq_idx,
        cp_group,
        cp_size,
        num_heads,
        head_dim,
    )
