# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Fused forward + backward for the pre-gated-delta-rule front-end.

This module is the high-performance replacement for the unfused
``torch_native_pre_gated_delta_rule`` reference path that
``GatedDeltaNet.pre_gated_delta_rule`` previously dispatched to. It takes
the dense ``qkvzba`` projection produced by ``in_proj`` and emits the six
tensors the gated delta rule consumes — ``query``, ``key``, ``value``,
``gate`` (z), ``beta`` (sigmoid), ``g`` (decay) — fully on the GPU through
hand-written Triton kernels plus one delegation to the upstream
``causal_conv1d`` C++ kernel for the conv backward.

Both the forward and the backward are designed to reduce redundant HBM
traffic and kernel launch overhead relative to the unfused reference path.
Detailed benchmark numbers are intentionally kept outside this source file
because they are environment-specific.

High-level design
-----------------

Forward (``_triton_pre_gated_delta_rule_forward``)
    Four Triton kernels on four CUDA side streams overlap the four
    sub-computations:

    * ``_conv_silu_project_kernel`` QK specialization (1 launch on s_qk).
      Q and K are one logical branch. The grid selects one Q/K group per
      program tile, then performs depthwise conv1d → silu → l2norm →
      REPEAT-way head broadcast directly into the final query/key layouts.
      The launch persists the bf16-rounded ``silu(conv(x))`` scratch for
      both Q and K so the current backward can keep using its saved-QK path.

    * ``_conv_silu_project_kernel`` V specialization (1 launch on s_v).
      Depthwise conv1d → silu directly into the final value layout. This
      shares the source-level conv+silu template with QK but compiles out
      QK-only l2norm/repeat logic.

    * ``_copy_z_kernel`` (1 launch on s_z). Directly copies the z slice into
      the final gate layout.

    * ``_compute_g_and_beta_kernel`` (1 launch on s_gbeta).
      ``g = -exp(A_log) * softplus(alpha + dt_bias)`` and
      ``beta = sigmoid(beta_raw)``.

Backward (``_triton_pre_gated_delta_rule_backward``)
    Five Triton kernels + one delegated C++ call, also fanned out on side
    streams:

    * ``_l2norm_repeat_backward_kernel`` (2 launches: Q on s_q, K on s_k).
      Reduces REPEAT v-heads of d_normed back to one per qk-head, then
      applies the l2norm backward against the saved ``silu_bf16`` to
      produce d_silu_conv.

    * ``_v_layout_to_conv_kernel`` (s_v). Streams ``dv`` straight into
      the V channel slice of d_silu_conv with a coalesced read/write —
      replaces a torch ``reshape``+``transpose``+``copy_`` that
      previously produced a non-overlapping ~150 μs ``direct_copy_kernel``.

    * ``_z_layout_to_qkvzba_kernel`` (s_z). Same idea for the z-slice
      gradient. The torch ``copy_`` it replaced was ~90 μs and didn't
      respect stream context.

    * ``_g_beta_backward_kernel`` (s_gbeta). Inverse of the forward g+beta
      kernel; produces d_A_log, d_dt_bias, d_alpha, d_beta_raw.

    * ``causal_conv1d_bwd_function`` (default stream). Hand-tuned C++ from
      the upstream ``causal_conv1d`` package. Computes d_x and d_w from
      d_silu_conv directly — bypassing PyTorch's autograd-engine entry
      point avoids a forward re-execution that ``CausalConv1dFn.backward``
      would otherwise trigger.

Stream wiring
-------------
Both forward and backward use five long-lived side streams
(``_get_side_stream`` with slot 0–4). The slot assignment is the same in
both directions so the same physical CUDA stream stays warm across a
forward / backward pair. Concurrent launches don't get more HBM
bandwidth than serial ones, but they do let the kernel scheduler
overlap kernels that have spare SMs.

Numerical precision
-------------------
* Conv accumulation runs in fp32 inside the kernel. ``acc`` is rounded
  through bf16 before silu to match the bit-exact ``F.conv1d → F.silu``
  reference within one bf16 ULP.
* ``silu_bf16`` is rounded again before the l2norm reduction so the
  fused l2norm sees the same rounded tensor the reference does.
* All gradient buffers we write into are bf16; reductions in the g+beta
  backward use fp32 atomic accumulators, cast to the parameter dtype
  before being returned.
* Validated against torch autograd by the targeted GDN unit tests in
  ``tests/unit_tests/ssm/test_gated_delta_net.py::TestFusedPreGatedDeltaRule``.

Requirements / unsupported configs
----------------------------------
* CUDA input only. CPU tensors raise ``NotImplementedError`` — there is
  no torch reference fallback in this file. (The unfused reference is in
  the GDN module if you need it for testing.)
* ``causal_conv1d`` package must be importable. The 1.6.1+ entry point
  ``causal_conv1d.cpp_functions.causal_conv1d_bwd_function`` is preferred;
  older builds expose the same function as
  ``causal_conv1d_cuda.causal_conv1d_bwd``.
* Conv bias is not supported. The production GDN config has none.
* ``use_qk_l2norm=False`` is not supported; the backward closes over the
  saved ``silu_bf16`` produced under the l2norm path.
* Packed sequences (``cu_seqlens != None``) and context parallelism (CP)
  are not supported.

Future directions (NOT implemented)
-----------------------------------
1. **Single-load conv input.** Each program currently issues K_W shifted
   ``tl.load`` calls of shape ``(BLOCK_S, HEAD_DIM)``, of which only
   ``BLOCK_S + K_W - 1`` rows are unique. Loading the union once and
   slicing K_W windows out of the register tile would halve conv-input
   HBM traffic. Blocked on current Triton limitations around slicing
   2D register-resident tensors. Revisit when the compiler/runtime
   supports this pattern directly.

2. **Multiple heads per program (``HEADS_PER_PROG > 1``).** Widens the
   per-row HBM transaction and lets adjacent programs on the same SM
   share L2 lines for the overlapping ``s + i`` conv-window reads.
   Structural rewrite — the l2norm reduction has to stay per-head.

3. **Drop the bf16 round-trip after conv.** Today the kernel does
   ``acc.to(bf16).to(fp32)`` to match the unfused F.conv1d → F.silu
   rounding boundary bit-for-bit. If the downstream gated_delta_rule
   kernel tolerates fp32 precision, the round-trip can go. Needs a
   tolerance study — current UT bound is too tight to absorb.

4. **Fused-streamed backward.** Forward now treats QK as one logical branch
   and keeps V/Z/G-Beta separate. Backward still uses the previous staged
   l2norm/V/Z transforms plus the external causal-conv backward. A future
   pass should replace that with QK/V branch kernels that compute conv input
   gradients and channel weight gradients in the same scope.

5. **Reduce HBM traffic in backward.** The backward is bandwidth-bound;
   stream-overlap tuning (Phase 11) was a no-op for that reason. The
   remaining levers are either saving more intermediates from forward
   (Phase 9 did silu_bf16; could also save rstd) or restructuring the
   l2norm kernel to keep more data SM-local.

6. **Hand-rolled CUDA kernel (ldmatrix / TMA).** Bypass Triton for the
   conv kernel with CUTLASS-style SMEM staging. This is a larger
   maintenance burden — only pursue if (1) or (2) hit a wall in Triton.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# The 1.6.1+ ``causal_conv1d`` package exposes the lower-level binding via
# ``causal_conv1d.cpp_functions.causal_conv1d_bwd_function``; older builds
# (still common in some older environments) expose the same
# function under ``causal_conv1d_cuda.causal_conv1d_bwd``. Try both so the
# fast path is taken everywhere the package is installed.
from torch import Tensor

try:
    from causal_conv1d.cpp_functions import (
        causal_conv1d_bwd_function as _causal_conv1d_bwd_function,
    )
except ImportError:
    import causal_conv1d_cuda as _causal_conv1d_cuda

    _causal_conv1d_bwd_function = _causal_conv1d_cuda.causal_conv1d_bwd


_L2NORM_EPS = 1e-6


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


@triton.autotune(
    configs=_conv_autotune_configs(),
    key=["seq_len", "HEAD_DIM", "K_W", "APPLY_L2", "REPEAT", "NUM_GROUPS"],
)
@triton.jit
def _conv_silu_project_kernel(
    qkvzba_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    silu_save_ptr,
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
    eps,
    HEAD_DIM: tl.constexpr,
    K_W: tl.constexpr,
    REPEAT: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
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
        x_ptr = (
            qkvzba_ptr
            + x_s[:, None] * qkvzba_s_stride
            + batch_id * qkvzba_b_stride
            + chan[None, :] * qkvzba_c_stride
        )
        x_val = tl.load(x_ptr, mask=x_mask[:, None], other=0.0).to(tl.float32)
        w_tap = tl.load(
            weight_ptr + chan * weight_c_stride + i * weight_w_stride
        ).to(tl.float32)
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
            # Phase 9: persist silu_bf16 so the backward can skip the
            # ``causal_conv1d_fn(activation="silu")`` recompute. Layout
            # mirrors what causal_conv1d would have returned: (b, c, s)
            # bf16. We only save the QK slice (V doesn't need l2norm
            # backward).
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
                silu_save_ptrs,
                silu_out.to(silu_save_ptr.dtype.element_ty),
                mask=s_mask[:, None],
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
    # softplus(x) = log(1 + exp(x)); torch's softplus thresholds at x>20 but we
    # rely on fp32 evaluation here, which stays well within range for typical
    # GDN inputs (the unfused path computes the same expression).
    softplus_val = tl.log(1.0 + tl.exp(pre))
    g = -tl.exp(A_log)[None, :] * softplus_val
    beta_sig = tl.sigmoid(beta)

    g_ptr = (
        g_out_ptr
        + pid_b * g_b_stride
        + s_offs[:, None] * g_s_stride
        + h_offs[None, :] * g_h_stride
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


@triton.jit
def _conv_silu_l2norm_backward_kernel(
    qkvzba_ptr,
    weight_ptr,
    d_out_ptr,
    d_qkvzba_ptr,
    d_w_partial_ptr,
    seq_len,
    num_qk_heads,
    in_channel_offset,
    eps,
    d_out_scale,
    qkvzba_s_stride,
    qkvzba_b_stride,
    qkvzba_c_stride,
    weight_c_stride,
    weight_w_stride,
    d_out_b_stride,
    d_out_s_stride,
    d_out_h_stride,
    d_wp_b_stride,
    d_wp_h_stride,
    d_wp_s_stride,
    d_wp_c_stride,
    d_wp_w_stride,
    HEAD_DIM: tl.constexpr,
    K_W: tl.constexpr,
    REPEAT: tl.constexpr,
    BLOCK_S: tl.constexpr,
    USE_L2NORM: tl.constexpr,
    V_HEAD_SHARED: tl.constexpr,
):
    """Backward for the Q / K / V branches of ``_conv_silu_project_kernel``.

    ``USE_L2NORM`` is a constexpr branch: ``True`` for the QK branches (with
    l2norm) and ``False`` for the V branch. The V case skips the l2norm
    intermediates entirely — Triton DCE drops them at compile time. Phase
    8.7's ``REPEAT=2`` workaround for the channel-collapse codegen bug still
    applies in both branches.

    Forward (no bias, with l2norm, with REPEAT-way head broadcast):
        acc       = depthwise_conv(qkvzba_qk_slice, weight_qk_slice)
        acc_bf16  = acc.to(bf16).to(fp32)              # F.conv1d rounding
        silu_out  = acc_bf16 * sigmoid(acc_bf16)
        silu_bf16 = silu_out.to(bf16).to(fp32)         # round before l2norm
        norm_sq   = sum_c silu_bf16^2
        rstd      = 1 / sqrt(norm_sq + eps)
        out       = silu_bf16 * rstd
        # out is stored identically to REPEAT adjacent value heads.

    Backward (given d_out for each v_head):
        d_qk_out  = Σ_{r in REPEAT} d_v_out[head_id * REPEAT + r]
        S         = Σ_c d_qk_out_c * silu_bf16_c
        d_silu_c  = rstd * d_qk_out_c - rstd^3 * silu_bf16_c * S
        d_acc_c   = d_silu_c * silu'(acc_bf16)
        d_w[c, i] = Σ_{b, t} d_acc[t, c] * x[t + i - (K_W - 1), c]
        d_x[u, c] += Σ_i d_acc[u + (K_W - 1) - i, c] * w[c, i]

    ``d_w`` uses the per-program partial-buffer pattern (see the V backward
    kernel). ``d_qkvzba`` uses bf16 atomic_add because the K_W − 1 boundary
    input rows cross seq-block programs.
    """

    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)

    batch_id = pid_bh // num_qk_heads
    head_id = pid_bh - batch_id * num_qk_heads

    chan_off = tl.arange(0, HEAD_DIM)
    chan = in_channel_offset + head_id * HEAD_DIM + chan_off

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < seq_len

    # ----- Forward recompute (conv + silu + l2norm) -----
    acc = tl.zeros([BLOCK_S, HEAD_DIM], dtype=tl.float32)
    for i in tl.static_range(K_W):
        x_s = s_offs - (K_W - 1) + i
        x_mask = (x_s >= 0) & (x_s < seq_len)
        x_ptr = (
            qkvzba_ptr
            + x_s[:, None] * qkvzba_s_stride
            + batch_id * qkvzba_b_stride
            + chan[None, :] * qkvzba_c_stride
        )
        x_val = tl.load(x_ptr, mask=x_mask[:, None], other=0.0).to(tl.float32)
        w_tap = tl.load(
            weight_ptr + chan * weight_c_stride + i * weight_w_stride
        ).to(tl.float32)
        acc += w_tap[None, :] * x_val
    acc = acc.to(d_qkvzba_ptr.dtype.element_ty).to(tl.float32)

    # ----- Sum d_out across REPEAT v_heads -----
    # For QK: v_head = head_id*REPEAT + r, summing REPEAT distinct heads.
    # For V (V_HEAD_SHARED=True): both r iterations load the SAME v_head
    # (head_id), so d_qk_out = REPEAT * d_value[head_id]; the host passes
    # d_out_scale = 1/REPEAT to recover d_value[head_id]. The duplicate
    # load goes through L2, so the kernel-side cost is roughly one load;
    # the trick was needed to avoid the REPEAT=1 codegen bug without
    # having to allocate a doubled d_value tensor on the host (~134MB).
    d_qk_out = tl.zeros([BLOCK_S, HEAD_DIM], dtype=tl.float32)
    for r in tl.static_range(REPEAT):
        if V_HEAD_SHARED:
            v_head = head_id
        else:
            v_head = head_id * REPEAT + r
        d_out_ptrs = (
            d_out_ptr
            + batch_id * d_out_b_stride
            + s_offs[:, None] * d_out_s_stride
            + v_head * d_out_h_stride
            + chan_off[None, :]
        )
        d_qk_out += tl.load(d_out_ptrs, mask=s_mask[:, None], other=0.0).to(tl.float32)
    d_qk_out = d_qk_out * d_out_scale

    # ----- l2norm backward gated by USE_L2NORM constexpr. -----
    if USE_L2NORM:
        silu_out = acc * tl.sigmoid(acc)
        silu_bf16 = silu_out.to(d_qkvzba_ptr.dtype.element_ty).to(tl.float32)
        norm_sq = tl.sum(silu_bf16 * silu_bf16, axis=1)
        rstd = 1.0 / tl.sqrt(norm_sq + eps)
        s_row = tl.sum(d_qk_out * silu_bf16, axis=1)
        rstd3 = rstd * rstd * rstd
        d_silu = rstd[:, None] * d_qk_out - rstd3[:, None] * silu_bf16 * s_row[:, None]
    else:
        d_silu = d_qk_out

    # ----- silu backward -----
    sig_acc = tl.sigmoid(acc)
    silu_prime = sig_acc + acc * sig_acc * (1.0 - sig_acc)
    d_acc = d_silu * silu_prime
    d_acc = tl.where(s_mask[:, None], d_acc, 0.0)

    # ----- d_w via per-program partial, d_x via atomic_add -----
    partial_base = (
        d_w_partial_ptr
        + batch_id * d_wp_b_stride
        + head_id * d_wp_h_stride
        + pid_s * d_wp_s_stride
    )

    for i in tl.static_range(K_W):
        x_s = s_offs - (K_W - 1) + i
        x_mask_inner = (x_s >= 0) & (x_s < seq_len)
        x_ptr = (
            qkvzba_ptr
            + x_s[:, None] * qkvzba_s_stride
            + batch_id * qkvzba_b_stride
            + chan[None, :] * qkvzba_c_stride
        )
        x_val = tl.load(x_ptr, mask=x_mask_inner[:, None], other=0.0).to(tl.float32)
        d_w_partial = tl.sum(d_acc * x_val, axis=0)
        tl.store(
            partial_base + chan_off * d_wp_c_stride + i * d_wp_w_stride,
            d_w_partial,
        )

        w_tap = tl.load(
            weight_ptr + chan * weight_c_stride + i * weight_w_stride
        ).to(tl.float32)
        contribution = d_acc * w_tap[None, :]
        d_qkvzba_target = (
            d_qkvzba_ptr
            + x_s[:, None] * qkvzba_s_stride
            + batch_id * qkvzba_b_stride
            + chan[None, :] * qkvzba_c_stride
        )
        tl.atomic_add(
            d_qkvzba_target,
            contribution.to(d_qkvzba_ptr.dtype.element_ty),
            mask=x_mask_inner[:, None],
        )


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
def _l2norm_repeat_backward_kernel(
    d_qk_out_ptr,        # (b, s, num_v_heads, head_dim) — gradient from downstream
    silu_bf16_ptr,        # (b, conv_dim, s) — silu(conv(x)) recomputed for QK channels
    d_silu_bf16_ptr,      # (b, conv_dim, s) — output gradient w.r.t. silu(conv(x))
    seq_len,
    num_qk_heads,
    channel_offset,       # 0 for Q, qk_channels for K — indexes into conv_dim
    eps,
    d_qk_b_stride,
    d_qk_s_stride,
    d_qk_h_stride,
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
    """l2norm + REPEAT-way head broadcast backward (Phase 8.15).

    Forward (per QK head):
        silu_bf16 ∈ R^{HEAD_DIM}   # silu(conv(x)) rounded to bf16
        norm_sq   = Σ_c silu_bf16_c^2
        rstd      = 1 / sqrt(norm_sq + eps)
        out       = silu_bf16 * rstd
        # out is broadcast identically to REPEAT adjacent v_heads.

    Backward (given d_qk_out for each v_head):
        d_normed  = Σ_{r in REPEAT} d_qk_out[head_id * REPEAT + r]
        S         = Σ_c d_normed_c * silu_bf16_c
        d_silu_c  = rstd * d_normed_c - rstd^3 * silu_bf16_c * S

    The output ``d_silu_bf16`` is the gradient w.r.t. ``silu(conv(x))`` —
    exactly what ``causal_conv1d_bwd_function`` consumes as its ``dout``
    argument when ``activation="silu"`` is in effect on the forward.
    """

    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)

    batch_id = pid_bh // num_qk_heads
    head_id = pid_bh - batch_id * num_qk_heads

    chan_off = tl.arange(0, HEAD_DIM)
    chan = channel_offset + head_id * HEAD_DIM + chan_off

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < seq_len

    # ----- Sum d_qk_out across REPEAT v_heads -----
    d_normed = tl.zeros([BLOCK_S, HEAD_DIM], dtype=tl.float32)
    for r in tl.static_range(REPEAT):
        v_head = head_id * REPEAT + r
        d_out_ptrs = (
            d_qk_out_ptr
            + batch_id * d_qk_b_stride
            + s_offs[:, None] * d_qk_s_stride
            + v_head * d_qk_h_stride
            + chan_off[None, :]
        )
        d_normed += tl.load(d_out_ptrs, mask=s_mask[:, None], other=0.0).to(tl.float32)

    # ----- Load silu_bf16 -----
    silu_ptrs = (
        silu_bf16_ptr
        + batch_id * silu_b_stride
        + chan[None, :] * silu_c_stride
        + s_offs[:, None] * silu_s_stride
    )
    silu_bf16 = tl.load(silu_ptrs, mask=s_mask[:, None], other=0.0).to(tl.float32)

    # ----- l2norm backward -----
    norm_sq = tl.sum(silu_bf16 * silu_bf16, axis=1)
    rstd = 1.0 / tl.sqrt(norm_sq + eps)
    s_row = tl.sum(d_normed * silu_bf16, axis=1)
    rstd3 = rstd * rstd * rstd
    d_silu = rstd[:, None] * d_normed - rstd3[:, None] * silu_bf16 * s_row[:, None]

    # ----- Store d_silu (same (b, conv_dim, s) layout as silu_bf16_ptr) -----
    d_silu_ptrs = (
        d_silu_bf16_ptr
        + batch_id * d_silu_b_stride
        + chan[None, :] * d_silu_c_stride
        + s_offs[:, None] * d_silu_s_stride
    )
    tl.store(d_silu_ptrs, d_silu.to(d_silu_bf16_ptr.dtype.element_ty), mask=s_mask[:, None])


@triton.jit
def _v_layout_to_conv_kernel(
    dv_ptr,                # (b, s, num_v_heads, value_head_dim)
    d_silu_conv_ptr,        # (b, conv_dim, s) — write into V channel slice
    seq_len,
    num_v_heads,
    v_channel_offset,       # = 2 * qk_channels
    dv_b_stride,
    dv_s_stride,
    dv_h_stride,
    d_silu_b_stride,
    d_silu_c_stride,
    d_silu_s_stride,
    HEAD_DIM: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Phase 8.15a: V-branch layout transform without a torch ``copy_``.

    ``dv`` is the gradient of ``value`` (forward layout
    ``(b, s, num_v_heads, value_head_dim)``). The conv backward needs
    ``d_silu_conv`` in layout ``(b, conv_dim, s)`` for the V channel
    slice. A one-shot Triton kernel streams dv directly into the V slice
    with a single launch, no intermediate buffer, and a contiguous write
    pattern at the head granularity.
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
    dgate_ptr,             # (b, s, num_v_heads, value_head_dim)
    d_qkvzba_ptr,           # (s, b, total_channels) — write into z channel slice
    seq_len,
    num_v_heads,
    z_channel_offset,       # = 2 * qk_channels + v_channels
    dgate_b_stride,
    dgate_s_stride,
    dgate_h_stride,
    d_qkvzba_s_stride,
    d_qkvzba_b_stride,
    d_qkvzba_c_stride,
    HEAD_DIM: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Phase 8.15b: z-slice gradient layout transform without a torch ``copy_``.

    ``dgate`` is the autograd-supplied gradient of ``gate`` (= the z
    slice of qkvzba in forward) with layout
    ``(b, s, num_v_heads, value_head_dim)``. We need to write it into
    ``d_qkvzba``'s z slice — layout ``(s, b, total_channels)`` with
    channels in ``[z_channel_offset, z_channel_offset + v_channels)``.

    The previous ``d_qkvzba[..., z:z+v_c].copy_(dgate.reshape(b, s, v_c)
    .transpose(0, 1))`` used strided tensors, so TensorIterator fell back
    to a scalar-vectorized direct_copy_kernel. Stream context also doesn't
    always propagate through that path, so the launch wasn't overlapping
    with the other side-stream kernels in the backward.

    A single Triton kernel launch with explicit coalesced reads and
    writes solves both problems at once.
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
        softplus_pre = log(1 + exp(pre))
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
    sigmoid_pre = tl.sigmoid(pre)
    softplus_pre = tl.log(1.0 + tl.exp(pre))
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
    tl.store(
        d_alpha_ptrs, d_alpha.to(d_qkvzba_ptr.dtype.element_ty), mask=mask
    )
    tl.store(
        d_beta_ptrs, d_beta_raw.to(d_qkvzba_ptr.dtype.element_ty), mask=mask
    )

    # ----- Atomic-add (b, s) partials into per-head accumulators -----
    tl.atomic_add(d_A_log_ptr + h_offs, d_A_log_partial, mask=h_mask)
    tl.atomic_add(d_dt_bias_ptr + h_offs, d_dt_bias_partial, mask=h_mask)


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


def _triton_l2norm_repeat_backward(
    d_qk_out: Tensor,
    silu_bf16: Tensor,
    d_silu_bf16: Tensor,
    *,
    is_query: bool,
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    eps: float = 1e-6,
    stream: Optional["torch.cuda.Stream"] = None,
) -> Tensor:
    """l2norm + REPEAT backward (Phase 8.15).

    ``silu_bf16`` is the (b, conv_dim, s) bf16 tensor produced by re-running
    causal_conv1d_fn (forward, no-grad). Output ``d_silu_bf16`` is written
    in place; only the matching channel slice (Q or K) is filled in.
    """

    batch = d_qk_out.shape[0]
    seq_len = d_qk_out.shape[1]
    qk_channels = num_key_heads * key_head_dim
    repeat = num_value_heads // num_key_heads
    channel_offset = 0 if is_query else qk_channels

    device = d_qk_out.device

    # Phase 10: BLOCK_S / num_warps / num_stages are chosen by the
    # autotune decorator on the kernel — see configs there.
    grid = lambda meta: (
        batch * num_key_heads,
        triton.cdiv(seq_len, meta["BLOCK_S"]),
    )

    if stream is not None:
        stream.wait_stream(torch.cuda.current_stream(device))

    launch_context = (
        torch.cuda.stream(stream) if stream is not None else _NullContext()
    )
    with launch_context:
        _l2norm_repeat_backward_kernel[grid](
            d_qk_out,
            silu_bf16,
            d_silu_bf16,
            seq_len,
            num_key_heads,
            channel_offset,
            eps,
            d_qk_out.stride(0),
            d_qk_out.stride(1),
            d_qk_out.stride(2),
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
    """Phase 8.15a: write ``dv`` into ``d_silu_conv``'s V channel slice."""

    batch, seq_len, _, _ = dv.shape
    device = dv.device

    BLOCK_S = 64
    num_seq_blocks = triton.cdiv(seq_len, BLOCK_S)
    grid = (batch * num_value_heads, num_seq_blocks)

    if stream is not None:
        stream.wait_stream(torch.cuda.current_stream(device))

    launch_context = (
        torch.cuda.stream(stream) if stream is not None else _NullContext()
    )
    with launch_context:
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
    """Phase 8.15b: write ``dgate`` into ``d_qkvzba``'s z channel slice.

    Replaces the torch ``copy_`` that previously sat in the backward and
    didn't reliably overlap with the other side-stream kernels.
    """

    batch, seq_len, _, _ = dgate.shape
    device = dgate.device

    BLOCK_S = 64
    num_seq_blocks = triton.cdiv(seq_len, BLOCK_S)
    grid = (batch * num_value_heads, num_seq_blocks)

    if stream is not None:
        stream.wait_stream(torch.cuda.current_stream(device))

    launch_context = (
        torch.cuda.stream(stream) if stream is not None else _NullContext()
    )
    with launch_context:
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
        and zero the buffer (the other backward kernels fill the rest).
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

    if stream is not None:
        stream.wait_stream(torch.cuda.current_stream(device))

    gb_grid = lambda meta: (
        batch,
        triton.cdiv(seq_len, meta["BLOCK_S"]),
        triton.cdiv(num_value_heads, meta["BLOCK_H"]),
    )
    launch_context = (
        torch.cuda.stream(stream) if stream is not None else _NullContext()
    )
    with launch_context:
        d_A_log = torch.zeros(num_value_heads, dtype=torch.float32, device=device)
        d_dt_bias = torch.zeros(num_value_heads, dtype=torch.float32, device=device)
        _g_beta_backward_kernel[gb_grid](
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


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


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
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Triton-backed forward for the pre-gated-delta-rule front-end.

    Returns ``(query, key, value, gate, beta, g, silu_qk_save)``. The last
    element is the bf16-rounded ``silu(conv(x))`` for the QK channel range
    laid out channel-last so the backward can feed it straight into
    ``causal_conv1d_bwd_function`` — see module docstring.
    """

    seq_len, batch, total_channels = qkvzba.shape
    qk_channels = num_key_heads * key_head_dim
    v_channels = num_value_heads * value_head_dim
    repeat_factor = num_value_heads // num_key_heads
    k_w = conv1d_weight.shape[-1]
    assert _is_power_of_two(key_head_dim), (
        "Triton kernel currently expects key_head_dim to be a power of two; "
        f"got {key_head_dim=}."
    )
    assert _is_power_of_two(value_head_dim), (
        "Triton kernel currently expects value_head_dim to be a power of two; "
        f"got {value_head_dim=}."
    )

    expected_channels = 2 * qk_channels + 2 * v_channels + 2 * num_value_heads
    assert total_channels == expected_channels, (
        f"qkvzba last-dim mismatch: got {total_channels}, expected {expected_channels}."
    )

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
    beta_out = torch.empty(batch, seq_len, num_value_heads, dtype=out_dtype, device=device)

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

    # Persist the QK silu(conv(x)) intermediate (Phase 9) so the backward
    # can skip the causal_conv1d_fn silu recompute. Channel-last
    # (stride(1)==1) layout so causal_conv1d_bwd_function consumes it
    # without an internal layout transform. ~16 MB at mbs=1 / 32 MB at
    # mbs=2 of scratch; dwarfed by the ~100 μs backward saving.
    silu_qk_save = torch.empty(
        (batch, seq_len, 2 * qk_channels), dtype=out_dtype, device=device
    ).permute(0, 2, 1)  # → (b, 2*qk_c, s) with stride(1)==1
    silu_save_b_stride = silu_qk_save.stride(0)
    silu_save_c_stride = silu_qk_save.stride(1)
    silu_save_s_stride = silu_qk_save.stride(2)

    # Stream setup. Each side stream handles one of the four sub-computations
    # (QK conv+l2norm, V conv, Z copy, g/beta).
    main_stream = torch.cuda.current_stream(device=device)
    qk_stream = _get_side_stream(device, slot=0)
    v_stream = _get_side_stream(device, slot=2)
    gb_stream = _get_side_stream(device, slot=3)
    z_stream = _get_side_stream(device, slot=4)
    for s in (qk_stream, v_stream, gb_stream, z_stream):
        s.wait_stream(main_stream)

    # --- QK conv + silu + l2norm + repeat ---
    qk_grid = lambda meta: (
        batch * 2 * num_key_heads,
        triton.cdiv(seq_len, meta["BLOCK_S"]),
    )
    with torch.cuda.stream(qk_stream):
        _conv_silu_project_kernel[qk_grid](
            qkvzba,
            weight_2d,
            bias_tensor,
            qk_out,
            silu_qk_save,
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
            _L2NORM_EPS,
            HEAD_DIM=key_head_dim,
            K_W=k_w,
            REPEAT=repeat_factor,
            NUM_GROUPS=2,
            HAS_BIAS=False,
            SAVE_SILU=True,
            APPLY_L2=True,
        )

    # --- V conv + silu (no l2norm, no repeat) ---
    v_channel_offset = 2 * qk_channels
    z_channel_offset = 2 * qk_channels + v_channels
    v_grid = lambda meta: (batch * num_value_heads, triton.cdiv(seq_len, meta["BLOCK_S"]))
    with torch.cuda.stream(v_stream):
        _conv_silu_project_kernel[v_grid](
            qkvzba,
            weight_2d,
            bias_tensor,
            value,
            qkvzba,  # silu_save unused (SAVE_SILU=False)
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
            _L2NORM_EPS,
            HEAD_DIM=value_head_dim,
            K_W=k_w,
            REPEAT=1,
            NUM_GROUPS=1,
            HAS_BIAS=False,
            SAVE_SILU=False,
            APPLY_L2=False,
        )

    # --- Z copy ---
    BLOCK_Z_S = 64
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

    # --- g and beta ---
    beta_channel_offset = 2 * qk_channels + 2 * v_channels
    alpha_channel_offset = beta_channel_offset + num_value_heads
    gb_grid = lambda meta: (
        batch,
        triton.cdiv(seq_len, meta["BLOCK_S"]),
        triton.cdiv(num_value_heads, meta["BLOCK_H"]),
    )
    with torch.cuda.stream(gb_stream):
        _compute_g_and_beta_kernel[gb_grid](
            qkvzba,
            A_log,
            dt_bias,
            g,
            beta_out,
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
            beta_out.stride(0),
            beta_out.stride(1),
            beta_out.stride(2),
        )

    # Re-join the side streams so the caller's stream observes the writes.
    main_stream.wait_stream(qk_stream)
    main_stream.wait_stream(v_stream)
    main_stream.wait_stream(z_stream)
    main_stream.wait_stream(gb_stream)

    return query, key, value, gate, beta_out, g, silu_qk_save


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
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Triton-backed backward for the pre-gated-delta-rule front-end.

    Mirror of :func:`_triton_pre_gated_delta_rule_forward`. Takes upstream
    gradients (``dq``/``dk``/``dv``/``dgate``/``dbeta``/``dg``) plus the
    saved forward intermediates and returns input/parameter gradients
    ``(d_qkvzba, d_weight, d_A_log, d_dt_bias)``.

    Five Triton kernels + one C++ ``causal_conv1d_bwd_function`` call,
    fanned out on five side streams so memory-bound work overlaps while
    the conv backward runs on the default stream. See module docstring
    for the overall design.
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

    # ``silu_qk_save`` is the (b, 2*qk_channels, s) bf16 buffer the
    # forward wrote ``silu(conv(x))`` into for QK (Phase 9). We re-use
    # it directly as the silu input to the l2norm backward, avoiding a
    # ~100 μs ``causal_conv1d_fn`` re-execution on the critical path.
    silu_conv = silu_qk_save

    # Allocate d_silu_conv channel-last (stride(1)==1) — that's what
    # ``causal_conv1d_channellast_bwd_kernel`` consumes natively. A naive
    # ``torch.empty(b, c, s)`` would be contiguous (stride(2)==1) and
    # trigger a ~500 μs internal layout-transform direct_copy.
    d_silu_conv = torch.empty(
        (batch, seq_len, conv_dim), dtype=qkvzba.dtype, device=device
    ).permute(0, 2, 1)

    # Side streams: Q=0, K=1, V=2, gb=3, z=4 — same slots as the forward
    # so the physical CUDA streams stay warm across forward / backward.
    s_q = _get_side_stream(device, slot=0)
    s_k = _get_side_stream(device, slot=1)
    s_v = _get_side_stream(device, slot=2)
    s_gbeta = _get_side_stream(device, slot=3)
    s_z = _get_side_stream(device, slot=4)

    # Q + K: l2norm + REPEAT backward writes into d_silu_conv's Q/K slices.
    _triton_l2norm_repeat_backward(
        dq,
        silu_conv,
        d_silu_conv,
        is_query=True,
        num_key_heads=num_key_heads,
        num_value_heads=num_value_heads,
        key_head_dim=key_head_dim,
        stream=s_q,
    )
    _triton_l2norm_repeat_backward(
        dk,
        silu_conv,
        d_silu_conv,
        is_query=False,
        num_key_heads=num_key_heads,
        num_value_heads=num_value_heads,
        key_head_dim=key_head_dim,
        stream=s_k,
    )

    # V: no l2norm and no REPEAT in forward, so d_silu_conv's V slice is
    # just dv re-laid-out from (b, s, num_v_heads, value_head_dim) to
    # (b, v_channels, s). A dedicated Triton kernel is ~3× faster than
    # the torch ``reshape``+``transpose``+``copy_`` it replaces.
    _triton_v_layout_to_conv(
        dv,
        d_silu_conv,
        v_channel_offset=2 * qk_channels,
        num_value_heads=num_value_heads,
        value_head_dim=value_head_dim,
        stream=s_v,
    )

    # g + beta backward fills d_qkvzba's alpha + beta slices in place via
    # atomic_add, plus per-head d_A_log / d_dt_bias. The conv slice and
    # z slice are overwritten by other kernels below, so only the alpha
    # and beta channel range needs to be zero-initialised.
    d_qkvzba = torch.empty_like(qkvzba)
    d_qkvzba[..., 2 * qk_channels + 2 * v_channels :].zero_()
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
        stream=s_gbeta,
    )

    # z slice gradient: stream dgate into d_qkvzba's z slice via a
    # dedicated Triton kernel. The previous torch ``copy_`` was ~90 μs
    # and didn't honour the stream context; the kernel is ~50 μs and
    # truly overlaps on s_z.
    _triton_z_layout_to_qkvzba(
        dgate,
        d_qkvzba,
        z_channel_offset=z_offset,
        num_value_heads=num_value_heads,
        value_head_dim=value_head_dim,
        stream=s_z,
    )

    # Join all the streams that wrote into d_silu_conv before we feed it
    # to causal_conv1d_bwd_function (which runs on the default stream).
    default_stream = torch.cuda.current_stream(device)
    for s in (s_q, s_k, s_v, s_gbeta, s_z):
        default_stream.wait_stream(s)

    # Pre-allocate d_x_conv as a strided view INTO d_qkvzba's conv slice.
    # d_qkvzba memory layout is (s, b, total_channels) contiguous, so
    # element [s, b, c] sits at offset s*b_stride + b*c_stride + c.
    # Re-interpreting that storage as (b, conv_dim, s) with strides
    # (c_stride, 1, b_stride) lets causal_conv1d_bwd_function write d_x
    # directly into the right cells — no separate layout-transform copy.
    b_stride = qkvzba.stride(0)   # = batch * total_channels
    c_stride = qkvzba.stride(1)   # = total_channels
    d_x_conv_view = d_qkvzba.as_strided(
        (batch, conv_dim, seq_len),
        (c_stride, 1, b_stride),
    )

    # Hand-tuned C++ conv backward. Internally folds the silu' factor and
    # computes both d_x and d_w in fp32; writes d_x directly into the
    # view above.
    _, d_weight_fp32, _, _ = _causal_conv1d_bwd_function(
        qkvzba_conv,
        weight_2d,
        None,              # no bias
        d_silu_conv,
        None,              # seq_idx
        None,              # initial_states
        None,              # dfinal_states
        d_x_conv_view,     # dx pre-allocated into d_qkvzba's conv slice
        False,             # return_dinitial_states
        True,              # activation (silu)
    )

    d_weight = d_weight_fp32.view(*conv1d_weight.shape).to(conv1d_weight.dtype)
    d_A_log = d_A_log_fp32.to(A_log.dtype)
    d_dt_bias = d_dt_bias_fp32.to(dt_bias.dtype)

    return d_qkvzba, d_weight, d_A_log, d_dt_bias


class _FusedPreGatedDeltaRuleFunction(torch.autograd.Function):
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
        num_key_heads,
        num_value_heads,
        key_head_dim,
        value_head_dim,
    ):
        ctx.num_key_heads = num_key_heads
        ctx.num_value_heads = num_value_heads
        ctx.key_head_dim = key_head_dim
        ctx.value_head_dim = value_head_dim
        query, key, value, gate, beta_out, g, silu_qk_save = (
            _triton_pre_gated_delta_rule_forward(
                qkvzba,
                conv1d_weight,
                A_log,
                dt_bias,
                num_key_heads=num_key_heads,
                num_value_heads=num_value_heads,
                key_head_dim=key_head_dim,
                value_head_dim=value_head_dim,
            )
        )
        ctx.save_for_backward(qkvzba, conv1d_weight, A_log, dt_bias, silu_qk_save)
        return query, key, value, gate, beta_out, g

    @staticmethod
    def backward(ctx, dq, dk, dv, dgate, dbeta, dg):
        qkvzba, conv1d_weight, A_log, dt_bias, silu_qk_save = ctx.saved_tensors
        d_qkvzba, d_weight, d_A_log, d_dt_bias = _triton_pre_gated_delta_rule_backward(
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
        )
        # Match forward inputs: (qkvzba, conv1d_weight, A_log, dt_bias,
        # num_key_heads, num_value_heads, key_head_dim, value_head_dim).
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
        )


def fused_pre_gated_delta_rule(
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
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Fused pre-gated-delta-rule entry point.

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
        cu_seqlens: Must be ``None``; packed sequence is not supported.

    Returns:
        ``(query, key, value, gate, beta, g)`` matching the unfused
        :meth:`GatedDeltaNet.pre_gated_delta_rule` API.
    """

    assert qkvzba.is_cuda, (
        "fused_pre_gated_delta_rule requires CUDA inputs; "
        f"got qkvzba.device={qkvzba.device}."
    )
    assert cu_seqlens is None, (
        "Packed sequence (cu_seqlens != None) is not supported by "
        "fused_pre_gated_delta_rule."
    )
    assert conv1d_bias is None, (
        "Conv bias is not supported by fused_pre_gated_delta_rule "
        "(production GDN config has none)."
    )
    assert use_qk_l2norm, (
        "use_qk_l2norm=False is not supported by fused_pre_gated_delta_rule "
        "(the backward closes over the l2norm path)."
    )
    assert num_value_heads % num_key_heads == 0, (
        f"{num_value_heads=} must be a multiple of {num_key_heads=}."
    )

    return _FusedPreGatedDeltaRuleFunction.apply(
        qkvzba,
        conv1d_weight,
        A_log,
        dt_bias,
        num_key_heads,
        num_value_heads,
        key_head_dim,
        value_head_dim,
    )
