# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Standalone Triton kernel that fuses the dt preprocessing for the CuteDSL SSD
front-end.

For the two hot (chunk-contiguous) paths — ``aligned`` and ``divisible`` — the
per-chunk ``delta`` and ``dA_cumsum`` inputs to the CuteDSL kernel are otherwise
built with a chain of elementwise torch ops (``dt.float()`` + bias add +
``softplus`` + clamp + a strided cast-copy + a multiply + ``torch.cumsum``).
Because the whole SSD path is CPU-dispatch bound (~tens of tiny launches), that
chain of ~6 launches is pure overhead. This kernel collapses it into a single
launch that reads the raw (token-packed) ``dt`` and writes ``delta`` and
``dA_cumsum`` (both fp32, matching Triton's fp32 dt) directly into the cached
workspace buffers, in the ``(B, H, C, L)`` chunk-major layout the CuteDSL
kernel expects.

The softplus/clamp/cumsum arithmetic mirrors the Triton reference
``_chunk_cumsum_fwd`` (see :mod:`megatron.core.ssm.ops.ssd_chunk_state`) so the
result is numerically equivalent.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _softplus_cumsum_kernel(
    dt_ptr,
    bias_ptr,
    a_ptr,
    delta_ptr,
    cumsum_ptr,
    H,
    C,
    stride_dt_token,
    stride_dt_h,
    stride_bias_h,
    stride_a_h,
    stride_ob,
    stride_oh,
    stride_oc,
    dt_min,
    dt_max,
    HAS_BIAS: tl.constexpr,
    SOFTPLUS: tl.constexpr,
    L: tl.constexpr,
):
    pid = tl.program_id(0)
    c = pid % C
    tmp = pid // C
    h = tmp % H
    b = tmp // H

    offs = tl.arange(0, L)
    # Chunk-contiguous token packing: aligned uses seqlen0 == C * L, divisible
    # uses B == 1 with C == total_chunks.
    token = (b * C + c) * L + offs
    dt = tl.load(dt_ptr + token * stride_dt_token + h * stride_dt_h).to(tl.float32)
    if HAS_BIAS:
        dt = dt + tl.load(bias_ptr + h * stride_bias_h).to(tl.float32)
    if SOFTPLUS:
        # log1p is unavailable on some Triton versions; the dt <= 20 guard matches
        # the Triton reference _chunk_cumsum_fwd (identity above to avoid overflow).
        dt = tl.where(dt <= 20.0, tl.math.log(tl.math.exp(dt) + 1.0), dt)
    dt = tl.minimum(tl.maximum(dt, dt_min), dt_max)

    a = tl.load(a_ptr + h * stride_a_h).to(tl.float32)
    dA_cumsum = tl.cumsum(dt * a, axis=0)

    out = b * stride_ob + h * stride_oh + c * stride_oc + offs
    tl.store(delta_ptr + out, dt.to(delta_ptr.dtype.element_ty))
    tl.store(cumsum_ptr + out, dA_cumsum)


def fused_softplus_cumsum(
    dt: torch.Tensor,
    A: torch.Tensor,
    dt_bias: torch.Tensor | None,
    dt_softplus: bool,
    dt_limit: tuple[float, float],
    delta_out: torch.Tensor,
    cumsum_out: torch.Tensor,
    B: int,
    H: int,
    C: int,
) -> None:
    """Fill ``delta_out`` and ``cumsum_out`` from raw ``dt`` in a single launch.

    Args:
        dt: Raw dt, shape ``(T, H)``, any float dtype.
        A: State-decay ``A``, shape ``(H,)``.
        dt_bias: Optional per-head bias, shape ``(H,)`` or ``None``.
        dt_softplus: Whether to apply ``softplus`` (matches the Triton reference).
        dt_limit: ``(dt_min, dt_max)`` clamp applied after softplus.
        delta_out: Output ``delta`` buffer, shape ``(B, H, C, L)``, fp32.
        cumsum_out: Output ``dA_cumsum`` buffer, shape ``(B, H, C, L)``, fp32.
        B: Batch dim of the output layout (num sequences for the aligned path, 1
            for the divisible path).
        H: Number of heads.
        C: Chunk dim of the output layout (Cmax for aligned, total_chunks for
            divisible).
    """
    L = delta_out.shape[-1]
    assert delta_out.shape == cumsum_out.shape, "delta/cumsum must share layout"
    assert delta_out.stride() == cumsum_out.stride(), "delta/cumsum must share strides"
    assert delta_out.stride(-1) == 1 and cumsum_out.stride(-1) == 1, "L must be contiguous"
    dt_min, dt_max = dt_limit
    has_bias = dt_bias is not None
    grid = (B * H * C,)
    # TODO(perf): write a CuTe DSL fused kernel
    _softplus_cumsum_kernel[grid](
        dt,
        dt_bias if has_bias else dt,
        A,
        delta_out,
        cumsum_out,
        H,
        C,
        dt.stride(0),
        dt.stride(1),
        dt_bias.stride(0) if has_bias else 0,
        A.stride(0),
        delta_out.stride(0),
        delta_out.stride(1),
        delta_out.stride(2),
        float(dt_min),
        float(dt_max),
        HAS_BIAS=has_bias,
        SOFTPLUS=dt_softplus,
        L=L,
    )
