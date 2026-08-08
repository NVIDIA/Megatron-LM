# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
    n_valid_tokens,
    chunk_token_base_ptr,
    chunk_valid_start_ptr,
    chunk_valid_end_ptr,
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
    RAGGED: tl.constexpr,
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
    if RAGGED:
        # General ragged: each workspace chunk carries its own token base and
        # real-token window (a chunk shared by two sequences appears once per
        # owner, each masked to its own tokens).
        token = tl.load(chunk_token_base_ptr + c) + offs
        valid_start = tl.load(chunk_valid_start_ptr + c)
        valid_end = tl.load(chunk_valid_end_ptr + c)
        valid = (token >= valid_start) & (token < valid_end)
    else:
        token = (b * C + c) * L + offs
        # A tail-ragged batch pads the LAST chunk: those lanes are not real
        # tokens. Masking the load keeps it in bounds, and forcing delta = 0
        # below removes them from the scan exactly (zero state contribution,
        # and cumsum stays flat past the last real token so the chunk's decay
        # is unchanged).
        valid = token < n_valid_tokens
    dt = tl.load(dt_ptr + token * stride_dt_token + h * stride_dt_h, mask=valid, other=0.0).to(
        tl.float32
    )
    if HAS_BIAS:
        dt = dt + tl.load(bias_ptr + h * stride_bias_h).to(tl.float32)
    if SOFTPLUS:
        # log1p is unavailable on some Triton versions; the dt <= 20 guard matches
        # the Triton reference _chunk_cumsum_fwd (identity above to avoid overflow).
        dt = tl.where(dt <= 20.0, tl.math.log(tl.math.exp(dt) + 1.0), dt)
    dt = tl.minimum(tl.maximum(dt, dt_min), dt_max)
    dt = tl.where(valid, dt, 0.0)

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
    n_valid_tokens: int | None = None,
    ragged_chunks: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
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
        n_valid_tokens: Number of REAL tokens in ``dt``. Lanes at or beyond it
            (the pad tail of a ragged final chunk) get ``delta = 0``, which
            removes them from the scan. Defaults to the full ``B * C * L``.
    """
    L = delta_out.shape[-1]
    assert delta_out.shape == cumsum_out.shape, "delta/cumsum must share layout"
    assert delta_out.stride() == cumsum_out.stride(), "delta/cumsum must share strides"
    assert delta_out.stride(-1) == 1 and cumsum_out.stride(-1) == 1, "L must be contiguous"
    dt_min, dt_max = dt_limit
    has_bias = dt_bias is not None
    if n_valid_tokens is None:
        n_valid_tokens = B * C * L
    ragged = ragged_chunks is not None
    base_p, lo_p, hi_p = ragged_chunks if ragged else (dt, dt, dt)
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
        n_valid_tokens,
        base_p,
        lo_p,
        hi_p,
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
        RAGGED=ragged,
        L=L,
    )
