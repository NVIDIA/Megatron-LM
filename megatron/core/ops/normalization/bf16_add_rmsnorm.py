# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Inference residual-add/RMSNorm with BF16 rounding before normalization.

The arithmetic follows Transformer Engine 2.15's native BF16 RMSNorm kernels
for hidden sizes divisible by eight up to 8192. General and tuned dispatch
use different reductions; both are reproduced, including row-count selection.
cuDNN and FP32 residuals have different contracts. This primitive does not
inspect or change the caller's normalization backend.
"""

import math

import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]


@triton.jit
def _vector_elements(value, THREADS: tl.constexpr, ELEMENTS: tl.constexpr):
    # Unpack each thread's contiguous BF16 values without a cross-thread
    # gather. Preserve TE's sequential FMA order; a tree over these values is
    # not equivalent. The split operations only rearrange registers.
    if ELEMENTS == 4:
        even, odd = tl.split(tl.reshape(value, (THREADS, 2, 2)))
        v0, v2 = tl.split(even)
        v1, v3 = tl.split(odd)
        return v0, v1, v2, v3
    else:
        even, odd = tl.split(tl.reshape(value, (THREADS, 4, 2)))
        even_low, even_high = tl.split(tl.reshape(even, (THREADS, 2, 2)))
        odd_low, odd_high = tl.split(tl.reshape(odd, (THREADS, 2, 2)))
        v0, v4 = tl.split(even_low)
        v2, v6 = tl.split(even_high)
        v1, v5 = tl.split(odd_low)
        v3, v7 = tl.split(odd_high)
        return v0, v1, v2, v3, v4, v5, v6, v7


@triton.jit
def _warp_allreduce(value, THREADS: tl.constexpr):
    # TE's single-warp allreduce is XOR 1,2,4,8,16, unlike its multi-warp
    # reduction's shuffle-down 16,8,4,2,1. Association is observable in BF16.
    for step in tl.static_range(5):
        value += tl.gather(value, tl.arange(0, THREADS) ^ (1 << step), 0)
    return value


@triton.jit
def _warp_leaders(value, THREADS: tl.constexpr):
    values = tl.reshape(value, (THREADS // 32, 32))
    return tl.sum(tl.where(tl.arange(0, 32)[None, :] == 0, values, 0), 1)


@triton.jit
def _tuned_inverse_rms(chunks, HIDDEN: tl.constexpr, THREADS: tl.constexpr, EPS: tl.constexpr):
    partial = tl.full((THREADS,), 0, tl.float32)
    for chunk in tl.static_range(len(chunks)):
        values = _vector_elements(chunks[chunk], THREADS, 8)
        for element in tl.static_range(8):
            partial += values[element]
    mean = _warp_allreduce(partial, THREADS) * (THREADS / (32.0 * HIDDEN))
    partial = tl.full((THREADS,), 0, tl.float32)
    for chunk in tl.static_range(len(chunks)):
        values = _vector_elements(chunks[chunk], THREADS, 8)
        for element in tl.static_range(8):
            delta = values[element] - mean
            partial = tl.fma(delta, delta, partial)
    m2 = _warp_leaders(_warp_allreduce(partial, THREADS), THREADS)
    means = _warp_leaders(mean, THREADS)
    if THREADS == 128:
        # TE merges four equal-size warp populations using Chan's formulas,
        # shuffle-down 2 then 1. Preserve the CUDA expression's explicit FMAs.
        count = tl.full((4,), HIDDEN / 4.0, tl.float32)
        for step in tl.static_range(2, 0, -1):
            offset = 1 << (step - 1)
            index = tl.minimum(tl.arange(0, 4) + offset, 3)
            other_mean = tl.gather(means, index, 0)
            other_m2 = tl.gather(m2, index, 0)
            other_count = tl.gather(count, index, 0)
            combined_count = count + other_count
            reciprocal = 1.0 / combined_count
            delta = means - other_mean
            correction = ((delta * delta) * count) * other_count
            m2 = tl.fma(correction, reciprocal, m2 + other_m2)
            means = tl.fma(count, means, other_count * other_mean) * reciprocal
            count = combined_count
        mean_scalar = tl.sum(tl.where(tl.arange(0, 4) == 0, means, 0), 0)
        m2_scalar = tl.sum(tl.where(tl.arange(0, 4) == 0, m2, 0), 0)
    else:
        mean_scalar = tl.sum(means, 0)
        m2_scalar = tl.sum(m2, 0)
    return tl.rsqrt(tl.fma(mean_scalar, mean_scalar, (1.0 / HIDDEN) * m2_scalar) + EPS)


@triton.jit
def _bf16_add_rmsnorm_kernel(
    X,
    RESIDUAL,
    WEIGHT,
    OUTPUT,
    NEXT_RESIDUAL,
    EPS: tl.constexpr,
    ZERO_CENTERED_GAMMA: tl.constexpr,
    HIDDEN: tl.constexpr,
    THREADS: tl.constexpr,
    ELEMENTS: tl.constexpr,
    TUNED: tl.constexpr,
):
    row = tl.program_id(0)
    CHUNKS: tl.constexpr = (HIDDEN + THREADS * ELEMENTS - 1) // (THREADS * ELEMENTS)
    columns = tl.arange(0, THREADS)[:, None] * ELEMENTS + tl.arange(0, ELEMENTS)[None, :]
    partial = tl.full((THREADS,), 0, tl.float32)
    rounded_chunks = ()
    for chunk in tl.static_range(CHUNKS):
        col = columns + chunk * THREADS * ELEMENTS
        offset = row * HIDDEN + col
        x = tl.load(X + offset, col < HIDDEN, 0).to(tl.float32)
        residual = tl.load(RESIDUAL + offset, col < HIDDEN, 0).to(tl.float32)
        # Both numerator and variance consume the BF16-rounded sum. Retain
        # these values in registers until normalization and residual storage.
        rounded = (x + residual).to(tl.bfloat16).to(tl.float32)
        rounded_chunks += (rounded,)  # type: ignore[assignment]  # Triton unrolls this tuple.
        if not TUNED:
            values = _vector_elements(rounded, THREADS, ELEMENTS)
            for element in tl.static_range(ELEMENTS):
                partial = tl.fma(values[element], values[element], partial)

    # TE reduces each warp by shuffle-down offsets 16, 8, 4, 2, 1, then adds
    # the four warp totals sequentially. Do not replace this by a 128-way sum.
    if TUNED:
        inverse_rms = _tuned_inverse_rms(rounded_chunks, HIDDEN, THREADS, EPS)
    else:
        if THREADS == 128:
            warp_totals = tl.sum(tl.reshape(partial, (4, 32)), 1)
            total = tl.full((), 0, tl.float32)
            for warp in tl.static_range(4):
                total += tl.sum(tl.where(tl.arange(0, 4) == warp, warp_totals, 0), 0)
        else:
            total = tl.sum(_warp_leaders(_warp_allreduce(partial, THREADS), THREADS), 0)
        inverse_rms = tl.rsqrt(tl.fma(total, 1.0 / HIDDEN, EPS))
    for chunk in tl.static_range(CHUNKS):
        col = columns + chunk * THREADS * ELEMENTS
        offset = row * HIDDEN + col
        weight = tl.load(WEIGHT + col, col < HIDDEN, 0).to(tl.float32)
        if ZERO_CENTERED_GAMMA:
            weight += 1.0
        output = (rounded_chunks[chunk] * inverse_rms) * weight
        tl.store(OUTPUT + offset, output, col < HIDDEN)
        tl.store(NEXT_RESIDUAL + offset, rounded_chunks[chunk], col < HIDDEN)


def fused_bf16_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
    zero_centered_gamma: bool = False,
    *,
    inplace: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(normalized_output, BF16(x + residual))`` in one CUDA kernel.

    This inference-only primitive preserves the pre-normalization BF16
    rounding used by BF16 training residual addition. Its reduction
    order targets the TE native RMSNorm kernels, not cuDNN or an
    arbitrary implementation of the same formula. Callers must select a
    compatible backend; this function does not inspect or change TE state.

    Args:
        x: Contiguous CUDA BF16 branch output, last dimension divisible by 8
            in the range [8, 8192].
        residual: Contiguous CUDA BF16 skip input, same shape/device as x.
        weight: Contiguous CUDA BF16 RMSNorm weight of shape (x.shape[-1],).
        eps: Positive finite RMSNorm epsilon, representable in FP32.
        zero_centered_gamma: Whether to use ``1 + weight`` in FP32.
        inplace: Reuse x as the returned residual, like native inference BDA.
            The caller must own x and have no remaining readers of its old
            contents. x must not share storage with residual or weight.

    Returns:
        Two BF16 tensors with the input shape. With inplace=False both are
        newly allocated and inputs are unchanged; otherwise the second output
        aliases and overwrites x. No FP32 residual buffer is allocated. Warm
        up before CUDA graph capture, as for other Triton ops.

    Raises:
        ValueError: If inputs are outside the supported numerical contract.
        RuntimeError: If called with an input requiring autograd while grad
            mode is enabled. No backward is implemented; training is unchanged.
    """
    if x.ndim < 1 or not 8 <= x.shape[-1] <= 8192 or x.shape[-1] % 8:
        raise ValueError(
            "fused_bf16_add_rmsnorm requires a hidden size divisible by 8 in [8, 8192]"
        )
    hidden = x.shape[-1]
    if residual.shape != x.shape or weight.shape != (hidden,):
        raise ValueError("Residual and weight shapes must match the input")
    for tensor in (x, residual, weight):
        if not tensor.is_cuda or tensor.device != x.device:
            raise ValueError("All inputs must be on the same CUDA device")
        if tensor.dtype != torch.bfloat16 or not tensor.is_contiguous():
            raise ValueError("All inputs must be contiguous BF16 tensors")
    if not math.isfinite(eps) or not 1.1754943508222875e-38 <= eps <= 3.4028234663852886e38:
        raise ValueError("eps must be a finite positive normal FP32 value")
    if torch.is_grad_enabled() and any(t.requires_grad for t in (x, residual, weight)):
        raise RuntimeError("fused_bf16_add_rmsnorm is inference-only; use torch.no_grad()")
    if inplace and x.numel():
        x_storage = x.untyped_storage().data_ptr()
        if any(t.untyped_storage().data_ptr() == x_storage for t in (residual, weight)):
            raise ValueError("In-place output must not share storage with residual or weight")

    output = torch.empty_like(x)
    next_residual = x if inplace else torch.empty_like(x)
    rows = x.numel() // hidden
    if rows:
        with torch.cuda.device(x.device):
            _bf16_add_rmsnorm_kernel[(rows,)](
                x,
                residual,
                weight,
                output,
                next_residual,
                eps,
                zero_centered_gamma,
                HIDDEN=hidden,
                THREADS=32 if hidden <= 2048 else 128,
                ELEMENTS=4 if hidden <= 128 else 8,
                TUNED=(
                    hidden in (512, 768, 1024, 2048, 4096, 8192)
                    and rows % 4 == 0
                    and weight.data_ptr() % 16 == 0
                ),
                num_warps=4,
                # FMA is explicit only where TE uses it. Preserve association
                # of the norm/weight multiplications and reduction additions.
                enable_fp_fusion=False,
            )
    return output, next_residual
