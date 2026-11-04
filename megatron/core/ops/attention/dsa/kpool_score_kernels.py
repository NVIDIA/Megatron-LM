# Copyright (c) ModelScope Contributors. All rights reserved.
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detached FP32 KPool scoring postprocessing; the GEMM remains unchanged.

Adapted from FunJim/mcore-bridge src/mcore_bridge/kpool_relu_mul.py at revision
8e63e6452b54631cf1edfad434dcbd6325606ba4 (Apache-2.0).
https://github.com/FunJim/mcore-bridge/blob/8e63e6452b54631cf1edfad434dcbd6325606ba4/src/mcore_bridge/kpool_relu_mul.py

Strided heads use four accumulators; a single-key pool uses the contiguous-head tree.
Parity with the torch reference is tested separately from bit-exact replay;
reproducibility alone does not establish identical discrete top-k selections.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _kpool_score_kernel(
    scores,
    weights,
    output,
    elements,
    batch,
    keys,
    SQ: tl.constexpr,
    SB: tl.constexpr,
    SH: tl.constexpr,
    WQ: tl.constexpr,
    WB: tl.constexpr,
    WH: tl.constexpr,
    USE_RELU: tl.constexpr,
    SINGLE_KEY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    key = offsets % keys
    b = (offsets // keys) % batch
    q = offsets // (keys * batch)
    valid = offsets < elements
    if SINGLE_KEY:
        lanes = tl.arange(0, 32)
        heads = (
            ((lanes & 1) << 4)
            | ((lanes & 2) << 2)
            | (lanes & 4)
            | ((lanes & 8) >> 2)
            | ((lanes & 16) >> 4)
        )
        value = tl.load(
            scores + q[:, None] * SQ + b[:, None] * SB + heads[None, :] * SH,
            valid[:, None],
            other=0.0,
        )
        weight = tl.load(
            weights + q[:, None] * WQ + b[:, None] * WB + heads[None, :] * WH,
            valid[:, None],
            other=0.0,
        )
        if USE_RELU:
            value = tl.where(value < 0.0, 0.0, value)
        product = tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [value, weight],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        partial = (tl.full((BLOCK, 32), 0.0, tl.float32) + product).reshape((BLOCK, 2, 2, 2, 2, 2))
        for stage in tl.static_range(5):
            left, right = tl.split(partial)
            partial = left + right
        result = partial
    else:
        a0 = tl.full((BLOCK,), 0.0, tl.float32)
        a1 = tl.full((BLOCK,), 0.0, tl.float32)
        a2 = tl.full((BLOCK,), 0.0, tl.float32)
        a3 = tl.full((BLOCK,), 0.0, tl.float32)
        for h in tl.static_range(32):
            value = tl.load(scores + q * SQ + b * SB + h * SH + key, valid, other=0.0)
            weight = tl.load(weights + q * WQ + b * WB + h * WH, valid, other=0.0)
            if USE_RELU:
                value = tl.where(value < 0.0, 0.0, value)
            product = tl.inline_asm_elementwise(
                "mul.rn.f32 $0, $1, $2;",
                "=f,f,f",
                [value, weight],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
            if h % 4 == 0:
                a0 = a0 + product
            elif h % 4 == 1:
                a1 = a1 + product
            elif h % 4 == 2:
                a2 = a2 + product
            else:
                a3 = a3 + product
        result = ((a0 + a1) + a2) + a3
    tl.store(output + offsets, result, valid)


def triton_kpool_score_postprocess(
    scores: torch.Tensor, weights: torch.Tensor, use_relu: bool
) -> torch.Tensor:
    """Fuse optional ReLU, weighting and 32-head sum; reject unsupported inputs."""
    if scores.dtype != torch.float32 or weights.dtype != torch.float32:
        raise ValueError("Triton KPool score postprocessing requires FP32 scores and weights")
    if scores.ndim != 4 or tuple(weights.shape) != tuple(scores.shape[:3]):
        raise ValueError("Expected scores[query,batch,heads,keys] and matching weights")
    if scores.device != weights.device or not scores.is_cuda:
        raise ValueError(
            "Triton KPool score postprocessing requires inputs on the same CUDA device"
        )
    if torch.is_grad_enabled() and (scores.requires_grad or weights.requires_grad):
        raise ValueError("Triton KPool score postprocessing requires detached selection inputs")
    if scores.shape[2] != 32:
        raise ValueError("Triton KPool score postprocessing requires 32 heads")
    if scores.numel():
        if scores.stride(-1) != 1 or scores.stride(2) != scores.shape[-1]:
            raise ValueError("Triton KPool scores require contiguous head/key axes")
        expected = 1
        for stride, size in sorted(zip(scores.stride(), scores.shape)):
            if size > 1:
                if stride != expected:
                    raise ValueError("Triton KPool scores must be dense and non-overlapping")
                expected *= size
    if any(stride < 0 for stride in weights.stride()):
        raise ValueError("Triton KPool weights cannot have negative strides")
    scores, weights = scores.detach(), weights.detach()
    queries, batch, _, keys = scores.shape
    output = torch.empty((queries, batch, keys), device=scores.device, dtype=torch.float32)
    if output.numel():
        with torch.cuda.device(scores.device):
            _kpool_score_kernel[(triton.cdiv(output.numel(), 256),)](
                scores,
                weights,
                output,
                output.numel(),
                batch,
                keys,
                *scores.stride()[:3],
                *weights.stride(),
                USE_RELU=use_relu,
                SINGLE_KEY=keys == 1,
                BLOCK=256,
                enable_fp_fusion=False,
                num_warps=4,
            )
    return output
