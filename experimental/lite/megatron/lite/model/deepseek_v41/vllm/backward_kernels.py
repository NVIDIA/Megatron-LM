# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Batched FP32 expert weight adjoint; imported only from CUDA backward."""
import triton
import triton.language as tl


@triton.jit
def grouped_weight_vjp(
    grad,
    x,
    ends,
    result,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr = 32,
    BK: tl.constexpr = 64,
    BM: tl.constexpr = 32,
):
    expert = tl.program_id(0)
    nr = tl.program_id(1) * BN + tl.arange(0, BN)
    kr = tl.program_id(2) * BK + tl.arange(0, BK)
    start = tl.load(ends + expert - 1, mask=expert > 0, other=0)
    stop = tl.load(ends + expert)
    acc = tl.full((BN, BK), 0, tl.float32)
    for offset in range(start, stop, BM):
        rows = offset + tl.arange(0, BM)
        a = tl.load(
            grad + rows[None, :] * N + nr[:, None],
            (rows[None, :] < stop) & (nr[:, None] < N),
            other=0,
        )
        b = tl.load(
            x + rows[:, None] * K + kr[None, :], (rows[:, None] < stop) & (kr[None, :] < K), other=0
        )
        acc = tl.dot(a, b, acc)
    tl.store(
        result + (expert * N + nr[:, None]) * K + kr[None, :],
        acc,
        (nr[:, None] < N) & (kr[None, :] < K),
    )
