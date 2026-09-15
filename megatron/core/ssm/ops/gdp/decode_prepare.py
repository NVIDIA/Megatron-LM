# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Fused decode-step preparation for Gated Delta Product.

Between the short-conv update and the recurrent kernel, GDP decode does nothing
but reshape and gate: it splits the post-conv activations into `value`, `key`
and `query`, GQA-expands `query`/`key`, interleaves the Householder copies
(`query` on the last copy, the decay on the first), and turns the `ba` slice
into `beta = sigmoid(b)` and `g = -exp(A_log) * softplus(a + dt_bias)`. Each of
those steps is a separate elementwise or copy kernel in PyTorch, and a decode
step carries one token per request, so their launch overhead outweighs the work
they do.

`gdp_decode_prepare` does all of it in one Triton kernel, writing the five
tensors the recurrent kernel consumes in exactly the layouts it wants. It is
CUDA-graph safe for the same reasons as the rest of this package: static
shapes, no host synchronization, and padding handled by the conv kernel
upstream (a `-1` slot zeroes its conv output, which propagates here).

It is deterministic: a pure elementwise map where each program owns a disjoint
slice of every output, with no atomics, no reductions and no autotuning, so
nothing depends on scheduling order.

It also aims to be bitwise identical to the eager path, which the inference
functional tests rely on. `query`/`key`/`value` are plain copies and match for
free; `beta` and `g` match only because the transcendentals and the division go
through libdevice rather than Triton's fp32 defaults. `test_decode_prepare.py`
asserts that with `torch.equal`, so a Triton or torch upgrade that breaks the
agreement fails there rather than in a functional test.
"""

import torch

from .common import HAVE_TRITON, tl, triton

if HAVE_TRITON:
    # Everything numeric here goes through libdevice rather than `tl`: on fp32
    # `tl.exp` lowers to `ex2.approx.f32` and `/` to `div.full.f32`, both of which
    # cost bitwise agreement with the eager path. `log1p` is libdevice-only
    # regardless; `triton.language.math` does not carry it.
    # The `triton.language.extra.libdevice` path dates to Triton 3.0 -- 2.x
    # exposed it as `extra.cuda.libdevice`, which this package does not support.
    try:
        from triton.language.extra import libdevice

        HAVE_TRITON_LIBDEVICE = True
    except ImportError:
        libdevice = None
        HAVE_TRITON_LIBDEVICE = False
else:
    libdevice = None
    HAVE_TRITON_LIBDEVICE = False


@triton.jit
def softplus(x):
    """`log1p(exp(x))` in fp32, saturating to the identity above 20 like torch's.

    `log1p` rather than `log(1 + ...)` so the far-negative tail, where `exp(x)`
    falls below fp32's epsilon and `1 + exp(x)` rounds to exactly 1, keeps its
    significant digits instead of flushing to zero.

    Mirrors `F.softplus`'s `(x * beta) > threshold ? x : log1p(exp(x * beta)) / beta`
    at the default `beta == 1.0`, where the multiply and divide are exact.
    """
    return tl.where(x > 20.0, x, libdevice.log1p(libdevice.exp(x)))


@triton.jit
def gdp_decode_prepare_kernel(
    x,
    x_n_stride,
    ba,
    ba_n_stride,
    A_log,
    dt_bias,
    q,
    k,
    v,
    beta,
    g,
    H: tl.constexpr,
    G: tl.constexpr,
    P: tl.constexpr,
    N: tl.constexpr,
    M: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    K_BASE: tl.constexpr,
    Q_BASE: tl.constexpr,
    BP: tl.constexpr,
    BN: tl.constexpr,
):
    """One program per (request, Householder copy, head).

    `x` is the post-conv `[n, M*H*P + M*G*N + G*N]` row -- value, key, query
    concatenated -- and `ba` is the `[n, M*H + H]` gating row. Each program
    emits one head-slice of every output.
    """
    i_n, i_m, i_h = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_g = i_h // HEADS_PER_GROUP

    o_p = tl.arange(0, BP)
    o_n = tl.arange(0, BN)
    mask_p = o_p < P
    mask_n = o_n < N

    p_x = x + i_n * x_n_stride
    # Output row for this (request, copy, head), shared by every output below
    # because they are all laid out as [n, M, H, ...].
    i_o = (i_n * M + i_m) * H + i_h

    # Value: the only tensor that is a straight copy, no GQA expansion.
    b_v = tl.load(p_x + (i_m * H + i_h) * P + o_p, mask=mask_p, other=0.0)
    tl.store(v + i_o * P + o_p, b_v, mask=mask_p)

    # Key: one per Householder copy, GQA-expanded from group `i_g`.
    b_k = tl.load(p_x + K_BASE + (i_m * G + i_g) * N + o_n, mask=mask_n, other=0.0)
    tl.store(k + i_o * N + o_n, b_k, mask=mask_n)

    # Query: a single vector per token, placed on the *last* Householder copy so
    # that it reads the state after all M updates have been applied. The other
    # copies get zeros, which the recurrent kernel's in-kernel L2 norm leaves at
    # zero (0 / sqrt(eps)).
    b_q = tl.load(p_x + Q_BASE + i_g * N + o_n, mask=mask_n, other=0.0).to(tl.float32)
    b_q = tl.where(i_m == M - 1, b_q, 0.0)
    tl.store(q + i_o * N + o_n, b_q.to(q.dtype.element_ty), mask=mask_n)

    p_ba = ba + i_n * ba_n_stride
    # beta is per (copy, head); `b` is laid out as (M, H) inside `ba`.
    b_b = tl.load(p_ba + i_m * H + i_h).to(tl.float32)
    # `1 / (1 + exp(-x))` in fp32, as torch's sigmoid computes it; `div_rn` because
    # Triton's fp32 `/` is the approximate `div.full.f32`.
    b_beta = libdevice.div_rn(1.0, 1.0 + libdevice.exp(-b_b))
    tl.store(beta + i_o, b_beta.to(beta.dtype.element_ty))

    # The decay is per token, not per copy, and belongs on the *first* copy: the
    # state decays once per step, before the M Householder updates.
    b_a = tl.load(p_ba + M * H + i_h).to(tl.float32)
    b_dt = tl.load(dt_bias + i_h).to(tl.float32)
    b_A = tl.load(A_log + i_h).to(tl.float32)
    b_g = tl.where(i_m == 0, -libdevice.exp(b_A) * softplus(b_a + b_dt), 0.0)
    tl.store(g + i_o, b_g)


def gdp_decode_prepare(
    x: torch.Tensor,
    ba: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_householder: int,
    num_heads: int,
    num_groups: int,
    head_dim: int,
    state_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split, reshape and gate one decode step's post-conv activations.

    Args:
        x: Post-conv activations `[n, 1, M*H*P + M*G*N + G*N]`, value/key/query
            concatenated along the last dimension.
        ba: The `ba` slice of the input projection, `[n, 1, M*H + H]`. May be a
            non-contiguous view; only the last dimension must be contiguous.
        A_log: Log decay rates `[H]`.
        dt_bias: Softplus bias `[H]`.
        num_householder: Number of Householder copies `M`.
        num_heads: TP/CP-local head count `H`.
        num_groups: TP/CP-local group count `G`; `H` must be a multiple of it.
        head_dim: Value head dimension `P`.
        state_dim: Key/query head dimension `N`.

    Returns `(query, key, value, beta, g)` shaped `[n, M, H, N]`, `[n, M, H, N]`,
    `[n, M, H, P]`, `[n, M, H]` and `[n, M, H]` (fp32) -- the exact layouts
    `fused_recurrent_gated_delta_rule_update` expects for an `M`-length
    sequence per request.
    """
    assert HAVE_TRITON, "gdp_decode_prepare requires Triton"
    assert HAVE_TRITON_LIBDEVICE, (
        "gdp_decode_prepare needs `triton.language.extra.libdevice` for its math, which "
        f"requires Triton >= 3.0; found {getattr(triton, '__version__', 'unknown')}"
    )
    assert x.shape[1] == 1 and ba.shape[1] == 1, "decode runs one token per request"
    assert num_heads % num_groups == 0, "num_heads must be a multiple of num_groups"

    M, H, G, P, N = num_householder, num_heads, num_groups, head_dim, state_dim
    n = x.shape[0]
    assert x.shape[2] == M * H * P + (M + 1) * G * N, "unexpected post-conv width"
    assert ba.shape[2] == (M + 1) * H, "unexpected gating width"
    # The kernel indexes rows with a single stride and reads each row densely.
    assert x.stride(2) == 1 and ba.stride(2) == 1, "the feature dimension must be contiguous"

    query = x.new_empty(n, M, H, N)
    key = x.new_empty(n, M, H, N)
    value = x.new_empty(n, M, H, P)
    beta = ba.new_empty(n, M, H)
    g = torch.empty(n, M, H, device=x.device, dtype=torch.float32)

    gdp_decode_prepare_kernel[(n, M, H)](
        x=x,
        x_n_stride=x.stride(0),
        ba=ba,
        ba_n_stride=ba.stride(0),
        A_log=A_log,
        dt_bias=dt_bias,
        q=query,
        k=key,
        v=value,
        beta=beta,
        g=g,
        H=H,
        G=G,
        P=P,
        N=N,
        M=M,
        HEADS_PER_GROUP=H // G,
        K_BASE=M * H * P,
        Q_BASE=M * H * P + M * G * N,
        BP=triton.next_power_of_2(P),
        BN=triton.next_power_of_2(N),
        num_warps=2,
    )
    return query, key, value, beta, g
