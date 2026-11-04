# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Fused decode- and prefill-step preparation for Gated Delta Product.

Between the short conv and the recurrence, GDP dynamic inference does nothing
but reshape and gate: it splits the post-conv activations into `value`, `key`
and `query`, GQA-expands `query`/`key`, lays `value`/`key` out with the
Householder copies folded into the sequence dimension, and turns the `ba` slice
into `beta = sigmoid(b)` and `g = -exp(A_log) * softplus(a + dt_bias)`. Each of
those steps is a separate elementwise or copy kernel in PyTorch -- ~14 per layer
in prefill, three of them strided copies of most of the post-conv tensor -- and a
decode step carries one token per request, so their launch overhead outweighs
the work they do.

`gdp_decode_prepare` and `gdp_prefill_prepare` each do all of it in one launch
of the same Triton kernel. They differ only in what the downstream kernel wants:

* Decode feeds `fused_recurrent_gated_delta_rule_update`, which walks the
  Householder copies as a sequence, so `query` and `g` are interleaved onto the
  copies (`query` on the last, the decay on the first, zeros elsewhere). One
  program covers one head, which keeps a small decode batch spread over the GPU.
* Prefill feeds `chunk_gated_delta_product_varlen`, which places `query` and `g`
  on the copies itself, so they stay one per token. One program covers every
  head of a (token, copy), because a prefill step carries thousands of tokens.

Both are CUDA-graph safe for the same reasons as the rest of this package:
static shapes, no host synchronization, and padding handled by the conv kernel
upstream (a `-1` slot zeroes its conv output, which propagates here). The
prefill token count is a runtime argument, so prefill lengths do not each
compile a kernel; decode's (1 plus the draft length) is a compile-time constant.

Both are deterministic: a pure elementwise map where each program owns a
disjoint slice of every output, with no atomics, no reductions and no
autotuning, so nothing depends on scheduling order.

Both also aim to be bitwise identical to the eager path, which the inference
functional tests rely on. `query`/`key`/`value` are plain copies and match for
free; `beta` and `g` match only because the transcendentals and the division go
through libdevice rather than Triton's fp32 defaults. `test_decode_prepare.py`
and `test_prefill_prepare.py` assert that with `torch.equal`, so a Triton or
torch upgrade that breaks the agreement fails there rather than in a functional
test.
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
def gdp_prepare_kernel(
    x,
    x_n_stride,
    x_t_stride,
    ba,
    ba_n_stride,
    ba_t_stride,
    A_log,
    dt_bias,
    q,
    k,
    v,
    beta,
    g,
    T,
    T_CONST: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    P: tl.constexpr,
    N: tl.constexpr,
    M: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    K_BASE: tl.constexpr,
    Q_BASE: tl.constexpr,
    BH: tl.constexpr,
    BP: tl.constexpr,
    BN: tl.constexpr,
    INTERLEAVE_QG: tl.constexpr,
):
    """One program per (request-token, Householder copy, block of `BH` heads).

    `x` is the post-conv `[n, T, M*H*P + M*G*N + G*N]` tensor -- value, key,
    query concatenated -- and `ba` is the `[n, T, M*H + H]` gating tensor. For
    decode `T` is 1 plus the speculative draft length, passed as `T_CONST` so
    the per-program split into (request, token) is a constant division; prefill
    passes `T_CONST == 0` and its packed token count at runtime, so step sizes
    do not each compile a kernel.
    """
    i_nt = tl.program_id(0).to(tl.int64)
    i_m = tl.program_id(1)
    if T_CONST > 0:
        i_n, i_t = i_nt // T_CONST, i_nt % T_CONST
    else:
        i_n, i_t = i_nt // T, i_nt % T

    o_h = tl.program_id(2) * BH + tl.arange(0, BH)
    o_p = tl.arange(0, BP)
    o_n = tl.arange(0, BN)
    mask_h = o_h < H
    mask_hp = mask_h[:, None] & (o_p < P)[None, :]
    mask_hn = mask_h[:, None] & (o_n < N)[None, :]
    # GQA expansion: head `h` reads group `h // (H // G)`, as `repeat_interleave`
    # on the group axis does.
    o_g = o_h // HEADS_PER_GROUP

    p_x = x + i_n * x_n_stride + i_t * x_t_stride
    p_ba = ba + i_n * ba_n_stride + i_t * ba_t_stride
    # Row of this (request, token, copy) in the `[n, T*M, H, ...]` outputs: the
    # Householder copies are folded into the sequence dimension behind each
    # token, which is the order the recurrent kernel walks them in.
    i_o = i_nt * M + i_m

    # Value: the only tensor that is a straight copy, no GQA expansion.
    b_v = tl.load(p_x + (i_m * H + o_h[:, None]) * P + o_p[None, :], mask=mask_hp, other=0.0)
    tl.store(v + (i_o * H + o_h[:, None]) * P + o_p[None, :], b_v, mask=mask_hp)

    # Key: one per Householder copy, GQA-expanded.
    b_k = tl.load(
        p_x + K_BASE + (i_m * G + o_g[:, None]) * N + o_n[None, :], mask=mask_hn, other=0.0
    )
    tl.store(k + (i_o * H + o_h[:, None]) * N + o_n[None, :], b_k, mask=mask_hn)

    # beta is per (copy, head); `b` is laid out as (M, H) inside `ba`.
    # `1 / (1 + exp(-x))` in fp32, as torch's sigmoid computes it; `div_rn` because
    # Triton's fp32 `/` is the approximate `div.full.f32`.
    b_b = tl.load(p_ba + i_m * H + o_h, mask=mask_h, other=0.0).to(tl.float32)
    b_beta = libdevice.div_rn(1.0, 1.0 + libdevice.exp(-b_b))
    tl.store(beta + i_o * H + o_h, b_beta.to(beta.dtype.element_ty), mask=mask_h)

    # Query and the decay are per token, not per copy.
    p_q = p_x + Q_BASE + o_g[:, None] * N + o_n[None, :]
    if INTERLEAVE_QG:
        # Decode: interleaved onto the copies in the `[n, T*M, H, ...]` layout.
        # The query goes on the *last* copy so that it reads the state after all
        # M updates have been applied; the other copies get zeros, which the
        # recurrent kernel's in-kernel L2 norm leaves at zero (0 / sqrt(eps)).
        if i_m == M - 1:
            b_q = tl.load(p_q, mask=mask_hn, other=0.0).to(tl.float32)
        else:
            b_q = tl.full((BH, BN), 0.0, tl.float32)
        tl.store(
            q + (i_o * H + o_h[:, None]) * N + o_n[None, :],
            b_q.to(q.dtype.element_ty),
            mask=mask_hn,
        )
        # The decay belongs on the *first* copy: the state decays once per step,
        # before the M Householder updates.
        if i_m == 0:
            b_a = tl.load(p_ba + M * H + o_h, mask=mask_h, other=0.0).to(tl.float32)
            b_dt = tl.load(dt_bias + o_h, mask=mask_h, other=0.0).to(tl.float32)
            b_A = tl.load(A_log + o_h, mask=mask_h, other=0.0).to(tl.float32)
            b_g = -libdevice.exp(b_A) * softplus(b_a + b_dt)
        else:
            b_g = tl.full((BH,), 0.0, tl.float32)
        tl.store(g + i_o * H + o_h, b_g, mask=mask_h)
    else:
        # Prefill: one per token in the compact `[n, T, H, ...]` layout, written
        # by the first copy's program; the chunk kernel places them itself.
        if i_m == 0:
            b_q = tl.load(p_q, mask=mask_hn, other=0.0)
            tl.store(q + (i_nt * H + o_h[:, None]) * N + o_n[None, :], b_q, mask=mask_hn)

            b_a = tl.load(p_ba + M * H + o_h, mask=mask_h, other=0.0).to(tl.float32)
            b_dt = tl.load(dt_bias + o_h, mask=mask_h, other=0.0).to(tl.float32)
            b_A = tl.load(A_log + o_h, mask=mask_h, other=0.0).to(tl.float32)
            b_g = -libdevice.exp(b_A) * softplus(b_a + b_dt)
            tl.store(g + i_nt * H + o_h, b_g, mask=mask_h)


def _check_prepare_args(name, x, ba, M, H, G, P, N):
    assert HAVE_TRITON, f"{name} requires Triton"
    assert HAVE_TRITON_LIBDEVICE, (
        f"{name} needs `triton.language.extra.libdevice` for its math, which "
        f"requires Triton >= 3.0; found {getattr(triton, '__version__', 'unknown')}"
    )
    assert H % G == 0, "num_heads must be a multiple of num_groups"
    assert ba.shape[:2] == x.shape[:2], f"x and ba disagree on [n, T]: {x.shape} vs {ba.shape}"
    assert x.shape[2] == M * H * P + (M + 1) * G * N, "unexpected post-conv width"
    assert ba.shape[2] == (M + 1) * H, "unexpected gating width"
    # The kernel indexes rows with strides and reads each row densely.
    assert x.stride(2) == 1 and ba.stride(2) == 1, "the feature dimension must be contiguous"


def _launch_prepare(x, ba, A_log, dt_bias, outputs, M, H, G, P, N, interleave_qg):
    query, key, value, beta, g = outputs
    n, T = x.shape[0], x.shape[1]
    if n * T == 0:
        return
    # Decode: one head per program. Prefill: every head of a (token, copy).
    BH = 1 if interleave_qg else triton.next_power_of_2(H)
    gdp_prepare_kernel[(n * T, M, triton.cdiv(H, BH))](
        x=x,
        x_n_stride=x.stride(0),
        x_t_stride=x.stride(1),
        ba=ba,
        ba_n_stride=ba.stride(0),
        ba_t_stride=ba.stride(1),
        A_log=A_log,
        dt_bias=dt_bias,
        q=query,
        k=key,
        v=value,
        beta=beta,
        g=g,
        T=T,
        T_CONST=T if interleave_qg else 0,
        H=H,
        G=G,
        P=P,
        N=N,
        M=M,
        HEADS_PER_GROUP=H // G,
        K_BASE=M * H * P,
        Q_BASE=M * H * P + M * G * N,
        BH=BH,
        BP=triton.next_power_of_2(P),
        BN=triton.next_power_of_2(N),
        INTERLEAVE_QG=interleave_qg,
        num_warps=2 if interleave_qg else 4,
    )


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
        x: Post-conv activations `[n, S, M*H*P + M*G*N + G*N]`, value/key/query
            concatenated along the last dimension. `S` is 1 plus the number of
            speculative draft tokens.
        ba: The `ba` slice of the input projection, `[n, S, M*H + H]`. May be a
            non-contiguous view; only the last dimension must be contiguous.
        A_log: Log decay rates `[H]`.
        dt_bias: Softplus bias `[H]`.
        num_householder: Number of Householder copies `M`.
        num_heads: TP/CP-local head count `H`.
        num_groups: TP/CP-local group count `G`; `H` must be a multiple of it.
        head_dim: Value head dimension `P`.
        state_dim: Key/query head dimension `N`.

    Returns `(query, key, value, beta, g)` shaped `[n, S*M, H, N]`,
    `[n, S*M, H, N]`, `[n, S*M, H, P]`, `[n, S*M, H]` and `[n, S*M, H]` (fp32)
    -- the exact layouts `fused_recurrent_gated_delta_rule_update` expects for
    an `S*M`-length sequence per request.
    """
    M, H, G, P, N = num_householder, num_heads, num_groups, head_dim, state_dim
    _check_prepare_args("gdp_decode_prepare", x, ba, M, H, G, P, N)
    n, S = x.shape[0], x.shape[1]
    outputs = (
        x.new_empty(n, S * M, H, N),
        x.new_empty(n, S * M, H, N),
        x.new_empty(n, S * M, H, P),
        ba.new_empty(n, S * M, H),
        torch.empty(n, S * M, H, device=x.device, dtype=torch.float32),
    )
    _launch_prepare(x, ba, A_log, dt_bias, outputs, M, H, G, P, N, interleave_qg=True)
    return outputs


def gdp_prefill_prepare(
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
    """Split, reshape and gate a prefill step's post-conv activations.

    Args:
        x: Post-conv activations `[n, T, M*H*P + M*G*N + G*N]`, value/key/query
            concatenated along the last dimension (dynamic-batching prefill packs
            every request into `n == 1`). Rows may be strided (e.g. a view of the
            conv output); only the last dimension must be contiguous.
        ba: The `ba` slice of the input projection, `[n, T, M*H + H]`. May be a
            non-contiguous view; only the last dimension must be contiguous.
        A_log: Log decay rates `[H]`.
        dt_bias: Softplus bias `[H]`.
        num_householder: Number of Householder copies `M`.
        num_heads: TP/CP-local head count `H`.
        num_groups: TP/CP-local group count `G`; `H` must be a multiple of it.
        head_dim: Value head dimension `P`.
        state_dim: Key/query head dimension `N`.

    Returns `(query, key, value, beta, g)` shaped `[n, T, H, N]`, `[n, T*M, H, N]`,
    `[n, T*M, H, P]`, `[n, T*M, H]` and `[n, T, H]` (fp32) -- what
    `chunk_gated_delta_product_varlen` consumes, with the L2 norm left to it.
    """
    M, H, G, P, N = num_householder, num_heads, num_groups, head_dim, state_dim
    _check_prepare_args("gdp_prefill_prepare", x, ba, M, H, G, P, N)
    n, T = x.shape[0], x.shape[1]
    outputs = (
        x.new_empty(n, T, H, N),
        x.new_empty(n, T * M, H, N),
        x.new_empty(n, T * M, H, P),
        ba.new_empty(n, T * M, H),
        torch.empty(n, T, H, device=x.device, dtype=torch.float32),
    )
    _launch_prepare(x, ba, A_log, dt_bias, outputs, M, H, G, P, N, interleave_qg=False)
    return outputs
