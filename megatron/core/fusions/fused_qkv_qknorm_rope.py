# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Fused QKV split + Q/K RMSNorm + RoPE (sbhd, non-interleaved) in Triton.

Input is the unsplit linear_qkv output viewed as [sq, b, ng, (nq + 2) * hn], per group laid out as
nq query heads, one key head, one value head. The unfused path splits it with strided views,
copies Q and K into contiguous tensors for TE RMSNorm, applies TE fused RoPE and copies V for
attention. Here one program reads each token-group once and writes contiguous
q [sq, b, ng * nq, hn], k [sq, b, ng, hn] and v [sq, b, ng, hn].

Numerics follow the unfused TE path: the norm runs in fp32 and is rounded to the activation
dtype (TE RMSNorm output), then RoPE runs in fp32 on the rounded value (TE fused RoPE). The
backward mirrors it: RoPE dgrad rounded to the activation dtype, then RMSNorm dgrad in fp32 from
the saved input with rsigma recomputed. dgamma is reduced per program into an fp32 buffer and
summed once, so it is deterministic. Q/K/V dgrads are written straight into one
[sq, b, ng, (nq + 2) * hn] tensor: no concat.
"""

from typing import Optional, Tuple
from unittest.mock import MagicMock

import torch

from megatron.core.utils import null_decorator

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

if not HAVE_TRITON:
    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()

try:  # cosf/sinf/rsqrtf as TE's RoPE and RMSNorm kernels compute them
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover
    libdevice = MagicMock()


@triton.jit
def _cos(x):
    return libdevice.cos(x)


@triton.jit
def _sin(x):
    return libdevice.sin(x)


@triton.jit
def _rsqrt(x):
    return libdevice.rsqrt(x)


# Launches pin enable_fp_fusion=True (Triton's default until 3.10): the TE parity is tested with it.
# Launch shape: tokens handled per program (looped) and warps per program. Module-level so a
# benchmark can sweep them; no autotuning, so the first call is safe ahead of CUDA graph capture.
# Tuned at head dim 128 with 8 and 16 query heads per group.
# BWD_WARPS None: 1 warp up to 8 query heads per group, else 2.
FWD_TOKENS, FWD_WARPS = 1, 1
BWD_TOKENS, BWD_WARPS = 16, None


@triton.jit
def _qkv_qknorm_rope_fwd_kernel(
    qkv_ptr,
    qf_ptr,
    kf_ptr,
    qw_ptr,
    kw_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    T,
    B,
    stride_qkv_s,
    stride_qkv_b,
    stride_qkv_g,
    stride_qf_s,
    stride_kf_s,
    eps,
    W_OFFSET: tl.constexpr,
    NG: tl.constexpr,
    NQ: tl.constexpr,
    NQ_P2: tl.constexpr,
    HN: tl.constexpr,
    SAME_FREQS: tl.constexpr,
    TOKENS: tl.constexpr,
):
    HALF: tl.constexpr = HN // 2
    pid_t = tl.program_id(0).to(tl.int64)
    g = tl.program_id(1)
    d = tl.arange(0, HALF)
    h = tl.arange(0, NQ_P2)
    hmask = h[:, None] < NQ
    out_dtype = q_ptr.dtype.element_ty
    w1 = tl.load(qw_ptr + d).to(tl.float32) + W_OFFSET
    w2 = tl.load(qw_ptr + HALF + d).to(tl.float32) + W_OFFSET
    u1 = tl.load(kw_ptr + d).to(tl.float32) + W_OFFSET
    u2 = tl.load(kw_ptr + HALF + d).to(tl.float32) + W_OFFSET

    for i in range(TOKENS):
        tok = pid_t * TOKENS + i
        if tok < T:
            s = tok // B
            bi = tok % B
            base = qkv_ptr + s * stride_qkv_s + bi * stride_qkv_b + g * stride_qkv_g

            # Query heads: [NQ_P2, HALF] halves.
            qoff = h[:, None] * HN + d[None, :]
            x1 = tl.load(base + qoff, mask=hmask, other=0.0).to(tl.float32)
            x2 = tl.load(base + qoff + HALF, mask=hmask, other=0.0).to(tl.float32)
            f1 = tl.load(qf_ptr + s * stride_qf_s + d)
            f2 = tl.load(qf_ptr + s * stride_qf_s + HALF + d)
            c1, s1, c2, s2 = _cos(f1), _sin(f1), _cos(f2), _sin(f2)
            rs = _rsqrt((tl.sum(x1 * x1, axis=1) + tl.sum(x2 * x2, axis=1)) / HN + eps)
            n1 = (x1 * rs[:, None] * w1[None, :]).to(out_dtype).to(tl.float32)
            n2 = (x2 * rs[:, None] * w2[None, :]).to(out_dtype).to(tl.float32)
            o1 = n1 * c1[None, :] - n2 * s1[None, :]
            o2 = n2 * c2[None, :] + n1 * s2[None, :]
            qout = q_ptr + (tok * NG * NQ + g * NQ) * HN + qoff
            tl.store(qout, o1.to(out_dtype), mask=hmask)
            tl.store(qout + HALF, o2.to(out_dtype), mask=hmask)

            # Key head.
            kin = base + NQ * HN
            y1 = tl.load(kin + d).to(tl.float32)
            y2 = tl.load(kin + HALF + d).to(tl.float32)
            if not SAME_FREQS:
                f1 = tl.load(kf_ptr + s * stride_kf_s + d)
                f2 = tl.load(kf_ptr + s * stride_kf_s + HALF + d)
                c1, s1, c2, s2 = _cos(f1), _sin(f1), _cos(f2), _sin(f2)
            rk = _rsqrt((tl.sum(y1 * y1, axis=0) + tl.sum(y2 * y2, axis=0)) / HN + eps)
            m1 = (y1 * rk * u1).to(out_dtype).to(tl.float32)
            m2 = (y2 * rk * u2).to(out_dtype).to(tl.float32)
            kout = k_ptr + (tok * NG + g) * HN
            tl.store(kout + d, (m1 * c1 - m2 * s1).to(out_dtype))
            tl.store(kout + HALF + d, (m2 * c2 + m1 * s2).to(out_dtype))

            # Value head: plain copy.
            vin = base + (NQ + 1) * HN
            vout = v_ptr + (tok * NG + g) * HN
            tl.store(vout + d, tl.load(vin + d))
            tl.store(vout + HALF + d, tl.load(vin + HALF + d))


@triton.jit
def _qkv_qknorm_rope_bwd_kernel(
    qkv_ptr,
    qf_ptr,
    kf_ptr,
    qw_ptr,
    kw_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    dqkv_ptr,
    dqw_part_ptr,
    dkw_part_ptr,
    T,
    B,
    stride_qkv_s,
    stride_qkv_b,
    stride_qkv_g,
    stride_qf_s,
    stride_kf_s,
    eps,
    W_OFFSET: tl.constexpr,
    NG: tl.constexpr,
    NQ: tl.constexpr,
    NQ_P2: tl.constexpr,
    HN: tl.constexpr,
    TOKENS: tl.constexpr,
    SAME_FREQS: tl.constexpr,
):
    HALF: tl.constexpr = HN // 2
    ROW: tl.constexpr = (NQ + 2) * HN
    pid_t = tl.program_id(0).to(tl.int64)
    g = tl.program_id(1)
    d = tl.arange(0, HALF)
    h = tl.arange(0, NQ_P2)
    hmask = h[:, None] < NQ
    qoff = h[:, None] * HN + d[None, :]
    grad_dtype = dqkv_ptr.dtype.element_ty

    w1 = tl.load(qw_ptr + d).to(tl.float32) + W_OFFSET
    w2 = tl.load(qw_ptr + HALF + d).to(tl.float32) + W_OFFSET
    u1 = tl.load(kw_ptr + d).to(tl.float32) + W_OFFSET
    u2 = tl.load(kw_ptr + HALF + d).to(tl.float32) + W_OFFSET
    dqw1 = tl.zeros([HALF], dtype=tl.float32)
    dqw2 = tl.zeros([HALF], dtype=tl.float32)
    dkw1 = tl.zeros([HALF], dtype=tl.float32)
    dkw2 = tl.zeros([HALF], dtype=tl.float32)

    for i in range(TOKENS):
        tok = pid_t * TOKENS + i
        if tok < T:
            s = tok // B
            bi = tok % B
            base = qkv_ptr + s * stride_qkv_s + bi * stride_qkv_b + g * stride_qkv_g
            dbase = dqkv_ptr + (tok * NG + g) * ROW
            f1 = tl.load(qf_ptr + s * stride_qf_s + d)
            f2 = tl.load(qf_ptr + s * stride_qf_s + HALF + d)
            c1, s1, c2, s2 = _cos(f1), _sin(f1), _cos(f2), _sin(f2)

            # Query heads: RoPE dgrad (rounded like TE's output), then RMSNorm dgrad.
            gq = dq_ptr + (tok * NG * NQ + g * NQ) * HN + qoff
            e1 = tl.load(gq, mask=hmask, other=0.0).to(tl.float32)
            e2 = tl.load(gq + HALF, mask=hmask, other=0.0).to(tl.float32)
            g1 = (e1 * c1[None, :] + e2 * s2[None, :]).to(grad_dtype).to(tl.float32)
            g2 = (e2 * c2[None, :] - e1 * s1[None, :]).to(grad_dtype).to(tl.float32)
            x1 = tl.load(base + qoff, mask=hmask, other=0.0).to(tl.float32)
            x2 = tl.load(base + qoff + HALF, mask=hmask, other=0.0).to(tl.float32)
            rs = _rsqrt((tl.sum(x1 * x1, axis=1) + tl.sum(x2 * x2, axis=1)) / HN + eps)
            xh1 = x1 * rs[:, None]
            xh2 = x2 * rs[:, None]
            z1 = g1 * w1[None, :]
            z2 = g2 * w2[None, :]
            mdot = (tl.sum(z1 * xh1, axis=1) + tl.sum(z2 * xh2, axis=1)) / HN
            tl.store(
                dbase + qoff, (rs[:, None] * (z1 - xh1 * mdot[:, None])).to(grad_dtype), mask=hmask
            )
            tl.store(
                dbase + qoff + HALF,
                (rs[:, None] * (z2 - xh2 * mdot[:, None])).to(grad_dtype),
                mask=hmask,
            )
            dqw1 += tl.sum(g1 * xh1, axis=0)
            dqw2 += tl.sum(g2 * xh2, axis=0)

            # Key head.
            if not SAME_FREQS:
                f1 = tl.load(kf_ptr + s * stride_kf_s + d)
                f2 = tl.load(kf_ptr + s * stride_kf_s + HALF + d)
                c1, s1, c2, s2 = _cos(f1), _sin(f1), _cos(f2), _sin(f2)
            gk = dk_ptr + (tok * NG + g) * HN
            a1 = tl.load(gk + d).to(tl.float32)
            a2 = tl.load(gk + HALF + d).to(tl.float32)
            b1 = (a1 * c1 + a2 * s2).to(grad_dtype).to(tl.float32)
            b2 = (a2 * c2 - a1 * s1).to(grad_dtype).to(tl.float32)
            y1 = tl.load(base + NQ * HN + d).to(tl.float32)
            y2 = tl.load(base + NQ * HN + HALF + d).to(tl.float32)
            rk = _rsqrt((tl.sum(y1 * y1, axis=0) + tl.sum(y2 * y2, axis=0)) / HN + eps)
            yh1 = y1 * rk
            yh2 = y2 * rk
            v1 = b1 * u1
            v2 = b2 * u2
            kdot = (tl.sum(v1 * yh1, axis=0) + tl.sum(v2 * yh2, axis=0)) / HN
            tl.store(dbase + NQ * HN + d, (rk * (v1 - yh1 * kdot)).to(grad_dtype))
            tl.store(dbase + NQ * HN + HALF + d, (rk * (v2 - yh2 * kdot)).to(grad_dtype))
            dkw1 += b1 * yh1
            dkw2 += b2 * yh2

            # Value head: plain copy of its gradient.
            gv = dv_ptr + (tok * NG + g) * HN
            tl.store(dbase + (NQ + 1) * HN + d, tl.load(gv + d))
            tl.store(dbase + (NQ + 1) * HN + HALF + d, tl.load(gv + HALF + d))

    part = (pid_t * NG + g) * HN
    tl.store(dqw_part_ptr + part + d, dqw1)
    tl.store(dqw_part_ptr + part + HALF + d, dqw2)
    tl.store(dkw_part_ptr + part + d, dkw1)
    tl.store(dkw_part_ptr + part + HALF + d, dkw2)


def _freqs_2d(freqs: torch.Tensor, sq: int, hn: int) -> torch.Tensor:
    """[sq', 1, 1, hn] (or [sq', hn]) fp32 RoPE angles -> [sq, hn] view with unit last stride."""
    f = freqs.reshape(freqs.shape[0], -1)
    assert (
        f.shape[1] == hn
    ), f"RoPE freqs last dim {f.shape[1]} != head dim {hn} (rotary_percent < 1?)"
    assert f.shape[0] >= sq, f"RoPE freqs cover {f.shape[0]} positions < sequence length {sq}"
    if f.dtype != torch.float32 or f.stride(1) != 1:
        f = f.float().contiguous()
    return f


class _FusedQKVQKNormRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mixed_qkv, q_freqs, k_freqs, q_weight, k_weight, nq, eps, zero_centered_gamma):
        """Split, normalize and rotate; saves the unsplit input for the backward."""
        sq, b, ng, row = mixed_qkv.shape
        hn = row // (nq + 2)
        assert (
            row == (nq + 2) * hn and hn % 2 == 0 and hn & (hn - 1) == 0
        ), f"unsupported head layout: row {row}, {nq} query heads per group, head dim {hn}"
        assert mixed_qkv.stride(3) == 1, "mixed_qkv must have a unit last-dim stride"
        qf = _freqs_2d(q_freqs, sq, hn)
        kf = qf if k_freqs is q_freqs else _freqs_2d(k_freqs, sq, hn)
        q_weight = q_weight.contiguous()
        k_weight = k_weight.contiguous()
        q = mixed_qkv.new_empty(sq, b, ng * nq, hn)
        k = mixed_qkv.new_empty(sq, b, ng, hn)
        v = mixed_qkv.new_empty(sq, b, ng, hn)
        _qkv_qknorm_rope_fwd_kernel[(triton.cdiv(sq * b, FWD_TOKENS), ng)](
            mixed_qkv,
            qf,
            kf,
            q_weight,
            k_weight,
            q,
            k,
            v,
            sq * b,
            b,
            mixed_qkv.stride(0),
            mixed_qkv.stride(1),
            mixed_qkv.stride(2),
            qf.stride(0),
            kf.stride(0),
            eps,
            W_OFFSET=1.0 if zero_centered_gamma else 0.0,
            NG=ng,
            NQ=nq,
            NQ_P2=triton.next_power_of_2(nq),
            HN=hn,
            SAME_FREQS=kf is qf,
            TOKENS=FWD_TOKENS,
            num_warps=FWD_WARPS,
            enable_fp_fusion=True,
        )
        ctx.save_for_backward(mixed_qkv, qf, kf, q_weight, k_weight)
        ctx.cfg = (nq, eps, zero_centered_gamma)
        return q, k, v

    @staticmethod
    def backward(ctx, dq, dk, dv):
        """Q/K/V dgrads into one unsplit tensor, plus the Q/K norm weight grads."""
        mixed_qkv, qf, kf, q_weight, k_weight = ctx.saved_tensors
        nq, eps, zero_centered_gamma = ctx.cfg
        sq, b, ng, row = mixed_qkv.shape
        hn = row // (nq + 2)
        T = sq * b
        n_t = triton.cdiv(T, BWD_TOKENS)
        dqkv = mixed_qkv.new_empty(sq, b, ng, row)
        dqw_part = torch.empty(n_t * ng, hn, device=mixed_qkv.device, dtype=torch.float32)
        dkw_part = torch.empty_like(dqw_part)
        _qkv_qknorm_rope_bwd_kernel[(n_t, ng)](
            mixed_qkv,
            qf,
            kf,
            q_weight,
            k_weight,
            dq.contiguous(),
            dk.contiguous(),
            dv.contiguous(),
            dqkv,
            dqw_part,
            dkw_part,
            T,
            b,
            mixed_qkv.stride(0),
            mixed_qkv.stride(1),
            mixed_qkv.stride(2),
            qf.stride(0),
            kf.stride(0),
            eps,
            W_OFFSET=1.0 if zero_centered_gamma else 0.0,
            NG=ng,
            NQ=nq,
            NQ_P2=triton.next_power_of_2(nq),
            HN=hn,
            TOKENS=BWD_TOKENS,
            SAME_FREQS=kf is qf,
            num_warps=BWD_WARPS or (1 if nq <= 8 else 2),
            enable_fp_fusion=True,
        )
        dqw = dqw_part.sum(0).to(q_weight.dtype)
        dkw = dkw_part.sum(0).to(k_weight.dtype)
        return dqkv, None, None, dqw, dkw, None, None, None


def fused_qkv_qknorm_rope(
    mixed_qkv: torch.Tensor,
    q_freqs: torch.Tensor,
    k_freqs: Optional[torch.Tensor],
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    num_query_heads_per_group: int,
    eps: float,
    zero_centered_gamma: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split mixed QKV, RMSNorm Q and K per head, apply RoPE to Q and K.

    Args:
        mixed_qkv: [sq, b, ng, (nq + 2) * hn] linear_qkv output (nq query, 1 key, 1 value head per
            group), unit last-dim stride.
        q_freqs, k_freqs: RoPE angles, [sq, 1, 1, hn] fp32 (rotary_percent 1, non-interleaved,
            mscale 1). k_freqs None means the same as q_freqs.
        q_weight, k_weight: [hn] RMSNorm weights of q_layernorm / k_layernorm.
        num_query_heads_per_group: nq.
        eps: RMSNorm epsilon.
        zero_centered_gamma: RMSNorm weight is stored as gamma - 1.

    Returns:
        Contiguous query [sq, b, ng * nq, hn], key [sq, b, ng, hn], value [sq, b, ng, hn].
    """
    if not HAVE_TRITON:
        raise ImportError("fused_qkv_qknorm_rope requires Triton")
    if k_freqs is None:
        k_freqs = q_freqs
    return _FusedQKVQKNormRoPE.apply(
        mixed_qkv,
        q_freqs,
        k_freqs,
        q_weight,
        k_weight,
        num_query_heads_per_group,
        eps,
        zero_centered_gamma,
    )
