# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fused Engram context gate: Triton forward and backward for ``Engram.forward``'s gating.

``Engram.forward`` uses these kernels instead of its eager FP32 gate when
``MCORE_ENGRAM_FUSED_GATE=1`` (read once at import, off by default), Triton is available and
its tensors are on CUDA; otherwise the eager code runs, and it stays the reference. Per token
(s, b) and residual stream i, with h the stream's hidden state, k its key, v the value shared
by all streams, gamma = q_weight * k_weight, d the hidden size and c0 = d ** -0.5:

    rh = rsqrt(mean(h^2) + eps),  rk = rsqrt(mean(k^2) + eps)
    dot = (sum(h * gamma * k) * (rh * rk)) * c0
    gate = sigmoid(copysign(sqrt(clamp_min(|dot|, 1e-6)), dot)), and 0 for masked tokens
    out = h + gate * v

Everything is computed in FP32 like the eager code. The gate and its derivative take the steps
of PyTorch's ops and autograd formulas with the same routines (correctly rounded sqrt and
division, expf), so given the same ``dot`` they are bitwise equal to eager autograd's, signed
zeros, saturated gates and non-finite values included. Masked tokens get a zero gradient for
dot even where eager autograd would multiply a non-finite value by zero. The sums over d (and
over streams and tokens in backward) run in a different order than PyTorch's reductions, so
results are not bitwise equal to the eager path; they are deterministic run to run (no
atomics, fixed launch configurations, fixed-order sums).

kv is either [sequence, batch, (n + 1) d] or, as the MXFP8 lookup's projection output,
[rows, (n + 1) d] with the tokens batch-major in its first rows and padding after them. The
latter goes in whole: its gradient comes back in that shape with zero pad rows, so autograd
records no slice (whose backward would zero-fill and copy the whole gradient), view or
transpose of it.

Backward keeps only the inputs (as given, typically BF16) and four FP32 scalars per token and
stream (rh, rk, dot, gate). It runs three kernels: per-token gate-gradient coefficients, the
input gradients by token chunk and hidden block (also writing FP32 partial sums of the
gamma gradient), and a fixed-order sum of those partials into the q_weight/k_weight gradients.
No host synchronization and only static-shape allocations, so the gate can be captured in
CUDA graphs, including full-iteration graphs.
"""

import functools
import warnings
from typing import Optional

import torch
from torch import Tensor

from megatron.core.env_flags import env_flag

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

# Read once at import; Engram.forward dispatches on it (and on HAVE_TRITON).
FUSED_ENGRAM_GATE = env_flag("MCORE_ENGRAM_FUSED_GATE", False)
if FUSED_ENGRAM_GATE and not HAVE_TRITON:
    warnings.warn(
        "MCORE_ENGRAM_FUSED_GATE=1 but Triton is unavailable; Engram keeps its eager gate"
    )

# torch.clamp_min(|dot|, 1e-6) on FP32 compares with 1e-6 rounded to FP32; so do the kernels.
_GATE_THRESHOLD = 1e-6
# Launch configurations are functions of the shapes only (no autotuning), so every reduction
# has a fixed order. Forward: hidden block of one (token, stream) program. Backward: elements
# per [streams, hidden block] register tile, and the token chunks of the q_weight/k_weight
# gradient partials (summed in chunk order by the weight kernel).
_FWD_BLOCK = 4096
_TILE_ELEMENTS = 1024
_MAX_TOKEN_CHUNKS = 256
_MIN_TOKENS_PER_CHUNK = 128
_MAIN_STAGES = 3
_WEIGHT_BLOCK = 128
_WEIGHT_UNROLL = 16


if HAVE_TRITON:

    @triton.jit
    def _gate_from_dot(dot, THRESHOLD: tl.constexpr):
        """sigmoid(copysign(sqrt(clamp_min(|dot|, THRESHOLD)), dot)) as PyTorch computes it."""
        magnitude = tl.maximum(tl.abs(dot), THRESHOLD, propagate_nan=tl.PropagateNan.ALL)
        root = libdevice.copysign(tl.sqrt_rn(magnitude), dot)
        return tl.div_rn(tl.full(dot.shape, 1.0, tl.float32), 1.0 + libdevice.exp(-root))

    @triton.jit
    def _dot_grad_from_gate_grad(
        grad_gate, gate, dot, THRESHOLD: tl.constexpr, AT_THRESHOLD: tl.constexpr
    ):
        """Chain autograd's sigmoid, copysign, sqrt, clamp_min and abs formulas, step by step.

        sigmoid: (grad * (1 - gate)) * gate; copysign: grad * (result / self), which is +-1,
        or NaN for an infinite dot; sqrt: grad / (2 * result); clamp_min passes the gradient
        where |dot| > THRESHOLD, and also at equality when AT_THRESHOLD (the installed
        PyTorch's rule, see _clamp_min_grad_at_min); abs: grad * sign(dot), with sign 0 for
        +-0 and NaN. Autograd then adds copysign's zero gradient for its sign input, so the
        final + 0.0 turns -0 (saturated gates) into +0 as well.
        """
        grad_root = (grad_gate * (1.0 - gate)) * gate
        magnitude = tl.abs(dot)
        root = tl.sqrt_rn(tl.maximum(magnitude, THRESHOLD, propagate_nan=tl.PropagateNan.ALL))
        grad_root = grad_root * tl.div_rn(libdevice.copysign(root, dot), root)
        grad_magnitude = tl.div_rn(grad_root, 2.0 * root)
        if AT_THRESHOLD:
            passes = magnitude >= THRESHOLD
        else:
            passes = magnitude > THRESHOLD
        sign = (dot > 0).to(tl.float32) - (dot < 0).to(tl.float32)
        return tl.where(passes, grad_magnitude, 0.0) * sign + 0.0

    @triton.jit
    def _engram_gate_fwd_kernel(
        hidden,
        kv,
        q_weight,
        k_weight,
        token_mask,
        out,
        stats,
        batch,
        stride_hs,
        stride_hb,
        stride_ks,
        stride_kb,
        stride_ms,
        stride_mb,
        eps,
        scale,
        inv_d,
        N: tl.constexpr,
        D: tl.constexpr,
        NP: tl.constexpr,
        BLOCK_D: tl.constexpr,
        HAS_MASK: tl.constexpr,
        THRESHOLD: tl.constexpr,
    ):
        """One program per (token, stream): the stream's gate, then out = h + gate * v.

        Streams vary fastest across programs, so the N programs of a token run together and
        share the value row in cache; the second pass re-reads only rows just read.
        """
        pid = tl.program_id(0).to(tl.int64)
        token = pid // N
        stream = pid - token * N
        s = token // batch
        b = token - s * batch
        cols = tl.arange(0, BLOCK_D)
        h_row = hidden + s * stride_hs + b * stride_hb + stream * D
        kv_row = kv + s * stride_ks + b * stride_kb
        sum_hh = tl.zeros((BLOCK_D,), tl.float32)
        sum_kk = tl.zeros((BLOCK_D,), tl.float32)
        sum_hgk = tl.zeros((BLOCK_D,), tl.float32)
        for start in tl.static_range(0, D, BLOCK_D):
            col = start + cols
            valid = col < D
            h = tl.load(h_row + col, valid, other=0.0).to(tl.float32)
            k = tl.load(kv_row + stream * D + col, valid, other=0.0).to(tl.float32)
            gamma = tl.load(q_weight + stream * D + col, valid, other=0.0).to(tl.float32) * tl.load(
                k_weight + stream * D + col, valid, other=0.0
            ).to(tl.float32)
            sum_hh += h * h
            sum_kk += k * k
            sum_hgk += (h * gamma) * k
        # PyTorch's reductions start from +0.0, so an all -0.0 sum is +0.0; adding +0.0 matches.
        rh = libdevice.rsqrt((tl.sum(sum_hh, axis=0) + 0.0) * inv_d + eps)
        rk = libdevice.rsqrt((tl.sum(sum_kk, axis=0) + 0.0) * inv_d + eps)
        dot = ((tl.sum(sum_hgk, axis=0) + 0.0) * (rh * rk)) * scale
        gate = _gate_from_dot(dot, THRESHOLD)
        if HAS_MASK:
            keep = tl.load(token_mask + s * stride_ms + b * stride_mb)
            gate = tl.where(keep != 0, gate, 0.0)
        row = stats + token * (4 * NP) + stream
        tl.store(row, rh)
        tl.store(row + NP, rk)
        tl.store(row + 2 * NP, dot)
        tl.store(row + 3 * NP, gate)
        out_row = out + token * (N * D) + stream * D
        for start in tl.static_range(0, D, BLOCK_D):
            col = start + cols
            valid = col < D
            h = tl.load(h_row + col, valid, other=0.0).to(tl.float32)
            v = tl.load(kv_row + N * D + col, valid, other=0.0).to(tl.float32)
            tl.store(out_row + col, (h + gate * v).to(out.dtype.element_ty), valid)

    @triton.jit
    def _engram_gate_bwd_coef_kernel(
        grad_out,
        kv,
        token_mask,
        stats,
        coef,
        batch,
        stride_gs,
        stride_gb,
        stride_ks,
        stride_kb,
        stride_ms,
        stride_mb,
        scale,
        inv_d,
        N: tl.constexpr,
        D: tl.constexpr,
        NP: tl.constexpr,
        BLOCK_D: tl.constexpr,
        HAS_MASK: tl.constexpr,
        THRESHOLD: tl.constexpr,
        AT_THRESHOLD: tl.constexpr,
    ):
        """One program per token: d loss / d gate and the per-stream input-gradient coefficients.

        With S = sum(h * gamma * k) and grad_dot = d loss / d dot:
        d dot / d S = (rh * rk) * c0, and d dot / d rh = dot / rh with d rh / d h = -rh^3 h / d
        (likewise for k), so d loss / d h gets cross * gamma * k - grad_dot * dot * rh^2 / d * h.
        """
        token = tl.program_id(0).to(tl.int64)
        s = token // batch
        b = token - s * batch
        streams = tl.arange(0, NP)[:, None]
        cols = tl.arange(0, BLOCK_D)[None, :]
        g_row = grad_out + s * stride_gs + b * stride_gb
        v_row = kv + s * stride_ks + b * stride_kb + N * D
        acc = tl.zeros((NP, BLOCK_D), tl.float32)
        for start in tl.static_range(0, D, BLOCK_D):
            col = start + cols
            valid = (streams < N) & (col < D)
            grad = tl.load(g_row + streams * D + col, valid, other=0.0).to(tl.float32)
            v = tl.load(v_row + col, col < D, other=0.0).to(tl.float32)
            acc += grad * v
        grad_gate = tl.sum(acc, axis=1) + 0.0
        # Padding lanes (N not a power of two) were never written: read them as zeros, which
        # makes all of their coefficients zero.
        lanes = tl.arange(0, NP)
        real = lanes < N
        row = stats + token * (4 * NP) + lanes
        rh = tl.load(row, real, other=0.0)
        rk = tl.load(row + NP, real, other=0.0)
        dot = tl.load(row + 2 * NP, real, other=0.0)
        gate = tl.load(row + 3 * NP, real, other=0.0)
        grad_dot = _dot_grad_from_gate_grad(grad_gate, gate, dot, THRESHOLD, AT_THRESHOLD)
        if HAS_MASK:
            keep = tl.load(token_mask + s * stride_ms + b * stride_mb)
            grad_dot = tl.where(keep != 0, grad_dot, 0.0)
        radial = (grad_dot * dot) * inv_d
        out_row = coef + token * (4 * NP) + lanes
        tl.store(out_row, (grad_dot * scale) * (rh * rk))
        tl.store(out_row + NP, radial * (rh * rh))
        tl.store(out_row + 2 * NP, radial * (rk * rk))
        tl.store(out_row + 3 * NP, gate)

    @triton.jit
    def _engram_gate_bwd_main_kernel(
        grad_out,
        hidden,
        kv,
        q_weight,
        k_weight,
        coef,
        grad_hidden,
        grad_kv,
        partial,
        num_tokens,
        batch,
        stride_gs,
        stride_gb,
        stride_hs,
        stride_hb,
        stride_ks,
        stride_kb,
        stride_dks,
        stride_dkb,
        N: tl.constexpr,
        D: tl.constexpr,
        NP: tl.constexpr,
        BLOCK_D: tl.constexpr,
        TOKENS: tl.constexpr,
        STAGES: tl.constexpr,
    ):
        """Token-chunk x hidden-block programs: input gradients and FP32 gamma-gradient partials.

        The value gradient sums the N streams in registers, and every program owns its slice
        of the partial buffer, so nothing is accumulated across programs.
        """
        chunk = tl.program_id(0).to(tl.int64)
        streams = tl.arange(0, NP)[:, None]
        col = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)[None, :]
        valid = (streams < N) & (col < D)
        offset = streams * D + col
        gamma = tl.load(q_weight + offset, valid, other=0.0).to(tl.float32) * tl.load(
            k_weight + offset, valid, other=0.0
        ).to(tl.float32)
        grad_gamma = tl.zeros((NP, BLOCK_D), tl.float32)
        lanes = tl.arange(0, NP)
        for i in tl.range(TOKENS, num_stages=STAGES):
            token = chunk * TOKENS + i
            live = token < num_tokens
            s = token // batch
            b = token - s * batch
            row = coef + token * (4 * NP) + lanes
            lane_live = (lanes < N) & live
            cross = tl.load(row, lane_live, other=0.0)[:, None]
            radial_h = tl.load(row + NP, lane_live, other=0.0)[:, None]
            radial_k = tl.load(row + 2 * NP, lane_live, other=0.0)[:, None]
            gate = tl.load(row + 3 * NP, lane_live, other=0.0)[:, None]
            mask = valid & live
            grad = tl.load(grad_out + s * stride_gs + b * stride_gb + offset, mask, other=0.0).to(
                tl.float32
            )
            h = tl.load(hidden + s * stride_hs + b * stride_hb + offset, mask, other=0.0).to(
                tl.float32
            )
            k = tl.load(kv + s * stride_ks + b * stride_kb + offset, mask, other=0.0).to(tl.float32)
            cross_k = cross * k
            grad_h = grad + cross_k * gamma - radial_h * h
            grad_k = cross * (h * gamma) - radial_k * k
            grad_gamma += cross_k * h
            grad_v = tl.sum(gate * grad, axis=0, keep_dims=True) + 0.0
            tl.store(
                grad_hidden + token * (N * D) + offset,
                grad_h.to(grad_hidden.dtype.element_ty),
                mask,
            )
            dkv_row = grad_kv + s * stride_dks + b * stride_dkb
            tl.store(dkv_row + offset, grad_k.to(grad_kv.dtype.element_ty), mask)
            tl.store(dkv_row + N * D + col, grad_v.to(grad_kv.dtype.element_ty), (col < D) & live)
        tl.store(partial + chunk * (N * D) + offset, grad_gamma, valid)

    @triton.jit
    def _engram_gate_bwd_weight_kernel(
        partial,
        q_weight,
        k_weight,
        grad_q,
        grad_k,
        num_chunks,
        NUMEL: tl.constexpr,
        BLOCK: tl.constexpr,
        UNROLL: tl.constexpr,
    ):
        """Sum the token-chunk partials in chunk order, then apply d gamma / d q and d gamma / d k.

        UNROLL partial rows are loaded together and then added one after another, so the
        loads overlap while the sum keeps its fixed chunk order (missing rows add +0.0, which
        leaves a sum that started from +0.0 unchanged).
        """
        offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offset < NUMEL
        total = tl.zeros((BLOCK,), tl.float32)
        for first in range(0, num_chunks, UNROLL):
            for u in tl.static_range(UNROLL):
                c = first + u
                total += tl.load(
                    partial + c.to(tl.int64) * NUMEL + offset, valid & (c < num_chunks), other=0.0
                )
        q = tl.load(q_weight + offset, valid, other=0.0).to(tl.float32)
        k = tl.load(k_weight + offset, valid, other=0.0).to(tl.float32)
        tl.store(grad_q + offset, (total * k).to(grad_q.dtype.element_ty), valid)
        tl.store(grad_k + offset, (total * q).to(grad_k.dtype.element_ty), valid)


@functools.lru_cache(maxsize=None)
def _clamp_min_grad_at_min() -> bool:
    """Whether autograd's ``clamp_min(x, min)`` passes the gradient where ``x == min``.

    PyTorch's formula was ``where(self >= min, grad, 0)`` and is ``where(self > min, grad, 0)``
    in recent versions; the eager gate inherits whichever is installed, and so does the fused
    one. Evaluated once, on a one-element CPU tensor.
    """
    with torch.inference_mode(False), torch.enable_grad():
        x = torch.tensor([_GATE_THRESHOLD], dtype=torch.float32, requires_grad=True)
        (grad,) = torch.autograd.grad(x.clamp_min(_GATE_THRESHOLD), x, torch.ones(1))
    return bool(grad.item() != 0)


def _fwd_block(hidden_size: int) -> tuple:
    """Hidden block and warps of the forward's (token, stream) programs."""
    block = min(triton.next_power_of_2(hidden_size), _FWD_BLOCK)
    return block, max(1, min(4, block // 1024))


def _tile(num_streams: int, hidden_size: int) -> tuple:
    """Padded stream count, hidden block and warps of the backward's [streams, block] tiles."""
    streams = triton.next_power_of_2(num_streams)
    block = min(triton.next_power_of_2(hidden_size), max(16, _TILE_ELEMENTS // streams))
    return streams, block, max(1, min(8, streams * block // 512))


def _tokens_per_chunk(num_tokens: int) -> int:
    """Tokens per backward program; a function of the token count only (fixed sum order)."""
    return max(_MIN_TOKENS_PER_CHUNK, triton.cdiv(num_tokens, _MAX_TOKEN_CHUNKS))


def _last_dim_contiguous(tensor: Tensor) -> Tensor:
    return tensor if tensor.stride(-1) == 1 else tensor.contiguous()


def _token_strides(tensor: Tensor, length: int) -> tuple:
    """Element strides from token (s, b) to (s + 1, b) and to (s, b + 1).

    A [sequence, batch, width] tensor has its own strides; a [rows, width] tensor holds the
    tokens batch-major in its first sequence * batch rows.
    """
    if tensor.dim() == 2:
        return tensor.stride(0), length * tensor.stride(0)
    return tensor.stride(0), tensor.stride(1)


def _mask_args(token_mask: Optional[Tensor], reference: Tensor) -> tuple:
    """A uint8 view of the [batch, sequence] mask and its (sequence, batch) strides."""
    if token_mask is None:
        return reference, 0, 0
    mask = token_mask.view(torch.uint8)
    return mask, mask.stride(1), mask.stride(0)


def _check_inputs(hidden_states, kv, q_weight, k_weight, token_mask, num_streams, hidden_size):
    if hidden_states.dim() != 3:
        raise ValueError("hidden states must be [sequence, batch, num_streams * hidden_size]")
    length, batch, width = hidden_states.shape
    if width != num_streams * hidden_size:
        raise ValueError(f"hidden states width {width} != {num_streams} x {hidden_size}")
    kv_width = (num_streams + 1) * hidden_size
    if kv.dim() == 2:
        if kv.shape[1] != kv_width or kv.shape[0] < length * batch:
            raise ValueError(
                f"kv shape {tuple(kv.shape)}: [rows, width] kv needs at least "
                f"{length * batch} rows of width {kv_width}"
            )
    elif tuple(kv.shape) != (length, batch, kv_width):
        raise ValueError(f"kv shape {tuple(kv.shape)} does not match hidden states")
    if q_weight.numel() != width or k_weight.numel() != width:
        raise ValueError("q_weight and k_weight must have num_streams * hidden_size elements")
    if token_mask is not None and (
        token_mask.dtype != torch.bool
        or token_mask.dim() != 2
        or any(size not in (1, full) for size, full in zip(token_mask.shape, (batch, length)))
    ):
        raise ValueError("token_mask must be a bool tensor broadcastable to [batch, sequence]")
    tensors = (hidden_states, kv, q_weight, k_weight) + (
        () if token_mask is None else (token_mask,)
    )
    if any(t.device != hidden_states.device or not t.is_cuda for t in tensors):
        raise ValueError("the fused Engram gate needs all tensors on one CUDA device")


class FusedEngramGate(torch.autograd.Function):
    """Engram's context gate with fused Triton forward and backward (see the module docstring)."""

    @staticmethod
    def forward(
        ctx, hidden_states, kv, q_weight, k_weight, token_mask, num_streams, hidden_size, eps
    ):
        """Return ``h + gate * v`` as [sequence, batch, num_streams * hidden_size]."""
        length, batch = hidden_states.shape[:2]
        hidden = _last_dim_contiguous(hidden_states)
        kv = _last_dim_contiguous(kv)
        stride_ks, stride_kb = _token_strides(kv, length)
        q_weight, k_weight = q_weight.contiguous(), k_weight.contiguous()
        streams = triton.next_power_of_2(num_streams)
        block, warps = _fwd_block(hidden_size)
        out = torch.empty(
            (length, batch, num_streams * hidden_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        stats = torch.empty(
            (length * batch, 4, streams), dtype=torch.float32, device=hidden_states.device
        )
        mask, stride_ms, stride_mb = _mask_args(token_mask, stats)
        if length * batch:
            _engram_gate_fwd_kernel[(length * batch * num_streams,)](
                hidden,
                kv,
                q_weight,
                k_weight,
                mask,
                out,
                stats,
                batch,
                hidden.stride(0),
                hidden.stride(1),
                stride_ks,
                stride_kb,
                stride_ms,
                stride_mb,
                eps,
                hidden_size**-0.5,
                1.0 / hidden_size,
                N=num_streams,
                D=hidden_size,
                NP=streams,
                BLOCK_D=block,
                HAS_MASK=token_mask is not None,
                THRESHOLD=_GATE_THRESHOLD,
                num_warps=warps,
            )
        ctx.save_for_backward(hidden, kv, q_weight, k_weight, token_mask, stats)
        ctx.num_streams, ctx.hidden_size = num_streams, hidden_size
        ctx.grad_at_threshold = _clamp_min_grad_at_min()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """Input, kv and q_weight/k_weight gradients; no atomics, fixed-order weight sums."""
        hidden, kv, q_weight, k_weight, token_mask, stats = ctx.saved_tensors
        num_streams, hidden_size = ctx.num_streams, ctx.hidden_size
        length, batch = hidden.shape[:2]
        tokens = length * batch
        grad_output = _last_dim_contiguous(grad_output)
        streams, block, warps = _tile(num_streams, hidden_size)
        mask, stride_ms, stride_mb = _mask_args(token_mask, stats)
        stride_ks, stride_kb = _token_strides(kv, length)
        coef = torch.empty_like(stats)
        grad_hidden = torch.empty(
            (length, batch, num_streams * hidden_size), dtype=hidden.dtype, device=hidden.device
        )
        # Keep kv's shape and layout (a [batch, sequence] buffer seen as [sequence, batch], or
        # the MXFP8 lookup's padded [rows, width] projection output), so the gradient needs no
        # copy on its way to the projection; the pad rows of a [rows, width] kv are zeroed here.
        grad_kv = torch.empty_like(kv)
        stride_dks, stride_dkb = _token_strides(grad_kv, length)
        if kv.dim() == 2 and kv.shape[0] > tokens:
            grad_kv[tokens:].zero_()
        chunk = _tokens_per_chunk(tokens)
        chunks = triton.cdiv(tokens, chunk)
        partial = torch.empty(
            (chunks, num_streams * hidden_size), dtype=torch.float32, device=hidden.device
        )
        grad_q, grad_k = torch.empty_like(q_weight), torch.empty_like(k_weight)
        if tokens:
            _engram_gate_bwd_coef_kernel[(tokens,)](
                grad_output,
                kv,
                mask,
                stats,
                coef,
                batch,
                grad_output.stride(0),
                grad_output.stride(1),
                stride_ks,
                stride_kb,
                stride_ms,
                stride_mb,
                hidden_size**-0.5,
                1.0 / hidden_size,
                N=num_streams,
                D=hidden_size,
                NP=streams,
                BLOCK_D=block,
                HAS_MASK=token_mask is not None,
                THRESHOLD=_GATE_THRESHOLD,
                AT_THRESHOLD=ctx.grad_at_threshold,
                num_warps=warps,
            )
            _engram_gate_bwd_main_kernel[(chunks, triton.cdiv(hidden_size, block))](
                grad_output,
                hidden,
                kv,
                q_weight,
                k_weight,
                coef,
                grad_hidden,
                grad_kv,
                partial,
                tokens,
                batch,
                grad_output.stride(0),
                grad_output.stride(1),
                hidden.stride(0),
                hidden.stride(1),
                stride_ks,
                stride_kb,
                stride_dks,
                stride_dkb,
                N=num_streams,
                D=hidden_size,
                NP=streams,
                BLOCK_D=block,
                TOKENS=chunk,
                STAGES=_MAIN_STAGES,
                num_warps=warps,
            )
            _engram_gate_bwd_weight_kernel[(triton.cdiv(q_weight.numel(), _WEIGHT_BLOCK),)](
                partial,
                q_weight,
                k_weight,
                grad_q,
                grad_k,
                chunks,
                NUMEL=q_weight.numel(),
                BLOCK=_WEIGHT_BLOCK,
                UNROLL=_WEIGHT_UNROLL,
                num_warps=1,
            )
        else:
            grad_q.zero_()
            grad_k.zero_()
        needs = ctx.needs_input_grad
        return (
            grad_hidden if needs[0] else None,
            grad_kv if needs[1] else None,
            grad_q if needs[2] else None,
            grad_k if needs[3] else None,
            None,
            None,
            None,
            None,
        )


def fused_engram_gate(
    hidden_states: Tensor,
    kv: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    token_mask: Optional[Tensor],
    num_streams: int,
    hidden_size: int,
    eps: float,
) -> Tensor:
    """Engram's gated memory update of all residual streams, fused.

    Args:
        hidden_states: [sequence, batch, num_streams * hidden_size] residual streams.
        kv: the projection output, keys of the streams followed by the shared value, either as
            [sequence, batch, (num_streams + 1) * hidden_size] in any layout whose last
            dimension is dense (Engram passes a [batch, sequence] buffer transposed to
            [sequence, batch]), or as [rows, (num_streams + 1) * hidden_size] with
            rows >= sequence * batch, the tokens batch-major in the first rows and the rest
            padding that is never read (the MXFP8 lookup's padded rows, passed whole). The
            kv gradient has kv's shape, with zero pad rows.
        q_weight, k_weight: [num_streams * hidden_size] gate weights.
        token_mask: optional bool tensor broadcastable to [batch, sequence] (as in the eager
            gate); masked tokens get gate 0.
        num_streams: residual streams n.
        hidden_size: hidden size d (any positive integer).
        eps: epsilon of the two RMS normalizations.

    Returns:
        [sequence, batch, num_streams * hidden_size] tensor in ``hidden_states.dtype``.
    """
    if not HAVE_TRITON:
        raise RuntimeError("the fused Engram gate needs Triton")
    _check_inputs(hidden_states, kv, q_weight, k_weight, token_mask, num_streams, hidden_size)
    if token_mask is not None:
        # Broadcast views (stride 0) work as they are: the kernels index the mask by strides.
        token_mask = token_mask.expand(hidden_states.shape[1], hidden_states.shape[0])
    args = (hidden_states, kv, q_weight, k_weight, token_mask, num_streams, hidden_size, eps)
    if hidden_states.device.index != torch.cuda.current_device():
        # Triton launches on the current device; autograd runs backward on the tensors' one.
        with torch.cuda.device(hidden_states.device):
            return FusedEngramGate.apply(*args)
    return FusedEngramGate.apply(*args)
