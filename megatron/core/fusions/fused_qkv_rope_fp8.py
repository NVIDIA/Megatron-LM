# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Fused RoPE + FP8 current-scaling quantization of the attention inputs.

Takes the QKV GEMM output ``[s, b, g, (hq + 2) * d]`` (bf16) and returns the rotated Q, K and V as
TransformerEngine ``Float8Tensor`` s sharing one current-scaling scale, ready to be passed to
TE ``DotProductAttention`` as FP8 inputs. Two Triton passes are needed because current scaling
needs the global amax before the first cast:

* amax pass: persistent, rotates Q/K (rounded to bf16 like TE's RoPE output) and reduces
  amax(|Q|, |K|, |V|) with one atomic per CTA; the last CTA turns the amax into the scale.
* cast pass: rotates again and writes e4m3 values, visiting tokens in reverse order so the tail
  of the amax pass is still in L2.

Forward results are bitwise identical to TE's fused QKV RoPE followed by current-scaling
quantization. The backward is TE's fused QKV RoPE backward applied to the gradients returned by
attention in bf16.
"""

from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import torch
from packaging import version

from megatron.core.utils import null_decorator

try:
    import triton
    import triton.language as tl

    if version.parse(triton.__version__) < version.parse("3.4.0") and not torch.cuda.is_available():
        HAVE_TRITON = False
    else:
        HAVE_TRITON = tl.constexpr(version.parse(triton.__version__) >= version.parse("2.0.0"))
except ImportError:
    HAVE_TRITON = False

if not HAVE_TRITON:
    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()
else:
    try:
        from triton.language.extra import libdevice
    except ImportError:  # older Triton
        from triton.language.extra.cuda import libdevice

try:
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.cpp_extensions.fused_attn import QKVFormat
    from transformer_engine.pytorch.quantized_tensor import QuantizedTensor
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

_HEAD_DIM = 128
_FP8_E4M3_MAX = 448.0
_AMAX_WARPS = 2
_CAST_WARPS = 4
_AMAX_STAGES = 3
_CTAS_PER_SM = 4


@triton.jit
def _finalize_scale(ws_ptr, scale_inv_ptr, n_prog, eps, FP8MAX: tl.constexpr, POW2: tl.constexpr):
    """Run by every CTA after its amax atomic; the last one turns amax into the scale (TE's
    compute_scale_from_amax) and hands the workspace [amax, counter, scale] back zeroed."""
    cnt = tl.atomic_add(ws_ptr + 1, 1.0)
    if cnt == n_prog - 1:
        amax = tl.atomic_xchg(ws_ptr, 0.0)
        tl.atomic_xchg(ws_ptr + 1, 0.0)
        bad = (amax == float("inf")) | (amax != amax)
        amax = tl.where(amax < eps, eps, amax)
        bad = bad | (amax == 0.0)
        scale = libdevice.div_rn(FP8MAX, amax)
        scale = tl.where(scale == float("inf"), 3.4028234663852886e38, scale)
        if POW2:
            scale = (scale.to(tl.uint32, bitcast=True) & 0xFF800000).to(tl.float32, bitcast=True)
        scale = tl.where(bad, 1.0, scale)
        tl.store(ws_ptr + 2, scale)
        tl.store(scale_inv_ptr, libdevice.rcp_rn(scale))


@triton.jit
def _trig_table_kernel(f_ptr, cos_ptr, sin_ptr, n, BLOCK: tl.constexpr):
    """cos/sin with the libdevice functions TE's sincosf resolves to, once per angle."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    f = tl.load(f_ptr + offs, mask=m, other=0.0)
    tl.store(cos_ptr + offs, libdevice.cos(f), mask=m)
    tl.store(sin_ptr + offs, libdevice.sin(f), mask=m)


@triton.jit
def _rot(x, c, xr, s):
    # fma(x, c, xr * s): matches nvcc's contraction in TE's RoPE kernels
    return tl.fma(x, c, xr * s)


@triton.jit
def _rot_fwd(lo, hi, c_lo, s_lo, c_hi, s_hi):
    # TE fused_qkv_rope_block_forward: out[d<h] = x[d]cos[d] + (-x[d+h])sin[d];
    #                                  out[d>=h] = x[d]cos[d] + x[d-h]sin[d]; bf16 output
    y_lo = _rot(lo, c_lo, -hi, s_lo).to(tl.bfloat16).to(tl.float32)
    y_hi = _rot(hi, c_hi, lo, s_hi).to(tl.bfloat16).to(tl.float32)
    return y_lo, y_hi


@triton.jit
def _rope_qkv_token(
    x_ptr,
    cos_ptr,
    sin_ptr,
    q_out,
    k_out,
    v_out,
    sc,
    tok,
    B: tl.constexpr,
    G: tl.constexpr,
    HQ: tl.constexpr,
    D: tl.constexpr,
    CAST: tl.constexpr,
):
    """Rotate Q/K of one (s, b) token for all heads. CAST=False returns the token's amax,
    CAST=True stores the scaled FP8 values."""
    W: tl.constexpr = (HQ + 2) * D
    HD: tl.constexpr = D // 2
    pos = tok // B
    g = tl.arange(0, G)
    jq = tl.arange(0, HQ)
    dl = tl.arange(0, HD)
    d = tl.arange(0, D)
    tb = pos * D + dl
    c_lo = tl.load(cos_ptr + tb)
    s_lo = tl.load(sin_ptr + tb)
    c_hi = tl.load(cos_ptr + tb + HD)
    s_hi = tl.load(sin_ptr + tb + HD)
    # Q: [g, jq, dl]; K: [g, dl]; V: [g, d]
    row = tok * (G * W) + g * W
    q_src = x_ptr + row[:, None, None] + jq[None, :, None] * D + dl[None, None, :]
    k_src = x_ptr + row[:, None] + HQ * D + dl[None, :]
    v_src = x_ptr + row[:, None] + (HQ + 1) * D + d[None, :]
    q_lo, q_hi = _rot_fwd(
        tl.load(q_src).to(tl.float32),
        tl.load(q_src + HD).to(tl.float32),
        c_lo[None, None, :],
        s_lo[None, None, :],
        c_hi[None, None, :],
        s_hi[None, None, :],
    )
    k_lo, k_hi = _rot_fwd(
        tl.load(k_src).to(tl.float32),
        tl.load(k_src + HD).to(tl.float32),
        c_lo[None, :],
        s_lo[None, :],
        c_hi[None, :],
        s_hi[None, :],
    )
    v = tl.load(v_src).to(tl.float32)
    if CAST:
        q_dst = (
            q_out
            + tok * (G * HQ * D)
            + g[:, None, None] * (HQ * D)
            + jq[None, :, None] * D
            + dl[None, None, :]
        )
        k_dst = k_out + tok * (G * D) + g[:, None] * D + dl[None, :]
        v_dst = v_out + tok * (G * D) + g[:, None] * D + d[None, :]
        ety = q_out.dtype.element_ty
        tl.store(q_dst, (q_lo * sc).to(ety))
        tl.store(q_dst + HD, (q_hi * sc).to(ety))
        tl.store(k_dst, (k_lo * sc).to(ety))
        tl.store(k_dst + HD, (k_hi * sc).to(ety))
        tl.store(v_dst, (v * sc).to(ety))
        return 0.0
    else:
        aq = tl.max(tl.max(tl.max(tl.maximum(tl.abs(q_lo), tl.abs(q_hi)), 2), 1), 0)
        ak = tl.max(tl.max(tl.maximum(tl.abs(k_lo), tl.abs(k_hi)), 1), 0)
        av = tl.max(tl.max(tl.abs(v), 1), 0)
        return tl.maximum(tl.maximum(aq, ak), av)


@triton.jit
def _rope_qkv_quantize_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    q_out,
    k_out,
    v_out,
    ws_ptr,
    scale_inv_ptr,
    n_tok,
    eps,
    B: tl.constexpr,
    G: tl.constexpr,
    HQ: tl.constexpr,
    D: tl.constexpr,
    CAST: tl.constexpr,
    FP8MAX: tl.constexpr,
    POW2: tl.constexpr,
    NUM_STAGES: tl.constexpr = 1,
):
    """CAST=False: persistent amax pass (one atomic per CTA, last CTA computes the scale).
    CAST=True: one token per CTA, in the reverse order of the amax pass."""
    n_prog = tl.num_programs(0)
    pid = tl.program_id(0)
    if CAST:
        sc = tl.load(ws_ptr + 2)
        _rope_qkv_token(
            x_ptr, cos_ptr, sin_ptr, q_out, k_out, v_out, sc, n_prog - 1 - pid, B, G, HQ, D, True
        )
    else:
        amax = 0.0
        for tok in tl.range(pid, n_tok, n_prog, num_stages=NUM_STAGES):
            a = _rope_qkv_token(
                x_ptr, cos_ptr, sin_ptr, q_out, k_out, v_out, 1.0, tok, B, G, HQ, D, False
            )
            amax = tl.maximum(amax, a)
        tl.atomic_max(ws_ptr, amax)
        _finalize_scale(ws_ptr, scale_inv_ptr, n_prog, eps, FP8MAX, POW2)


# One [amax, counter, scale, pad] workspace per (device, stream): the last CTA of every launch
# re-zeroes it, but launches on different streams must not share one.
_WORKSPACES: Dict[Tuple[int, int], torch.Tensor] = {}
_NUM_SMS: Dict[int, int] = {}
# (freqs, version, cos, sin) per device; holding freqs keeps its address from being reused.
_TRIG_CACHE: Dict[int, Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]] = {}


def _workspace(device: torch.device) -> torch.Tensor:
    key = (device.index, torch.cuda.current_stream(device).cuda_stream)
    ws = _WORKSPACES.get(key)
    if ws is None:
        ws = _WORKSPACES[key] = torch.zeros(4, dtype=torch.float32, device=device)
    return ws


def _trig_tables(freqs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """cos/sin of the (contiguous fp32) RoPE angles; all layers of a step share one tensor."""
    ent = _TRIG_CACHE.get(freqs.device.index)
    if ent is not None and ent[0] is freqs and ent[1] == freqs._version:
        return ent[2], ent[3]
    cos = torch.empty_like(freqs)
    sin = torch.empty_like(freqs)
    n = freqs.numel()
    _trig_table_kernel[(triton.cdiv(n, 1024),)](freqs, cos, sin, n, BLOCK=1024, num_warps=4)
    _TRIG_CACHE[freqs.device.index] = (freqs, freqs._version, cos, sin)
    return cos, sin


def _rope_freqs(
    mixed_qkv: torch.Tensor,
    q_freqs: torch.Tensor,
    k_freqs: torch.Tensor,
    split: List[int],
) -> Optional[torch.Tensor]:
    """The contiguous fp32 angles if the fused kernel supports these inputs, else None."""
    if not (
        HAVE_TRITON
        and HAVE_TE
        and len(split) == 3
        and split[1] == split[2] == _HEAD_DIM
        and split[0] % split[1] == 0
        and mixed_qkv.dim() == 4
        and mixed_qkv.is_cuda
        and mixed_qkv.dtype == torch.bfloat16
        and mixed_qkv.is_contiguous()
        and mixed_qkv.shape[3] == sum(split)
        and mixed_qkv.numel() < 2**31
        and q_freqs.data_ptr() == k_freqs.data_ptr()
        and q_freqs.shape == k_freqs.shape
        and q_freqs.stride() == k_freqs.stride()
        and q_freqs.dim() == 4
        and q_freqs.shape[1] == 1
        and q_freqs.shape[2] == 1
        and q_freqs.shape[3] == split[1]
        and q_freqs.shape[0] >= mixed_qkv.shape[0]
        and q_freqs.is_cuda
    ):
        return None
    freqs = q_freqs if q_freqs.dtype == torch.float32 else q_freqs.float()
    return freqs if freqs.is_contiguous() else freqs.contiguous()


class _FusedQKVRoPEFP8(torch.autograd.Function):
    """RoPE + e4m3 current-scaling quantization of Q/K/V; backward is TE's fused RoPE backward."""

    @staticmethod
    def forward(ctx, mixed_qkv, freqs, split, quantizer):
        s, b, g, _ = mixed_qkv.shape
        d = split[1]
        hq = split[0] // d
        n_tok = s * b
        q_shape, kv_shape = (s, b, g * hq, d), (s, b, g, d)

        q_fp8 = quantizer.make_empty(q_shape, dtype=mixed_qkv.dtype, device=mixed_qkv.device)
        # K and V get their own buffers (not views of one payload) so TE infers the plain
        # sbhd_sbhd_sbhd layout from their pointers; the scale tensor is shared with Q.
        kv_data = [
            torch.empty(kv_shape, dtype=torch.uint8, device=mixed_qkv.device) for _ in range(2)
        ]
        k_fp8 = Float8Tensor.make_like(q_fp8, data=kv_data[0], dtype=mixed_qkv.dtype)
        v_fp8 = Float8Tensor.make_like(q_fp8, data=kv_data[1], dtype=mixed_qkv.dtype)

        cos, sin = _trig_tables(freqs)
        device = mixed_qkv.device
        if device.index not in _NUM_SMS:
            _NUM_SMS[device.index] = torch.cuda.get_device_properties(device).multi_processor_count
        ws = _workspace(device)
        fp8 = torch.float8_e4m3fn
        common = dict(
            B=b,
            G=g,
            HQ=hq,
            D=d,
            FP8MAX=_FP8_E4M3_MAX,
            POW2=bool(quantizer.force_pow_2_scales),
        )
        args = (
            mixed_qkv,
            cos,
            sin,
            q_fp8._data.view(fp8),
            kv_data[0].view(fp8),
            kv_data[1].view(fp8),
            ws,
            q_fp8._scale_inv,
            n_tok,
            float(quantizer.amax_epsilon),
        )
        _rope_qkv_quantize_kernel[(min(n_tok, _NUM_SMS[device.index] * _CTAS_PER_SM),)](
            *args, CAST=False, NUM_STAGES=_AMAX_STAGES, num_warps=_AMAX_WARPS, **common
        )
        _rope_qkv_quantize_kernel[(n_tok,)](*args, CAST=True, num_warps=_CAST_WARPS, **common)

        ctx.freqs = freqs
        ctx.split = split
        return q_fp8, k_fp8, v_fp8

    @staticmethod
    def backward(ctx, grad_q, grad_k, grad_v):
        # torch.Tensor also has a dequantize() (it upcasts to fp32), so test for TE's type
        grads = [
            g.dequantize(dtype=torch.bfloat16) if isinstance(g, QuantizedTensor) else g
            for g in (grad_q, grad_k, grad_v)
        ]
        grad_input = tex.fused_qkv_rope_backward(
            *[g.contiguous() for g in grads],
            ctx.freqs,
            ctx.freqs,
            ctx.split,
            QKVFormat["sbhd"],
            False,
            1,
            0,
        )
        return grad_input, None, None, None


def fused_qkv_rope_fp8_quantize(
    mixed_qkv: torch.Tensor,
    q_freqs: torch.Tensor,
    k_freqs: torch.Tensor,
    qkv_split_arg_list: List[int],
    quantizer,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Rotate and FP8-quantize the QKV GEMM output, or return None for unsupported inputs.

    Args:
        mixed_qkv: bf16 ``[s, b, g, (hq + 2) * d]`` QKV GEMM output (``d == 128``).
        q_freqs, k_freqs: RoPE angles ``[>= s, 1, 1, d]``; must be the same tensor.
        qkv_split_arg_list: ``[hq * d, d, d]``.
        quantizer: TE rowwise-only e4m3 ``Float8CurrentScalingQuantizer`` without amax reduction.

    Returns:
        ``Float8Tensor`` Q ``[s, b, g * hq, d]``, K and V ``[s, b, g, d]`` with a shared scale.
        Their gradients are expected in bf16 (attention with ``bf16_backward=True``).
    """
    if quantizer.columnwise_usage or not quantizer.rowwise_usage:
        return None
    freqs = _rope_freqs(mixed_qkv, q_freqs, k_freqs, list(qkv_split_arg_list))
    if freqs is None:
        return None
    return _FusedQKVRoPEFP8.apply(mixed_qkv, freqs, list(qkv_split_arg_list), quantizer)
