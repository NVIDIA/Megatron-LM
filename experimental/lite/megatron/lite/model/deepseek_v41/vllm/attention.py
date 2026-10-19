# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Caller-owned SWA using the pinned inference producers and sparse FlashMLA.

No inference engine or persistent KV state is constructed. Each THD document
owns its pages. Native checkpoint parameter names and owners remain unchanged.
"""
import torch

from .linear import linear


def _rotate(x, positions, cache, inverse=False):
    width = cache.shape[-1]
    angles = cache[positions].float()
    cos, sin = angles.chunk(2, -1)
    while cos.ndim < x.ndim:
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    if inverse:
        sin = -sin
    tail = x[..., -width:].float().unflatten(-1, (-1, 2))
    a, b = tail.unbind(-1)
    rotated = torch.stack((a * cos - b * sin, b * cos + a * sin), -1).flatten(-2)
    return torch.cat((x[..., :-width], rotated.to(x.dtype)), -1)


class _QueryInputs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, raw, qw, kw, eps):
        from vllm.model_executor.kernels.batch_invariant import rms_norm_cuda
        from vllm.models.deepseek_v41.common.ops.query_quant import fused_q_kv_rmsnorm_quant

        q, kv = raw.split((qw.numel(), kw.numel()), -1)
        qa, native_kv = fused_q_kv_rmsnorm_quant(q, kv, qw.bfloat16(), kw.bfloat16(), eps)
        qr = rms_norm_cuda(q.contiguous(), qw.bfloat16(), eps)
        ctx.save_for_backward(q, kv, qw, kw)
        ctx.eps = eps
        ctx.mark_non_differentiable(qa.data, qa.scale)
        return qr, native_kv, qa.data, qa.scale

    @staticmethod
    def backward(ctx, dq, dk, unused_codes, unused_scales):
        q, kv, qw, kw = ctx.saved_tensors

        def norm_vjp(x, w, grad):
            x = x.float()
            g = grad.float()
            inv = torch.rsqrt(x.square().mean(-1, keepdim=True) + ctx.eps)
            scaled = g * w.bfloat16().float()
            dx = inv * (scaled - x * inv.square() * (scaled * x).mean(-1, keepdim=True))
            return dx, (g * x * inv).sum(0).to(w.dtype)

        a, b = norm_vjp(q, qw, dq)
        c, d = norm_vjp(kv, kw, dk)
        return torch.cat((a, c), -1).to(q.dtype), b, d, None


class _SWAInsert(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, positions, cache, eps):
        from vllm.models.deepseek_v41.common.ops.cache_utils import dequantize_and_gather_k_cache

        rows, heads, dim = q.shape
        block = 64
        pages = (rows + block - 1) // block
        table = torch.arange(pages, dtype=torch.int32, device=q.device)[None]
        slots = torch.arange(rows, dtype=torch.int64, device=q.device)
        packed = torch.zeros((pages, block, 528), dtype=torch.uint8, device=q.device)
        padded = 64 if heads <= 64 else 128
        rotated = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            q.contiguous(),
            kv.contiguous(),
            packed.view(pages, -1),
            slots,
            positions,
            cache,
            padded,
            eps,
            block,
            False,
            True,
            True,
            False,
        )
        decoded = kv.new_empty((1, rows, dim))
        lengths = torch.tensor([rows], dtype=torch.int32, device=q.device)
        dequantize_and_gather_k_cache(decoded, packed, lengths, None, table, block, 0)
        ctx.save_for_backward(positions, cache)
        ctx.heads = heads
        return rotated[:, :heads], decoded[0]

    @staticmethod
    def backward(ctx, dq, dkv):
        positions, cache = ctx.saved_tensors
        return (
            _rotate(dq, positions, cache, True),
            _rotate(dkv, positions, cache, True),
            None,
            None,
            None,
        )


class _Sparse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, sink, indices, lengths, scale):
        from vllm.v1.attention.ops.flashmla import flash_mla_sparse_fwd

        heads = q.shape[1]
        padded = 64 if heads <= 64 else 128
        query = torch.nn.functional.pad(q, (0, 0, 0, padded - heads)).contiguous()
        bias = torch.nn.functional.pad(sink.float(), (0, padded - heads), value=-torch.inf)
        output = torch.empty_like(query)
        flash_mla_sparse_fwd(
            q=query,
            kv=kv[:, None].contiguous(),
            indices=indices[:, None],
            sm_scale=scale,
            attn_sink=bias,
            topk_length=lengths,
            out=output,
        )
        ctx.save_for_backward(q, kv, sink, indices)
        ctx.scale = scale
        return output[:, :heads]

    @staticmethod
    def backward(ctx, grad):
        q, kv, sink, indices = ctx.saved_tensors
        # Fixed-slot VJP: FP32 accumulated shared K/V gradients, rounded once.
        dq = torch.empty_like(q)
        dkv = torch.zeros_like(kv, dtype=torch.float32)
        dsink = torch.zeros_like(sink, dtype=torch.float32)
        with torch.enable_grad():
            values = kv.detach().float().requires_grad_()
            bias = sink.detach().float().requires_grad_()
            for lo in range(0, q.shape[0], 32):
                hi = min(lo + 32, q.shape[0])
                query = q[lo:hi].detach().float().requires_grad_()
                selected = indices[lo:hi]
                gathered = values[selected.clamp_min(0).long()]
                logits = torch.einsum('shd,std->sht', query, gathered) * ctx.scale
                logits = logits.masked_fill((selected < 0)[:, None], -torch.inf)
                logits = torch.cat((logits, bias[None, :, None].expand(hi - lo, -1, 1)), -1)
                prob = logits.softmax(-1)[..., :-1]
                out = torch.einsum('sht,std->shd', prob, gathered)
                a, b, c = torch.autograd.grad(out, (query, values, bias), grad[lo:hi].float())
                dq[lo:hi], dkv, dsink = a, dkv + b, dsink + c
        return dq, dkv.to(kv.dtype), dsink.to(sink.dtype), None, None, None


class _InverseRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, positions, cache, groups):
        from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
            fused_inv_rope_fp8_quant,
        )

        ctx.save_for_backward(positions, cache)
        ctx.shape = x.shape
        out, _ = fused_inv_rope_fp8_quant(
            x,
            positions,
            cache,
            n_groups=groups,
            heads_per_group=x.shape[1] // groups,
            nope_dim=x.shape[-1] - cache.shape[-1],
            rope_dim=cache.shape[-1],
            quant_group_size=128,
            tma_aligned_scales=True,
            quantize=False,
        )
        return out

    @staticmethod
    def backward(ctx, grad):
        positions, cache = ctx.saved_tensors
        return _rotate(grad.reshape(ctx.shape), positions, cache), None, None, None


def swa_forward(attention, hidden, positions, rope_cache, *, observe=None):
    """One complete ratio-zero document, including native weight quantization."""
    emit = observe or (lambda name, value: None)
    x = hidden.reshape(-1, hidden.shape[-1])
    fused = torch.cat((attention.linear_q_down_proj.weight, attention.linear_kv_proj.weight))
    raw = linear(x, fused, fp8=True)
    emit('qkv_projection', raw)
    from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic

    qr, kv, codes, scales = _QueryInputs.apply(
        raw,
        attention.q_layernorm.weight,
        attention.kv_layernorm.weight,
        attention.config.layernorm_epsilon,
    )
    qa = QuantizedActivation(codes, scales, qr.dtype, qr.shape, kMxfp8Dynamic)
    emit('q_norm', qr)
    emit('kv_norm', kv)
    q = linear(qr, attention.linear_q_up_proj.weight, fp8=True, activation=qa).bfloat16()
    q = q.reshape(-1, attention.num_attention_heads_per_partition, attention.q_head_dim)
    emit('q_projection', q)
    q, kv = _SWAInsert.apply(q, kv, positions, rope_cache, attention.config.layernorm_epsilon)
    emit('q_rope', q)
    emit('kv_cache_decoded', kv)
    # Use the same native metadata producer, including padding and slot order.
    from vllm.models.deepseek_v41.common.ops import combine_topk_swa_indices

    rows = x.shape[0]
    cu = torch.tensor([0, rows], device=x.device, dtype=torch.int32)
    lengths = torch.tensor([rows], device=x.device, dtype=torch.int32)
    topk = torch.empty((rows, 0), device=x.device, dtype=torch.int32)
    indices, lens = combine_topk_swa_indices(
        topk, cu, lengths, lengths, attention.config.csa_window_size, 0, 0, rows, 0
    )
    emit('swa_indices', indices)
    sink = attention.core_attention.native.attn_sink
    out = _Sparse.apply(q, kv, sink, indices, lens, attention.q_head_dim**-0.5)
    emit('core_attn', out)
    weight = attention.linear_o_group_proj
    groups = attention.config.o_groups
    unrotated = _InverseRoPE.apply(out, positions, rope_cache, groups)
    emit('inverse_rope', unrotated)
    weight = weight.reshape(groups, attention.config.o_lora_rank, -1)
    z = torch.stack(
        [
            linear(unrotated[:, g].contiguous(), weight[g], fp8=False).bfloat16()
            for g in range(groups)
        ],
        1,
    )
    emit('wo_a', z)
    result = linear(z.flatten(1), attention.linear_proj.weight, fp8=True).bfloat16()
    emit('wo_b', result)
    return result.reshape_as(hidden)
