# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Native ratio-two compression/indexing; owned pooling and codec VJPs."""
from types import SimpleNamespace

import torch

from .linear import linear


def metadata(rows, ratio, device):
    if ratio != 2:
        raise NotImplementedError('Deployment compressed provider currently validates ratio two')
    p = torch.arange(rows, device=device, dtype=torch.int64)
    return SimpleNamespace(
        positions=p,
        cu=torch.tensor([0, rows], device=device, dtype=torch.int32),
        requests=torch.zeros(rows, device=device, dtype=torch.int32),
        ring_slots=p.remainder(ratio),
        slots=torch.where((p + 1).remainder(ratio) == 0, p // ratio, -1),
        lengths=torch.tensor([rows // ratio], device=device, dtype=torch.int32),
        table=torch.arange((rows // ratio + 63) // 64, device=device, dtype=torch.int32)[None],
    )


def pool_vjp(raw, weight, ratio, grad, eps):
    """Fixed groups, FP32 softmax/RMS adjoint; unwritten/open rows contribute zero."""
    count = raw.shape[0] // ratio * ratio
    with torch.enable_grad():
        z = raw[:count].detach().float().requires_grad_()
        w = weight.detach().float().requires_grad_()
        values, scores = z.reshape(-1, ratio, 1024).split(512, -1)
        pooled = (values * scores.softmax(1)).sum(1)
        normed = pooled * torch.rsqrt(pooled.square().mean(-1, keepdim=True) + eps) * w
        dz, dw = torch.autograd.grad(normed, (z, w), grad[ratio - 1 : count : ratio].float())
    result = torch.zeros_like(raw)
    result[:count] = dz.to(raw.dtype)
    return result, dw.to(weight.dtype)


class _Pool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, raw, weight, ratio, eps):
        from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import (
            fused_save_compress_norm,
        )

        m = metadata(raw.shape[0], ratio, raw.device)
        ring = raw.new_zeros((1, ratio, 1024))
        latent = torch.zeros((raw.shape[0], 512), dtype=torch.bfloat16, device=raw.device)
        fused_save_compress_norm(
            raw, m.positions, ring, m.ring_slots, m.cu, m.requests, weight, eps, ratio, latent
        )
        ctx.save_for_backward(raw, weight)
        ctx.ratio, ctx.eps = ratio, eps
        return latent

    @staticmethod
    def backward(ctx, grad):
        raw, weight = ctx.saved_tensors
        dr, dw = pool_vjp(raw, weight, ctx.ratio, grad, ctx.eps)
        return dr, dw, None, None


class _Record(torch.autograd.Function):
    @staticmethod
    def forward(ctx, latent, rope, ratio):
        from vllm.models.deepseek_v41.common.ops import dequantize_and_gather_k_cache
        from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import rope_quant_insert

        m = metadata(latent.shape[0], ratio, latent.device)
        cache = torch.zeros((m.table.numel(), 64, 288), dtype=torch.uint8, device=latent.device)
        rope_quant_insert(latent, m.positions, rope, cache, m.slots, ratio)
        decoded = latent.new_empty((1, latent.shape[0] // ratio, 512))
        dequantize_and_gather_k_cache(decoded, cache, m.lengths, None, m.table, 64, 0)
        ctx.save_for_backward(m.positions[ratio - 1 :: ratio] // ratio * ratio, rope)
        ctx.rows, ctx.ratio = latent.shape[0], ratio
        return decoded[0]

    @staticmethod
    def backward(ctx, grad):
        from .attention import _rotate

        positions, rope = ctx.saved_tensors
        result = grad.new_zeros((ctx.rows, 512))
        result[ctx.ratio - 1 :: ctx.ratio] = _rotate(grad, positions, rope, True)
        return result, None, None


def index_keys(latent, wk, norm, rope, ratio, eps):
    from vllm.models.deepseek_v41.common.ops import indexer_k_norm_rope_store

    m = metadata(latent.shape[0], ratio, latent.device)
    cache = torch.zeros((m.table.numel(), 64, 68), dtype=torch.uint8, device=latent.device)
    k = linear(latent.detach(), wk, fp8=False).bfloat16()
    indexer_k_norm_rope_store(k, m.positions, rope, norm, eps, cache, m.slots, ratio, True)
    # Native indexer pages segregate values then scales, unlike the main cache.
    page = cache.flatten(1)
    count = latent.shape[0] // ratio
    values = page[:, : 64 * 64].reshape(-1, 64)[:count].contiguous()
    scales = page[:, 64 * 64 :].reshape(-1, 4)[:count].contiguous()
    return values, scales


def index_topk(x, qr, qa, weights, keys, rope, ratio, emit):
    from vllm.model_executor.kernels.batch_invariant import top_k_per_row_prefill_index_order
    from vllm.model_executor.layers.sparse_attn_indexer import as_fp32_logits
    from vllm.models.deepseek_v41.common.ops import fused_indexer_q_rope_quant
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits, mqa_logits_mx_inputs

    wq, ww = weights
    positions = torch.arange(x.shape[0], device=x.device, dtype=torch.int64)
    q = linear(qr.detach(), wq, fp8=True, activation=qa).bfloat16().reshape(-1, 32, 128)
    w = linear(x.detach(), ww, fp8=False).bfloat16()
    emit('indexer_q_projection', q)
    emit('indexer_weights_projection', w)
    quant, w = fused_indexer_q_rope_quant(
        positions, q, rope, w, 128**-0.5, 32**-0.5, use_fp4=True, weights_out_dtype=torch.float32
    )
    qvalues, qscales = quant
    emit('indexer_q_codes', qvalues)
    emit('indexer_q_scales', qscales)
    values, scales = keys
    qscales, w = mqa_logits_mx_inputs(qvalues.view(torch.int8), qscales, w)
    emit('indexer_scoring_weights', w)
    starts = torch.zeros(x.shape[0], device=x.device, dtype=torch.int32)
    ends = ((positions + 1) // ratio).int()
    logits = as_fp32_logits(
        fp8_fp4_mqa_logits(
            (qvalues.view(torch.int8), qscales),
            (values.view(torch.int8), scales.view(torch.int32).squeeze(-1)),
            w,
            starts,
            ends,
            values.shape[0],
        ),
        ends,
    )
    emit(
        'indexer_logits_valid',
        logits[torch.arange(logits.shape[1], device=x.device)[None] < ends[:, None]],
    )
    topk = torch.full((x.shape[0], 512), -1, device=x.device, dtype=torch.int32)
    top_k_per_row_prefill_index_order(
        logits, starts, ends, topk, x.shape[0], logits.stride(0), logits.stride(1), 512
    )
    return topk


def source(attention, x, qr, qa, rope, document, emit):
    """Reuse Lite's source-state lifetime and native parameter ownership."""
    core = attention.core_attention
    context, policy = core.context, core.policy
    state, layer = context.state, core.layer
    native = core.native
    ratio, eps = core.ratio, attention.config.layernorm_epsilon
    if ratio != 2 or 0 <= context.release.candidate_source_layer_id <= layer:
        raise NotImplementedError('Candidate and ratio-one deployment need separate module gates')
    geometry = (document, tuple(x.shape), x.device, x.dtype, state.grad_mode)

    def produce():
        compressor = native.compressor
        weight = torch.cat((compressor.linear_wkv.weight, compressor.linear_wgate.weight))
        raw = linear(x, weight, fp8=False)
        emit('compressor_projection', raw)
        latent = _Pool.apply(raw, compressor.norm.weight, ratio, eps)
        emit('compressor_latent', latent[ratio - 1 :: ratio])
        decoded = _Record.apply(latent, rope, ratio)
        key = native.indexer.compressor
        with torch.no_grad():
            keys = index_keys(latent, key.wk.weight, key.norm.weight, rope, ratio, eps)
        emit('indexer_k_values', keys[0])
        emit('indexer_k_scales', keys[1])
        return decoded, keys

    decoded, keys = state.resolve(layer, policy.kv, ('vllm_kv', geometry), produce)

    def select():
        indexer = native.indexer
        weights = (indexer.linear_wq_b.linear.weight, indexer.linear_weights_proj.linear.weight)
        with torch.no_grad():
            return index_topk(x, qr, qa, weights, keys, rope, ratio, emit)

    topk = state.resolve(layer, policy.index, ('vllm_index', geometry), select)
    emit('compressed_kv_decoded', decoded)
    emit('indexer_topk', topk)
    return decoded, topk
