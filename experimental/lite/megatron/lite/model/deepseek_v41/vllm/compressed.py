# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Native compression/indexing; owned pooling and codec VJPs."""
from types import SimpleNamespace

import torch

from .linear import linear


def metadata(lengths, ratio, device):
    """Packed THD documents (an int is one document) as independent vLLM requests.

    Each document's compressed rows are contiguous and in document order, so
    every request reads exactly the rows it would own alone."""
    if ratio not in (1, 2):
        raise NotImplementedError('Deployment compression requires literal ratio one or two')
    lengths = [lengths] if isinstance(lengths, int) else [int(n) for n in lengths]
    counts = [n // ratio for n in lengths]
    starts = [sum(counts[:d]) for d in range(len(counts))]
    total = sum(counts)
    p = torch.cat(
        [torch.zeros(0, dtype=torch.int64)] + [torch.arange(n, dtype=torch.int64) for n in lengths]
    ).to(device)
    doc = torch.cat(
        [torch.zeros(0, dtype=torch.int32)]
        + [torch.full((n,), d, dtype=torch.int32) for d, n in enumerate(lengths)]
    ).to(device)
    closing = (p + 1).remainder(ratio) == 0
    first = torch.tensor(starts + [0], dtype=torch.int64, device=device)[doc.long()]
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    return SimpleNamespace(
        positions=p,
        cu=torch.tensor(cu, device=device, dtype=torch.int32),
        requests=doc,
        ring_slots=doc.long() * ratio + p.remainder(ratio),
        slots=torch.where(closing, first + p // ratio, -1),
        closers=closing.nonzero().flatten(),
        lengths=torch.tensor([total], device=device, dtype=torch.int32),
        table=torch.arange((total + 63) // 64, device=device, dtype=torch.int32)[None],
        documents=len(lengths),
        counts=counts,
        starts=starts,
    )


def pool_vjp(raw, weight, ratio, grad, eps, closers=None):
    """Fixed groups ending at ``closers`` (default: one document from position 0),
    FP32 softmax/RMS adjoint; unwritten/open rows contribute zero."""
    if closers is None:
        closers = torch.arange(raw.shape[0], device=raw.device)[ratio - 1 :: ratio]
    with torch.enable_grad():
        z = raw.detach().float().requires_grad_()
        w = weight.detach().float().requires_grad_()
        if ratio == 1:
            pooled = z[closers]
        else:
            values = torch.stack((z[closers - 1, :512], z[closers, :512]), 1)
            scores = torch.stack((z[closers - 1, 512:], z[closers, 512:]), 1)
            pooled = (values * scores.softmax(1)).sum(1)
        normed = pooled * torch.rsqrt(pooled.square().mean(-1, keepdim=True) + eps) * w
        dz, dw = torch.autograd.grad(normed, (z, w), grad[closers].float())
    return dz.to(raw.dtype), dw.to(weight.dtype)


class _Pool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, raw, weight, ratio, eps, m):
        from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import (
            fused_save_compress_norm,
        )

        ring = raw.new_zeros((max(m.documents, 1), ratio, 1024)) if ratio == 2 else None
        latent = torch.zeros((raw.shape[0], 512), dtype=torch.bfloat16, device=raw.device)
        fused_save_compress_norm(
            raw,
            m.positions,
            ring,
            m.ring_slots if ratio == 2 else m.slots,
            m.cu,
            m.requests,
            weight,
            eps,
            ratio,
            latent,
        )
        ctx.save_for_backward(raw, weight, m.closers)
        ctx.ratio, ctx.eps = ratio, eps
        return latent

    @staticmethod
    def backward(ctx, grad):
        raw, weight, closers = ctx.saved_tensors
        dr, dw = pool_vjp(raw, weight, ctx.ratio, grad, ctx.eps, closers)
        return dr, dw, None, None, None


class _Record(torch.autograd.Function):
    @staticmethod
    def forward(ctx, latent, rope, ratio, m):
        from vllm.models.deepseek_v41.common.ops import dequantize_and_gather_k_cache
        from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import rope_quant_insert

        cache = torch.zeros((m.table.numel(), 64, 288), dtype=torch.uint8, device=latent.device)
        rope_quant_insert(latent, m.positions, rope, cache, m.slots, ratio)
        decoded = latent.new_empty((1, sum(m.counts), 512))
        dequantize_and_gather_k_cache(decoded, cache, m.lengths, None, m.table, 64, 0)
        ctx.save_for_backward(m.positions[m.closers] // ratio * ratio, rope, m.closers)
        ctx.rows = latent.shape[0]
        return decoded[0]

    @staticmethod
    def backward(ctx, grad):
        from .attention import _rotate

        positions, rope, closers = ctx.saved_tensors
        result = grad.new_zeros((ctx.rows, 512))
        result[closers] = _rotate(grad, positions, rope, True)
        return result, None, None, None


def index_keys(latent, wk, norm, rope, ratio, eps, m):
    from vllm.models.deepseek_v41.common.ops import indexer_k_norm_rope_store

    cache = torch.zeros((m.table.numel(), 64, 68), dtype=torch.uint8, device=latent.device)
    k = linear(latent.detach(), wk, fp8=False).bfloat16()
    indexer_k_norm_rope_store(k, m.positions, rope, norm, eps, cache, m.slots, ratio, True)
    # Native indexer pages segregate values then scales, unlike the main cache.
    page = cache.flatten(1)
    count = sum(m.counts)
    values = page[:, : 64 * 64].reshape(-1, 64)[:count].contiguous()
    scales = page[:, 64 * 64 :].reshape(-1, 4)[:count].contiguous()
    return values, scales


def index_topk(q, w, keys, rope, ratio, emit, candidates=None):
    """One document's indexer top-k from its projected rows (``q``/``w`` are row-wise
    GEMM outputs, so slices of a batched projection are the document's own bits)."""
    from vllm.model_executor.kernels.batch_invariant import top_k_per_row_prefill_index_order
    from vllm.model_executor.layers.sparse_attn_indexer import as_fp32_logits
    from vllm.models.deepseek_v41.common.ops import fused_indexer_q_rope_quant
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits, mqa_logits_mx_inputs

    rows = q.shape[0]
    positions = torch.arange(rows, device=q.device, dtype=torch.int64)
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
    starts = torch.zeros(rows, device=q.device, dtype=torch.int32)
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
        logits[torch.arange(logits.shape[1], device=q.device)[None] < ends[:, None]],
    )
    if candidates is not None:
        candidates(logits, starts, ends)
    topk = torch.full((rows, 512), -1, device=q.device, dtype=torch.int32)
    top_k_per_row_prefill_index_order(
        logits, starts, ends, topk, rows, logits.stride(0), logits.stride(1), 512
    )
    return topk


def source(attention, x, qr, qa, rope, documents, emit):
    """Compressed KV and indexer top-k of all packed documents at once.

    ``documents``: ``[(lo, hi), ...]`` THD row ranges. Returns the decoded compressed
    rows of every document in order and document-local top-k per row. Reuses Lite's
    source-state lifetime and native parameter ownership."""
    core = attention.core_attention
    context, policy = core.context, core.policy
    state, layer = context.state, core.layer
    native = core.native
    ratio, eps = core.ratio, attention.config.layernorm_epsilon
    if ratio not in (1, 2):
        raise NotImplementedError('Deployment compression requires literal ratio one or two')
    lengths = tuple(hi - lo for lo, hi in documents)
    geometry = (lengths, tuple(x.shape), x.device, x.dtype, state.grad_mode)
    m = metadata(lengths, ratio, x.device)

    def produce():
        if ratio == 1:
            weight, norm = (core.main_projection.weight,), core.main_norm.weight
            wk, knorm = core.index_key[0].weight, core.index_key[1].weight
        else:
            compressor = native.compressor
            weight = (compressor.linear_wkv.weight, compressor.linear_wgate.weight)
            norm = compressor.norm.weight
            key = native.indexer.compressor
            wk, knorm = key.wk.weight, key.norm.weight
        raw = linear(x, weight, fp8=False)
        emit('compressor_projection', raw)
        latent = _Pool.apply(raw, norm, ratio, eps, m)
        emit('compressor_latent', latent[m.closers])
        decoded = _Record.apply(latent, rope, ratio, m)
        with torch.no_grad():
            keys = index_keys(latent, wk, knorm, rope, ratio, eps, m)
        emit('indexer_k_values', keys[0])
        emit('indexer_k_scales', keys[1])
        return decoded, keys

    decoded, keys = state.resolve(layer, policy.kv, ('vllm_kv', geometry), produce)

    def candidates(document, rows):
        release = context.release
        owner = release.candidate_source_layer_id
        if owner < 0 or layer < owner:
            return None
        from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
            apply_candidate_mask,
            select_candidate_blocks,
        )

        def apply(logits, starts, ends):
            def publish():
                out = torch.full(
                    (rows, release.candidate_topk_blocks), -1, dtype=torch.int32, device=x.device
                )
                select_candidate_blocks(
                    logits,
                    starts,
                    ends,
                    release.candidate_topk_blocks,
                    release.candidate_block_size,
                    out,
                )
                return out

            key = ('vllm_candidate', (document, rows) + geometry)
            blocks = state.resolve(layer, owner, key, publish)
            if layer != owner:
                apply_candidate_mask(logits, starts, ends, blocks, release.candidate_block_size)
            emit('candidate_blocks', blocks)

        return apply

    def select():
        if ratio == 1:
            weights = (core.index_q.linear.weight, core.index_weights.linear.weight)
        else:
            indexer = native.indexer
            weights = (indexer.linear_wq_b.linear.weight, indexer.linear_weights_proj.linear.weight)
        with torch.no_grad():
            q = linear(qr.detach(), weights[0], fp8=True, activation=qa).bfloat16()
            q = q.reshape(-1, 32, 128)
            w = linear(x.detach(), weights[1], fp8=False).bfloat16()
            values, scales = keys
            pieces = [torch.full((0, 512), -1, dtype=torch.int32, device=x.device)]
            for document, (lo, hi) in enumerate(documents):
                if hi > lo:
                    a, b = m.starts[document], m.starts[document] + m.counts[document]
                    pieces.append(
                        index_topk(
                            q[lo:hi],
                            w[lo:hi],
                            # DeepGEMM needs 16-byte aligned operands; copy the slice.
                            (values[a:b].clone(), scales[a:b].clone()),
                            rope,
                            ratio,
                            emit,
                            candidates(document, hi - lo),
                        )
                    )
            return torch.cat(pieces)

    topk = state.resolve(layer, policy.index, ('vllm_index', geometry), select)
    emit('compressed_kv_decoded', decoded)
    emit('indexer_topk', topk)
    return decoded, topk
