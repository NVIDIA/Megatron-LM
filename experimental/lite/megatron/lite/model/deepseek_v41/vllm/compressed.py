# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Native compression/indexing; owned pooling and codec VJPs."""
from types import SimpleNamespace

import torch

from .linear import linear

INDEXER_LOGITS_BYTES = 512 * 1024**2


def metadata(lengths, ratio, device):
    """Packed compression metadata computed without per-document CUDA launches."""
    if ratio not in (1, 2):
        raise NotImplementedError('Deployment compression requires literal ratio one or two')
    n = torch.as_tensor(
        [lengths] if isinstance(lengths, int) else lengths, dtype=torch.int64, device='cpu'
    )
    counts = n // ratio
    starts = torch.cat((n.new_zeros(1), counts.cumsum(0)))[:-1]
    cu = torch.cat((n.new_zeros(1), n.cumsum(0)))
    rows, total = int(cu[-1]), int(counts.sum())
    doc = torch.repeat_interleave(torch.arange(n.numel()), n, output_size=rows)
    pos = torch.arange(rows) - cu[doc]
    closing = (pos + 1).remainder(ratio) == 0
    return SimpleNamespace(
        positions=pos.to(device),
        cu=cu.to(device=device, dtype=torch.int32),
        requests=doc.to(device=device, dtype=torch.int32),
        ring_slots=(doc * ratio + pos.remainder(ratio)).to(device),
        slots=torch.where(closing, starts[doc] + pos // ratio, -1).to(device),
        closers=closing.nonzero().flatten().to(device),
        closing=closing.to(device),
        lengths=torch.tensor([total], dtype=torch.int32, device=device),
        table=torch.arange((total + 63) // 64, dtype=torch.int32, device=device)[None],
        documents=n.numel(),
        counts=counts.tolist(),
        starts=starts.tolist(),
        key_starts=starts.to(device=device, dtype=torch.int32),
        query_positions=pos.to(device),
        query_requests=doc.to(device),
    )


class _BoundaryRows(torch.autograd.Function):
    """``latent[closers]`` (the mega core's input) with a sort-free backward.

    Boundary rows are unique and ascending, so the gradient fills them in order by a
    masked scatter instead of the sort-based deterministic advanced-indexing VJP."""

    @staticmethod
    def forward(ctx, latent, closers, closing):
        ctx.save_for_backward(closing)
        return latent.index_select(0, closers)

    @staticmethod
    def backward(ctx, grad):
        (closing,) = ctx.saved_tensors
        mask = closing[:, None].expand(closing.shape[0], grad.shape[1])
        return grad.new_zeros(mask.shape).masked_scatter_(mask, grad), None, None


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
        ctx.mark_non_differentiable(cache)
        return decoded[0], cache

    @staticmethod
    def backward(ctx, grad, _records):
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


def index_topk(q, w, keys, rope, ratio, emit, candidates=None, *, m=None):
    """One batched MQA/top-k call; per-row key starts retain document-local columns."""
    from vllm.model_executor.kernels.batch_invariant import top_k_per_row_prefill_index_order
    from vllm.model_executor.layers.sparse_attn_indexer import as_fp32_logits
    from vllm.models.deepseek_v41.common.ops import fused_indexer_q_rope_quant
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits, mqa_logits_mx_inputs

    observer, emit = emit, emit or (lambda *_: None)
    rows = q.shape[0]
    positions = (
        torch.arange(rows, device=q.device, dtype=torch.int64) if m is None else m.query_positions
    )
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
    starts = (
        torch.zeros(rows, device=q.device, dtype=torch.int32)
        if m is None
        else m.key_starts[m.query_requests]
    )
    row_ends = ((positions + 1) // ratio).int()
    ends = starts + row_ends
    width = values.shape[0] if m is None else max(m.counts, default=0)
    chunk_rows = max(1, INDEXER_LOGITS_BYTES // max(4 * width, 1))
    topk = torch.full((rows, 512), -1, device=q.device, dtype=torch.int32)
    diagnostics = []
    # Bound the query dimension; every row still sees its complete key interval.
    for lo in range(0, rows, chunk_rows):
        span = slice(lo, min(lo + chunk_rows, rows))
        logits = as_fp32_logits(
            fp8_fp4_mqa_logits(
                (qvalues[span].view(torch.int8), None if qscales is None else qscales[span]),
                (values.view(torch.int8), scales.view(torch.int32).squeeze(-1)),
                w[span],
                starts[span],
                ends[span],
                width,
            ),
            row_ends[span],
        )
        if observer is not None:
            diagnostics.append(
                logits[torch.arange(logits.shape[1], device=q.device)[None] < row_ends[span, None]]
            )
        zero_starts = torch.zeros_like(row_ends[span])
        if candidates is not None:
            candidates(logits, zero_starts, row_ends[span], offset=lo)
        top_k_per_row_prefill_index_order(
            logits,
            zero_starts,
            row_ends[span],
            topk[span],
            logits.shape[0],
            logits.stride(0),
            logits.stride(1),
            512,
            **_zero_starts(),
        )
        del logits
    if observer is not None:
        emit('indexer_logits_valid', torch.cat(diagnostics))
    return topk


def _zero_starts():
    """Rows start at column zero; lets vLLM's VLLM_BI_INDEXER_DEEPSELECT pick the
    same top-k kernel as rollout (older vLLM lacks the keyword)."""
    import inspect

    from vllm.model_executor.kernels.batch_invariant import top_k_per_row_prefill_index_order

    params = inspect.signature(top_k_per_row_prefill_index_order).parameters
    return {'zero_starts': True} if 'zero_starts' in params else {}


def source(attention, x, qr, qa, rope, documents, emit):
    """Compressed KV and indexer top-k of all packed documents at once.

    ``documents`` is cached global THD metadata with a local query span. Returns
    globally ordered compressed rows (decoded, cache records and the pre-RoPE latent of
    the group-boundary rows) and document-local top-k per query. Reuses Lite's
    source-state lifetime and native parameter ownership."""
    observer, emit = emit, emit or (lambda *_: None)
    core = attention.core_attention
    context, policy = core.context, core.policy
    state, layer = context.state, core.layer
    native = core.native
    ratio, eps = core.ratio, attention.config.layernorm_epsilon
    if ratio not in (1, 2):
        raise NotImplementedError('Deployment compression requires literal ratio one or two')
    from .parallel import gather_rows

    layout = documents
    lengths = tuple(layout.lengths)
    geometry = (lengths, tuple(x.shape), x.device, x.dtype, state.grad_mode, layout.offset)
    if ratio not in layout.compression:
        m = metadata(lengths, ratio, x.device)
        span = slice(layout.offset, layout.offset + x.shape[0])
        m.query_positions, m.query_requests = m.positions[span], m.requests[span].long()
        layout.compression[ratio] = m
    m = layout.compression[ratio]

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
        raw = gather_rows(linear(x, weight, fp8=False), layout.cp_group)
        emit('compressor_projection', raw)
        latent = _Pool.apply(raw, norm, ratio, eps, m)
        emit('compressor_latent', latent[m.closers])
        decoded, records = _Record.apply(latent, rope, ratio, m)
        with torch.no_grad():
            keys = index_keys(latent, wk, knorm, rope, ratio, eps, m)
        emit('indexer_k_values', keys[0])
        emit('indexer_k_scales', keys[1])
        # The mega core inserts and quantizes the boundary-row latent itself.
        return decoded, keys, records, _BoundaryRows.apply(latent, m.closers, m.closing)

    decoded, keys, records, latent = state.resolve(layer, policy.kv, ('vllm_kv', geometry), produce)

    def candidates(rows):
        release = context.release
        owner = release.candidate_source_layer_id
        if owner < 0 or layer < owner:
            return None
        from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
            apply_candidate_mask,
            select_candidate_blocks,
        )

        def apply(logits, starts, ends, *, offset=0):
            rows = logits.shape[0]

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

            key = ('vllm_candidate', (rows, offset) + geometry)
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
            return index_topk(q, w, keys, rope, ratio, observer, candidates(x.shape[0]), m=m)

    topk = state.resolve(layer, policy.index, ('vllm_index', geometry), select)
    emit('compressed_kv_decoded', decoded)
    emit('indexer_topk', topk)
    return decoded, topk, records, latent
