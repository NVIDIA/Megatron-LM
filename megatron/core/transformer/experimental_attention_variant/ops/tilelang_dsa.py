# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""TileLang-backed DSA hook implementations.

This module keeps TileLang-specific batching, chunking, and sparse-KL streaming
out of the backend-neutral DSA control flow in ``dsa.py``.
"""

from collections import OrderedDict
from typing import Optional, Tuple

import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.experimental_attention_variant import dsa_layout, dsa_masking

try:
    from megatron.core.transformer.experimental_attention_variant.ops.indexer import (
        lighting_indexer,
    )
except (ImportError, OSError):
    lighting_indexer = None

try:
    from megatron.core.transformer.experimental_attention_variant.ops.sparse_mla import SparseMLA
except (ImportError, OSError):
    SparseMLA = None


# Reusable no-grad scratch buffers keyed by (name, shape, dtype, device).
_DSA_SCRATCH_CACHE_MAX_ENTRIES = 128
_DSA_SCRATCH_CACHE_MAX_BYTES = 512 * 1024 * 1024
_DSA_SCRATCH_CACHE = OrderedDict()
_DSA_SCRATCH_CACHE_TOTAL_BYTES = 0


def _scratch_buffer_bytes(buf: torch.Tensor) -> int:
    return buf.numel() * buf.element_size()


def _scratch_cache_total_bytes() -> int:
    """Return total bytes held by cached scratch tensors."""
    return _DSA_SCRATCH_CACHE_TOTAL_BYTES


def _evict_scratch_cache_if_needed() -> None:
    """Bound scratch cache growth by LRU eviction."""
    global _DSA_SCRATCH_CACHE_TOTAL_BYTES
    while (
        len(_DSA_SCRATCH_CACHE) > _DSA_SCRATCH_CACHE_MAX_ENTRIES
        or _DSA_SCRATCH_CACHE_TOTAL_BYTES > _DSA_SCRATCH_CACHE_MAX_BYTES
    ):
        _, buf = _DSA_SCRATCH_CACHE.popitem(last=False)
        _DSA_SCRATCH_CACHE_TOTAL_BYTES -= _scratch_buffer_bytes(buf)


def _get_scratch_buffer(
    name: str, shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Get a reusable scratch tensor for temporary no-grad workspaces."""
    global _DSA_SCRATCH_CACHE_TOTAL_BYTES
    key = (name, shape, dtype, device)
    buf = _DSA_SCRATCH_CACHE.pop(key, None)
    if buf is not None:
        _DSA_SCRATCH_CACHE_TOTAL_BYTES -= _scratch_buffer_bytes(buf)
    else:
        buf = torch.empty(shape, dtype=dtype, device=device)
    _DSA_SCRATCH_CACHE[key] = buf
    _DSA_SCRATCH_CACHE_TOTAL_BYTES += _scratch_buffer_bytes(buf)
    _evict_scratch_cache_if_needed()
    return buf


def _sanitize_fused_topk_outputs(
    topk_indices: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    topk_scores: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Mask fused indexer outputs to valid row-wise [start, end) key bounds."""
    idx_i64 = topk_indices.to(dtype=torch.int64)
    starts_i64 = starts.to(device=idx_i64.device, dtype=torch.int64).unsqueeze(-1)
    ends_i64 = ends.to(device=idx_i64.device, dtype=torch.int64).unsqueeze(-1)
    valid = (idx_i64 >= starts_i64) & (idx_i64 < ends_i64)
    sanitized_indices = idx_i64.masked_fill(~valid, -1)
    if topk_indices.dtype != torch.int64:
        sanitized_indices = sanitized_indices.to(dtype=topk_indices.dtype)
    if topk_scores is not None:
        topk_scores = topk_scores.masked_fill(~valid, float("-inf"))
    return sanitized_indices, topk_scores


def fused_qk_topk_lighting(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    index_topk: int,
    starts: torch.Tensor,
    ends: torch.Tensor,
    block_size: int,
    use_relu: bool = True,
) -> Optional[torch.Tensor]:
    """Run fused TileLang indexer and return top-k indices [b, sq, topk]."""
    if lighting_indexer is None:
        return None
    if q.ndim != 4 or k.ndim != 3 or weights.ndim != 3:
        return None

    sq, b = q.size(0), q.size(1)
    if k.size(1) != b or weights.size(1) != b:
        return None
    starts = starts.contiguous()
    ends = ends.contiguous()

    topk_out = None
    for bi in range(b):
        index_q = q[:, bi].contiguous()
        index_k = k[:, bi].contiguous()
        index_w = weights[:, bi].float().contiguous()
        for start in range(0, sq, block_size):
            end = min(start + block_size, sq)
            _, topk_indices = lighting_indexer(
                index_q[start:end],
                index_k,
                index_w[start:end],
                starts[start:end],
                ends[start:end],
                index_topk,
                topk_indices=None,
                use_relu=use_relu,
            )
            topk_indices, _ = _sanitize_fused_topk_outputs(
                topk_indices=topk_indices, starts=starts[start:end], ends=ends[start:end]
            )
            if topk_out is None:
                topk_out = torch.empty(
                    (b, sq, topk_indices.size(-1)),
                    dtype=topk_indices.dtype,
                    device=topk_indices.device,
                )
            topk_out[bi, start:end].copy_(topk_indices)

    if topk_out is None:
        return None
    return topk_out


def _compute_topk_target_chunk_sum(
    *,
    query_h: torch.Tensor,
    key_shared: Optional[torch.Tensor],
    key_per_head: Optional[torch.Tensor],
    s0: int,
    s1: int,
    idx_seq: torch.Tensor,
    valid_seq: torch.Tensor,
    softmax_scale: float,
    head_chunk_size: int,
    topk_chunk_size: int,
    sk: int,
    hn: int,
) -> torch.Tensor:
    """Compute unnormalized target probability mass on top-k support for one sequence chunk."""
    s_len = s1 - s0
    topk = idx_seq.size(-1)
    device = query_h.device
    np = query_h.size(0)

    attn_chunk_sum = _get_scratch_buffer("kl_attn_chunk_sum", (s_len, topk), torch.float32, device)
    attn_chunk_sum.zero_()

    for h0 in range(0, np, head_chunk_size):
        h1 = min(h0 + head_chunk_size, np)
        h_chunk = h1 - h0
        q_chunk = query_h[h0:h1, s0:s1, :]

        if key_shared is None:
            key_chunk = key_per_head[h0:h1]
            flat_keys = key_chunk.reshape(h_chunk * sk, hn)
            head_offsets = (
                torch.arange(h_chunk, device=device, dtype=torch.int64).view(-1, 1, 1) * sk
            )
        else:
            flat_keys = None
            head_offsets = None

        # Two-pass online softmax over top-k chunks:
        # 1) compute row-wise max and denominator; 2) recompute and accumulate probabilities.
        m = _get_scratch_buffer("kl_m", (h_chunk, s_len), torch.float32, device)
        l = _get_scratch_buffer("kl_l", (h_chunk, s_len), torch.float32, device)
        m.fill_(float("-inf"))
        l.zero_()

        for t0 in range(0, topk, topk_chunk_size):
            t1 = min(t0 + topk_chunk_size, topk)
            idx_topk = idx_seq[:, t0:t1]
            valid_topk_chunk = valid_seq[:, t0:t1]

            if key_shared is not None:
                key_sel = key_shared.index_select(0, idx_topk.reshape(-1)).view(s_len, t1 - t0, hn)
                logits = (
                    torch.einsum('hsd,skd->hsk', q_chunk.float(), key_sel.float()) * softmax_scale
                )
            else:
                flat_idx = idx_topk.unsqueeze(0) + head_offsets
                key_sel = flat_keys.index_select(0, flat_idx.reshape(-1)).view(
                    h_chunk, s_len, t1 - t0, hn
                )
                logits = (q_chunk.float().unsqueeze(2) * key_sel.float()).sum(
                    dim=-1
                ) * softmax_scale

            logits = logits.masked_fill(~valid_topk_chunk.unsqueeze(0), float("-inf"))
            chunk_max = logits.max(dim=-1).values
            m_new = torch.maximum(m, chunk_max)
            m_new_for_exp = torch.where(torch.isfinite(m_new), m_new, torch.zeros_like(m_new))
            alpha = torch.exp(m - m_new_for_exp)
            p_chunk = torch.exp(logits - m_new_for_exp.unsqueeze(-1))
            l = l * alpha + p_chunk.sum(dim=-1)
            m = m_new

        stable_m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        inv_l = l.clamp_min(1e-10).reciprocal()
        for t0 in range(0, topk, topk_chunk_size):
            t1 = min(t0 + topk_chunk_size, topk)
            idx_topk = idx_seq[:, t0:t1]
            valid_topk_chunk = valid_seq[:, t0:t1]

            if key_shared is not None:
                key_sel = key_shared.index_select(0, idx_topk.reshape(-1)).view(s_len, t1 - t0, hn)
                logits = (
                    torch.einsum('hsd,skd->hsk', q_chunk.float(), key_sel.float()) * softmax_scale
                )
            else:
                flat_idx = idx_topk.unsqueeze(0) + head_offsets
                key_sel = flat_keys.index_select(0, flat_idx.reshape(-1)).view(
                    h_chunk, s_len, t1 - t0, hn
                )
                logits = (q_chunk.float().unsqueeze(2) * key_sel.float()).sum(
                    dim=-1
                ) * softmax_scale

            logits = logits.masked_fill(~valid_topk_chunk.unsqueeze(0), float("-inf"))
            probs = torch.exp(logits - stable_m.unsqueeze(-1)) * inv_l.unsqueeze(-1)
            attn_chunk_sum[:, t0:t1] += probs.sum(dim=0)

    return attn_chunk_sum


def _compute_sparse_topk_kl_chunk(
    target_chunk: torch.Tensor, index_logits_chunk: torch.Tensor, valid_seq: torch.Tensor
) -> torch.Tensor:
    """Compute KL(target || index) sum for one [s_chunk, topk] chunk."""
    index_logits_chunk = index_logits_chunk.to(dtype=torch.float32, device=target_chunk.device)
    index_logits_chunk = index_logits_chunk.masked_fill(~valid_seq, float("-inf"))
    no_valid_rows = ~valid_seq.any(dim=-1, keepdim=True)
    if no_valid_rows.any():
        index_logits_chunk = index_logits_chunk.masked_fill(
            no_valid_rows.expand_as(index_logits_chunk), 0.0
        )
    index_scores_chunk = torch.nn.functional.softmax(
        index_logits_chunk, dim=-1, dtype=torch.float32
    )
    kl_chunk = target_chunk * (
        torch.log(target_chunk + 1e-10) - torch.log(index_scores_chunk + 1e-10)
    )
    return kl_chunk.sum()


def _normalize_topk_target_chunk(target_chunk: torch.Tensor) -> torch.Tensor:
    """Normalize target probability mass over top-k support."""
    return target_chunk / target_chunk.sum(dim=-1, keepdim=True).clamp_min(1e-10)


def _stage_topk_target_chunk(
    target_chunk: torch.Tensor,
    *,
    slot_prefix: str,
    slot: int,
    device: torch.device,
    tp_group: torch.distributed.ProcessGroup,
    tp_size: int,
) -> Tuple[torch.Tensor, Optional[torch.distributed.Work]]:
    """Copy chunk into scratch slot and optionally launch async TP all-reduce."""
    target_chunk_work = _get_scratch_buffer(
        f"{slot_prefix}_slot{slot}", tuple(target_chunk.shape), torch.float32, device
    )
    target_chunk_work.copy_(target_chunk)
    if tp_size > 1:
        handle = torch.distributed.all_reduce(target_chunk_work, group=tp_group, async_op=True)
    else:
        handle = None
    return target_chunk_work, handle


def _consume_pending_topk_kl_chunk(
    *,
    pending_handle: Optional[torch.distributed.Work],
    pending_target_chunk: Optional[torch.Tensor],
    pending_index_logits: Optional[torch.Tensor],
    pending_valid_seq: Optional[torch.Tensor],
    kl_sum: torch.Tensor,
) -> Tuple[
    torch.Tensor,
    Optional[torch.distributed.Work],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Finalize one pending chunk and accumulate KL."""
    if pending_target_chunk is None:
        return kl_sum, pending_handle, pending_target_chunk, pending_index_logits, pending_valid_seq
    if pending_handle is not None:
        pending_handle.wait()
    normalized_target = _normalize_topk_target_chunk(pending_target_chunk)
    kl_sum = kl_sum + _compute_sparse_topk_kl_chunk(
        target_chunk=normalized_target,
        index_logits_chunk=pending_index_logits,
        valid_seq=pending_valid_seq,
    )
    return kl_sum, None, None, None, None


def _enqueue_topk_kl_chunk(
    *,
    target_chunk: torch.Tensor,
    index_logits_chunk: torch.Tensor,
    valid_seq: torch.Tensor,
    slot_prefix: str,
    chunk_id: int,
    device: torch.device,
    tp_group: torch.distributed.ProcessGroup,
    tp_size: int,
    pending_handle: Optional[torch.distributed.Work],
    pending_target_chunk: Optional[torch.Tensor],
    pending_index_logits: Optional[torch.Tensor],
    pending_valid_seq: Optional[torch.Tensor],
    kl_sum: torch.Tensor,
) -> Tuple[
    torch.Tensor,
    int,
    Optional[torch.distributed.Work],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Stage a new KL chunk, consume previous pending chunk, and update pending state."""
    slot = chunk_id & 1
    target_chunk_work, current_handle = _stage_topk_target_chunk(
        target_chunk,
        slot_prefix=slot_prefix,
        slot=slot,
        device=device,
        tp_group=tp_group,
        tp_size=tp_size,
    )
    kl_sum, _, _, _, _ = _consume_pending_topk_kl_chunk(
        pending_handle=pending_handle,
        pending_target_chunk=pending_target_chunk,
        pending_index_logits=pending_index_logits,
        pending_valid_seq=pending_valid_seq,
        kl_sum=kl_sum,
    )
    return (kl_sum, chunk_id + 1, current_handle, target_chunk_work, index_logits_chunk, valid_seq)


def _flush_pending_topk_kl_chunk(
    *,
    pending_handle: Optional[torch.distributed.Work],
    pending_target_chunk: Optional[torch.Tensor],
    pending_index_logits: Optional[torch.Tensor],
    pending_valid_seq: Optional[torch.Tensor],
    kl_sum: torch.Tensor,
) -> torch.Tensor:
    """Consume the final pending KL chunk and return updated kl_sum."""
    kl_sum, _, _, _, _ = _consume_pending_topk_kl_chunk(
        pending_handle=pending_handle,
        pending_target_chunk=pending_target_chunk,
        pending_index_logits=pending_index_logits,
        pending_valid_seq=pending_valid_seq,
        kl_sum=kl_sum,
    )
    return kl_sum


def fused_qk_topk_lighting_with_streaming_sparse_kl(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    index_topk: int,
    starts: torch.Tensor,
    ends: torch.Tensor,
    block_size: int,
    query: torch.Tensor,
    key: torch.Tensor,
    softmax_scale: float,
    loss_coeff: float,
    pg_collection: ProcessGroupCollection,
    query_valid_rows: Optional[torch.Tensor] = None,
    calculate_per_token_loss: bool = False,
    seq_chunk_size: int = 512,
    head_chunk_size: int = 16,
    topk_chunk_size: int = 1024,
    use_relu: bool = True,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Run fused TileLang indexer and stream top-k logits into sparse KL accumulation."""
    if lighting_indexer is None:
        return None
    if q.ndim != 4 or k.ndim != 3 or weights.ndim != 3:
        return None

    query, _ = dsa_layout.ensure_sbhd(query, "query")
    key, _ = dsa_layout.ensure_sbhd(key, "key")
    sq, b = q.size(0), q.size(1)
    sq_q, b_q, np, hn = query.size()
    sk, b_k, nk, hk = key.size()
    if k.size(1) != b or weights.size(1) != b:
        return None
    if sq_q != sq or b_q != b or b_k != b or hk != hn:
        return None
    if nk != 1 and nk != np:
        return None
    query_valid_rows = dsa_masking.normalize_query_valid_rows(
        query_valid_rows, b=b, sq=sq, device=query.device
    )

    starts = starts.contiguous()
    ends = ends.contiguous()

    topk_out = None
    kl_sum = torch.zeros((), dtype=torch.float32, device=q.device)
    tp_size = pg_collection.tp.size()
    pending_handle = None
    pending_target_chunk = None
    pending_index_logits = None
    pending_valid_seq = None
    chunk_id = 0
    for bi in range(b):
        query_h = query[:, bi].permute(1, 0, 2).contiguous()
        if nk == 1:
            key_shared = key[:, bi, 0].contiguous()
            key_per_head = None
        else:
            key_shared = None
            key_per_head = key[:, bi].permute(1, 0, 2).contiguous()

        index_q = q[:, bi].contiguous()
        index_k = k[:, bi].contiguous()
        index_w = weights[:, bi].float().contiguous()

        for start in range(0, sq, block_size):
            end = min(start + block_size, sq)
            topk_scores, topk_indices = lighting_indexer(
                index_q[start:end],
                index_k,
                index_w[start:end],
                starts[start:end],
                ends[start:end],
                index_topk,
                topk_indices=None,
                use_relu=use_relu,
            )
            topk_indices, topk_scores = _sanitize_fused_topk_outputs(
                topk_indices=topk_indices,
                starts=starts[start:end],
                ends=ends[start:end],
                topk_scores=topk_scores,
            )

            if topk_out is None:
                topk_out = torch.empty(
                    (b, sq, topk_indices.size(-1)),
                    dtype=topk_indices.dtype,
                    device=topk_indices.device,
                )
            topk_out[bi, start:end].copy_(topk_indices)

            s_len = end - start
            for rel_start in range(0, s_len, seq_chunk_size):
                rel_end = min(rel_start + seq_chunk_size, s_len)
                abs_start = start + rel_start
                abs_end = start + rel_end

                idx_seq_raw = topk_indices[rel_start:rel_end].to(
                    dtype=torch.int64, device=query.device
                )
                valid_seq = idx_seq_raw >= 0
                idx_seq = idx_seq_raw.clamp(min=0)
                target_chunk = _compute_topk_target_chunk_sum(
                    query_h=query_h,
                    key_shared=key_shared,
                    key_per_head=key_per_head,
                    s0=abs_start,
                    s1=abs_end,
                    idx_seq=idx_seq,
                    valid_seq=valid_seq,
                    softmax_scale=softmax_scale,
                    head_chunk_size=head_chunk_size,
                    topk_chunk_size=topk_chunk_size,
                    sk=sk,
                    hn=hn,
                )
                index_logits_chunk = topk_scores[rel_start:rel_end]
                if query_valid_rows is not None:
                    row_valid = query_valid_rows[bi, abs_start:abs_end]
                    if not row_valid.any():
                        continue
                    target_chunk = target_chunk[row_valid]
                    index_logits_chunk = index_logits_chunk[row_valid]
                    valid_seq = valid_seq[row_valid]
                (
                    kl_sum,
                    chunk_id,
                    pending_handle,
                    pending_target_chunk,
                    pending_index_logits,
                    pending_valid_seq,
                ) = _enqueue_topk_kl_chunk(
                    target_chunk=target_chunk,
                    index_logits_chunk=index_logits_chunk,
                    valid_seq=valid_seq,
                    slot_prefix="stream_kl_target",
                    chunk_id=chunk_id,
                    device=query.device,
                    tp_group=pg_collection.tp,
                    tp_size=tp_size,
                    pending_handle=pending_handle,
                    pending_target_chunk=pending_target_chunk,
                    pending_index_logits=pending_index_logits,
                    pending_valid_seq=pending_valid_seq,
                    kl_sum=kl_sum,
                )
    kl_sum = _flush_pending_topk_kl_chunk(
        pending_handle=pending_handle,
        pending_target_chunk=pending_target_chunk,
        pending_index_logits=pending_index_logits,
        pending_valid_seq=pending_valid_seq,
        kl_sum=kl_sum,
    )

    if topk_out is None:
        return None
    if calculate_per_token_loss:
        kl_div = kl_sum
    else:
        valid_row_count = (
            query_valid_rows.sum().to(dtype=torch.float32, device=q.device).clamp_min(1.0)
            if query_valid_rows is not None
            else torch.tensor(float(b * sq), dtype=torch.float32, device=q.device)
        )
        kl_div = kl_sum / valid_row_count
    return topk_out, kl_div * loss_coeff


def fused_sparse_mla_absorbed(
    query: torch.Tensor,
    key: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    v_channels: int,
) -> Optional[torch.Tensor]:
    """Run fused SparseMLA kernel for absorbed-MLA path."""
    if SparseMLA is None:
        return None

    if query.ndim != 4 or key.ndim != 4 or topk_indices.ndim != 3:
        return None
    if key.size(2) != 1:
        return None
    if query.size(1) != key.size(1) or topk_indices.size(0) != query.size(1):
        return None
    if topk_indices.size(1) != query.size(0):
        return None
    if query.size(-1) != key.size(-1):
        return None
    if query.size(-1) != 576 or v_channels != 512:
        # Current copied TileLang kernels are specialized for GLM5/DeepSeek V3.2 absorbed dims.
        return None
    if topk_indices.size(-1) % 64 != 0:
        return None

    batch_outputs = None
    for bi in range(query.size(1)):
        q_t = query[:, bi].contiguous()
        kv_t = key[:, bi].contiguous()
        idx_t = topk_indices[bi].unsqueeze(1).to(torch.int32).contiguous()
        out, _ = SparseMLA.apply(q_t, kv_t, idx_t, softmax_scale)
        if out.ndim != 3 or out.size(-1) != v_channels:
            return None
        if batch_outputs is None:
            batch_outputs = torch.empty(
                (out.size(0), query.size(1), out.size(1), out.size(2)),
                dtype=out.dtype,
                device=out.device,
            )
        batch_outputs[:, bi].copy_(out)

    if batch_outputs is None:
        return None
    return batch_outputs.contiguous()


def run_fused_qk_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    index_topk: int,
    starts: torch.Tensor,
    ends: torch.Tensor,
    block_size: int,
    use_relu: bool = True,
) -> Optional[torch.Tensor]:
    """Optional fused indexer hook backed by TileLang."""
    return fused_qk_topk_lighting(q, k, weights, index_topk, starts, ends, block_size, use_relu)


def run_fused_qk_topk_with_loss(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    index_topk: int,
    starts: torch.Tensor,
    ends: torch.Tensor,
    block_size: int,
    query: torch.Tensor,
    key: torch.Tensor,
    softmax_scale: float,
    loss_coeff: float,
    pg_collection: ProcessGroupCollection,
    query_valid_rows: Optional[torch.Tensor] = None,
    calculate_per_token_loss: bool = False,
    use_relu: bool = True,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Optional fused indexer+loss hook backed by TileLang."""
    return fused_qk_topk_lighting_with_streaming_sparse_kl(
        q=q,
        k=k,
        weights=weights,
        index_topk=index_topk,
        starts=starts,
        ends=ends,
        block_size=block_size,
        query=query,
        key=key,
        softmax_scale=softmax_scale,
        loss_coeff=loss_coeff,
        pg_collection=pg_collection,
        query_valid_rows=query_valid_rows,
        calculate_per_token_loss=calculate_per_token_loss,
        use_relu=use_relu,
    )


def run_fused_absorbed_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    v_channels: int,
) -> Optional[torch.Tensor]:
    """Optional fused sparse-attention hook backed by TileLang."""
    return fused_sparse_mla_absorbed(query, key, topk_indices, softmax_scale, v_channels)
