# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""TileLang-backed DSA hook implementations.

This module keeps TileLang-specific batching, chunking, and sparse-KL streaming
out of the backend-neutral DSA control flow in ``dsa.py``.
"""

from collections import OrderedDict
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.experimental_attention_variant import (
    dsa_indexer_loss,
    dsa_layout,
    dsa_masking,
)
from megatron.core.utils import get_pg_size

if TYPE_CHECKING:
    from megatron.core.packed_seq_params import PackedSeqParams

try:
    from megatron.core.transformer.experimental_attention_variant.ops.indexer import (
        lighting_indexer,
        lighting_indexer_indices,
    )
    from megatron.core.transformer.experimental_attention_variant.ops.tilelang_indexer_bwd import (
        is_supported_indexer_bwd_head_count,
    )
except (ImportError, OSError):
    is_supported_indexer_bwd_head_count = None
    lighting_indexer = None
    lighting_indexer_indices = None

try:
    from megatron.core.transformer.experimental_attention_variant.ops.sparse_mla import SparseMLA
except (ImportError, OSError):
    SparseMLA = None

try:
    from megatron.core.transformer.experimental_attention_variant.ops.tilelang_indexer_loss import (
        SparseIndexerKLLoss,
        sparse_indexer_target_interface,
    )
except (ImportError, OSError):
    SparseIndexerKLLoss = None
    sparse_indexer_target_interface = None


# Reusable no-grad scratch buffers keyed by (name, shape, dtype, device).
_DSA_SCRATCH_CACHE_MAX_ENTRIES = 128
_DSA_SCRATCH_CACHE_MAX_BYTES = 512 * 1024 * 1024
_DSA_SCRATCH_CACHE = OrderedDict()
_DSA_SCRATCH_CACHE_TOTAL_BYTES = 0


def _scratch_buffer_bytes(buf: torch.Tensor) -> int:
    return buf.numel() * buf.element_size()


def _is_supported_sparse_mla_head_count(heads: int, kv_group: int = 1) -> bool:
    """Return whether TileLang SparseMLA supports this query/KV head grouping.

    The forward and backward kernels pad ``head_kv`` to ``max(next_power_of_2(head_kv), 16)``
    and index the unpadded head dimension by that padded count with no head-dim bound, so they
    only stay in bounds when no padding occurs. That requires ``head_kv`` (= ``heads //
    kv_group``) to be a power of two and at least 16; any other value (e.g. 48, 192, or < 16)
    must fall back to the unfused path rather than read/write past the real head count.
    """
    if kv_group <= 0 or heads % kv_group != 0:
        return False
    head_kv = heads // kv_group
    return head_kv >= 16 and (head_kv & (head_kv - 1)) == 0


def _all_bfloat16(*tensors: torch.Tensor) -> bool:
    return all(tensor.dtype == torch.bfloat16 for tensor in tensors)


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


def _topk_valid_mask(
    topk_indices: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor
) -> torch.Tensor:
    """Compute the row-wise [start, end) validity mask for fused indexer outputs."""
    starts_for_cmp = starts.to(device=topk_indices.device, dtype=topk_indices.dtype).unsqueeze(-1)
    ends_for_cmp = ends.to(device=topk_indices.device, dtype=topk_indices.dtype).unsqueeze(-1)
    return (topk_indices >= starts_for_cmp) & (topk_indices < ends_for_cmp)


def _sanitize_fused_topk_indices(
    topk_indices: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor
) -> torch.Tensor:
    """Mask fused indexer outputs in place and return the validity mask."""
    valid = _topk_valid_mask(topk_indices, starts, ends)
    topk_indices.masked_fill_(~valid, -1)
    return valid


def _sanitize_fused_topk_outputs(
    topk_indices: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    topk_scores: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Mask fused indexer outputs and optional scores to row-wise key bounds."""
    valid = _topk_valid_mask(topk_indices, starts, ends)
    sanitized_indices = topk_indices.masked_fill(~valid, -1)
    if topk_scores is not None:
        topk_scores = topk_scores.masked_fill(~valid, float("-inf"))
    return sanitized_indices, topk_scores


def _build_packed_cp_indexer_inputs(
    index_k: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    *,
    packed_seq_params: "PackedSeqParams",
    cp_size: int,
    cp_rank: int,
    single_packed_thd_sequence: bool,
    local_query_start: int,
    local_query_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack CP front/back key prefixes and translate query bounds to the packed key space."""
    if cp_size <= 1 or not 0 <= cp_rank < cp_size:
        raise RuntimeError("packed CP TileLang indexer requires a valid CP rank and cp_size > 1")
    if local_query_start < 0 or local_query_start + starts.numel() > local_query_len:
        raise RuntimeError(
            "packed CP TileLang indexer received an invalid local query slice: "
            f"start={local_query_start}, rows={starts.numel()}, local_rows={local_query_len}"
        )

    cu_q, cu_k = dsa_layout.get_packed_qk_cu_seqlens(packed_seq_params)
    if cu_q.shape != cu_k.shape or cu_q.numel() < 2:
        raise RuntimeError("packed CP TileLang indexer requires matching non-empty q/k cu_seqlens")

    device = index_k.device
    cu_q = cu_q.to(device=device, dtype=torch.int64).contiguous()
    cu_k = cu_k.to(device=device, dtype=torch.int64).contiguous()
    q_lengths = cu_q[1:] - cu_q[:-1]
    k_lengths = cu_k[1:] - cu_k[:-1]
    segment_divisor = 2 * cp_size
    q_half = q_lengths // segment_divisor
    segment_q_lengths = torch.stack((q_half, q_half), dim=1).reshape(-1)

    sk = index_k.size(0)
    if single_packed_thd_sequence and sk == local_query_len:
        if cu_q.numel() != 2 or local_query_len % 2 != 0:
            raise RuntimeError(
                "local-key packed CP TileLang indexer requires one sequence and even query rows"
            )
        half = local_query_len // 2
        segment_q_lengths = torch.full((2,), half, dtype=torch.int64, device=device)
        segment_k_lengths = torch.tensor((half, local_query_len), dtype=torch.int64, device=device)
        segment_key_starts = torch.zeros(2, dtype=torch.int64, device=device)
        total_segment_k = local_query_len + half
    else:
        if sk % segment_divisor != 0:
            raise RuntimeError(
                f"packed CP TileLang key length must be divisible by {segment_divisor}, got {sk}"
            )
        k_half = k_lengths // segment_divisor
        segment_k_lengths = torch.stack(
            ((cp_rank + 1) * k_half, k_lengths - cp_rank * k_half), dim=1
        ).reshape(-1)
        segment_key_starts = cu_k[:-1].repeat_interleave(2)
        total_segment_k = sk + sk // segment_divisor

    zero = torch.zeros(1, dtype=torch.int64, device=device)
    segment_cu_q = torch.cat((zero, segment_q_lengths.cumsum(dim=0))).contiguous()
    segment_cu_k = torch.cat((zero, segment_k_lengths.cumsum(dim=0))).contiguous()

    segment_ids_k = torch.repeat_interleave(
        torch.arange(segment_k_lengths.numel(), device=device),
        segment_k_lengths,
        output_size=total_segment_k,
    )
    segment_offsets_k = torch.arange(total_segment_k, device=device, dtype=torch.int64)
    segment_offsets_k -= torch.repeat_interleave(
        segment_cu_k[:-1], segment_k_lengths, output_size=total_segment_k
    )
    source_indices = segment_key_starts.index_select(0, segment_ids_k) + segment_offsets_k
    segmented_k = index_k.index_select(0, source_indices).contiguous()

    segment_ids_q = torch.repeat_interleave(
        torch.arange(segment_q_lengths.numel(), device=device),
        segment_q_lengths,
        output_size=local_query_len,
    )
    row_start = local_query_start
    row_end = row_start + starts.numel()
    row_segment_ids = segment_ids_q[row_start:row_end]
    row_segment_starts = segment_cu_k[:-1].index_select(0, row_segment_ids)
    row_segment_ends = row_segment_starts + segment_k_lengths.index_select(0, row_segment_ids)
    row_global_starts = segment_key_starts.index_select(0, row_segment_ids)

    local_starts = row_segment_starts + starts.to(torch.int64) - row_global_starts
    local_ends = row_segment_starts + ends.to(torch.int64) - row_global_starts
    local_starts = torch.maximum(local_starts, row_segment_starts)
    local_starts = torch.minimum(local_starts, row_segment_ends)
    local_ends = torch.maximum(local_ends, local_starts)
    local_ends = torch.minimum(local_ends, row_segment_ends)
    return (
        segmented_k,
        local_starts.to(torch.int32).contiguous(),
        local_ends.to(torch.int32).contiguous(),
        source_indices,
    )


def _remap_segmented_topk_indices(
    topk_indices: torch.Tensor, source_indices: torch.Tensor
) -> torch.Tensor:
    """Map valid indices from a segmented key tensor back to the original packed key tensor."""
    valid = topk_indices >= 0
    safe_indices = topk_indices.clamp(min=0).reshape(-1).to(torch.int64)
    global_indices = source_indices.index_select(0, safe_indices).view_as(topk_indices)
    return torch.where(valid, global_indices.to(topk_indices.dtype), topk_indices)


def fused_qk_topk_lighting(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    index_topk: int,
    starts: torch.Tensor,
    ends: torch.Tensor,
    block_size: int,
    use_relu: bool = True,
    use_local_indexer_varlen: bool = False,
    single_packed_thd_sequence: bool = False,
    local_packed_cp_rank: int = 0,
    local_packed_cp_query_start: int = 0,
    local_packed_cp_query_len: Optional[int] = None,
    packed_seq_params: Optional["PackedSeqParams"] = None,
    cp_size: int = 1,
) -> Optional[torch.Tensor]:
    """Run fused TileLang indexer and return top-k indices [b, sq, topk]."""
    if lighting_indexer_indices is None:
        return None
    if q.ndim != 4 or k.ndim != 3 or weights.ndim != 3:
        return None
    if not _all_bfloat16(q, k):
        return None

    sq, b = q.size(0), q.size(1)
    if k.size(1) != b or weights.size(1) != b:
        return None
    starts = starts.contiguous()
    ends = ends.contiguous()

    topk_k = min(index_topk, k.size(0))
    topk_out = torch.empty((b, sq, topk_k), dtype=torch.int32, device=q.device)
    for bi in range(b):
        index_q = q[:, bi].contiguous()
        index_k = k[:, bi].contiguous()
        index_w = weights[:, bi].float().contiguous()
        local_starts = starts
        local_ends = ends
        source_indices = None
        if b == 1 and use_local_indexer_varlen and packed_seq_params is not None and cp_size > 1:
            local_query_len = (
                local_packed_cp_query_len if local_packed_cp_query_len is not None else sq
            )
            index_k, local_starts, local_ends, source_indices = _build_packed_cp_indexer_inputs(
                index_k,
                starts,
                ends,
                packed_seq_params=packed_seq_params,
                cp_size=cp_size,
                cp_rank=local_packed_cp_rank,
                single_packed_thd_sequence=single_packed_thd_sequence,
                local_query_start=local_packed_cp_query_start,
                local_query_len=local_query_len,
            )
        for start in range(0, sq, block_size):
            end = min(start + block_size, sq)
            topk_indices = lighting_indexer_indices(
                index_q[start:end],
                index_k,
                index_w[start:end],
                local_starts[start:end],
                local_ends[start:end],
                topk_k,
                use_relu=use_relu,
            )
            _sanitize_fused_topk_indices(
                topk_indices, starts=local_starts[start:end], ends=local_ends[start:end]
            )
            if source_indices is not None:
                topk_indices = _remap_segmented_topk_indices(topk_indices, source_indices)
            topk_out[bi, start:end].copy_(topk_indices)

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
        q_chunk_float = q_chunk.float()

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
        # These accumulators are rebound to fresh tensors each top-k chunk, so they
        # cannot reuse a scratch buffer in place.
        running_max = torch.full(
            (h_chunk, s_len), float("-inf"), dtype=torch.float32, device=device
        )
        running_denom = torch.zeros((h_chunk, s_len), dtype=torch.float32, device=device)

        def _chunk_logits(idx_topk, valid_topk_chunk, k_len):
            if key_shared is not None:
                key_sel = key_shared.index_select(0, idx_topk.reshape(-1)).view(s_len, k_len, hn)
                logits = torch.einsum("hsd,skd->hsk", q_chunk_float, key_sel.float())
            else:
                flat_idx = idx_topk.unsqueeze(0) + head_offsets
                key_sel = flat_keys.index_select(0, flat_idx.reshape(-1)).view(
                    h_chunk, s_len, k_len, hn
                )
                logits = (q_chunk_float.unsqueeze(2) * key_sel.float()).sum(dim=-1)
            logits = logits * softmax_scale
            return logits.masked_fill(~valid_topk_chunk.unsqueeze(0), float("-inf"))

        for t0 in range(0, topk, topk_chunk_size):
            t1 = min(t0 + topk_chunk_size, topk)
            logits = _chunk_logits(idx_seq[:, t0:t1], valid_seq[:, t0:t1], t1 - t0)
            chunk_max = logits.max(dim=-1).values
            new_running_max = torch.maximum(running_max, chunk_max)
            max_for_exp = torch.where(
                torch.isfinite(new_running_max), new_running_max, torch.zeros_like(new_running_max)
            )
            alpha = torch.exp(running_max - max_for_exp)
            p_chunk = torch.exp(logits - max_for_exp.unsqueeze(-1))
            running_denom = running_denom * alpha + p_chunk.sum(dim=-1)
            running_max = new_running_max

        stable_max = torch.where(
            torch.isfinite(running_max), running_max, torch.zeros_like(running_max)
        )
        inverse_denom = running_denom.clamp_min(1e-10).reciprocal()
        for t0 in range(0, topk, topk_chunk_size):
            t1 = min(t0 + topk_chunk_size, topk)
            logits = _chunk_logits(idx_seq[:, t0:t1], valid_seq[:, t0:t1], t1 - t0)
            probs = torch.exp(logits - stable_max.unsqueeze(-1)) * inverse_denom.unsqueeze(-1)
            attn_chunk_sum[:, t0:t1] += probs.sum(dim=0)

    return attn_chunk_sum


def _compute_sparse_topk_kl_chunk(
    target_chunk: torch.Tensor, index_logits_chunk: torch.Tensor, valid_seq: torch.Tensor
) -> torch.Tensor:
    """Compute KL(target || index) sum for one [s_chunk, topk] chunk."""
    index_logits_chunk = index_logits_chunk.to(dtype=torch.float32, device=target_chunk.device)
    target_chunk = target_chunk.to(dtype=torch.float32, device=index_logits_chunk.device)
    with torch.no_grad():
        index_log_scores_chunk = dsa_masking.masked_log_softmax(
            index_logits_chunk.detach(), valid_seq, dim=-1
        )
        index_scores_chunk = index_log_scores_chunk.exp().masked_fill(~valid_seq, 0.0)
        kl_value = dsa_indexer_loss.indexer_kl_sum(target_chunk, index_log_scores_chunk, valid_seq)
        grad_logits = (index_scores_chunk - target_chunk).masked_fill(~valid_seq, 0.0)
    index_logits_for_grad = index_logits_chunk.masked_fill(~valid_seq, 0.0)
    grad_surrogate = (index_logits_for_grad * grad_logits).sum()
    return grad_surrogate + (kl_value - grad_surrogate).detach()


def _can_use_fused_sparse_indexer_target(
    query: torch.Tensor, key: Optional[torch.Tensor], topk_indices: torch.Tensor
) -> bool:
    """Return whether the fused TileLang target kernel supports these tensors."""
    return (
        sparse_indexer_target_interface is not None
        and key is not None
        and query.is_cuda
        and key.is_cuda
        and topk_indices.is_cuda
        and query.ndim == 3
        and key.ndim == 2
        and query.dtype == torch.bfloat16
        and key.dtype == torch.bfloat16
        and query.size(-1) == key.size(-1)
        and query.size(-1) % 16 == 0
        and topk_indices.ndim == 2
        and topk_indices.size(-1) % 64 == 0
    )


def _can_use_fused_sparse_indexer_kl(
    target: torch.Tensor, index_logits: torch.Tensor, valid_mask: torch.Tensor
) -> bool:
    """Return whether the fused TileLang KL/score-gradient kernel supports these tensors."""
    return (
        SparseIndexerKLLoss is not None
        and target.is_cuda
        and index_logits.is_cuda
        and valid_mask.is_cuda
        and target.dtype == torch.float32
        and index_logits.dtype == torch.float32
        and valid_mask.dtype == torch.bool
        and target.shape == index_logits.shape == valid_mask.shape
        and target.ndim == 2
        and target.size(-1) % 256 == 0
    )


def _canonicalize_topk_scores_for_tp_reduce(
    topk_indices: torch.Tensor, topk_scores: torch.Tensor, *, sk: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sort selected top-k slots by key index before slot-wise TP reductions."""
    valid = topk_indices >= 0
    sort_key = torch.where(valid, topk_indices, torch.full_like(topk_indices, sk))
    order = sort_key.argsort(dim=-1)
    topk_indices = torch.gather(topk_indices, dim=-1, index=order)
    topk_scores = torch.gather(topk_scores, dim=-1, index=order)
    valid = torch.gather(valid, dim=-1, index=order)
    topk_indices = topk_indices.masked_fill(~valid, -1)
    topk_scores = topk_scores.masked_fill(~valid, float("-inf"))
    return topk_indices.contiguous(), topk_scores.contiguous()


def _accumulate_topk_kl_chunk(
    *,
    target_chunk: torch.Tensor,
    index_logits_chunk: torch.Tensor,
    valid_seq: torch.Tensor,
    kl_sum: torch.Tensor,
) -> torch.Tensor:
    """Normalize one target chunk and accumulate its sparse KL contribution."""
    if _can_use_fused_sparse_indexer_kl(target_chunk, index_logits_chunk, valid_seq):
        return kl_sum + SparseIndexerKLLoss.apply(
            target_chunk.contiguous(), index_logits_chunk.contiguous(), valid_seq.contiguous()
        )
    normalized_target = dsa_indexer_loss.normalize_indexer_target_(target_chunk)
    return kl_sum + _compute_sparse_topk_kl_chunk(
        target_chunk=normalized_target, index_logits_chunk=index_logits_chunk, valid_seq=valid_seq
    )


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
) -> torch.Tensor:
    """Finalize one pending chunk and accumulate its KL contribution into ``kl_sum``."""
    if pending_target_chunk is None:
        return kl_sum
    if pending_handle is not None:
        pending_handle.wait()
    return _accumulate_topk_kl_chunk(
        target_chunk=pending_target_chunk,
        index_logits_chunk=pending_index_logits,
        valid_seq=pending_valid_seq,
        kl_sum=kl_sum,
    )


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
    kl_sum = _consume_pending_topk_kl_chunk(
        pending_handle=pending_handle,
        pending_target_chunk=pending_target_chunk,
        pending_index_logits=pending_index_logits,
        pending_valid_seq=pending_valid_seq,
        kl_sum=kl_sum,
    )
    return (kl_sum, chunk_id + 1, current_handle, target_chunk_work, index_logits_chunk, valid_seq)


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
    use_local_indexer_varlen: bool = False,
    single_packed_thd_sequence: bool = False,
    local_packed_cp_rank: int = 0,
    local_packed_cp_query_start: int = 0,
    local_packed_cp_query_len: Optional[int] = None,
    packed_seq_params: Optional["PackedSeqParams"] = None,
    cp_size: int = 1,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Run the fused TileLang indexer with streaming sparse KL accumulation.

    The objective matches ``compute_dsa_indexer_loss`` on the selected top-k support. TileLang
    streams query/head/top-k chunks and overlaps TP target reduction to avoid materializing dense
    scores; its custom gradient surrogate supplies the same log-softmax gradient for fused indexer
    logits. Target normalization, KL evaluation, and token reduction use the shared backend-neutral
    helpers in ``dsa_indexer_loss``.
    """
    if lighting_indexer is None:
        return None
    if q.ndim != 4 or k.ndim != 3 or weights.ndim != 3:
        return None
    if not _all_bfloat16(q, k):
        return None
    if is_supported_indexer_bwd_head_count is None or not is_supported_indexer_bwd_head_count(
        q.size(2)
    ):
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
    tp_size = get_pg_size(pg_collection.tp)
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
        local_starts = starts
        local_ends = ends
        source_indices = None
        if b == 1 and use_local_indexer_varlen and packed_seq_params is not None and cp_size > 1:
            local_query_len = (
                local_packed_cp_query_len if local_packed_cp_query_len is not None else sq
            )
            index_k, local_starts, local_ends, source_indices = _build_packed_cp_indexer_inputs(
                index_k,
                starts,
                ends,
                packed_seq_params=packed_seq_params,
                cp_size=cp_size,
                cp_rank=local_packed_cp_rank,
                single_packed_thd_sequence=single_packed_thd_sequence,
                local_query_start=local_packed_cp_query_start,
                local_query_len=local_query_len,
            )

        for start in range(0, sq, block_size):
            end = min(start + block_size, sq)
            topk_scores, topk_indices = lighting_indexer(
                index_q[start:end],
                index_k,
                index_w[start:end],
                local_starts[start:end],
                local_ends[start:end],
                min(index_topk, k.size(0)),
                topk_indices=None,
                use_relu=use_relu,
            )
            topk_indices, topk_scores = _sanitize_fused_topk_outputs(
                topk_indices=topk_indices,
                starts=local_starts[start:end],
                ends=local_ends[start:end],
                topk_scores=topk_scores,
            )
            if source_indices is not None:
                topk_indices = _remap_segmented_topk_indices(topk_indices, source_indices)
            if tp_size > 1:
                topk_indices, topk_scores = _canonicalize_topk_scores_for_tp_reduce(
                    topk_indices, topk_scores, sk=sk
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

                idx_seq_raw = topk_indices[rel_start:rel_end].to(device=query.device)
                valid_seq = idx_seq_raw >= 0
                if query_valid_rows is not None:
                    row_valid = query_valid_rows[bi, abs_start:abs_end]
                    valid_seq = valid_seq & row_valid.unsqueeze(-1)
                loss_topk_indices = idx_seq_raw.masked_fill(~valid_seq, -1).contiguous()
                query_chunk = query[abs_start:abs_end, bi].contiguous()
                if _can_use_fused_sparse_indexer_target(query_chunk, key_shared, loss_topk_indices):
                    target_chunk = sparse_indexer_target_interface(
                        query_chunk, key_shared, loss_topk_indices, softmax_scale
                    )
                else:
                    target_chunk = _compute_topk_target_chunk_sum(
                        query_h=query_h,
                        key_shared=key_shared,
                        key_per_head=key_per_head,
                        s0=abs_start,
                        s1=abs_end,
                        idx_seq=idx_seq_raw.clamp(min=0).to(torch.int64),
                        valid_seq=valid_seq,
                        softmax_scale=softmax_scale,
                        head_chunk_size=head_chunk_size,
                        topk_chunk_size=topk_chunk_size,
                        sk=sk,
                        hn=hn,
                    )
                index_logits_chunk = topk_scores[rel_start:rel_end]
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
    kl_sum = _consume_pending_topk_kl_chunk(
        pending_handle=pending_handle,
        pending_target_chunk=pending_target_chunk,
        pending_index_logits=pending_index_logits,
        pending_valid_seq=pending_valid_seq,
        kl_sum=kl_sum,
    )

    if topk_out is None:
        return None
    valid_row_count = query_valid_rows.sum() if query_valid_rows is not None else None
    kl_div = dsa_indexer_loss.reduce_indexer_kl_sum(
        kl_sum,
        num_rows=b * sq,
        calculate_per_token_loss=calculate_per_token_loss,
        valid_row_count=valid_row_count,
    )
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
    if not _all_bfloat16(query, key):
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
    query_heads = query.size(2)
    if query_heads <= 0:
        return None
    kernel_heads = max(query_heads, 16)
    if not _is_supported_sparse_mla_head_count(kernel_heads, kv_group=key.size(2)):
        return None
    if topk_indices.size(-1) % 64 != 0:
        return None

    query_bshd = query.permute(1, 0, 2, 3).contiguous()
    if kernel_heads != query_heads:
        # SparseMLA uses a minimum 16-head tile without head bounds. Pad the caller
        # tensor so small TP shards stay in bounds, then discard those heads below.
        query_bshd = torch.nn.functional.pad(query_bshd, (0, 0, 0, kernel_heads - query_heads))
    key_bshd = key.permute(1, 0, 2, 3).contiguous()
    indices_bsgk = topk_indices.unsqueeze(2).to(torch.int32).contiguous()
    out, _ = SparseMLA.apply(query_bshd, key_bshd, indices_bsgk, softmax_scale)
    if out.ndim != 4 or out.size(2) != kernel_heads or out.size(-1) != v_channels:
        return None
    out = out[:, :, :query_heads]
    return out.permute(1, 0, 2, 3).contiguous()


def run_fused_qk_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    index_topk: int,
    starts: torch.Tensor,
    ends: torch.Tensor,
    block_size: int,
    use_relu: bool = True,
    use_local_indexer_varlen: bool = False,
    single_packed_thd_sequence: bool = False,
    local_packed_cp_rank: int = 0,
    local_packed_cp_query_start: int = 0,
    local_packed_cp_query_len: Optional[int] = None,
    packed_seq_params: Optional["PackedSeqParams"] = None,
    cp_size: int = 1,
) -> Optional[torch.Tensor]:
    """Optional fused indexer hook backed by TileLang."""
    return fused_qk_topk_lighting(
        q,
        k,
        weights,
        index_topk,
        starts,
        ends,
        block_size,
        use_relu,
        use_local_indexer_varlen=use_local_indexer_varlen,
        single_packed_thd_sequence=single_packed_thd_sequence,
        local_packed_cp_rank=local_packed_cp_rank,
        local_packed_cp_query_start=local_packed_cp_query_start,
        local_packed_cp_query_len=local_packed_cp_query_len,
        packed_seq_params=packed_seq_params,
        cp_size=cp_size,
    )


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
    use_local_indexer_varlen: bool = False,
    single_packed_thd_sequence: bool = False,
    local_packed_cp_rank: int = 0,
    local_packed_cp_query_start: int = 0,
    local_packed_cp_query_len: Optional[int] = None,
    packed_seq_params: Optional["PackedSeqParams"] = None,
    cp_size: int = 1,
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
        use_local_indexer_varlen=use_local_indexer_varlen,
        single_packed_thd_sequence=single_packed_thd_sequence,
        local_packed_cp_rank=local_packed_cp_rank,
        local_packed_cp_query_start=local_packed_cp_query_start,
        local_packed_cp_query_len=local_packed_cp_query_len,
        packed_seq_params=packed_seq_params,
        cp_size=cp_size,
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
