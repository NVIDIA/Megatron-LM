# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import copy
import logging
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from megatron.core import parallel_state
from megatron.core.models.common.embeddings import (
    RotaryEmbedding,
    YarnRotaryEmbedding,
    apply_rotary_pos_emb,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig

logger = logging.getLogger(__name__)

try:
    from fast_hadamard_transform import hadamard_transform
except ImportError:
    hadamard_transform = None

try:
    from megatron.core.transformer.experimental_attention_variant.ops.indexer import (
        lighting_indexer,
    )
except (ImportError, OSError):
    logger.debug(
        "Failed to import fused TileLang indexer; lighting_indexer path disabled.", exc_info=True
    )
    lighting_indexer = None

try:
    from megatron.core.transformer.experimental_attention_variant.ops.sparse_mla import SparseMLA
except (ImportError, OSError):
    logger.debug(
        "Failed to import fused TileLang SparseMLA; SparseMLA path disabled.", exc_info=True
    )
    SparseMLA = None

# Reusable no-grad scratch buffers keyed by (name, shape, dtype, device).
_DSA_SCRATCH_CACHE_MAX_ENTRIES = 128
_DSA_SCRATCH_CACHE_MAX_BYTES = 512 * 1024 * 1024
_DSA_SCRATCH_CACHE = OrderedDict()


def _scratch_cache_total_bytes() -> int:
    """Return total bytes held by cached scratch tensors."""
    return sum(buf.numel() * buf.element_size() for buf in _DSA_SCRATCH_CACHE.values())


def _evict_scratch_cache_if_needed() -> None:
    """Bound scratch cache growth by LRU eviction."""
    while (
        len(_DSA_SCRATCH_CACHE) > _DSA_SCRATCH_CACHE_MAX_ENTRIES
        or _scratch_cache_total_bytes() > _DSA_SCRATCH_CACHE_MAX_BYTES
    ):
        _DSA_SCRATCH_CACHE.popitem(last=False)


def _get_scratch_buffer(
    name: str, shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Get a reusable scratch tensor for temporary no-grad workspaces."""
    key = (name, shape, dtype, device)
    buf = _DSA_SCRATCH_CACHE.pop(key, None)
    if buf is None:
        buf = torch.empty(shape, dtype=dtype, device=device)
    _DSA_SCRATCH_CACHE[key] = buf
    _evict_scratch_cache_if_needed()
    return buf


def _normalize_cp_comm_type(cp_comm_type: Optional[str]) -> str:
    """Normalize CP communication type to a canonical lowercase form."""
    if cp_comm_type is None:
        return "p2p"
    return cp_comm_type.replace("_", "").lower()


def _get_cp_positions_from_layout(
    sq: int,
    skv: int,
    cp_size: int,
    cp_rank: int,
    cp_comm_type: Optional[str],
    device: torch.device,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Infer query/key global token positions under CP layout.

    This helper currently supports allgather CP layout, where each rank owns a
    contiguous query chunk and sees gathered keys in global order.
    """
    if cp_size <= 1:
        query_pos = torch.arange(sq, device=device, dtype=torch.int64)
        key_pos = torch.arange(skv, device=device, dtype=torch.int64)
        return query_pos, key_pos

    if _normalize_cp_comm_type(cp_comm_type) != "allgather":
        raise NotImplementedError(
            "DSAttention context parallelism currently supports cp_comm_type=allgather only."
        )

    # Avoid assuming uniform per-rank query lengths (cp_rank * sq). When available,
    # gather local lengths to build the true global offset for this CP rank.
    query_offset = cp_rank * sq
    if (
        cp_group is not None
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and cp_group.size() == cp_size
    ):
        local_len = torch.tensor([sq], device=device, dtype=torch.int64)
        all_lens = [torch.empty_like(local_len) for _ in range(cp_size)]
        torch.distributed.all_gather(all_lens, local_len, group=cp_group)
        if cp_rank > 0:
            query_offset = int(torch.stack(all_lens[:cp_rank]).sum().item())
        else:
            query_offset = 0

    query_pos = torch.arange(sq, device=device, dtype=torch.int64) + query_offset
    key_pos = torch.arange(skv, device=device, dtype=torch.int64)
    return query_pos, key_pos


def _build_causal_mask_from_positions(
    query_pos: torch.Tensor, key_pos: torch.Tensor
) -> torch.Tensor:
    """Build a causal mask from explicit query/key global positions."""
    assert query_pos.dtype in (torch.int32, torch.int64), "query_pos must be integer tensor"
    assert key_pos.dtype in (torch.int32, torch.int64), "key_pos must be integer tensor"
    assert query_pos.device == key_pos.device, "query_pos and key_pos must be on the same device"

    # mask[q, k] = -inf if key_pos[k] > query_pos[q], else 0.
    invalid = key_pos.unsqueeze(0) > query_pos.unsqueeze(-1)
    mask = torch.zeros(
        (query_pos.numel(), key_pos.numel()), dtype=torch.float32, device=query_pos.device
    )
    mask.masked_fill_(invalid, float("-inf"))
    return mask


def _generate_varlen_mask_params(cu_seqlens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate row-wise [start, end) key bounds for packed causal masking."""
    assert cu_seqlens.ndim == 1 and cu_seqlens.numel() >= 2, "invalid cu_seqlens"
    cu_seqlens = cu_seqlens.to(dtype=torch.int64)
    seq_len = int(cu_seqlens[-1].item())
    q_indices = torch.arange(seq_len, dtype=torch.int64, device=cu_seqlens.device)
    seq_indices = torch.searchsorted(cu_seqlens, q_indices, right=True) - 1
    starts = cu_seqlens[seq_indices]
    ends = q_indices + 1
    return starts, ends


def _build_valid_mask_from_starts_ends(
    starts: torch.Tensor, ends: torch.Tensor, key_positions: torch.Tensor
) -> torch.Tensor:
    """Build boolean validity mask [sq, sk] from row-wise [start, end) bounds."""
    assert starts.ndim == ends.ndim == 1, "starts/ends must be 1D"
    assert starts.shape == ends.shape, "starts/ends shape mismatch"
    assert key_positions.ndim == 1, "key_positions must be 1D"
    assert starts.device == ends.device == key_positions.device, "device mismatch"
    assert starts.dtype in (torch.int32, torch.int64), "starts must be int tensor"
    assert ends.dtype in (torch.int32, torch.int64), "ends must be int tensor"
    assert key_positions.dtype in (torch.int32, torch.int64), "key_positions must be int tensor"
    key_positions = key_positions.to(dtype=torch.int64)
    starts = starts.to(dtype=torch.int64)
    ends = ends.to(dtype=torch.int64)
    return (key_positions.unsqueeze(0) >= starts.unsqueeze(-1)) & (
        key_positions.unsqueeze(0) < ends.unsqueeze(-1)
    )


def _apply_starts_ends_mask_to_scores(
    scores: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor, key_positions: torch.Tensor
) -> torch.Tensor:
    """Apply varlen starts/ends mask to score tensor.

    Supports scores with shape [b, sq, sk] or [b, np, sq, sk].
    """
    valid = _build_valid_mask_from_starts_ends(starts, ends, key_positions)
    if scores.ndim == 3:
        return scores.masked_fill(~valid.unsqueeze(0), float("-inf"))
    if scores.ndim == 4:
        return scores.masked_fill(~valid.unsqueeze(0).unsqueeze(0), float("-inf"))
    raise ValueError(f"Unsupported scores ndim={scores.ndim}, expected 3 or 4.")


def _build_default_causal_mask(sq: int, sk: int, device: torch.device) -> torch.Tensor:
    """Build standard upper-triangular additive causal mask."""
    return torch.triu(
        torch.full((sq, sk), float("-inf"), dtype=torch.float32, device=device), diagonal=1
    )


def _prepare_additive_mask(
    mask: Optional[torch.Tensor], *, sq: int, sk: int, b: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate/build additive mask and return useful broadcasted views.

    Returns:
        score_mask: [sq, sk] or [b, sq, sk]
        attn_score_mask: [1, 1, sq, sk] or [b, 1, sq, sk]
        index_score_mask: [1, sq, sk] or [b, sq, sk]
        valid_mask: [b, sq, sk] bool, True means finite (not masked)
    """
    if mask is None:
        score_mask = _build_default_causal_mask(sq, sk, device=device)
    else:
        assert mask.dtype == torch.float32, "mask dtype must be float32"
        assert mask.device == device, "mask device mismatch"
        assert mask.ndim in (2, 3), "mask must be 2D or 3D"
        if mask.ndim == 2:
            assert mask.shape == (sq, sk), "mask shape mismatch"
        else:
            assert mask.shape == (b, sq, sk), "mask shape mismatch"
        score_mask = mask

    if score_mask.ndim == 2:
        attn_score_mask = score_mask.view(1, 1, sq, sk)
        index_score_mask = score_mask.unsqueeze(0)
        valid_mask = torch.isfinite(score_mask).unsqueeze(0).expand(b, sq, sk)
    else:
        attn_score_mask = score_mask.view(b, 1, sq, sk)
        index_score_mask = score_mask
        valid_mask = torch.isfinite(score_mask)
    return score_mask, attn_score_mask, index_score_mask, valid_mask


def _prepare_sparse_mask_context(
    *,
    mask: Optional[torch.Tensor],
    varlen_starts: Optional[torch.Tensor],
    varlen_ends: Optional[torch.Tensor],
    key_positions: Optional[torch.Tensor],
    sq: int,
    sk: int,
    b: int,
    device: torch.device,
) -> Tuple[
    Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]
]:
    """Prepare shared sparse-mask context for unfused attention paths."""
    if mask is not None and varlen_starts is not None:
        raise ValueError("mask and varlen_starts are mutually exclusive")

    if varlen_starts is not None:
        if varlen_ends is None:
            raise ValueError("varlen_ends is required when varlen_starts is provided")
        varlen_starts_i64 = varlen_starts.to(device=device, dtype=torch.int64)
        varlen_ends_i64 = varlen_ends.to(device=device, dtype=torch.int64)
        if key_positions is None:
            key_positions_i64 = torch.arange(sk, dtype=torch.int64, device=device)
        else:
            key_positions_i64 = key_positions.to(device=device, dtype=torch.int64)
        return None, varlen_starts_i64, varlen_ends_i64, key_positions_i64

    _, _, index_score_mask, _ = _prepare_additive_mask(mask, sq=sq, sk=sk, b=b, device=device)
    return index_score_mask, None, None, None


def _apply_sparse_validity_to_index_mask(
    index_mask: torch.Tensor,
    *,
    row_mask: Optional[torch.Tensor],
    varlen_starts: Optional[torch.Tensor],
    varlen_ends: Optional[torch.Tensor],
    key_positions: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply either varlen or additive mask validity constraints to index_mask."""
    if varlen_starts is not None:
        if varlen_ends is None:
            raise ValueError("varlen_ends is required when varlen_starts is provided")
        if key_positions is None:
            raise ValueError("key_positions is required when varlen_starts is provided")
        valid_mask = _build_valid_mask_from_starts_ends(
            varlen_starts, varlen_ends, key_positions
        ).unsqueeze(0)
        return index_mask.masked_fill(~valid_mask, float("-inf"))

    if row_mask is None:
        raise ValueError("row_mask is required when varlen_starts is None")
    return index_mask + row_mask


def _gather_sparse_topk_validity_and_bias(
    *,
    idx_topk: torch.Tensor,
    valid_t: torch.Tensor,
    bi: int,
    s0: int,
    s1: int,
    row_mask: Optional[torch.Tensor],
    varlen_starts: Optional[torch.Tensor],
    varlen_ends: Optional[torch.Tensor],
    key_positions: Optional[torch.Tensor],
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Gather top-k validity mask and optional additive bias for one [s_chunk, topk] block."""
    if varlen_starts is not None:
        if varlen_ends is None:
            raise ValueError("varlen_ends is required when varlen_starts is provided")
        if key_positions is None:
            raise ValueError("key_positions is required when varlen_starts is provided")
        key_pos_sel = key_positions.index_select(0, idx_topk.reshape(-1)).view_as(idx_topk)
        valid_varlen = (key_pos_sel >= varlen_starts[s0:s1].unsqueeze(-1)) & (
            key_pos_sel < varlen_ends[s0:s1].unsqueeze(-1)
        )
        return valid_t & valid_varlen, None

    if row_mask is None:
        raise ValueError("row_mask is required when varlen_starts is None")
    mask_src = row_mask[0, s0:s1, :] if row_mask.size(0) == 1 else row_mask[bi, s0:s1, :]
    mask_bias = mask_src.gather(-1, idx_topk).to(dtype=dtype)
    return valid_t & torch.isfinite(mask_bias), mask_bias


def _scatter_topk_into_index_mask(
    index_mask: torch.Tensor, topk_indices: torch.Tensor, *, seq_chunk_size: int = 256
) -> None:
    """Scatter top-k supports into index_mask using chunk-wise int64 casts."""
    b, sq, _ = index_mask.shape
    assert topk_indices.ndim == 3, "topk_indices must be [b, sq, topk]"
    assert topk_indices.shape[:2] == (b, sq), "topk_indices shape mismatch"
    device = index_mask.device
    seq_chunk_size = max(1, int(seq_chunk_size))

    for s0 in range(0, sq, seq_chunk_size):
        s1 = min(s0 + seq_chunk_size, sq)
        idx_chunk = topk_indices[:, s0:s1]
        if idx_chunk.dtype != torch.int64 or idx_chunk.device != device:
            idx_chunk = idx_chunk.to(dtype=torch.int64, device=device)
        if torch.any(idx_chunk < 0):
            valid_topk = idx_chunk >= 0
            if valid_topk.any():
                b_idx, q_rel_idx, t_idx = torch.where(valid_topk)
                q_idx = q_rel_idx + s0
                k_idx = idx_chunk[b_idx, q_rel_idx, t_idx]
                index_mask[b_idx, q_idx, k_idx] = 0.0
        else:
            index_mask[:, s0:s1].scatter_(-1, idx_chunk, 0.0)


def _extract_query_positions_from_position_ids(
    position_ids: Optional[torch.Tensor], sq: int, device: torch.device
) -> Optional[torch.Tensor]:
    """Extract per-rank query positions from position_ids if compatible."""
    if position_ids is None:
        return None
    if position_ids.ndim == 2:
        if position_ids.size(0) > 1:
            assert torch.equal(
                position_ids[0], position_ids[-1]
            ), "Allgather-CP DSA expects identical position_ids across batch"
        query_pos = position_ids[0]
    elif position_ids.ndim == 1:
        query_pos = position_ids
    else:
        raise ValueError(f"position_ids should be 1D or 2D tensor, got {position_ids.ndim}D.")

    if query_pos.numel() != sq:
        return None
    return query_pos.to(device=device, dtype=torch.int64)


def _get_packed_qk_cu_seqlens(
    packed_seq_params: PackedSeqParams,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select packed cu_seqlens for query and key/value streams."""
    cu_seqlens_q = (
        packed_seq_params.cu_seqlens_q_padded
        if packed_seq_params.cu_seqlens_q_padded is not None
        else packed_seq_params.cu_seqlens_q
    )
    cu_seqlens = (
        packed_seq_params.cu_seqlens_kv_padded
        if packed_seq_params.cu_seqlens_kv_padded is not None
        else packed_seq_params.cu_seqlens_kv
    )
    cu_seqlens_kv = cu_seqlens

    if cu_seqlens_q is None and cu_seqlens_kv is None:
        raise ValueError("Packed sequence parameters must provide cu_seqlens for DSA masking.")
    if cu_seqlens_q is None:
        cu_seqlens_q = cu_seqlens_kv
    if cu_seqlens_kv is None:
        cu_seqlens_kv = cu_seqlens_q
    return cu_seqlens_q, cu_seqlens_kv


def _build_dsattention_forward_mask(
    *,
    sq: int,
    skv: int,
    b: int,
    device: torch.device,
    cp_size: int,
    cp_rank: int,
    cp_comm_type: str,
    cp_group: Optional[torch.distributed.ProcessGroup],
    attn_mask_type: Optional[AttnMaskType],
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    packed_seq_params: Optional[PackedSeqParams],
) -> Tuple[Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """Build DSAttention mask.

    Returns:
        float_mask: Optional additive mask [sq, skv] or [b, sq, skv].
        varlen_params: Optional (starts, ends, key_positions), each int64 tensor.
    """
    packed_thd = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
    if attn_mask_type is not None:
        assert attn_mask_type == AttnMaskType.causal, "Only causal mask is supported for now"
        if packed_thd:
            cu_seqlens_q, _ = _get_packed_qk_cu_seqlens(packed_seq_params)
            cu_seqlens_q = cu_seqlens_q.to(device=device, dtype=torch.int64)
            starts, ends = _generate_varlen_mask_params(cu_seqlens_q)
            if cp_size > 1:
                query_idx, key_idx = _get_cp_positions_from_layout(
                    sq=sq,
                    skv=skv,
                    cp_size=cp_size,
                    cp_rank=cp_rank,
                    cp_comm_type=cp_comm_type,
                    device=device,
                    cp_group=cp_group,
                )
            else:
                query_idx = torch.arange(sq, dtype=torch.int64, device=device)
                key_idx = torch.arange(skv, dtype=torch.int64, device=device)
            varlen_starts = starts.index_select(0, query_idx)
            varlen_ends = ends.index_select(0, query_idx)
            return None, (varlen_starts, varlen_ends, key_idx)

        if cp_size > 1:
            query_pos = _extract_query_positions_from_position_ids(position_ids, sq, device)
            if query_pos is None:
                query_pos, key_pos = _get_cp_positions_from_layout(
                    sq=sq,
                    skv=skv,
                    cp_size=cp_size,
                    cp_rank=cp_rank,
                    cp_comm_type=cp_comm_type,
                    device=device,
                    cp_group=cp_group,
                )
            else:
                key_pos = torch.arange(skv, dtype=torch.int64, device=device)
            return _build_causal_mask_from_positions(query_pos, key_pos), None

        return _build_default_causal_mask(sq, skv, device=device), None

    assert attention_mask is not None, "attention_mask is required when attn_mask_type is None"
    assert attention_mask.shape == (b, 1, sq, skv), "attention_mask shape mismatch"
    mask = attention_mask[:, 0, :, :]
    float_mask = torch.zeros_like(mask, dtype=torch.float32).masked_fill(mask, float("-inf"))
    return float_mask, None


def _build_fused_indexer_varlen_bounds(
    *,
    sq: int,
    skv: int,
    device: torch.device,
    mask: Optional[torch.Tensor],
    varlen_starts: Optional[torch.Tensor],
    varlen_ends: Optional[torch.Tensor],
    key_positions: Optional[torch.Tensor],
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Build starts/ends bounds for tilelang fused indexer.

    Fused indexer expects row-wise contiguous valid key ranges [start, end).
    """
    if varlen_starts is not None:
        if varlen_ends is None:
            raise ValueError("varlen_ends is required when varlen_starts is provided")
        if key_positions is None:
            key_positions = torch.arange(skv, dtype=torch.int64, device=device)
        expected_key_pos = torch.arange(skv, dtype=torch.int64, device=device)
        key_positions = key_positions.to(dtype=torch.int64, device=device)
        if not torch.equal(key_positions, expected_key_pos):
            return None
        return (
            varlen_starts.to(dtype=torch.int32, device=device),
            varlen_ends.to(dtype=torch.int32, device=device),
        )

    if mask is None:
        # Standard local causal mask.
        ends = torch.arange(1, sq + 1, dtype=torch.int64, device=device).clamp_max(skv)
        starts = torch.zeros_like(ends)
        return starts.to(dtype=torch.int32), ends.to(dtype=torch.int32)

    if mask.ndim == 3:
        # Fused indexer uses one shared starts/ends schedule. For batched masks, only
        # enable fused path when all batch masks are identical.
        if mask.size(0) > 1:
            ref_mask = mask[0]
            for bi in range(1, mask.size(0)):
                if not torch.equal(mask[bi], ref_mask):
                    return None
        row_mask = mask[0]
    else:
        row_mask = mask
    if row_mask.ndim != 2 or row_mask.shape != (sq, skv):
        return None

    finite = torch.isfinite(row_mask)
    ends = finite.sum(dim=-1, dtype=torch.int64)
    key_ids = torch.arange(skv, dtype=torch.int64, device=device).unsqueeze(0)
    expected = key_ids < ends.unsqueeze(-1)
    if not torch.equal(finite, expected):
        return None

    starts = torch.zeros_like(ends)
    return starts.to(dtype=torch.int32), ends.to(dtype=torch.int32)


def _fused_qk_topk_lighting(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    index_topk: int,
    starts: torch.Tensor,
    ends: torch.Tensor,
    block_size: int,
) -> Optional[torch.Tensor]:
    """Run fused tilelang indexer and return top-k indices [b, sq, topk]."""
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
        q_chunk = query_h[h0:h1, s0:s1, :]  # [h_chunk, s_len, hn]

        if key_shared is None:
            key_chunk = key_per_head[h0:h1]  # [h_chunk, sk, hn]
            flat_keys = key_chunk.reshape(h_chunk * sk, hn)
            head_offsets = (
                torch.arange(h_chunk, device=device, dtype=torch.int64).view(-1, 1, 1) * sk
            )
        else:
            flat_keys = None
            head_offsets = None

        # Two-pass online softmax over top-k chunks (per head):
        # 1) pass computes row-wise max and denominator;
        # 2) pass recomputes chunk logits and accumulates probabilities.
        m = _get_scratch_buffer("kl_m", (h_chunk, s_len), torch.float32, device)
        l = _get_scratch_buffer("kl_l", (h_chunk, s_len), torch.float32, device)
        m.fill_(float("-inf"))
        l.zero_()

        # Pass 1: row max + denominator.
        for t0 in range(0, topk, topk_chunk_size):
            t1 = min(t0 + topk_chunk_size, topk)
            tk = t1 - t0
            idx_topk = idx_seq[:, t0:t1]  # [s_len, tk]
            valid_topk_chunk = valid_seq[:, t0:t1]  # [s_len, tk]

            if key_shared is not None:
                key_sel = key_shared.index_select(0, idx_topk.reshape(-1)).view(s_len, tk, hn)
                logits = (
                    torch.einsum('hsd,skd->hsk', q_chunk.float(), key_sel.float()) * softmax_scale
                )
            else:
                flat_idx = idx_topk.unsqueeze(0) + head_offsets  # [h_chunk, s_len, tk]
                key_sel = flat_keys.index_select(0, flat_idx.reshape(-1)).view(
                    h_chunk, s_len, tk, hn
                )
                logits = (q_chunk.float().unsqueeze(2) * key_sel.float()).sum(
                    dim=-1
                ) * softmax_scale

            logits = logits.masked_fill(~valid_topk_chunk.unsqueeze(0), float("-inf"))

            chunk_max = logits.max(dim=-1).values
            m_new = torch.maximum(m, chunk_max)
            alpha = torch.exp(m - m_new)
            alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
            p_chunk = torch.exp(logits - m_new.unsqueeze(-1))
            p_chunk = torch.nan_to_num(p_chunk, nan=0.0, posinf=0.0, neginf=0.0)
            l = l * alpha + p_chunk.sum(dim=-1)
            m = m_new

        # Pass 2: probabilities accumulation per top-k chunk.
        stable_m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        inv_l = l.clamp_min(1e-10).reciprocal()
        for t0 in range(0, topk, topk_chunk_size):
            t1 = min(t0 + topk_chunk_size, topk)
            tk = t1 - t0
            idx_topk = idx_seq[:, t0:t1]  # [s_len, tk]
            valid_topk_chunk = valid_seq[:, t0:t1]  # [s_len, tk]

            if key_shared is not None:
                key_sel = key_shared.index_select(0, idx_topk.reshape(-1)).view(s_len, tk, hn)
                logits = (
                    torch.einsum('hsd,skd->hsk', q_chunk.float(), key_sel.float()) * softmax_scale
                )
            else:
                flat_idx = idx_topk.unsqueeze(0) + head_offsets  # [h_chunk, s_len, tk]
                key_sel = flat_keys.index_select(0, flat_idx.reshape(-1)).view(
                    h_chunk, s_len, tk, hn
                )
                logits = (q_chunk.float().unsqueeze(2) * key_sel.float()).sum(
                    dim=-1
                ) * softmax_scale

            logits = logits.masked_fill(~valid_topk_chunk.unsqueeze(0), float("-inf"))
            probs = torch.exp(logits - stable_m.unsqueeze(-1)) * inv_l.unsqueeze(-1)
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
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


def _fused_qk_topk_lighting_with_streaming_sparse_kl(
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
    seq_chunk_size: int = 512,
    head_chunk_size: int = 16,
    topk_chunk_size: int = 1024,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Run fused tilelang indexer and stream top-k logits directly into sparse KL accumulation."""
    if lighting_indexer is None:
        return None
    if q.ndim != 4 or k.ndim != 3 or weights.ndim != 3:
        return None

    query, _ = _ensure_sbhd(query, "query")
    key, _ = _ensure_sbhd(key, "key")
    sq, b = q.size(0), q.size(1)
    sq_q, b_q, np, hn = query.size()
    sk, b_k, nk, hk = key.size()
    if k.size(1) != b or weights.size(1) != b:
        return None
    if sq_q != sq or b_q != b or b_k != b or hk != hn:
        return None
    if nk != 1 and nk != np:
        return None

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
        query_h = query[:, bi].permute(1, 0, 2).contiguous()  # [np, sq, hn]
        if nk == 1:
            key_shared = key[:, bi, 0].contiguous()  # [sk, hn]
            key_per_head = None
        else:
            key_shared = None
            key_per_head = key[:, bi].permute(1, 0, 2).contiguous()  # [np, sk, hn]

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
                (
                    kl_sum,
                    chunk_id,
                    pending_handle,
                    pending_target_chunk,
                    pending_index_logits,
                    pending_valid_seq,
                ) = _enqueue_topk_kl_chunk(
                    target_chunk=target_chunk,
                    index_logits_chunk=topk_scores[rel_start:rel_end],
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
    kl_div = kl_sum / (b * sq)
    return topk_out, kl_div * loss_coeff


def _fused_sparse_mla_absorbed(
    query: torch.Tensor,
    key: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    v_channels: int,
) -> Optional[torch.Tensor]:
    """Run fused SparseMLA kernel for absorbed-MLA path.

    Inputs are expected in SBHD with MQA key heads (kv_group=1):
      query: [sq, b, np, d_total]
      key:   [skv, b, 1, d_total]
      topk:  [b, sq, topk]

    Returns:
      output: [sq, b, np, v_channels], or None if unsupported / unavailable.
    """
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
        # Current copied tilelang kernels are specialized for GLM5/DeepSeek V3.2 absorbed dims.
        return None

    # Kernel requires topk to be block-aligned.
    if topk_indices.size(-1) % 64 != 0:
        return None

    batch_outputs = None
    for bi in range(query.size(1)):
        q_t = query[:, bi].contiguous()  # [sq, np, d_total]
        kv_t = key[:, bi].contiguous()  # [skv, 1, d_total]
        idx_t = topk_indices[bi].unsqueeze(1).to(torch.int32).contiguous()  # [sq, 1, topk]
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


def _explain_absorbed_fused_skip(
    query: torch.Tensor, key: torch.Tensor, topk_indices: torch.Tensor, v_channels: int
) -> str:
    """Return first failing condition for fused absorbed SparseMLA path."""
    if SparseMLA is None:
        return "SparseMLA kernel unavailable (import failed)"
    if query.ndim != 4 or key.ndim != 4 or topk_indices.ndim != 3:
        return "invalid tensor rank (expected query/key 4D and topk 3D)"
    if key.size(2) != 1:
        return f"key head count {key.size(2)} != 1 (MQA required)"
    if query.size(1) != key.size(1) or topk_indices.size(0) != query.size(1):
        return "batch shape mismatch among query/key/topk"
    if topk_indices.size(1) != query.size(0):
        return "topk seqlen mismatch with query seqlen"
    if query.size(-1) != key.size(-1):
        return "query/key hidden dim mismatch"
    if query.size(-1) != 576 or v_channels != 512:
        return (
            f"kernel specialized for d_total=576,v_channels=512 but got "
            f"d_total={query.size(-1)}, v_channels={v_channels}"
        )
    if topk_indices.size(-1) % 64 != 0:
        return f"topk ({topk_indices.size(-1)}) is not block-aligned (must be multiple of 64)"
    return "unknown runtime fallback (SparseMLA.apply returned/raised failure)"


def _build_sparse_attn_reason(
    *,
    sparse_attn_path: str,
    absorbed_mla: bool,
    query: torch.Tensor,
    key: torch.Tensor,
    topk_indices: torch.Tensor,
    config: TransformerConfig,
) -> str:
    """Build a concise reason string for sparse attention path selection."""
    if sparse_attn_path == "fused_sparse_mla_absorbed":
        return "fused absorbed SparseMLA is active"
    if sparse_attn_path == "fused_sparse_mla_absorbed_upv":
        return "fused absorbed SparseMLA + up_v projection active"
    if sparse_attn_path == "unfused_absorbed_upv":
        return "absorbed QK path with up_v projection active; fused SparseMLA unavailable"
    if not absorbed_mla:
        return "absorbed=False; non-absorbed DSAttention currently uses unfused_dsa only"
    return _explain_absorbed_fused_skip(
        query=query,
        key=key,
        topk_indices=topk_indices,
        v_channels=int(getattr(config, "kv_lora_rank", 0) or 0),
    )


def _unfused_absorbed_dsa_fn(
    query: torch.Tensor,
    key: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    v_channels: int,
    mask: Optional[torch.Tensor] = None,
    varlen_starts: Optional[torch.Tensor] = None,
    varlen_ends: Optional[torch.Tensor] = None,
    key_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Unfused absorbed-MLA attention: output stays [sq, b, np, v_channels]."""
    sq, b, np, hn = query.size()
    skv = key.size(0)
    assert key.size(2) == 1, "Absorbed DSA expects MQA key head dimension = 1"
    assert key.size(-1) >= v_channels, "key last dim must contain latent value channels"
    row_mask, varlen_starts, varlen_ends, key_positions = _prepare_sparse_mask_context(
        mask=mask,
        varlen_starts=varlen_starts,
        varlen_ends=varlen_ends,
        key_positions=key_positions,
        sq=sq,
        sk=skv,
        b=b,
        device=query.device,
    )

    # [sq,b,np,hn] -> [b,np,sq,hn]
    q = query.permute(1, 2, 0, 3)
    # [skv,b,1,hn] -> [b,1,hn,skv]
    k = key.permute(1, 2, 3, 0)
    attention_scores = torch.matmul(q.float(), k.float()) * softmax_scale

    # Sparse + causal/varlen validity mask.
    index_mask = torch.full((b, sq, skv), float("-inf"), device=attention_scores.device)
    _scatter_topk_into_index_mask(index_mask, topk_indices, seq_chunk_size=256)
    index_mask = _apply_sparse_validity_to_index_mask(
        index_mask,
        row_mask=row_mask,
        varlen_starts=varlen_starts,
        varlen_ends=varlen_ends,
        key_positions=key_positions,
    )

    attention_scores += index_mask.unsqueeze(1)
    attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32)

    # Latent value is the first v_channels slice of absorbed key cache.
    value = key[..., :v_channels].permute(1, 2, 0, 3)  # [b,1,skv,v]
    output = torch.matmul(attention_scores.to(value.dtype), value)  # [b,np,sq,v]
    return output.permute(2, 0, 1, 3).contiguous()


def _run_sparse_attention(
    *,
    absorbed_mla: bool,
    query: torch.Tensor,
    key: torch.Tensor,
    value: Optional[torch.Tensor],
    up_v_weight: Optional[torch.Tensor],
    topk_indices: torch.Tensor,
    softmax_scale: float,
    config: TransformerConfig,
    mask: Optional[torch.Tensor],
    varlen_starts: Optional[torch.Tensor],
    varlen_ends: Optional[torch.Tensor],
    key_positions: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, str]:
    """Run sparse attention for absorbed and non-absorbed MLA paths."""
    if absorbed_mla:
        latent_v_channels = int(getattr(config, "kv_lora_rank", 0) or 0)
        if latent_v_channels <= 0:
            raise RuntimeError(
                "Invalid kv_lora_rank for absorbed-MLA DSAttention sparse attention."
            )
        if value is not None:
            raise RuntimeError(
                "Absorbed DSAttention expects value=None (latent path). "
                "Received absorbed layout with explicit value tensor."
            )
        output = _fused_sparse_mla_absorbed(
            query, key, topk_indices, softmax_scale, latent_v_channels
        )
        if output is None:
            output = _unfused_absorbed_dsa_fn(
                query,
                key,
                topk_indices,
                softmax_scale,
                latent_v_channels,
                mask=mask,
                varlen_starts=varlen_starts,
                varlen_ends=varlen_ends,
                key_positions=key_positions,
            )
            if up_v_weight is None:
                return output, "unfused_absorbed"
            output = torch.einsum("sbhc,hdc->sbhd", output, up_v_weight).contiguous()
            output = output.view(output.size(0), output.size(1), -1)
            return output, "unfused_absorbed_upv"
        if up_v_weight is None:
            return output, "fused_sparse_mla_absorbed"
        output = torch.einsum("sbhc,hdc->sbhd", output, up_v_weight).contiguous()
        output = output.view(output.size(0), output.size(1), -1)
        return output, "fused_sparse_mla_absorbed_upv"

    return (
        unfused_dsa_fn(
            query,
            key,
            value,
            topk_indices,
            softmax_scale,
            mask=mask,
            varlen_starts=varlen_starts,
            varlen_ends=varlen_ends,
            key_positions=key_positions,
        ),
        "unfused_dsa",
    )


def _ensure_sbhd(tensor: torch.Tensor, name: str) -> Tuple[torch.Tensor, bool]:
    """Ensure tensor is [s, b, h, d], allowing packed [t, h, d] input."""
    if tensor.ndim == 4:
        return tensor, False
    if tensor.ndim == 3:
        return tensor.unsqueeze(1), True
    raise ValueError(f"{name} must be 3D ([t,h,d]) or 4D ([s,b,h,d]), got {tensor.ndim}D")


def _normalize_dsattention_output_rank(output: torch.Tensor, target_ndim: int) -> torch.Tensor:
    """Normalize DSAttention output rank to match caller hidden-state rank."""
    if target_ndim not in (2, 3):
        raise RuntimeError(f"DSAttention expected x.ndim in (2, 3), got {target_ndim}")

    if output.ndim == 4:
        output = output.reshape(output.size(0), output.size(1), -1)
    elif output.ndim not in (2, 3):
        raise RuntimeError(
            f"DSAttention produced unexpected output rank {output.ndim}; expected 2D/3D/4D."
        )

    if target_ndim == 3 and output.ndim == 2:
        output = output.unsqueeze(1)
    elif target_ndim == 2 and output.ndim == 3:
        if output.size(1) != 1:
            raise RuntimeError(
                "DSAttention cannot squeeze non-singleton batch dim for packed output: "
                f"shape={tuple(output.shape)}"
            )
        output = output.squeeze(1)

    if output.ndim != target_ndim:
        raise RuntimeError(
            "DSAttention output rank mismatch after normalization: "
            f"target_ndim={target_ndim}, output_shape={tuple(output.shape)}"
        )
    return output


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Apply Hadamard rotation activation.
    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L424-L428

    Args:
        x: Input tensor (must be bfloat16).

    Returns:
        Rotated tensor.
    """
    assert (
        x.dtype == torch.bfloat16
    ), f"rotate_activation only support bf16 input, but got {x.dtype}"
    assert hadamard_transform is not None, "fast_hadamard_transform is not installed."
    hidden_size = x.size(-1)
    return hadamard_transform(x, scale=hidden_size**-0.5)


class DSAIndexerLossLoggingHelper:
    """Helper class for logging sparse attention indexer losses."""

    tracker = {}

    @staticmethod
    def save_loss_to_tracker(
        loss: torch.Tensor,
        layer_number: int,
        num_layers: int,
        reduce_group: torch.distributed.ProcessGroup = None,
        avg_group: torch.distributed.ProcessGroup = None,
    ):
        """Save the indexer loss for logging.

        Args:
            loss: The loss tensor.
            layer_number: Layer index of the loss, 1-indexed.
            num_layers: The number of total layers.
            reduce_group: The group for reducing the loss.
            avg_group: The group for averaging the loss.
        """
        # Skip indexer loss logging if layer_number is None.
        if layer_number is None:
            return

        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            tracker["values"] = torch.zeros(num_layers, device=torch.cuda.current_device())
        tracker["values"][layer_number - 1] += loss.detach()
        tracker["reduce_group"] = reduce_group
        tracker["avg_group"] = avg_group

    @staticmethod
    def clean_loss_in_tracker():
        """Clear the indexer losses."""
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" in tracker:
            tracker["values"].zero_()
        tracker["reduce_group"] = None
        tracker["avg_group"] = None

    @staticmethod
    def reduce_loss_in_tracker():
        """Collect and reduce the indexer losses across ranks."""
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            return
        values = tracker["values"]

        torch.distributed.all_reduce(
            values, group=parallel_state.get_pipeline_model_parallel_group()
        )
        # Reduce indexer losses across ranks.
        if tracker.get('reduce_group') is not None:
            torch.distributed.all_reduce(values, group=tracker.get('reduce_group'))
        if tracker.get('avg_group') is not None:
            torch.distributed.all_reduce(
                values, group=tracker['avg_group'], op=torch.distributed.ReduceOp.AVG
            )
        torch.distributed.all_reduce(
            values,
            group=parallel_state.get_data_parallel_group(with_context_parallel=False),
            op=torch.distributed.ReduceOp.AVG,
        )

    @staticmethod
    def track_indexer_metrics(
        loss_scale: float,
        iteration: int,
        writer,
        wandb_writer=None,
        total_loss_dict=None,
        per_layer_logging: bool = False,
    ):
        """Track the sparse attention indexer metrics for logging.

        Args:
            loss_scale: Scale factor for the loss.
            iteration: Current training iteration.
            writer: TensorBoard writer.
            wandb_writer: Weights & Biases writer.
            total_loss_dict: Dictionary to accumulate total losses.
            per_layer_logging: Whether to log per-layer losses.
        """
        DSAIndexerLossLoggingHelper.reduce_loss_in_tracker()
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            return

        indexer_loss_values = tracker["values"] * loss_scale
        num_layers = indexer_loss_values.shape[0]

        # Average across all layers (assuming all layers have sparse attention)
        avg_indexer_loss = indexer_loss_values.sum() / num_layers

        # Log average loss
        if total_loss_dict is not None:
            if "indexer loss" in total_loss_dict:
                total_loss_dict["indexer loss"] += avg_indexer_loss
            else:
                total_loss_dict["indexer loss"] = avg_indexer_loss

        if writer is not None:
            writer.add_scalar("indexer loss", avg_indexer_loss, iteration)

        if wandb_writer is not None:
            wandb_writer.log({"indexer loss": avg_indexer_loss}, iteration)

        DSAIndexerLossLoggingHelper.clean_loss_in_tracker()


def compute_dsa_indexer_loss(
    index_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    softmax_scale: float,
    loss_coeff: float,
    sparse_loss: bool,
    pg_collection: ProcessGroupCollection,
    mask: Optional[torch.Tensor] = None,
    varlen_starts: Optional[torch.Tensor] = None,
    varlen_ends: Optional[torch.Tensor] = None,
    key_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute KL divergence loss between index_scores and true attention_scores.

    This loss trains the indexer to predict which tokens are important by matching the distribution
    of true attention scores.

    Reference: Section 2.1 of
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf

    Args:
        index_scores: Scores predicted by indexer [batch, seqlen_q, seqlen_k].
        topk_indices: Top-k indices [batch, seqlen_q, index_topk].
        query: Query tensor [seqlen_q, batch, heads, dim].
        key: Key tensor [seqlen_k, batch, heads, dim].
        softmax_scale: Scale coefficient after q @ k^T.
        loss_coeff: Coefficient for the indexer KL divergence loss.
        sparse_loss: bool, whether to use sparse indexer loss. If True, only the topk
            indices will be used to compute the loss.
        pg_collection: Process group collection, must have TP process group.
        mask: Optional additive attention mask. Supports shape [sq, sk] or [b, sq, sk].
            Invalid positions should be -inf.
        varlen_starts: Optional row-wise key start bounds [sq] for packed THD.
        varlen_ends: Optional row-wise key end bounds [sq] for packed THD.
        key_positions: Optional global key positions [sk] for packed THD.

    Returns:
        index_loss: KL divergence loss (scalar).
    """
    query, _ = _ensure_sbhd(query, "query")
    key, _ = _ensure_sbhd(key, "key")
    if mask is not None and varlen_starts is not None:
        raise ValueError("mask and varlen_starts are mutually exclusive")

    sq, b, np, hn = query.size()
    sk = key.size(0)

    # [sq, b, np, hn] -> [b, np, sq, hn] -> [b * np, sq, hn]
    query = query.permute(1, 2, 0, 3).reshape(b * np, sq, hn)
    # [sk, b, np, hn] -> [b, np, hn, sk] -> [b * np, hn, sk]
    key = key.permute(1, 2, 3, 0).reshape(b * np, hn, sk)
    # Compute attention scores [b * np, sq, sk]
    attention_scores = torch.bmm(query.float(), key.float()) * softmax_scale
    # Reshape to [b, np, sq, sk]
    attention_scores = attention_scores.reshape(b, np, sq, sk)

    if varlen_starts is not None:
        if varlen_ends is None:
            raise ValueError("varlen_ends is required when varlen_starts is provided")
        if key_positions is None:
            key_positions = torch.arange(sk, dtype=torch.int64, device=attention_scores.device)
        attention_scores = _apply_starts_ends_mask_to_scores(
            attention_scores, varlen_starts, varlen_ends, key_positions
        )
        index_scores = _apply_starts_ends_mask_to_scores(
            index_scores, varlen_starts, varlen_ends, key_positions
        )
    else:
        _, attn_score_mask, _, _ = _prepare_additive_mask(
            mask, sq=sq, sk=sk, b=b, device=attention_scores.device
        )
        # [b, np, sq, sk] + [1/b, 1, sq, sk] -> [b, np, sq, sk]
        attention_scores += attn_score_mask

    # index_mask [b, sq, sk]
    index_mask = torch.full(
        (b, sq, sk), float("-inf"), dtype=torch.float32, device=attention_scores.device
    ).scatter_(-1, topk_indices, 0)

    if sparse_loss:
        # [b, np, sq, sk] + [b, 1, sq, sk] -> [b, np, sq, sk]
        attention_scores += index_mask.view(b, 1, sq, sk)
        # [b, sq, sk] + [b, sq, sk] -> [b, sq, sk]
        index_scores += index_mask

    # [b, np, sq, sk] -> [b, np, sq, sk]
    attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32)
    # [b, sq, sk] -> [b, sq, sk]
    index_scores = torch.nn.functional.softmax(index_scores, dim=-1, dtype=torch.float32)

    # Sum attention scores across heads.
    # [batch, heads, seqlen_q, seqlen_k] -> [batch, seqlen_q, seqlen_k]
    attention_scores = attention_scores.sum(dim=1)
    if pg_collection.tp.size() > 1:
        # attention scores are scattered to TP ranks in head dimension.
        torch.distributed.all_reduce(attention_scores.contiguous(), group=pg_collection.tp)
    # L1 normalize target on the last dimension. Doesn't use abs() because attention_scores are
    # obtained from softmax so they are already non-negative.
    attention_scores = attention_scores / attention_scores.sum(dim=-1, keepdim=True)

    # Compute KL divergence: KL(target || index) = target(x) * log(target(x) / index(x))
    # kl_per_element [b, sq, sk]
    kl_per_element = attention_scores * (
        torch.log(attention_scores + 1e-10) - torch.log(index_scores + 1e-10)
    )

    # [b, sq, sk] -> [b, sq] -> [1]
    # Each token has same weight in the loss.
    kl_div = kl_per_element.sum(dim=-1).mean()

    # Scale by coefficient.
    indexer_loss = kl_div * loss_coeff

    return indexer_loss


def compute_dsa_indexer_loss_topk_sparse(
    index_topk_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    softmax_scale: float,
    loss_coeff: float,
    pg_collection: ProcessGroupCollection,
) -> torch.Tensor:
    """Compute sparse top-k KL loss using fused indexer top-k logits.

    This matches fused indexer semantics where indexer logits are
    only materialized at selected top-k positions.
    The implementation streams sequence chunks to reduce peak memory.
    """
    query, _ = _ensure_sbhd(query, "query")
    key, _ = _ensure_sbhd(key, "key")

    sq, b, np, hn = query.size()
    sk, bk, nk, hk = key.size()
    assert bk == b and hk == hn, "query/key shape mismatch"
    assert index_topk_scores.shape[:2] == (b, sq), "index_topk_scores shape mismatch"
    assert topk_indices.shape[:2] == (b, sq), "topk_indices shape mismatch"

    if nk != 1:
        assert nk == np, "key head count must be 1 (MQA) or match query heads"

    # Compute KL in streaming chunks to avoid materializing full [b, sq, topk]
    # and avoid full-size valid/safe top-k tensors.
    seq_chunk_size = 512
    head_chunk_size = 16
    topk_chunk_size = 1024
    kl_sum = torch.zeros((), dtype=torch.float32, device=query.device)
    tp_size = pg_collection.tp.size()
    pending_handle = None
    pending_target_chunk = None
    pending_index_logits = None
    pending_valid_seq = None
    chunk_id = 0

    for bi in range(b):
        query_h = query[:, bi].permute(1, 0, 2).contiguous()  # [np, sq, hn]
        if nk == 1:
            key_shared = key[:, bi, 0].contiguous()  # [sk, hn]
            key_per_head = None
        else:
            key_shared = None
            key_per_head = key[:, bi].permute(1, 0, 2).contiguous()  # [np, sk, hn]

        for s0 in range(0, sq, seq_chunk_size):
            s1 = min(s0 + seq_chunk_size, sq)
            idx_seq_raw = topk_indices[bi, s0:s1]  # [s_len, topk]
            if idx_seq_raw.dtype != torch.int64 or idx_seq_raw.device != query.device:
                idx_seq_raw = idx_seq_raw.to(dtype=torch.int64, device=query.device)
            valid_seq = idx_seq_raw >= 0
            idx_seq = idx_seq_raw.clamp(min=0)

            target_chunk = _compute_topk_target_chunk_sum(
                query_h=query_h,
                key_shared=key_shared,
                key_per_head=key_per_head,
                s0=s0,
                s1=s1,
                idx_seq=idx_seq,
                valid_seq=valid_seq,
                softmax_scale=softmax_scale,
                head_chunk_size=head_chunk_size,
                topk_chunk_size=topk_chunk_size,
                sk=sk,
                hn=hn,
            )
            (
                kl_sum,
                chunk_id,
                pending_handle,
                pending_target_chunk,
                pending_index_logits,
                pending_valid_seq,
            ) = _enqueue_topk_kl_chunk(
                target_chunk=target_chunk,
                index_logits_chunk=index_topk_scores[bi, s0:s1],
                valid_seq=valid_seq,
                slot_prefix="topk_sparse_kl_target",
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

    kl_div = kl_sum / (b * sq)
    return kl_div * loss_coeff


def _compute_index_scores(q: torch.Tensor, weights: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    Perform index score using BF16 precision.

    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/kernel.py#L254-L274
    This is a BF16 implementation of the `fp8_index` logic:
        1. Compute attention scores: q @ k^T;
        2. Apply ReLU activation;
        3. Weight by attention weights;
        4. Sum across attention heads.

    Args:
        q: BF16 [seqlen_q, batch, index_n_heads, index_head_dim], the query tensor.
        weights: BF16 [seqlen_q, batch, index_n_heads], the attention weights.
        k: BF16 [seqlen_k, batch, index_head_dim], the key tensor.

    Returns:
        index_scores: FP32 [batch, seqlen_q, seqlen_k], the index scores.
    """
    # Compute attention scores: q @ k^T
    # [seqlen_q, batch, index_n_heads, index_head_dim] @ [seqlen_k, batch, index_head_dim]^T
    #   -> [seqlen_q, batch, index_n_heads, seqlen_k]
    index_scores = torch.einsum('sbhd,tbd->sbht', q.float(), k.float())

    # Apply ReLU activation.
    index_scores = torch.relu(index_scores)

    # Weight each head by attention weights.
    # [seqlen_q, batch, index_n_heads, seqlen_k] * [seqlen_q, batch, index_n_heads, 1]
    #   -> [seqlen_q, batch, index_n_heads, seqlen_k]
    index_scores = index_scores * weights.unsqueeze(-1)

    # Sum across attention heads.
    # [seqlen_q, batch, index_n_heads, seqlen_k] -> [seqlen_q, batch, seqlen_k]
    index_scores = index_scores.sum(dim=2)

    # Transpose to [batch, seqlen_q, seqlen_k].
    index_scores = index_scores.transpose(0, 1)

    return index_scores


def fused_qk_topk_naive(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    index_topk: int,
    mask: Optional[torch.Tensor] = None,
    varlen_starts: Optional[torch.Tensor] = None,
    varlen_ends: Optional[torch.Tensor] = None,
    key_positions: Optional[torch.Tensor] = None,
):
    """Naive implementation of QK Topk."""
    sk = k.size(0)
    # =========================================
    # Compute index scores
    # =========================================
    # [batch, seqlen, seqlen]
    index_scores = _compute_index_scores(q, weights, k)
    if mask is not None and varlen_starts is not None:
        raise ValueError("mask and varlen_starts are mutually exclusive")
    if varlen_starts is not None:
        if varlen_ends is None:
            raise ValueError("varlen_ends is required when varlen_starts is provided")
        if key_positions is None:
            key_positions = torch.arange(sk, dtype=torch.int64, device=index_scores.device)
        index_scores = _apply_starts_ends_mask_to_scores(
            index_scores, varlen_starts, varlen_ends, key_positions
        )
    elif mask is not None:
        assert mask.dtype == index_scores.dtype, "Mask dtype must match index scores dtype"
        index_scores = index_scores + mask

    # =========================================
    # Select top-k indices
    # =========================================
    topk_k = min(index_topk, sk)
    # [batch, seqlen, index_topk]
    topk_indices = index_scores.topk(topk_k, dim=-1)[1]

    return index_scores, topk_indices


def fwd_fused_indexer_loss_naive(
    q,
    weights,
    k,
    query,
    key,
    topk,
    softmax_scale,
    loss_coeff,
    mask,
    sparse_loss,
    pg_collection,
    varlen_starts=None,
    varlen_ends=None,
    key_positions=None,
):
    """Naive implementation of forward pass for indexer loss."""
    index_scores, topk_indices = fused_qk_topk_naive(
        q,
        k,
        weights,
        topk,
        mask=mask,
        varlen_starts=varlen_starts,
        varlen_ends=varlen_ends,
        key_positions=key_positions,
    )

    indexer_loss = compute_dsa_indexer_loss(
        index_scores,
        topk_indices,
        query,
        key,
        softmax_scale,
        loss_coeff,
        sparse_loss,
        pg_collection,
        mask=mask,
        varlen_starts=varlen_starts,
        varlen_ends=varlen_ends,
        key_positions=key_positions,
    )

    return topk_indices, indexer_loss


def bwd_fused_indexer_loss_naive(
    q,
    weights,
    k,
    query,
    key,
    topk_indices,
    softmax_scale,
    loss_coeff,
    sparse_loss,
    mask,
    grad_loss,
    pg_collection,
    varlen_starts=None,
    varlen_ends=None,
    key_positions=None,
):
    """Naive implementation of backward pass for indexer loss."""
    query, _ = _ensure_sbhd(query, "query")
    key, _ = _ensure_sbhd(key, "key")
    if mask is not None and varlen_starts is not None:
        raise ValueError("mask and varlen_starts are mutually exclusive")

    index_scores = _compute_index_scores(q, weights, k)  # [B, Sq, Sk]

    sq, b, np, hn = query.size()
    sk = key.size(0)

    # [sq, b, np, hn] -> [b, np, sq, hn] -> [b * np, sq, hn]
    query_reshaped = query.permute(1, 2, 0, 3).reshape(b * np, sq, hn)
    # [sk, b, np, hn] -> [b, np, hn, sk] -> [b * np, hn, sk]
    key_reshaped = key.permute(1, 2, 3, 0).reshape(b * np, hn, sk)
    # Compute attention scores [b * np, sq, sk]
    attention_scores = torch.bmm(query_reshaped.float(), key_reshaped.float()) * softmax_scale
    # Free reshaped tensors - no longer needed after bmm
    del query_reshaped, key_reshaped

    # Reshape to [b, np, sq, sk]
    attention_scores = attention_scores.reshape(b, np, sq, sk)

    if varlen_starts is not None:
        if varlen_ends is None:
            raise ValueError("varlen_ends is required when varlen_starts is provided")
        if key_positions is None:
            key_positions = torch.arange(sk, dtype=torch.int64, device=attention_scores.device)
        attention_scores = _apply_starts_ends_mask_to_scores(
            attention_scores, varlen_starts, varlen_ends, key_positions
        )
        index_scores = _apply_starts_ends_mask_to_scores(
            index_scores, varlen_starts, varlen_ends, key_positions
        )
        base_valid_mask = (
            _build_valid_mask_from_starts_ends(varlen_starts, varlen_ends, key_positions)
            .unsqueeze(0)
            .expand(b, sq, sk)
        )
    else:
        _, attn_score_mask, index_score_mask, base_valid_mask = _prepare_additive_mask(
            mask, sq=sq, sk=sk, b=b, device=attention_scores.device
        )
        # [b, np, sq, sk] + [1/b, 1, sq, sk] -> [b, np, sq, sk]
        attention_scores = attention_scores + attn_score_mask
        # [b, sq, sk] + [1/b, sq, sk] -> [b, sq, sk]
        index_scores = index_scores + index_score_mask

    # index_mask [b, sq, sk]
    index_mask = torch.full(
        (b, sq, sk), float("-inf"), dtype=torch.float32, device=attention_scores.device
    ).scatter_(-1, topk_indices, 0)

    if sparse_loss:
        # [b, np, sq, sk] + [b, 1, sq, sk] -> [b, np, sq, sk]
        attention_scores = attention_scores + index_mask.view(b, 1, sq, sk)
        # [b, sq, sk] + [b, sq, sk] -> [b, sq, sk]
        index_scores = index_scores + index_mask

    # Compute softmax for both
    attention_scores_softmax = torch.nn.functional.softmax(
        attention_scores, dim=-1, dtype=torch.float32
    )
    # Free attention_scores immediately
    del attention_scores

    index_scores_softmax = torch.nn.functional.softmax(index_scores, dim=-1, dtype=torch.float32)
    # Free index_scores - no longer needed after softmax
    del index_scores

    # Sum attention scores across heads: [b, np, sq, sk] -> [b, sq, sk]
    attention_scores_sum = attention_scores_softmax.sum(dim=1)
    # Free attention_scores_softmax
    del attention_scores_softmax

    if pg_collection.tp.size() > 1:
        # attention scores are scattered to TP ranks in head dimension.
        torch.distributed.all_reduce(attention_scores_sum.contiguous(), group=pg_collection.tp)

    # L1 normalize
    attention_scores_normalized = attention_scores_sum / attention_scores_sum.sum(
        dim=-1, keepdim=True
    )
    # Free attention_scores_sum - no longer needed after normalization
    del attention_scores_sum

    # Backward through loss = kl_div * loss_coeff
    # where kl_div = kl_per_element.sum(dim=-1).mean()
    grad_kl_div = grad_loss * loss_coeff  # scalar

    # Backward through mean: distribute gradient equally
    grad_kl_per_row = grad_kl_div / (b * sq)  # scalar value for each row

    # Backward through sum(dim=-1): broadcast back to [b, sq, sk]
    # Each element in a row contributes to the sum, so gradient is same for all
    grad_kl_per_element = grad_kl_per_row.view(1, 1, 1).expand(b, sq, sk)

    # Backward through kl_per_element = target * (log(target) - log(index))
    # ∂kl/∂index_softmax = -target / index_softmax
    grad_index_scores_softmax = (
        -attention_scores_normalized / (index_scores_softmax + 1e-10) * grad_kl_per_element
    )
    # Free attention_scores_normalized - no longer needed
    del attention_scores_normalized

    # Backward through softmax: ∂L/∂x = softmax * (∂L/∂softmax - sum(∂L/∂softmax * softmax))
    sum_grad = (grad_index_scores_softmax * index_scores_softmax).sum(dim=-1, keepdim=True)
    grad_index_scores_logits = index_scores_softmax * (grad_index_scores_softmax - sum_grad)
    # Free intermediate tensors
    del index_scores_softmax, grad_index_scores_softmax, sum_grad

    # Zero out gradients for masked positions.
    if sparse_loss:
        # Also apply index mask - only topk positions are valid.
        index_valid_mask = index_mask == 0  # [b, sq, sk]
        del index_mask
        valid_mask = base_valid_mask & index_valid_mask  # [b, sq, sk]
        del index_valid_mask
    else:
        del index_mask
        valid_mask = base_valid_mask  # [b, sq, sk]
    del base_valid_mask

    grad_index_scores_logits = grad_index_scores_logits * valid_mask.float()
    del valid_mask

    # Transpose from [b, sq, sk] to [sq, b, sk]
    grad_index_scores = grad_index_scores_logits.transpose(0, 1)  # [sq, b, sk]
    del grad_index_scores_logits

    # Backward through sum over heads: expand gradient
    grad_weighted_scores = grad_index_scores.unsqueeze(2)  # [sq, b, 1, sk]
    del grad_index_scores

    # Compute forward values needed for backward
    scores = torch.einsum('sbhd,tbd->sbht', q.float(), k.float())  # [sq, b, h, sk]
    # Compute relu_mask before relu (saves memory vs keeping both scores and relu output)
    relu_mask = scores > 0
    scores_after_relu = torch.relu(scores)
    del scores

    # Backward through multiplication by weights: index_scores_per_head * weights
    # ∂L/∂weights = grad * relu_scores (sum over sk)
    grad_weights = (grad_weighted_scores * scores_after_relu).sum(dim=-1)  # [sq, b, h]

    # ∂L/∂relu_scores = grad * weights
    grad_scores_after_relu = grad_weighted_scores * weights.unsqueeze(-1)  # [sq, b, h, sk]
    del grad_weighted_scores, scores_after_relu

    # Backward through ReLU
    grad_scores = grad_scores_after_relu * relu_mask.float()  # [sq, b, h, sk]
    del grad_scores_after_relu, relu_mask

    # Backward through einsum 'sbhd,tbd->sbht'
    # ∂L/∂q = einsum('sbht,tbd->sbhd', grad_scores, k)
    grad_q = torch.einsum('sbht,tbd->sbhd', grad_scores, k.float())  # [sq, b, h, d]
    # ∂L/∂k = einsum('sbht,sbhd->tbd', grad_scores, q)
    grad_k = torch.einsum('sbht,sbhd->tbd', grad_scores, q.float())  # [sk, b, d]
    del grad_scores

    return grad_q.to(q.dtype), grad_weights.to(weights.dtype), grad_k.to(k.dtype)


class FusedDSAIndexerLoss(torch.autograd.Function):
    """Fused implementation of DSA Indexer Loss."""

    @staticmethod
    def forward(
        ctx,
        q,
        weights,
        k,
        query,
        key,
        softmax_scale,
        topk,
        loss_coeff,
        mask,
        sparse_loss,
        pg_collection,
        varlen_starts=None,
        varlen_ends=None,
        key_positions=None,
    ):
        """
        Fused forward: index_scores never materialized in full.
        """
        topk_indices, loss = fwd_fused_indexer_loss_naive(
            q,
            weights,
            k,
            query,
            key,
            topk,
            softmax_scale,
            loss_coeff,
            mask,
            sparse_loss,
            pg_collection,
            varlen_starts=varlen_starts,
            varlen_ends=varlen_ends,
            key_positions=key_positions,
        )

        # Save for backward (recomputation strategy)
        ctx.save_for_backward(q, weights, k, query, key, topk_indices)
        ctx.softmax_scale = softmax_scale
        ctx.loss_coeff = loss_coeff
        ctx.sparse_loss = sparse_loss
        ctx.mask = mask
        ctx.pg_collection = pg_collection
        ctx.varlen_starts = varlen_starts
        ctx.varlen_ends = varlen_ends
        ctx.key_positions = key_positions

        return topk_indices, loss

    @staticmethod
    def backward(ctx, grad_topk_indices, grad_loss):
        """
        Backward: Recompute what we need.
        """
        q, weights, k, query, key, topk_indices = ctx.saved_tensors

        grad_q, grad_weights, grad_k = bwd_fused_indexer_loss_naive(
            q,
            weights,
            k,
            query,
            key,
            topk_indices,
            ctx.softmax_scale,
            ctx.loss_coeff,
            ctx.sparse_loss,
            ctx.mask,
            grad_loss,
            ctx.pg_collection,
            varlen_starts=ctx.varlen_starts,
            varlen_ends=ctx.varlen_ends,
            key_positions=ctx.key_positions,
        )

        # query and key are detached in forward, so return None for their gradients
        grads = [
            grad_q,
            grad_weights,
            grad_k,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ]
        return tuple(grads[: len(ctx.needs_input_grad)])


class DSAIndexerLossAutoScaler(torch.autograd.Function):
    """An AutoScaler that triggers the backward pass and scales the grad for indexer loss.

    This custom autograd function attaches a KL divergence loss to the activation
    to train the indexer to predict attention scores without affecting the forward pass.
    """

    main_loss_backward_scale: torch.Tensor = None

    @staticmethod
    def forward(ctx, output: torch.Tensor, indexer_loss: torch.Tensor):
        """Preserve the indexer_loss by storing it in the context to avoid garbage collection.

        Args:
            output: The output tensor (activation).
            indexer_loss: The indexer KL divergence loss tensor.

        Returns:
            torch.Tensor: The output tensor unchanged.
        """
        ctx.save_for_backward(indexer_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Compute and scale the gradient for indexer loss.

        Args:
            grad_output: The gradient of the output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The gradient of the output, scaled indexer loss
                gradient.
        """
        (indexer_loss,) = ctx.saved_tensors
        if DSAIndexerLossAutoScaler.main_loss_backward_scale is None:
            DSAIndexerLossAutoScaler.main_loss_backward_scale = torch.tensor(
                1.0, device=indexer_loss.device
            )
        indexer_loss_backward_scale = DSAIndexerLossAutoScaler.main_loss_backward_scale
        scaled_indexer_loss_grad = torch.ones_like(indexer_loss) * indexer_loss_backward_scale
        return grad_output, scaled_indexer_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        """Set the scale of the indexer loss.

        Args:
            scale: The scale value to set.
        """
        if DSAIndexerLossAutoScaler.main_loss_backward_scale is None:
            DSAIndexerLossAutoScaler.main_loss_backward_scale = scale
        else:
            DSAIndexerLossAutoScaler.main_loss_backward_scale.copy_(scale)


@dataclass
class DSAIndexerSubmodules:
    """
    Configuration class for specifying the submodules of an DSA Indexer.

    Args:
        linear_wq_b: Linear projection for query bottleneck expansion.
        linear_wk: Linear projection for key.
        k_norm: Layer normalization for key.
        linear_weights_proj: Linear projection for attention weights.
    """

    linear_wq_b: Union[ModuleSpec, type] = None
    linear_wk: Union[ModuleSpec, type] = None
    k_norm: Union[ModuleSpec, type] = None
    linear_weights_proj: Union[ModuleSpec, type] = None


@dataclass
class DSAttentionSubmodules:
    """
    Configuration class for specifying the submodules of DSAttention.

    Args:
        indexer: DSA Indexer module for computing sparse attention indices.
    """

    indexer: Union[ModuleSpec, type] = None


class DSAIndexer(MegatronModule):
    """
    DSA Lightning Indexer for DeepSeek Sparse Attention.

    Computes index scores to identify the top-k most relevant key-value pairs for each query in
    sparse attention.

    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L431-L480
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: DSAIndexerSubmodules,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        """Initialize the indexer.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
            submodules (DSAIndexerSubmodules): Indexer submodules specification.
            pg_collection (ProcessGroupCollection, optional): Process groups for the indexer.
        """
        super().__init__(config=config)
        self.hidden_size = self.config.hidden_size
        self.qk_pos_emb_head_dim = self.config.qk_pos_emb_head_dim
        self.q_lora_rank = (
            self.config.q_lora_rank
            if self.config.q_lora_rank is not None
            else self.config.hidden_size
        )

        self.index_n_heads = self.config.dsa_indexer_n_heads
        self.index_head_dim = self.config.dsa_indexer_head_dim
        self.index_topk = self.config.dsa_indexer_topk

        self.softmax_scale: float = self.index_head_dim**-0.5

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        self.pg_collection = pg_collection

        # Initialize Position Embedding.
        if self.config.rope_type == 'rope':
            self.rotary_pos_emb = RotaryEmbedding(
                self.qk_pos_emb_head_dim,
                rotary_percent=self.config.rotary_percent,
                rotary_base=self.config.rotary_base,
                cp_group=self.pg_collection.cp,
            )
        elif self.config.rope_type == 'yarn':
            self.rotary_pos_emb = YarnRotaryEmbedding(
                self.qk_pos_emb_head_dim,
                rotary_base=self.config.rotary_base,
                scaling_factor=self.config.rotary_scaling_factor,
                original_max_position_embeddings=self.config.original_max_position_embeddings,
                beta_fast=self.config.beta_fast,
                beta_slow=self.config.beta_slow,
                mscale=self.config.mscale,
                mscale_all_dim=self.config.mscale_all_dim,
                cp_group=self.pg_collection.cp,
            )
        else:
            raise ValueError(
                f'Unsupported RoPE type: {self.config.rope_type}, supported types are "rope" and '
                f'"yarn"'
            )

        self.linear_wq_b = build_module(
            submodules.linear_wq_b,
            self.q_lora_rank,
            self.index_n_heads * self.index_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

        self.linear_wk = build_module(
            submodules.linear_wk,
            self.hidden_size,
            self.index_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

        k_norm_config = copy.copy(self.config)
        k_norm_config.normalization = "LayerNorm"
        self.k_norm = build_module(
            submodules.k_norm,
            config=k_norm_config,
            hidden_size=self.index_head_dim,
            eps=self.config.layernorm_epsilon,
        )

        self.linear_weights_proj = build_module(
            submodules.linear_weights_proj,
            self.hidden_size,
            self.index_n_heads,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

    def _apply_rope(
        self,
        x: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        mscale: float,
        cu_seqlens: Optional[torch.Tensor] = None,
    ):
        """Apply RoPE to the input tensor."""
        # x_nope [seqlen, batch, *, index_head_dim - qk_pos_emb_head_dim]
        # x_pe   [seqlen, batch, *, qk_pos_emb_head_dim]
        x_nope, x_pe = torch.split(
            x, [self.index_head_dim - self.qk_pos_emb_head_dim, self.qk_pos_emb_head_dim], dim=-1
        )
        squeezed_batch_dim = False
        if cu_seqlens is not None and cu_seqlens.device != x_pe.device:
            cu_seqlens = cu_seqlens.to(device=x_pe.device)
        # THD RoPE path expects [t, h, d], while indexer tensors are [t, 1, h, d].
        if cu_seqlens is not None and x_pe.ndim == 4 and x_pe.size(1) == 1:
            x_pe = x_pe.squeeze(1)
            squeezed_batch_dim = True
        x_pe = apply_rotary_pos_emb(
            x_pe,
            rotary_pos_emb,
            config=self.config,
            cu_seqlens=cu_seqlens,
            mscale=mscale,
            cp_group=self.pg_collection.cp,
        )
        if squeezed_batch_dim:
            x_pe = x_pe.unsqueeze(1)
        # [seqlen, batch, *, index_head_dim]
        x = torch.cat([x_nope, x_pe], dim=-1)
        return x

    def forward_before_topk(
        self, x: torch.Tensor, qr: torch.Tensor, packed_seq_params: Optional[PackedSeqParams] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """All computations before topk."""
        packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"

        # =========================================
        # Prepare RoPE params
        # =========================================
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            None, None, x, self.config, packed_seq_params
        )
        if self.config.rope_type == "rope":
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)
            mscale = 1.0
        else:
            rotary_pos_emb, mscale = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)
        if packed_seq:
            cu_seqlens_q, cu_seqlens_kv = _get_packed_qk_cu_seqlens(packed_seq_params)
        else:
            cu_seqlens_q = cu_seqlens_kv = None

        # =========================================
        # Gather inputs if sp is enabled
        # =========================================
        if self.config.sequence_parallel and self.pg_collection.tp.size() > 1:
            x = gather_from_sequence_parallel_region(x, group=self.pg_collection.tp)
            qr = gather_from_sequence_parallel_region(qr, group=self.pg_collection.tp)

        # =========================================
        # Get sequence length and batch size
        # =========================================
        seqlen, bsz, _ = x.size()

        # =========================================
        # q linear and apply rope to q
        # =========================================
        # [seqlen, batch, q_lora_rank] -> [seqlen, batch, index_n_heads * index_head_dim]
        q, _ = self.linear_wq_b(qr)
        # [seqlen, batch, index_n_heads * index_head_dim]
        #   -> [seqlen, batch, index_n_heads, index_head_dim]
        q = q.reshape(seqlen, bsz, self.index_n_heads, self.index_head_dim)
        q = self._apply_rope(q, rotary_pos_emb, mscale, cu_seqlens=cu_seqlens_q)

        # =========================================
        # k linear and apply rope to k
        # =========================================
        # [seqlen, batch, hidden_size] -> [seqlen, batch, index_head_dim]
        k, _ = self.linear_wk(x)
        k = self.k_norm(k.float()).to(dtype=k.dtype)
        # [seqlen, batch, index_head_dim] -> [seqlen, batch, 1, index_head_dim]
        k = k.reshape(seqlen, bsz, 1, self.index_head_dim)
        k = self._apply_rope(k, rotary_pos_emb, mscale, cu_seqlens=cu_seqlens_kv)
        # [seqlen, batch, 1, index_head_dim] -> [seqlen, batch, index_head_dim]
        k = k.reshape(seqlen, bsz, self.index_head_dim)

        # =========================================
        # Rotate activation
        # =========================================
        q = rotate_activation(q)
        k = rotate_activation(k)

        # =========================================
        # Prepare weights for index scores
        # =========================================
        # [seqlen, batch, hidden_size] -> [seqlen, batch, index_n_heads]
        weights, _ = self.linear_weights_proj(x)
        weights = weights * (self.index_n_heads**-0.5) * self.softmax_scale

        return q, k, weights

    def forward_with_scores(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for DSA Indexer that returns both index scores and top-k indices.

        This is used when KL loss is enabled to compare indexer scores with true attention scores.

        Args:
            x: hidden states [seqlen, batch, hidden_size].
            qr: Low-rank query tensor [seqlen, batch, q_lora_rank].
            mask: Optional additive attention mask [seqlen, seqlen] or
                [batch, seqlen, seqlen].
            packed_seq_params: Packed sequence parameters for variable length sequences.

        Returns:
            index_scores: Index scores [batch, seqlen, seqlen].
            topk_indices: Top-k indices [batch, seqlen, index_topk].
        """
        # [seqlen, batch, index_n_heads * index_head_dim]
        # [seqlen, batch, index_head_dim]
        # [seqlen, batch, index_n_heads]
        q, k, weights = self.forward_before_topk(x, qr, packed_seq_params)

        # [batch, seqlen, seqlen], [batch, seqlen, index_topk]
        index_scores, topk_indices = fused_qk_topk_naive(q, k, weights, self.index_topk, mask)

        return index_scores, topk_indices

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """
        Forward pass for DSA Indexer.

        Args:
            x: hidden states [seqlen, batch, hidden_size].
            qr: Low-rank query tensor [seqlen, batch, q_lora_rank].
            mask: Attention mask [batch, seqlen, seqlen].
            packed_seq_params: Packed sequence parameters for variable length sequences.

        Returns:
            topk_indices: Top-k indices for sparse attention [batch, seqlen, index_topk].
        """
        _, topk_indices = self.forward_with_scores(x, qr, mask, packed_seq_params)
        return topk_indices


def unfused_dsa_fn(
    query,
    key,
    value,
    topk_indices,
    softmax_scale,
    mask: Optional[torch.Tensor] = None,
    varlen_starts: Optional[torch.Tensor] = None,
    varlen_ends: Optional[torch.Tensor] = None,
    key_positions: Optional[torch.Tensor] = None,
):
    """
    Unfused sparse attention implementation.

    This path uses chunked sparse softmax accumulation over top-k selected keys
    to avoid materializing full [b, np, sq, skv] attention score tensors.
    """
    if value is None:
        raise NotImplementedError("DSAttention unfused path requires value tensor.")

    query, query_was_thd = _ensure_sbhd(query, "query")
    key, _ = _ensure_sbhd(key, "key")
    value, _ = _ensure_sbhd(value, "value")

    sq, b, np, hn = query.size()
    skv = key.size(0)
    nk = key.size(2)
    hnv = value.size(3)
    nv = value.size(2)

    # [sq, b, np, hn] -> [b, np, sq, hn]
    query_b = query.permute(1, 2, 0, 3).contiguous()
    # [skv, b, nk, hn] -> [b, nk, skv, hn]
    key_b = key.permute(1, 2, 0, 3).contiguous()
    # [skv, b, nv, hnv] -> [b, nv, skv, hnv]
    value_b = value.permute(1, 2, 0, 3).contiguous()
    if nk == 1 and np > 1:
        key_b = key_b.expand(b, np, skv, hn)
    else:
        assert nk == np, "key head count must be 1 (MQA) or match query heads"
    if nv == 1 and np > 1:
        value_b = value_b.expand(b, np, skv, hnv)
    else:
        assert nv == np, "value head count must be 1 (MQA) or match query heads"

    row_mask, varlen_starts, varlen_ends, key_positions = _prepare_sparse_mask_context(
        mask=mask,
        varlen_starts=varlen_starts,
        varlen_ends=varlen_ends,
        key_positions=key_positions,
        sq=sq,
        sk=skv,
        b=b,
        device=query.device,
    )

    seq_chunk_size = 512
    head_chunk_size = 16
    topk_chunk_size = 1024
    safe_k_max = max(0, skv - 1)
    output = torch.empty((sq, b, np * hnv), dtype=value.dtype, device=query.device)

    for bi in range(b):
        for h0 in range(0, np, head_chunk_size):
            h1 = min(h0 + head_chunk_size, np)
            h_chunk = h1 - h0
            out_h0 = h0 * hnv
            out_h1 = h1 * hnv
            k_chunk = key_b[bi, h0:h1, :, :].contiguous()  # [h_chunk, skv, hn]
            v_chunk = value_b[bi, h0:h1, :, :].contiguous()  # [h_chunk, skv, hnv]
            flat_k = k_chunk.reshape(h_chunk * skv, hn)
            flat_v = v_chunk.reshape(h_chunk * skv, hnv)
            head_offsets = (
                torch.arange(h_chunk, device=query.device, dtype=torch.int64).view(-1, 1, 1) * skv
            )

            for s0 in range(0, sq, seq_chunk_size):
                s1 = min(s0 + seq_chunk_size, sq)
                s_len = s1 - s0
                idx_seq_raw = topk_indices[bi, s0:s1]  # [s_len, topk]
                if idx_seq_raw.dtype != torch.int64 or idx_seq_raw.device != query.device:
                    idx_seq_raw = idx_seq_raw.to(dtype=torch.int64, device=query.device)
                valid_seq = idx_seq_raw >= 0
                idx_seq = idx_seq_raw.clamp(min=0, max=safe_k_max)
                q_chunk = query_b[bi, h0:h1, s0:s1, :]  # [h_chunk, s_len, hn]

                m = _get_scratch_buffer(
                    "unfused_dsa_m", (h_chunk, s_len), torch.float32, query.device
                )
                l = _get_scratch_buffer(
                    "unfused_dsa_l", (h_chunk, s_len), torch.float32, query.device
                )
                acc = _get_scratch_buffer(
                    "unfused_dsa_acc", (h_chunk, s_len, hnv), torch.float32, query.device
                )
                m.fill_(float("-inf"))
                l.zero_()
                acc.zero_()

                for t0 in range(0, idx_seq.size(-1), topk_chunk_size):
                    t1 = min(t0 + topk_chunk_size, idx_seq.size(-1))
                    idx_topk = idx_seq[:, t0:t1]  # [s_len, tk]
                    valid_t = valid_seq[:, t0:t1]  # [s_len, tk]
                    flat_idx = idx_topk.unsqueeze(0) + head_offsets  # [h_chunk, s_len, tk]
                    k_sel = flat_k.index_select(0, flat_idx.reshape(-1)).view(
                        h_chunk, s_len, -1, hn
                    )
                    v_sel = flat_v.index_select(0, flat_idx.reshape(-1)).view(
                        h_chunk, s_len, -1, hnv
                    )
                    logits = (q_chunk.float().unsqueeze(2) * k_sel.float()).sum(
                        dim=-1
                    ) * softmax_scale

                    valid_2d, mask_bias = _gather_sparse_topk_validity_and_bias(
                        idx_topk=idx_topk,
                        valid_t=valid_t,
                        bi=bi,
                        s0=s0,
                        s1=s1,
                        row_mask=row_mask,
                        varlen_starts=varlen_starts,
                        varlen_ends=varlen_ends,
                        key_positions=key_positions,
                        dtype=torch.float32,
                    )
                    if mask_bias is not None:
                        logits = logits + mask_bias.unsqueeze(0)
                    logits = logits.masked_fill(
                        ~valid_2d.unsqueeze(0).expand(h_chunk, -1, -1), float("-inf")
                    )
                    m_new = torch.maximum(m, logits.max(dim=-1).values)
                    alpha = torch.exp(m - m_new)
                    alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
                    p = torch.exp(logits - m_new.unsqueeze(-1))
                    p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
                    acc = acc * alpha.unsqueeze(-1) + torch.einsum(
                        "hst,hstd->hsd", p, v_sel.float()
                    )
                    l = l * alpha + p.sum(dim=-1)
                    m = m_new

                out_chunk = (acc / l.clamp_min(1e-10).unsqueeze(-1)).to(dtype=value.dtype)
                output[s0:s1, bi, out_h0:out_h1] = out_chunk.permute(1, 0, 2).reshape(
                    s_len, h_chunk * hnv
                )

    if query_was_thd:
        output = output.squeeze(1)
    return output


class DSAttention(MegatronModule):
    """
    This module implements sparse attention mechanism using an DSA Indexer to compute top-k
    attention indices for reducing computational complexity.

    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L491-L597
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: DSAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        k_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        cp_comm_type: str = "p2p",
        pg_collection: ProcessGroupCollection = None,
    ):
        super().__init__(config=config)

        self.layer_number = layer_number

        self.indexer = build_module(
            submodules.indexer, config=self.config, pg_collection=pg_collection
        )

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(
                k_channels if k_channels is not None else config.kv_channels
            )
        self.softmax_scale = softmax_scale
        self.cp_comm_type = _normalize_cp_comm_type(cp_comm_type)
        self._last_debug_path_msg = None

    def _debug_print_path(self, msg: str) -> None:
        """Print DSAttention path transitions for debugging."""
        if msg == self._last_debug_path_msg:
            return
        self._last_debug_path_msg = msg
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            logger.info("[DSAttention][L%s] %s", self.layer_number, msg)
            return
        if torch.distributed.get_rank() == 0:
            logger.info("[DSAttention][L%s] %s", self.layer_number, msg)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        x: torch.Tensor,
        qr: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attn_mask_type: AttnMaskType = None,
        attention_bias: torch.Tensor = None,
        packed_seq_params: PackedSeqParams = None,
        up_v_weight: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for Sparse Attention.

        Args:
            query: Query tensor [sq, b, np, hn] or packed [t, np, hn].
            key: Key tensor [skv, b, np, hn] or packed [t, np, hn].
            value: Value tensor [skv, b, np, hnv] or packed [t, np, hnv].
            x: Original hidden states [sq, b, hidden_size].
            qr: Low-rank query representation [sq, b, q_lora_rank].
            position_ids: Optional position ids [b, sq], used by allgather CP causal masking.
            attention_mask: Attention mask tensor [b, 1, sq, sk].
            attn_mask_type: Type of attention mask.
            attention_bias: Optional attention bias.
            packed_seq_params: Packed sequence parameters.

        Returns:
            output: Output tensor [sq, b, hidden_size]
        """
        query, _ = _ensure_sbhd(query, "query")
        key, _ = _ensure_sbhd(key, "key")
        if value is not None:
            value, _ = _ensure_sbhd(value, "value")
        if up_v_weight is not None:
            assert up_v_weight.ndim == 3, "up_v_weight must be [heads, v_head_dim, kv_lora_rank]"
            up_v_weight = up_v_weight.to(device=query.device, dtype=query.dtype).contiguous()
            if value is not None:
                raise RuntimeError(
                    "DSAttention received up_v_weight with explicit value tensor. "
                    "For absorbed DSA path, value must be None."
                )

        latent_v_channels = int(getattr(self.config, "kv_lora_rank", 0) or 0)
        qk_pos_dim = int(getattr(self.config, "qk_pos_emb_head_dim", 0) or 0)
        expected_absorbed_dim = latent_v_channels + qk_pos_dim
        absorbed_layout = (
            latent_v_channels > 0
            and expected_absorbed_dim > 0
            and key.size(2) == 1
            and query.size(-1) == key.size(-1) == expected_absorbed_dim
        )
        absorbed_mla = absorbed_layout
        if value is None and not absorbed_mla:
            raise RuntimeError(
                "DSAttention received value=None but query/key are not in absorbed layout. "
                f"query_hdim={query.size(-1)}, key_hdim={key.size(-1)}, key_heads={key.size(2)}, "
                f"expected_absorbed_dim={expected_absorbed_dim}"
            )
        if up_v_weight is not None and not absorbed_mla:
            raise RuntimeError(
                "DSAttention received up_v_weight but absorbed layout was not detected. "
                f"query_hdim={query.size(-1)}, key_hdim={key.size(-1)}, key_heads={key.size(2)}, "
                f"expected_absorbed_dim={expected_absorbed_dim}"
            )
        if self.training and absorbed_mla and up_v_weight is None:
            raise RuntimeError(
                "Absorbed DSAttention training requires up_v_weight for latent-to-value projection."
            )

        sq, b, _, _ = query.size()

        cp_group = getattr(self.indexer.pg_collection, "cp", None)
        cp_size = cp_group.size() if cp_group is not None else 1
        cp_rank = cp_group.rank() if cp_group is not None else 0

        if cp_size > 1:
            assert (
                self.cp_comm_type == "allgather"
            ), "DSAttention context parallelism currently supports cp_comm_type=allgather only."
            # For allgather CP, keys/values are expected in full-sequence order.
            # Gather only if inputs are local-sequence tensors.
            if key.size(0) == sq:
                key = gather_from_sequence_parallel_region(key, group=cp_group)
            if value is not None and value.size(0) == sq:
                value = gather_from_sequence_parallel_region(value, group=cp_group)

        skv = key.size(0)

        # Detach x and qr to prevent gradients of indexer from flowing back to the main model.
        x = x.detach()
        qr = qr.detach()

        indexer_loss_coeff = self.config.dsa_indexer_loss_coeff
        use_indexer_loss = self.training and torch.is_grad_enabled() and indexer_loss_coeff > 0
        float_mask, varlen_params = _build_dsattention_forward_mask(
            sq=sq,
            skv=skv,
            b=b,
            device=x.device,
            cp_size=cp_size,
            cp_rank=cp_rank,
            cp_comm_type=self.cp_comm_type,
            cp_group=cp_group,
            attn_mask_type=attn_mask_type,
            attention_mask=attention_mask,
            position_ids=position_ids,
            packed_seq_params=packed_seq_params,
        )
        if varlen_params is not None:
            varlen_starts, varlen_ends, key_positions = varlen_params
        else:
            varlen_starts = varlen_ends = key_positions = None

        # ===================================
        # Prepare indexer inputs / top-k
        # ===================================
        q, k, weights = self.indexer.forward_before_topk(x, qr, packed_seq_params)
        if cp_size > 1 and k.size(0) == sq:
            k = gather_from_sequence_parallel_region(k, group=cp_group)
        fused_bounds = _build_fused_indexer_varlen_bounds(
            sq=sq,
            skv=skv,
            device=q.device,
            mask=float_mask,
            varlen_starts=varlen_starts,
            varlen_ends=varlen_ends,
            key_positions=key_positions,
        )

        topk_indices = None
        indexer_path = "unknown" if use_indexer_loss else "naive_topk"
        indexer_loss = None

        if use_indexer_loss:
            # ===================================
            # Attach indexer topk and loss
            # ===================================
            sparse_indexer_loss = self.config.dsa_indexer_use_sparse_loss
            if sparse_indexer_loss and fused_bounds is not None:
                starts_i32, ends_i32 = fused_bounds
                block_size = int(getattr(self, "fused_indexer_block_size", 8192))
                fused_topk_with_loss = _fused_qk_topk_lighting_with_streaming_sparse_kl(
                    q,
                    k,
                    weights,
                    self.indexer.index_topk,
                    starts_i32,
                    ends_i32,
                    block_size=max(1, block_size),
                    query=query.detach(),
                    key=key.detach(),
                    softmax_scale=self.softmax_scale,
                    loss_coeff=indexer_loss_coeff,
                    pg_collection=self.indexer.pg_collection,
                )
                if fused_topk_with_loss is not None:
                    topk_indices, indexer_loss = fused_topk_with_loss
                    indexer_path = "fused_topk_sparse_kl"

            if topk_indices is None or indexer_loss is None:
                # Legacy dense path fallback.
                key_for_loss = key.detach()
                if absorbed_mla and key_for_loss.size(2) == 1 and query.size(2) > 1:
                    key_for_loss = key_for_loss.expand(-1, -1, query.size(2), -1).contiguous()
                topk_indices, indexer_loss = FusedDSAIndexerLoss.apply(
                    q,
                    weights,
                    k,
                    query.detach(),
                    key_for_loss,
                    self.softmax_scale,
                    self.indexer.index_topk,
                    indexer_loss_coeff,
                    float_mask,
                    sparse_indexer_loss,
                    self.indexer.pg_collection,
                    varlen_starts,
                    varlen_ends,
                    key_positions,
                )
                indexer_path = "dense_indexer_loss_fallback"

            # Save indexer loss for logging.
            if indexer_loss_coeff > 0:
                DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                    loss=indexer_loss,
                    layer_number=self.layer_number,
                    num_layers=self.config.num_layers,
                )
        else:
            # ===================================
            # Get top-k indices
            # ===================================
            if fused_bounds is not None:
                starts_i32, ends_i32 = fused_bounds
                block_size = int(getattr(self, "fused_indexer_block_size", 8192))
                topk_indices = _fused_qk_topk_lighting(
                    q,
                    k,
                    weights,
                    self.indexer.index_topk,
                    starts_i32,
                    ends_i32,
                    block_size=max(1, block_size),
                )
                if topk_indices is not None:
                    indexer_path = "fused_topk"

            if topk_indices is None:
                _, topk_indices = fused_qk_topk_naive(
                    q,
                    k,
                    weights,
                    self.indexer.index_topk,
                    mask=float_mask,
                    varlen_starts=varlen_starts,
                    varlen_ends=varlen_ends,
                    key_positions=key_positions,
                )

        # ===================================
        # Run sparse attention kernel
        # ===================================
        output, sparse_attn_path = _run_sparse_attention(
            absorbed_mla=absorbed_mla,
            query=query,
            key=key,
            value=value,
            up_v_weight=up_v_weight,
            topk_indices=topk_indices,
            softmax_scale=self.softmax_scale,
            config=self.config,
            mask=float_mask,
            varlen_starts=varlen_starts,
            varlen_ends=varlen_ends,
            key_positions=key_positions,
        )
        sparse_attn_reason = _build_sparse_attn_reason(
            sparse_attn_path=sparse_attn_path,
            absorbed_mla=absorbed_mla,
            query=query,
            key=key,
            topk_indices=topk_indices,
            config=self.config,
        )
        self._debug_print_path(
            f"use_indexer_loss={use_indexer_loss}, indexer={indexer_path}, "
            f"sparse_attn={sparse_attn_path}, cp_size={cp_size}, absorbed={absorbed_mla}, "
            f"sparse_attn_reason={sparse_attn_reason}"
        )

        if use_indexer_loss:
            if indexer_loss is None:
                raise RuntimeError("Indexer loss path did not produce a valid loss tensor.")
            output = DSAIndexerLossAutoScaler.apply(output, indexer_loss)

        return _normalize_dsattention_output_rank(output, x.ndim)
