# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Masking helpers for DeepSeek sparse attention."""

from typing import Optional, Tuple

import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant import dsa_layout

__all__ = [
    "apply_sparse_validity_to_index_mask",
    "apply_starts_ends_mask_to_scores",
    "build_causal_mask_from_positions",
    "build_dsattention_forward_mask",
    "build_fused_indexer_varlen_bounds",
    "build_valid_mask_from_starts_ends",
    "extract_query_valid_rows_from_packed_seq_params",
    "gather_sparse_topk_validity_and_bias",
    "generate_varlen_mask_params",
    "generate_varlen_mask_params_for_positions",
    "normalize_query_valid_rows",
    "normalize_varlen_bounds",
    "prepare_additive_mask",
    "prepare_sparse_mask_context",
    "scatter_topk_into_index_mask",
]


def build_causal_mask_from_positions(
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


def generate_varlen_mask_params(cu_seqlens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate row-wise [start, end) key bounds for packed causal masking."""
    assert cu_seqlens.ndim == 1 and cu_seqlens.numel() >= 2, "invalid cu_seqlens"
    cu_seqlens = cu_seqlens.to(dtype=torch.int64)
    seq_len = int(cu_seqlens[-1].item())
    q_indices = torch.arange(seq_len, dtype=torch.int64, device=cu_seqlens.device)
    seq_indices = torch.searchsorted(cu_seqlens, q_indices, right=True) - 1
    starts = cu_seqlens[seq_indices]
    ends = q_indices + 1
    return starts, ends


def generate_varlen_mask_params_for_positions(
    cu_seqlens: torch.Tensor, query_positions: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate packed causal bounds only for the requested query positions."""
    assert cu_seqlens.ndim == 1 and cu_seqlens.numel() >= 2, "invalid cu_seqlens"
    assert query_positions.dtype in (torch.int32, torch.int64), "query_positions must be integer"
    cu_seqlens = cu_seqlens.to(device=query_positions.device, dtype=torch.int64)
    query_positions = query_positions.to(dtype=torch.int64)
    seq_indices = torch.searchsorted(cu_seqlens[1:], query_positions, right=True)
    starts = cu_seqlens[seq_indices]
    ends = query_positions + 1
    return starts, ends


def build_valid_mask_from_starts_ends(
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


def apply_starts_ends_mask_to_scores(
    scores: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor, key_positions: torch.Tensor
) -> torch.Tensor:
    """Apply varlen starts/ends mask to score tensor.

    Supports scores with shape [b, sq, sk] or [b, np, sq, sk].
    """
    valid = build_valid_mask_from_starts_ends(starts, ends, key_positions)
    if scores.ndim == 3:
        return scores.masked_fill(~valid.unsqueeze(0), float("-inf"))
    if scores.ndim == 4:
        return scores.masked_fill(~valid.unsqueeze(0).unsqueeze(0), float("-inf"))
    raise ValueError(f"Unsupported scores ndim={scores.ndim}, expected 3 or 4.")


def normalize_varlen_bounds(
    *,
    mask: Optional[torch.Tensor],
    varlen_starts: Optional[torch.Tensor],
    varlen_ends: Optional[torch.Tensor],
    key_positions: Optional[torch.Tensor],
    sk: int,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Validate mask/varlen exclusivity and normalize varlen bounds to int64 tensors."""
    if mask is not None and varlen_starts is not None:
        raise ValueError("mask and varlen_starts are mutually exclusive")
    if varlen_starts is None:
        return None, None, None
    if varlen_ends is None:
        raise ValueError("varlen_ends is required when varlen_starts is provided")

    varlen_starts_i64 = varlen_starts.to(device=device, dtype=torch.int64)
    varlen_ends_i64 = varlen_ends.to(device=device, dtype=torch.int64)
    if key_positions is None:
        key_positions_i64 = torch.arange(sk, dtype=torch.int64, device=device)
    else:
        key_positions_i64 = key_positions.to(device=device, dtype=torch.int64)
    return varlen_starts_i64, varlen_ends_i64, key_positions_i64


def _build_default_causal_mask(sq: int, sk: int, device: torch.device) -> torch.Tensor:
    """Build standard upper-triangular additive causal mask."""
    return torch.triu(
        torch.full((sq, sk), float("-inf"), dtype=torch.float32, device=device), diagonal=1
    )


def prepare_additive_mask(
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


def prepare_sparse_mask_context(
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
    varlen_starts_i64, varlen_ends_i64, key_positions_i64 = normalize_varlen_bounds(
        mask=mask,
        varlen_starts=varlen_starts,
        varlen_ends=varlen_ends,
        key_positions=key_positions,
        sk=sk,
        device=device,
    )
    if varlen_starts_i64 is not None:
        return None, varlen_starts_i64, varlen_ends_i64, key_positions_i64

    _, _, index_score_mask, _ = prepare_additive_mask(mask, sq=sq, sk=sk, b=b, device=device)
    return index_score_mask, None, None, None


def apply_sparse_validity_to_index_mask(
    index_mask: torch.Tensor,
    *,
    row_mask: Optional[torch.Tensor],
    varlen_starts: Optional[torch.Tensor],
    varlen_ends: Optional[torch.Tensor],
    key_positions: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply either varlen or additive mask validity constraints to index_mask."""
    if varlen_starts is not None:
        varlen_starts, varlen_ends, key_positions = normalize_varlen_bounds(
            mask=None,
            varlen_starts=varlen_starts,
            varlen_ends=varlen_ends,
            key_positions=key_positions,
            sk=index_mask.size(-1),
            device=index_mask.device,
        )
        valid_mask = build_valid_mask_from_starts_ends(
            varlen_starts, varlen_ends, key_positions
        ).unsqueeze(0)
        return index_mask.masked_fill(~valid_mask, float("-inf"))

    if row_mask is None:
        raise ValueError("row_mask is required when varlen_starts is None")
    return index_mask + row_mask


def gather_sparse_topk_validity_and_bias(
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


def scatter_topk_into_index_mask(
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


def normalize_query_valid_rows(
    query_valid_rows: Optional[torch.Tensor], *, b: int, sq: int, device: torch.device
) -> Optional[torch.Tensor]:
    """Normalize optional query-row validity mask to shape [b, sq]."""
    if query_valid_rows is None:
        return None
    query_valid_rows = query_valid_rows.to(device=device, dtype=torch.bool)
    if query_valid_rows.ndim == 1:
        if query_valid_rows.numel() != sq:
            raise ValueError(
                f"query_valid_rows length mismatch: expected {sq}, got {query_valid_rows.numel()}"
            )
        return query_valid_rows.unsqueeze(0).expand(b, sq)
    if query_valid_rows.ndim == 2:
        if query_valid_rows.shape == (1, sq):
            return query_valid_rows.expand(b, sq)
        if query_valid_rows.shape != (b, sq):
            expected_shape = (b, sq)
            raise ValueError(
                f"query_valid_rows shape mismatch: expected {expected_shape}, "
                f"got {tuple(query_valid_rows.shape)}"
            )
        return query_valid_rows
    raise ValueError(f"query_valid_rows should be 1D or 2D tensor, got {query_valid_rows.ndim}D.")


def extract_query_valid_rows_from_packed_seq_params(
    packed_seq_params: Optional[PackedSeqParams], *, b: int, sq: int, device: torch.device
) -> Optional[torch.Tensor]:
    """Extract optional real-token query-row mask from packed sequence metadata."""
    if packed_seq_params is None:
        return None
    query_valid_rows = getattr(packed_seq_params, "real_token_mask_q", None)
    if query_valid_rows is None:
        return None
    return normalize_query_valid_rows(query_valid_rows, b=b, sq=sq, device=device)


def build_dsattention_forward_mask(
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
    packed_query_positions: Optional[torch.Tensor] = None,
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
            cu_seqlens_q, _ = dsa_layout.get_packed_qk_cu_seqlens(packed_seq_params)
            cu_seqlens_q = cu_seqlens_q.to(device=device, dtype=torch.int64)
            if cp_size > 1:
                if packed_query_positions is not None:
                    query_idx = packed_query_positions.to(device=device, dtype=torch.int64)
                    key_idx = torch.arange(skv, dtype=torch.int64, device=device)
                else:
                    query_idx, key_idx = dsa_layout.get_cp_positions_from_layout(
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
            varlen_starts, varlen_ends = generate_varlen_mask_params_for_positions(
                cu_seqlens_q, query_idx
            )
            return None, (varlen_starts, varlen_ends, key_idx)

        if cp_size > 1:
            query_pos = dsa_layout.extract_query_positions_from_position_ids(
                position_ids, sq, device
            )
            if query_pos is None:
                query_pos, key_pos = dsa_layout.get_cp_positions_from_layout(
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
            return build_causal_mask_from_positions(query_pos, key_pos), None

        return _build_default_causal_mask(sq, skv, device=device), None

    assert attention_mask is not None, "attention_mask is required when attn_mask_type is None"
    assert attention_mask.shape == (b, 1, sq, skv), "attention_mask shape mismatch"
    mask = attention_mask[:, 0, :, :]
    float_mask = torch.zeros_like(mask, dtype=torch.float32).masked_fill(mask, float("-inf"))
    return float_mask, None


def build_fused_indexer_varlen_bounds(
    *,
    sq: int,
    skv: int,
    device: torch.device,
    mask: Optional[torch.Tensor],
    varlen_starts: Optional[torch.Tensor],
    varlen_ends: Optional[torch.Tensor],
    key_positions: Optional[torch.Tensor],
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Build row-wise contiguous [start, end) key bounds for optional fused indexer kernels."""
    varlen_starts, varlen_ends, key_positions = normalize_varlen_bounds(
        mask=mask,
        varlen_starts=varlen_starts,
        varlen_ends=varlen_ends,
        key_positions=key_positions,
        sk=skv,
        device=device,
    )
    if varlen_starts is not None:
        expected_key_pos = torch.arange(skv, dtype=torch.int64, device=device)
        if not torch.equal(key_positions, expected_key_pos):
            return None
        return (
            varlen_starts.to(dtype=torch.int32, device=device),
            varlen_ends.to(dtype=torch.int32, device=device),
        )

    if mask is None:
        ends = torch.arange(1, sq + 1, dtype=torch.int64, device=device).clamp_max(skv)
        starts = torch.zeros_like(ends)
        return starts.to(dtype=torch.int32), ends.to(dtype=torch.int32)

    if mask.ndim == 3:
        # Fused indexers generally use one shared bounds schedule. For batched masks, only
        # enable a fused path when all batch masks are identical.
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
