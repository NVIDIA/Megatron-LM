# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from functools import lru_cache

import torch


def _pool_compressor_values(
    kv: torch.Tensor, score: torch.Tensor, output_dtype: torch.dtype
) -> torch.Tensor:
    """Pool compressor values with FP32 weights, products, and reduction."""
    weights = torch.softmax(score, dim=1, dtype=torch.float32)
    return (kv.float() * weights).sum(dim=1).to(output_dtype)


@lru_cache(maxsize=8)
def _get_window_topk_idxs_cached(window_size: int, seqlen: int, device_str: str) -> torch.Tensor:
    """Compute sliding-window indices for a single sequence (cached).

    Returns:
        indices: [seqlen, window_size] int tensor, -1 for invalid positions.
    """
    base = torch.arange(seqlen, device=device_str).unsqueeze(1)
    offsets = torch.arange(window_size, device=device_str)
    matrix = (base - window_size + 1).clamp(min=0) + offsets
    matrix = torch.where(matrix > base, -1, matrix)
    return matrix


def get_window_topk_idxs(
    window_size: int, batch_size: int, seqlen: int, device: torch.device
) -> torch.Tensor:
    """Sliding-window indices [batch, seqlen, window_size]."""
    matrix = _get_window_topk_idxs_cached(window_size, seqlen, str(device))
    return matrix.unsqueeze(0).expand(batch_size, -1, -1)


@lru_cache(maxsize=8)
def _get_compress_topk_idxs_cached(
    ratio: int, seqlen: int, offset: int, device_str: str
) -> torch.Tensor:
    """Compute all-compressed-positions indices for a single sequence (cached).

    Returns:
        indices: [seqlen, seqlen // ratio] int tensor, -1 for future positions.
    """
    n_compressed = seqlen // ratio
    matrix = torch.arange(n_compressed, device=device_str).repeat(seqlen, 1)
    mask = matrix >= torch.arange(1, seqlen + 1, device=device_str).unsqueeze(1) // ratio
    matrix = torch.where(mask, -1, matrix + offset)
    return matrix


def get_compress_topk_idxs(
    ratio: int, batch_size: int, seqlen: int, offset: int, device: torch.device
) -> torch.Tensor:
    """All-compressed-position indices [batch, seqlen, seqlen // ratio]."""
    matrix = _get_compress_topk_idxs_cached(ratio, seqlen, offset, str(device))
    return matrix.unsqueeze(0).expand(batch_size, -1, -1)


@lru_cache(maxsize=8)
def _get_compress_causal_mask_cached(
    ratio: int, seqlen: int, n_compressed: int, device_str: str
) -> torch.Tensor:
    """Return the additive causal mask for compressed positions (cached)."""
    compressed_positions = torch.arange(n_compressed, device=device_str).unsqueeze(0)
    valid_counts = torch.arange(1, seqlen + 1, device=device_str).unsqueeze(1) // ratio
    return torch.where(compressed_positions >= valid_counts, float("-inf"), 0.0)


@lru_cache(maxsize=8)
def _get_compress_valid_counts_cached(ratio: int, seqlen: int, device_str: str) -> torch.Tensor:
    """Return the number of causally valid compressed positions per query (cached)."""
    return torch.arange(1, seqlen + 1, device=device_str).unsqueeze(1) // ratio


def unfused_compressed_sparse_attn(
    query: torch.Tensor,
    kv_full: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Differentiable sparse attention with MQA and attention sink.

    Determinism:
        Unknown. Bit-exact forward and backward repeatability has not been certified.

    Args:
        query:        [sq, b, np, hn]   multi-head query.
        kv_full:      [n_kv, b, hn]     single-head KV (original + compressed).
        attn_sink:    [np]              per-head learnable bias.
        topk_indices: [b, sq, topk]     indices into kv_full (int32, -1 = invalid).
        softmax_scale: float

    Returns:
        output:       [sq, b, np * hn]
    """
    sq, b, np_, hn = query.size()
    if attn_sink.ndim != 1 or attn_sink.numel() != np_:
        raise ValueError(
            f"attn_sink must contain one value per query head ({np_}), "
            f"got shape {tuple(attn_sink.shape)}."
        )

    # --- Gather KV at topk positions ---
    # Flatten batch and KV position before gathering. Gathering from a logical
    # [b, sq, n_kv, hn] expanded view makes gather backward allocate that entire
    # dense shape before reducing the stride-0 query dimension.
    n_kv = kv_full.size(0)
    topk = topk_indices.size(-1)
    kv_flat = kv_full.permute(1, 0, 2).reshape(b * n_kv, hn)
    batch_offsets = (torch.arange(b, device=kv_full.device, dtype=torch.int64) * n_kv).view(b, 1, 1)
    safe_indices = topk_indices.clamp(min=0).to(dtype=torch.int64) + batch_offsets
    kv_gathered = kv_flat.index_select(0, safe_indices.reshape(-1)).view(b, sq, topk, hn)

    # --- Attention scores ---
    # query: [sq, b, np, hn] -> [b, np, sq, hn]
    q = query.permute(1, 2, 0, 3).float()
    kv_g = kv_gathered.float()  # [b, sq, topk, hn]

    # [b, np, sq, topk]
    scores = torch.einsum("bnsh,bskh->bnsk", q, kv_g) * softmax_scale

    # Mask invalid
    invalid_mask = (topk_indices < 0).unsqueeze(1)  # [b, 1, sq, topk]
    scores = scores.masked_fill(invalid_mask, float("-inf"))

    # --- Softmax with attention sink ---
    sink = attn_sink.view(1, np_, 1, 1).float()
    scores_max = scores.max(dim=-1, keepdim=True).values  # [b, np, sq, 1]
    scores_max = torch.max(scores_max, sink)

    exp_scores = torch.exp(scores - scores_max)  # [b, np, sq, topk]
    exp_sink = torch.exp(sink - scores_max)  # [1, np, 1, 1]

    sum_exp = exp_scores.sum(dim=-1, keepdim=True) + exp_sink
    attn_weights = exp_scores / sum_exp  # [b, np, sq, topk]

    # --- Weighted sum ---
    output = torch.einsum("bnsk,bskh->bnsh", attn_weights, kv_g)
    output = output.to(query.dtype)

    # [b, np, sq, hn] -> [sq, b, np, hn] -> [sq, b, np * hn]
    output = output.permute(2, 0, 1, 3).contiguous()
    output = output.reshape(sq, b, np_ * hn)
    return output


@torch.no_grad()
def _compute_unfused_csa_non_compressed_lse(
    query: torch.Tensor,
    kv_full: torch.Tensor,
    attn_sink: torch.Tensor,
    window_indices: torch.Tensor,
    softmax_scale: float,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Return the detached sliding-window-plus-sink log mass for the CSA teacher.

    Determinism:
        Unknown. Bit-exact CUDA reduction behavior has not been certified.

    Args:
        query: Query tensor in ``[sq, batch, heads, head_dim]`` layout.
        kv_full: Original (non-compressed) KV in ``[sk, batch, head_dim]`` layout.
        attn_sink: Per-head sink logits in ``[heads]`` layout.
        window_indices: Local per-batch window indices in ``[batch, sq, window]`` layout.
        softmax_scale: Scale applied to query-key logits.
        chunk_size: Maximum number of flattened query rows processed at once.

    Returns:
        Detached FP32 log-sum-exp values in ``[batch, heads, sq]`` layout.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if query.ndim != 4:
        raise ValueError(f"query must have shape [sq, batch, heads, dim], got {query.shape}")
    if attn_sink.ndim != 1:
        raise ValueError(f"attn_sink must be 1D, got shape {tuple(attn_sink.shape)}")

    seqlen_q, batch_size, num_heads, head_dim = query.shape
    if kv_full.ndim != 3 or kv_full.shape[1:] != (batch_size, head_dim):
        raise ValueError(
            "non-compressed KV must have shape "
            f"[sk, {batch_size}, {head_dim}], got {tuple(kv_full.shape)}"
        )
    if window_indices.ndim != 3 or window_indices.shape[:2] != (batch_size, seqlen_q):
        raise ValueError(
            "window_indices must have shape "
            f"[{batch_size}, {seqlen_q}, window], got {tuple(window_indices.shape)}"
        )
    if attn_sink.numel() != num_heads:
        raise ValueError(f"attn_sink must contain {num_heads} values, got {attn_sink.numel()}")
    if not (query.device == kv_full.device == attn_sink.device == window_indices.device):
        raise ValueError("query, kv_full, attn_sink, and window_indices must share a device")

    n_kv = kv_full.shape[0]
    q_flat = query.detach().permute(1, 0, 2, 3).reshape(-1, num_heads, head_dim)
    kv_flat = kv_full.detach().permute(1, 0, 2).reshape(-1, head_dim)
    batch_offsets = (
        torch.arange(batch_size, device=window_indices.device, dtype=torch.int64) * n_kv
    ).view(batch_size, 1, 1)
    window_indices_i64 = window_indices.to(dtype=torch.int64)
    global_indices = torch.where(
        window_indices_i64 >= 0, window_indices_i64 + batch_offsets, window_indices_i64
    ).reshape(batch_size * seqlen_q, -1)

    sink = attn_sink.detach().to(dtype=torch.float32).view(1, num_heads)
    lse_chunks = []
    for start in range(0, q_flat.shape[0], chunk_size):
        end = min(start + chunk_size, q_flat.shape[0])
        indices = global_indices[start:end]
        gathered_kv = kv_flat.index_select(0, indices.clamp(min=0).reshape(-1)).reshape(
            end - start, indices.shape[-1], head_dim
        )
        window_logits = torch.einsum("rhd,rkd->rhk", q_flat[start:end].float(), gathered_kv.float())
        window_logits = (window_logits * softmax_scale).masked_fill(
            (indices < 0).unsqueeze(1), float("-inf")
        )
        lse_chunks.append(torch.logaddexp(torch.logsumexp(window_logits, dim=-1), sink))

    if lse_chunks:
        lse_flat = torch.cat(lse_chunks, dim=0)
    else:
        lse_flat = torch.empty((0, num_heads), dtype=torch.float32, device=query.device)
    return lse_flat.reshape(batch_size, seqlen_q, num_heads).permute(0, 2, 1).contiguous()
