# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Qwen QSA compressed-block routing and PyTorch correctness references.

QSA scores complete compressed key blocks, selects a fixed number of blocks,
expands them back to logical token IDs, and appends the current causal tail.
These helpers favor explicit semantics over kernel performance. They are
correctness oracles for optimized backends and are intended for tests and
short sequences.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn

from megatron.core.transformer.experimental_attention_variant.ops.qsa_stable_topk import (
    qsa_stable_topk_indices,
)

_PYTORCH_ORACLE_QUERY_CHUNK_SIZE = 16


def _validate_qsa_parameters(token_budget: int, compress_ratio: int) -> None:
    if compress_ratio <= 1:
        raise ValueError(f"QSA compress_ratio must be greater than one, got {compress_ratio}.")
    if token_budget <= 0 or token_budget % compress_ratio != 0:
        raise ValueError(
            "QSA token_budget must be positive and divisible by compress_ratio, "
            f"got token_budget={token_budget}, compress_ratio={compress_ratio}."
        )


class QSAIndexerOutput(NamedTuple):
    """Discrete QSA routes and optional indexer-loss inputs.

    Attributes:
        routes: Logical token IDs ``[B, S, token_budget + R - 1]`` in int32.
        block_logits: Differentiable indexer logits ``[B, S, P]``, or ``None``
            when the indexer is frozen.
        support_mask: Boolean blocks entering the indexer KL ``[B, S, P]``, or
            ``None`` when the indexer is frozen.
    """

    routes: torch.Tensor
    block_logits: torch.Tensor | None
    support_mask: torch.Tensor | None


def qsa_raw_block_logits(index_query: torch.Tensor, compressed_key: torch.Tensor) -> torch.Tensor:
    """Return the unscaled block score from the QSA report.

    Args:
        index_query: Normalized and rotated index queries ``[..., Sq, H, D]``.
        compressed_key: Normalized and rotated block keys ``[..., P, D]``.

    Returns:
        ``sum_h relu(<q_i^h, k_b>)`` with shape ``[..., Sq, P]``.
    """
    scores = torch.einsum("...qhd,...pd->...qhp", index_query, compressed_key)
    return torch.relu(scores).sum(dim=-2)


def qsa_block_logits(index_query: torch.Tensor, compressed_key: torch.Tensor) -> torch.Tensor:
    """Return the scaled QSA logits consumed by routing and indexer KL.

    The released implementation divides the head-summed ReLU score by
    ``sqrt(D_index)``. This positive scale does not change Top-K ordering, but
    it is part of the student softmax temperature during indexer training.

    Args:
        index_query: Normalized and rotated index queries ``[..., Sq, H, D]``.
        compressed_key: Normalized and rotated block keys ``[..., P, D]``.

    Returns:
        Scaled block logits ``[..., Sq, P]``.
    """
    return qsa_raw_block_logits(index_query, compressed_key) / math.sqrt(index_query.size(-1))


def compute_block_scores(index_query: torch.Tensor, compressed_key: torch.Tensor) -> torch.Tensor:
    """Compute differentiable, unmasked QSA block logits.

    Args:
        index_query: Index queries ``[B, S, H, D]``.
        compressed_key: Compressed keys ``[B, P, 1, D]``.

    Returns:
        Block logits ``[B, S, P]``. FP64 is preserved for gradcheck and lower
        precision inputs are accumulated in FP32.
    """
    if index_query.ndim != 4:
        raise ValueError(f"index_query must be [B, S, H, D], got {tuple(index_query.shape)}.")
    if compressed_key.ndim != 4 or compressed_key.size(2) != 1:
        raise ValueError(f"compressed_key must be [B, P, 1, D], got {tuple(compressed_key.shape)}.")
    if compressed_key.size(0) != index_query.size(0) or compressed_key.size(-1) != index_query.size(
        -1
    ):
        raise ValueError("QSA index query/key batch and head dimensions must match.")
    compute_dtype = (
        index_query.dtype if index_query.dtype in (torch.float32, torch.float64) else torch.float32
    )
    return qsa_block_logits(
        index_query.to(compute_dtype), compressed_key[:, :, 0].to(compute_dtype)
    )


def qsa_block_causal_mask(
    sequence_length: int,
    num_blocks: int,
    sequence_lengths: torch.Tensor,
    *,
    compress_ratio: int,
    device: torch.device,
) -> torch.Tensor:
    """Return where each query may score a complete compressed block.

    Args:
        sequence_length: Physical query length ``S``.
        num_blocks: Number of compressed blocks ``P``.
        sequence_lengths: Right-padded logical lengths ``[B]``.
        compress_ratio: Tokens represented by one block.
        device: Device for the returned mask.

    Returns:
        Boolean causal and padding mask ``[B, S, P]``.
    """
    if sequence_lengths.ndim != 1:
        raise ValueError("sequence_lengths must be rank one.")
    lengths = sequence_lengths.to(device=device, dtype=torch.int64)
    positions = torch.arange(sequence_length, device=device)
    block_ids = torch.arange(num_blocks, device=device)
    visible_blocks = torch.div(positions + 1, compress_ratio, rounding_mode="floor")
    causal = block_ids.unsqueeze(0) < visible_blocks.unsqueeze(1)
    available = torch.div(lengths, compress_ratio, rounding_mode="floor")
    within_length = block_ids.unsqueeze(0) < available.unsqueeze(1)
    live_rows = positions.unsqueeze(0) < lengths.unsqueeze(1)
    return causal.unsqueeze(0) & within_length.unsqueeze(1) & live_rows.unsqueeze(-1)


def qsa_topk_block_support(
    block_logits: torch.Tensor, causal_mask: torch.Tensor, *, block_budget: int
) -> torch.Tensor:
    """Build the detached Stage-2 KL support from the indexer's Top-K.

    Rows that fit within ``block_budget`` keep every visible block. Sparse rows
    use the canonical ordering: score descending and block ID ascending for
    equal scores.

    Args:
        block_logits: Scaled block logits ``[B, S, P]``.
        causal_mask: Scorable blocks ``[B, S, P]``.
        block_budget: Maximum complete blocks in the support.

    Returns:
        Detached boolean support ``[B, S, P]``.
    """
    if block_logits.shape != causal_mask.shape:
        raise ValueError("block_logits and causal_mask must have the same shape.")
    if block_budget <= 0:
        raise ValueError(f"block_budget must be positive, got {block_budget}.")
    with torch.no_grad():
        masked = block_logits.detach().masked_fill(~causal_mask, -torch.inf)
        width = min(block_budget, block_logits.size(-1))
        chosen = torch.zeros_like(causal_mask)
        chosen.scatter_(-1, qsa_stable_topk_indices(masked, width), True)
        visible = causal_mask.sum(dim=-1, keepdim=True)
        return torch.where(visible > block_budget, causal_mask & chosen, causal_mask)


@torch.no_grad()
def expand_blocks_to_routes(
    top_blocks: torch.Tensor,
    valid_blocks: torch.Tensor,
    query_positions: torch.Tensor,
    *,
    compress_ratio: int,
    block_budget: int,
    route_width: int,
) -> torch.Tensor:
    """Expand selected blocks to token IDs and append the incomplete tail.

    Args:
        top_blocks: Selected block IDs ``[rows, W]``.
        valid_blocks: Valid selected slots ``[rows, W]``.
        query_positions: Logical position of each query row ``[rows]``.
        compress_ratio: Tokens represented by one block.
        block_budget: Maximum selected complete blocks.
        route_width: Fixed output width ``token_budget + compress_ratio - 1``.

    Returns:
        int32 token routes ``[rows, route_width]`` with ``-1`` padding.
    """
    if top_blocks.shape != valid_blocks.shape:
        raise ValueError("top_blocks and valid_blocks must have the same shape.")
    device = query_positions.device
    rows = query_positions.numel()
    routes = torch.full((rows, route_width), -1, dtype=torch.int32, device=device)
    block_offsets = torch.arange(compress_ratio, device=device, dtype=torch.int64)
    tail_offsets = torch.arange(compress_ratio - 1, device=device, dtype=torch.int64)
    visible_blocks = torch.div(query_positions + 1, compress_ratio, rounding_mode="floor")

    width = top_blocks.size(-1)
    if width > 0:
        expanded = top_blocks.unsqueeze(-1) * compress_ratio + block_offsets
        expanded = expanded.masked_fill(~valid_blocks.unsqueeze(-1), -1)
        routes[:, : width * compress_ratio] = expanded.flatten(1).to(torch.int32)

    tail_start = visible_blocks * compress_ratio
    tail_count = query_positions + 1 - tail_start
    valid_block_count = torch.minimum(visible_blocks, torch.full_like(visible_blocks, block_budget))
    tail_values = tail_start.unsqueeze(1) + tail_offsets.unsqueeze(0)
    tail_valid = tail_offsets.unsqueeze(0) < tail_count.unsqueeze(1)
    if bool(tail_valid.any()):
        destinations = valid_block_count.unsqueeze(1) * compress_ratio + tail_offsets.unsqueeze(0)
        row_ids = torch.arange(rows, device=device).unsqueeze(1).expand_as(tail_valid)
        routes[row_ids[tail_valid], destinations[tail_valid]] = tail_values[tail_valid].to(
            torch.int32
        )
    return routes


@torch.no_grad()
def select_qsa_token_ids(
    index_query: torch.Tensor,
    compressed_key: torch.Tensor,
    sequence_lengths: torch.Tensor,
    *,
    token_budget: int,
    compress_ratio: int,
    query_chunk_size: int = 128,
    block_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select logical key-token IDs used by QSA sparse attention.

    Complete blocks participate in Top-K. Tokens in the current incomplete
    block are appended directly, so every valid route remains causal.

    Args:
        index_query: Index queries ``[B, S, H, D]``.
        compressed_key: Compressed keys ``[B, P, 1, D]``.
        sequence_lengths: Right-padded logical lengths ``[B]``.
        token_budget: Maximum tokens selected from complete blocks.
        compress_ratio: Tokens represented by one block.
        query_chunk_size: Query rows scored in one oracle chunk.
        block_logits: Optional precomputed detached logits ``[B, S, P]``.

    Returns:
        int32 routes ``[B, S, token_budget + compress_ratio - 1]``.
    """
    _validate_qsa_parameters(token_budget, compress_ratio)
    if index_query.ndim != 4:
        raise ValueError(f"index_query must be [B, S, H, D], got {tuple(index_query.shape)}.")
    if compressed_key.ndim != 4 or compressed_key.size(2) != 1:
        raise ValueError(f"compressed_key must be [B, P, 1, D], got {tuple(compressed_key.shape)}.")
    batch_size, sequence_length, num_heads, head_dim = index_query.shape
    if compressed_key.size(0) != batch_size or compressed_key.size(-1) != head_dim:
        raise ValueError("QSA index query/key batch and head dimensions must match.")
    if sequence_lengths.shape != (batch_size,):
        raise ValueError(
            f"sequence_lengths must have shape [{batch_size}], got {tuple(sequence_lengths.shape)}."
        )
    if num_heads <= 0 or head_dim <= 0:
        raise ValueError("QSA index query head count and dimension must be positive.")
    if query_chunk_size <= 0:
        raise ValueError(f"query_chunk_size must be positive, got {query_chunk_size}.")
    if block_logits is not None and block_logits.shape != (
        batch_size,
        sequence_length,
        compressed_key.size(1),
    ):
        raise ValueError("block_logits must have shape [B, S, P].")

    lengths = sequence_lengths.to(device=index_query.device, dtype=torch.int64)
    if bool(((lengths < 0) | (lengths > sequence_length)).any()):
        raise ValueError(f"QSA sequence lengths must lie in [0, {sequence_length}].")
    required_blocks = torch.div(lengths, compress_ratio, rounding_mode="floor")
    if bool((required_blocks > compressed_key.size(1)).any()):
        raise ValueError(
            "compressed_key does not cover all complete logical blocks; "
            f"need {int(required_blocks.max())}, got {compressed_key.size(1)}."
        )

    block_budget = token_budget // compress_ratio
    route_width = token_budget + compress_ratio - 1
    selected = torch.full(
        (batch_size, sequence_length, route_width), -1, dtype=torch.int32, device=index_query.device
    )

    for batch_idx in range(batch_size):
        logical_length = int(lengths[batch_idx])
        available_blocks = logical_length // compress_ratio
        keys = compressed_key[batch_idx, :available_blocks, 0].float()

        for query_start in range(0, logical_length, query_chunk_size):
            query_end = min(query_start + query_chunk_size, logical_length)
            query_positions = torch.arange(query_start, query_end, device=index_query.device)
            visible_blocks = torch.div(query_positions + 1, compress_ratio, rounding_mode="floor")
            rows = query_end - query_start
            topk_width = min(block_budget, available_blocks)

            if topk_width > 0:
                candidate_blocks = torch.arange(topk_width, device=index_query.device)
                top_blocks = candidate_blocks.unsqueeze(0).expand(rows, -1).clone()
                valid_blocks = candidate_blocks.unsqueeze(0) < visible_blocks.unsqueeze(1)
                sparse_rows = visible_blocks > block_budget
                if bool(sparse_rows.any()):
                    if block_logits is None:
                        sparse_query = index_query[batch_idx, query_start:query_end][sparse_rows]
                        scores = qsa_block_logits(sparse_query.float(), keys)
                    else:
                        scores = block_logits[batch_idx, query_start:query_end][sparse_rows][
                            :, :available_blocks
                        ].float()
                    block_ids = torch.arange(available_blocks, device=index_query.device)
                    sparse_visible = visible_blocks[sparse_rows]
                    scores.masked_fill_(
                        block_ids.unsqueeze(0) >= sparse_visible.unsqueeze(1), -torch.inf
                    )
                    top_blocks[sparse_rows] = qsa_stable_topk_indices(scores, block_budget)
                    valid_blocks[sparse_rows] = True
                selected_blocks, selected_valid = top_blocks, valid_blocks
            else:
                selected_blocks = torch.zeros(
                    (rows, 0), dtype=torch.int64, device=index_query.device
                )
                selected_valid = torch.zeros((rows, 0), dtype=torch.bool, device=index_query.device)

            selected[batch_idx, query_start:query_end] = expand_blocks_to_routes(
                selected_blocks,
                selected_valid,
                query_positions,
                compress_ratio=compress_ratio,
                block_budget=block_budget,
                route_width=route_width,
            )
    return selected


def unfused_qsa_gqa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected_token_ids: torch.Tensor,
    *,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Run a differentiable PyTorch QSA GQA correctness oracle.

    Args:
        query: Main-attention queries ``[B, Sq, Hq, D]``.
        key: Main-attention keys ``[B, Skv, Hkv, D]``.
        value: Main-attention values ``[B, Skv, Hkv, D]``.
        selected_token_ids: Logical key IDs ``[B, Sq, K]`` with ``-1`` padding.
        softmax_scale: Score multiplier, defaulting to ``1 / sqrt(D)``.

    Returns:
        Sparse attention output ``[B, Sq, Hq, D]``.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("QSA query, key, and value must use [B, S, H, D] layout.")
    if key.shape != value.shape:
        raise ValueError(
            f"QSA key and value shapes must match, got {tuple(key.shape)} and {tuple(value.shape)}."
        )
    batch_size, query_length, num_query_heads, head_dim = query.shape
    key_length = key.size(1)
    num_kv_heads = key.size(2)
    if key.size(0) != batch_size or key.size(-1) != head_dim:
        raise ValueError("QSA main query/key batch and head dimensions must match.")
    if num_kv_heads <= 0 or num_query_heads % num_kv_heads != 0:
        raise ValueError(
            f"QSA requires Hq divisible by Hkv, got Hq={num_query_heads}, Hkv={num_kv_heads}."
        )
    if selected_token_ids.ndim != 3 or selected_token_ids.shape[:2] != (batch_size, query_length):
        raise ValueError("selected_token_ids must be [B, Sq, K] matching query.")
    if bool(((selected_token_ids < -1) | (selected_token_ids >= key_length)).any()):
        raise ValueError("QSA selected token IDs must be -1 or a valid key position.")
    if query_length == 0:
        return torch.empty_like(query)

    scale = head_dim**-0.5 if softmax_scale is None else softmax_scale
    query_groups = num_query_heads // num_kv_heads
    grouped_query = query.unflatten(2, (num_kv_heads, query_groups))
    outputs = []

    for query_start in range(0, query_length, _PYTORCH_ORACLE_QUERY_CHUNK_SIZE):
        query_end = min(query_start + _PYTORCH_ORACLE_QUERY_CHUNK_SIZE, query_length)
        chunk_query = grouped_query[:, query_start:query_end]
        token_ids = selected_token_ids[:, query_start:query_end].long()
        valid = token_ids >= 0
        safe_ids = token_ids.clamp_min(0)
        batch_ids = torch.arange(batch_size, device=query.device).view(batch_size, 1, 1)

        gathered_key = key[batch_ids, safe_ids].permute(0, 1, 3, 2, 4)
        gathered_value = value[batch_ids, safe_ids].permute(0, 1, 3, 2, 4)
        scores = torch.einsum("bqhgd,bqhkd->bqhgk", chunk_query.float(), gathered_key.float())
        scores.mul_(scale)
        score_valid = valid[:, :, None, None, :]
        scores.masked_fill_(~score_valid, -torch.inf)
        has_routes = valid.any(dim=-1)
        scores = torch.where(has_routes[:, :, None, None, None], scores, torch.zeros_like(scores))
        probabilities = torch.softmax(scores, dim=-1).masked_fill(~score_valid, 0.0)
        chunk_output = torch.einsum("bqhgk,bqhkd->bqhgd", probabilities, gathered_value.float())
        outputs.append(chunk_output.flatten(2, 3).to(query.dtype))

    return torch.cat(outputs, dim=1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_qsa_rope(states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    rotary_width = cos.size(-1)
    if rotary_width <= 0 or rotary_width % 2 != 0 or rotary_width > states.size(-1):
        raise ValueError(
            "QSA rotary width must be positive, even, and no wider than the index head; "
            f"got rotary_width={rotary_width}, head_dim={states.size(-1)}."
        )
    rotary, passthrough = states[..., :rotary_width], states[..., rotary_width:]
    rotary = rotary * cos + _rotate_half(rotary) * sin
    return torch.cat((rotary, passthrough), dim=-1)


class QSAZeroCenteredRMSNorm(nn.Module):
    """Qwen QSA zero-centered RMSNorm used by indexer checkpoints."""

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize the last dimension and apply zero-centered weight."""
        output = hidden_states.float()
        output = output * torch.rsqrt(output.square().mean(dim=-1, keepdim=True) + self.eps)
        return (output * (1.0 + self.weight.float())).to(hidden_states.dtype)


class QSAIndexer(nn.Module):
    """Qwen QSA block indexer for MCore ``[S, B, H]`` hidden states."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_query_heads: int,
        num_key_heads: int,
        head_dim: int,
        compress_ratio: int,
        token_budget: int,
        norm_epsilon: float = 1e-6,
        attention_scaling: float = 1.0,
        params_dtype: torch.dtype = torch.bfloat16,
        query_chunk_size: int = 128,
        qsa_indexer_loss_coeff: float = 0.0,
        qsa_indexer_use_sparse_loss: bool = True,
    ) -> None:
        super().__init__()
        _validate_qsa_parameters(token_budget, compress_ratio)
        if hidden_size <= 0 or num_query_heads <= 0 or head_dim <= 0:
            raise ValueError(
                "QSA hidden size, query head count, and head dimension must be positive."
            )
        if num_key_heads != 1:
            raise ValueError(f"Qwen QSA requires exactly one index key head, got {num_key_heads}.")
        if query_chunk_size <= 0:
            raise ValueError(f"query_chunk_size must be positive, got {query_chunk_size}.")
        if not isinstance(qsa_indexer_loss_coeff, (int, float)) or isinstance(
            qsa_indexer_loss_coeff, bool
        ):
            raise TypeError(
                f"qsa_indexer_loss_coeff must be a number, got {qsa_indexer_loss_coeff!r}."
            )
        coefficient = float(qsa_indexer_loss_coeff)
        if not math.isfinite(coefficient) or coefficient < 0.0:
            raise ValueError(
                "qsa_indexer_loss_coeff must be finite and non-negative, "
                f"got {qsa_indexer_loss_coeff!r}."
            )

        self.hidden_size = hidden_size
        self.num_query_heads = num_query_heads
        self.num_key_heads = num_key_heads
        self.head_dim = head_dim
        self.compress_ratio = compress_ratio
        self.token_budget = token_budget
        self.attention_scaling = attention_scaling
        self.query_chunk_size = query_chunk_size
        self.qsa_indexer_loss_coeff = coefficient
        self.qsa_indexer_use_sparse_loss = bool(qsa_indexer_use_sparse_loss)
        self.indexer_is_trainable = coefficient > 0.0

        self.index_qk_proj = nn.Linear(
            hidden_size,
            (num_query_heads + num_key_heads) * head_dim,
            bias=False,
            dtype=params_dtype,
        )
        self.q_layernorm = QSAZeroCenteredRMSNorm(head_dim, norm_epsilon, params_dtype)
        self.k_layernorm = QSAZeroCenteredRMSNorm(head_dim, norm_epsilon, params_dtype)
        self.requires_grad_(self.indexer_is_trainable)

    def index_states(
        self, hidden_states: torch.Tensor, rotary_pos_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project, normalize, pool, and rotate QSA index states.

        Keys are average-pooled before RoPE and each block key uses the block's
        start position. The RoPE tensor controls the rotary prefix width; the
        remaining index-head channels pass through unchanged.

        Args:
            hidden_states: Decoder input ``[S, B, hidden_size]``.
            rotary_pos_emb: Rotary angles ``[S, 1, 1, rotary_width]``.

        Returns:
            Queries ``[B, S, H, D]`` and compressed keys ``[B, P, 1, D]``.
        """
        if hidden_states.ndim != 3 or hidden_states.size(-1) != self.hidden_size:
            raise ValueError(
                f"hidden_states must be [S, B, {self.hidden_size}], "
                f"got {tuple(hidden_states.shape)}."
            )
        sequence_length, batch_size, _ = hidden_states.shape
        if rotary_pos_emb.size(0) < sequence_length:
            raise ValueError("rotary_pos_emb must cover every input position.")

        qk = self.index_qk_proj(hidden_states)
        query_width = self.num_query_heads * self.head_dim
        query, raw_key = torch.split(qk, [query_width, self.head_dim], dim=-1)
        query = query.view(
            sequence_length, batch_size, self.num_query_heads, self.head_dim
        ).permute(1, 0, 2, 3)
        raw_key = raw_key.view(sequence_length, batch_size, 1, self.head_dim).permute(1, 0, 2, 3)
        query = self.q_layernorm(query)

        frequencies = rotary_pos_emb.reshape(rotary_pos_emb.size(0), -1)[:sequence_length]
        cos = (torch.cos(frequencies) * self.attention_scaling).to(query.dtype)
        sin = (torch.sin(frequencies) * self.attention_scaling).to(query.dtype)
        query = _apply_qsa_rope(query, cos[None, :, None, :], sin[None, :, None, :])

        num_blocks = sequence_length // self.compress_ratio
        usable_tokens = num_blocks * self.compress_ratio
        pooled_key = raw_key[:, :usable_tokens].reshape(
            batch_size, num_blocks, self.compress_ratio, self.head_dim
        )
        pooled_key = pooled_key.float().mean(dim=2).to(raw_key.dtype).unsqueeze(2)
        pooled_key = self.k_layernorm(pooled_key)
        block_starts = torch.arange(num_blocks, device=hidden_states.device) * self.compress_ratio
        pooled_key = _apply_qsa_rope(
            pooled_key, cos[block_starts][None, :, None, :], sin[block_starts][None, :, None, :]
        )
        return query, pooled_key

    def route_and_score(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        sequence_lengths: torch.Tensor | None = None,
    ) -> QSAIndexerOutput:
        """Run one projection and return routes plus optional KL inputs.

        A zero loss coefficient keeps the indexer frozen and avoids allocating
        the quadratic reference logit matrix. A positive coefficient preserves
        differentiable logits while deriving routes and support from a detached
        view, so no gradient crosses the discrete Top-K.

        Args:
            hidden_states: Decoder input ``[S, B, hidden_size]``.
            rotary_pos_emb: Rotary angles ``[S, 1, 1, rotary_width]``.
            sequence_lengths: Optional right-padded logical lengths ``[B]``.

        Returns:
            Routes and optional differentiable KL inputs.
        """
        if hidden_states.ndim != 3:
            raise ValueError(
                f"route_and_score expects [S, B, hidden_size], got {tuple(hidden_states.shape)}."
            )
        sequence_length, batch_size, _ = hidden_states.shape
        if sequence_lengths is None:
            sequence_lengths = torch.full(
                (batch_size,), sequence_length, dtype=torch.int64, device=hidden_states.device
            )

        if not self.indexer_is_trainable:
            with torch.no_grad():
                routes = select_qsa_token_ids(
                    *self.index_states(hidden_states, rotary_pos_emb),
                    sequence_lengths,
                    token_budget=self.token_budget,
                    compress_ratio=self.compress_ratio,
                    query_chunk_size=self.query_chunk_size,
                )
            return QSAIndexerOutput(routes=routes, block_logits=None, support_mask=None)

        query, pooled_key = self.index_states(hidden_states, rotary_pos_emb)
        block_logits = compute_block_scores(query, pooled_key)
        detached_logits = block_logits.detach()
        with torch.no_grad():
            routes = select_qsa_token_ids(
                query,
                pooled_key,
                sequence_lengths,
                token_budget=self.token_budget,
                compress_ratio=self.compress_ratio,
                query_chunk_size=self.query_chunk_size,
                block_logits=detached_logits,
            )
            causal_mask = qsa_block_causal_mask(
                sequence_length,
                pooled_key.size(1),
                sequence_lengths,
                compress_ratio=self.compress_ratio,
                device=hidden_states.device,
            )
            support_mask = (
                qsa_topk_block_support(
                    detached_logits,
                    causal_mask,
                    block_budget=self.token_budget // self.compress_ratio,
                )
                if self.qsa_indexer_use_sparse_loss
                else causal_mask
            )
        return QSAIndexerOutput(routes=routes, block_logits=block_logits, support_mask=support_mask)

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        sequence_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return QSA token routes for a non-packed input.

        Args:
            hidden_states: Decoder input ``[S, B, hidden_size]``.
            rotary_pos_emb: Rotary angles ``[S, 1, 1, rotary_width]``.
            sequence_lengths: Optional right-padded logical lengths ``[B]``.

        Returns:
            int32 routes ``[B, S, token_budget + compress_ratio - 1]``.
        """
        if hidden_states.ndim != 3 or hidden_states.size(-1) != self.hidden_size:
            raise ValueError(
                f"hidden_states must be [S, B, {self.hidden_size}], "
                f"got {tuple(hidden_states.shape)}."
            )
        sequence_length, batch_size, _ = hidden_states.shape
        if sequence_lengths is None:
            sequence_lengths = torch.full(
                (batch_size,), sequence_length, dtype=torch.int64, device=hidden_states.device
            )
        query, pooled_key = self.index_states(hidden_states, rotary_pos_emb)
        return select_qsa_token_ids(
            query,
            pooled_key,
            sequence_lengths,
            token_budget=self.token_budget,
            compress_ratio=self.compress_ratio,
            query_chunk_size=self.query_chunk_size,
        )


__all__ = [
    "QSAIndexer",
    "QSAIndexerOutput",
    "QSAZeroCenteredRMSNorm",
    "compute_block_scores",
    "expand_blocks_to_routes",
    "qsa_block_logits",
    "qsa_block_causal_mask",
    "qsa_raw_block_logits",
    "qsa_topk_block_support",
    "select_qsa_token_ids",
    "unfused_qsa_gqa_attention",
]
