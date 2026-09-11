# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import math
from typing import Optional

import torch

from megatron.core.ops.attention.dsa import dsa_indexer_loss, dsa_layout, dsa_masking
from megatron.core.process_groups_config import ProcessGroupCollection


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
    row_mask, varlen_starts, varlen_ends, key_positions = dsa_masking.prepare_sparse_mask_context(
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
    dsa_masking.scatter_topk_into_index_mask(index_mask, topk_indices, seq_chunk_size=256)
    index_mask = dsa_masking.apply_sparse_validity_to_index_mask(
        index_mask,
        row_mask=row_mask,
        varlen_starts=varlen_starts,
        varlen_ends=varlen_ends,
        key_positions=key_positions,
    )

    attention_scores = attention_scores + index_mask.unsqueeze(1)
    valid_index_mask = torch.isfinite(index_mask)
    attention_scores = dsa_masking.masked_softmax(
        attention_scores.float(), valid_index_mask.unsqueeze(1).expand(b, np, sq, skv), dim=-1
    )

    # Latent value is the first v_channels slice of absorbed key cache.
    value = key[..., :v_channels].permute(1, 2, 0, 3)  # [b,1,skv,v]
    output = torch.matmul(attention_scores.to(value.dtype), value)  # [b,np,sq,v]
    return output.permute(2, 0, 1, 3).contiguous()


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
    query_valid_rows: Optional[torch.Tensor] = None,
    calculate_per_token_loss: bool = False,
    non_compressed_lse: torch.Tensor | None = None,
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
        non_compressed_lse: Optional detached FP32 log-sum-exp contribution
            [batch, heads, seqlen_q] from teacher keys that are intentionally
            omitted from ``key``. When provided, the selected ``key`` logits
            are normalized with this external mass before heads are summed.

    Returns:
        index_loss: KL divergence loss (scalar).
    """
    query, _ = dsa_layout.ensure_sbhd(query, "query")
    key, _ = dsa_layout.ensure_sbhd(key, "key")

    sq, b, np, hn = query.size()
    sk = key.size(0)
    query_valid_rows = dsa_masking.normalize_query_valid_rows(
        query_valid_rows, b=b, sq=sq, device=index_scores.device
    )

    varlen_starts, varlen_ends, key_positions = dsa_masking.normalize_varlen_bounds(
        mask=mask,
        varlen_starts=varlen_starts,
        varlen_ends=varlen_ends,
        key_positions=key_positions,
        sk=sk,
        device=index_scores.device,
    )

    # [sq, b, np, hn] -> [b, np, sq, hn] -> [b * np, sq, hn]
    query = query.permute(1, 2, 0, 3).reshape(b * np, sq, hn)
    # [sk, b, np, hn] -> [b, np, hn, sk] -> [b * np, hn, sk]
    key = key.permute(1, 2, 3, 0).reshape(b * np, hn, sk)
    # Compute attention scores [b * np, sq, sk]
    attention_scores = torch.bmm(query.float(), key.float()) * softmax_scale
    # Reshape to [b, np, sq, sk]
    attention_scores = attention_scores.reshape(b, np, sq, sk)
    if varlen_starts is not None:
        attention_scores = dsa_masking.apply_starts_ends_mask_to_scores(
            attention_scores, varlen_starts, varlen_ends, key_positions
        )
        index_scores = dsa_masking.apply_starts_ends_mask_to_scores(
            index_scores, varlen_starts, varlen_ends, key_positions
        )
        base_valid_mask = (
            dsa_masking.build_valid_mask_from_starts_ends(varlen_starts, varlen_ends, key_positions)
            .unsqueeze(0)
            .expand(b, sq, sk)
        )
    else:
        _, attn_score_mask, index_score_mask, base_valid_mask = dsa_masking.prepare_additive_mask(
            mask, sq=sq, sk=sk, b=b, device=attention_scores.device
        )
        # [b, np, sq, sk] + [1/b, 1, sq, sk] -> [b, np, sq, sk]
        attention_scores += attn_score_mask
        # [b, sq, sk] + [1/b, sq, sk] -> [b, sq, sk]
        index_scores += index_score_mask

    # index_mask [b, sq, sk]
    index_mask = torch.full(
        (b, sq, sk), float("-inf"), dtype=torch.float32, device=attention_scores.device
    )
    dsa_masking.scatter_topk_into_index_mask(index_mask, topk_indices, seq_chunk_size=256)

    if sparse_loss:
        # [b, np, sq, sk] + [b, 1, sq, sk] -> [b, np, sq, sk]
        attention_scores += index_mask.view(b, 1, sq, sk)
        # [b, sq, sk] + [b, sq, sk] -> [b, sq, sk]
        index_scores += index_mask
        index_valid_mask = base_valid_mask & (index_mask == 0)
    else:
        index_valid_mask = base_valid_mask
    attention_valid_mask = index_valid_mask if sparse_loss else base_valid_mask

    # [b, np, sq, sk] -> [b, np, sq, sk]
    attention_scores = _compute_indexer_teacher_probabilities(
        attention_scores, attention_valid_mask, non_compressed_lse=non_compressed_lse
    )
    # [b, sq, sk] -> [b, sq, sk]
    index_log_scores = dsa_masking.masked_log_softmax(
        index_scores.float(), index_valid_mask, dim=-1
    )

    # Sum attention scores across heads.
    # [batch, heads, seqlen_q, seqlen_k] -> [batch, seqlen_q, seqlen_k]
    attention_scores = attention_scores.sum(dim=1)
    if pg_collection.tp.size() > 1:
        # attention scores are scattered to TP ranks in head dimension.
        torch.distributed.all_reduce(attention_scores.contiguous(), group=pg_collection.tp)
    # The target is already non-negative because it is a sum of softmax probabilities.
    attention_scores = _normalize_indexer_teacher_target(attention_scores, non_compressed_lse)
    return dsa_indexer_loss.indexer_loss_from_target(
        attention_scores,
        index_log_scores,
        loss_coeff,
        query_valid_rows=query_valid_rows,
        calculate_per_token_loss=calculate_per_token_loss,
    )


def _compute_indexer_teacher_probabilities(
    attention_scores: torch.Tensor,
    attention_valid_mask: torch.Tensor,
    non_compressed_lse: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return selected-key teacher mass, optionally including omitted mass.

    ``non_compressed_lse`` is a sufficient statistic for teacher logits that
    must participate in the softmax denominator but must not appear in the
    compressed-key target returned by this helper. When the absolute compressed
    mass underflows FP32, all heads in a row receive the same log-domain shift;
    the returned weights remain proportional and the caller L1-normalizes them.
    """
    b, np, sq, sk = attention_scores.shape
    expanded_valid_mask = attention_valid_mask.unsqueeze(1).expand(b, np, sq, sk)
    if non_compressed_lse is None:
        return dsa_masking.masked_softmax(attention_scores.float(), expanded_valid_mask, dim=-1)

    expected_shape = (b, np, sq)
    if tuple(non_compressed_lse.shape) != expected_shape:
        raise ValueError(
            "non_compressed_lse must have shape [batch, heads, seqlen_q], "
            f"got {tuple(non_compressed_lse.shape)}, expected {expected_shape}"
        )
    if non_compressed_lse.device != attention_scores.device:
        raise ValueError(
            "non_compressed_lse and attention_scores must be on the same device, "
            f"got {non_compressed_lse.device} and {attention_scores.device}"
        )
    if non_compressed_lse.requires_grad:
        raise ValueError("non_compressed_lse must be detached")

    masked_scores = attention_scores.float().masked_fill(~expanded_valid_mask, float("-inf"))
    compressed_lse = torch.logsumexp(masked_scores, dim=-1)
    row_has_compressed_keys = expanded_valid_mask.any(dim=-1)
    # Avoid the undefined ``-inf - -inf`` intermediate on fully masked rows.
    # This is only a [batch, heads, seqlen] tensor, so it does not recreate the
    # full-size temporary that the log-domain formulation is designed to avoid.
    safe_compressed_lse = torch.where(
        row_has_compressed_keys, compressed_lse, torch.zeros_like(compressed_lse)
    )
    conditional_probabilities = torch.exp(masked_scores - safe_compressed_lse.unsqueeze(-1))
    del masked_scores

    full_lse = torch.logaddexp(non_compressed_lse.float(), compressed_lse)
    log_compressed_mass = (compressed_lse - full_lse).masked_fill(
        ~row_has_compressed_keys, float("-inf")
    )

    # The external window/sink mass can put every head's compressed mass below
    # the FP32 normal range. A common per-row shift across heads
    # preserves all relative teacher weights and cancels in the downstream L1
    # normalization. CSA currently requires TP1, so no cross-rank MAX is needed.
    row_max = log_compressed_mass.amax(dim=1, keepdim=True)
    needs_rescale = torch.isfinite(row_max) & (row_max < math.log(torch.finfo(torch.float32).tiny))
    common_shift = torch.where(needs_rescale, row_max, torch.zeros_like(row_max))
    compressed_mass = torch.exp(log_compressed_mass - common_shift)
    return conditional_probabilities * compressed_mass.unsqueeze(-1)


def _normalize_indexer_teacher_target(
    target: torch.Tensor, non_compressed_lse: torch.Tensor | None
) -> torch.Tensor:
    """L1-normalize teacher mass without changing the legacy DSA path."""
    if non_compressed_lse is None:
        return dsa_indexer_loss.normalize_indexer_target(target)
    row_mass = target.sum(dim=-1, keepdim=True)
    # External teacher mass can legitimately make the compressed mass smaller
    # than INDEXER_LOSS_EPS (or even float32 tiny). Only an exactly zero row is
    # degenerate; keep it zero rather than imposing a numerical floor.
    safe_row_mass = torch.where(row_mass > 0, row_mass, torch.ones_like(row_mass))
    return target / safe_row_mass


def _compute_index_scores(
    q: torch.Tensor, weights: torch.Tensor, k: torch.Tensor, use_relu: bool = True
) -> torch.Tensor:
    """
    Perform index score using BF16 precision.

    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/kernel.py#L254-L274
    This is a BF16 implementation of the `fp8_index` logic:
        1. Compute attention scores: q @ k^T;
        2. Optionally apply ReLU activation (DeepSeek V3.2 only; disabled for GLM5);
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

    # Optionally apply ReLU activation (used by DeepSeek V3.2, not GLM5).
    if use_relu:
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
    use_relu: bool = True,
):
    """Naive implementation of QK Topk."""
    sk = k.size(0)
    # =========================================
    # Compute index scores
    # =========================================
    # [batch, seqlen, seqlen]
    index_scores = _compute_index_scores(q, weights, k, use_relu=use_relu)
    varlen_starts, varlen_ends, key_positions = dsa_masking.normalize_varlen_bounds(
        mask=mask,
        varlen_starts=varlen_starts,
        varlen_ends=varlen_ends,
        key_positions=key_positions,
        sk=sk,
        device=index_scores.device,
    )
    if varlen_starts is not None:
        index_scores = dsa_masking.apply_starts_ends_mask_to_scores(
            index_scores, varlen_starts, varlen_ends, key_positions
        )
    elif mask is not None:
        assert mask.dtype == index_scores.dtype, "Mask dtype must match index scores dtype"
        index_scores = index_scores + mask

    # =========================================
    # Select top-k indices
    # =========================================
    topk_k = min(index_topk, sk)
    if topk_k > 0:
        topk_scores, topk_indices = index_scores.topk(topk_k, dim=-1)
        topk_indices = topk_indices.masked_fill(topk_scores == float("-inf"), -1)
    else:
        topk_indices = torch.empty(
            index_scores.shape[:-1] + (0,), dtype=torch.int64, device=index_scores.device
        )

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
    query_valid_rows=None,
    calculate_per_token_loss: bool = False,
    use_relu: bool = True,
    non_compressed_lse: torch.Tensor | None = None,
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
        use_relu=use_relu,
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
        query_valid_rows=query_valid_rows,
        calculate_per_token_loss=calculate_per_token_loss,
        non_compressed_lse=non_compressed_lse,
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
    query_valid_rows=None,
    calculate_per_token_loss: bool = False,
    use_relu: bool = True,
    non_compressed_lse: torch.Tensor | None = None,
):
    """Naive implementation of backward pass for indexer loss."""
    query, _ = dsa_layout.ensure_sbhd(query, "query")
    key, _ = dsa_layout.ensure_sbhd(key, "key")

    index_scores = _compute_index_scores(q, weights, k, use_relu=use_relu)  # [B, Sq, Sk]

    sq, b, np, hn = query.size()
    sk = key.size(0)
    query_valid_rows = dsa_masking.normalize_query_valid_rows(
        query_valid_rows, b=b, sq=sq, device=query.device
    )

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
    varlen_starts, varlen_ends, key_positions = dsa_masking.normalize_varlen_bounds(
        mask=mask,
        varlen_starts=varlen_starts,
        varlen_ends=varlen_ends,
        key_positions=key_positions,
        sk=sk,
        device=attention_scores.device,
    )

    if varlen_starts is not None:
        attention_scores = dsa_masking.apply_starts_ends_mask_to_scores(
            attention_scores, varlen_starts, varlen_ends, key_positions
        )
        index_scores = dsa_masking.apply_starts_ends_mask_to_scores(
            index_scores, varlen_starts, varlen_ends, key_positions
        )
        base_valid_mask = (
            dsa_masking.build_valid_mask_from_starts_ends(varlen_starts, varlen_ends, key_positions)
            .unsqueeze(0)
            .expand(b, sq, sk)
        )
    else:
        _, attn_score_mask, index_score_mask, base_valid_mask = dsa_masking.prepare_additive_mask(
            mask, sq=sq, sk=sk, b=b, device=attention_scores.device
        )
        # [b, np, sq, sk] + [1/b, 1, sq, sk] -> [b, np, sq, sk]
        attention_scores = attention_scores + attn_score_mask
        # [b, sq, sk] + [1/b, sq, sk] -> [b, sq, sk]
        index_scores = index_scores + index_score_mask

    # index_mask [b, sq, sk]
    index_mask = torch.full(
        (b, sq, sk), float("-inf"), dtype=torch.float32, device=attention_scores.device
    )
    dsa_masking.scatter_topk_into_index_mask(index_mask, topk_indices, seq_chunk_size=256)

    if sparse_loss:
        # [b, np, sq, sk] + [b, 1, sq, sk] -> [b, np, sq, sk]
        attention_scores = attention_scores + index_mask.view(b, 1, sq, sk)
        # [b, sq, sk] + [b, sq, sk] -> [b, sq, sk]
        index_scores = index_scores + index_mask

    # Compute softmax for both.
    if sparse_loss:
        index_valid_mask = base_valid_mask & (index_mask == 0)
    else:
        index_valid_mask = base_valid_mask
    attention_valid_mask = index_valid_mask if sparse_loss else base_valid_mask
    attention_scores_softmax = _compute_indexer_teacher_probabilities(
        attention_scores, attention_valid_mask, non_compressed_lse=non_compressed_lse
    )
    # Free attention_scores immediately
    del attention_scores

    index_scores_softmax = dsa_masking.masked_softmax(
        index_scores.float(), index_valid_mask, dim=-1
    )
    # Free index_scores - no longer needed after softmax
    del index_scores

    # Sum attention scores across heads: [b, np, sq, sk] -> [b, sq, sk]
    attention_scores_sum = attention_scores_softmax.sum(dim=1)
    # Free attention_scores_softmax
    del attention_scores_softmax

    if pg_collection.tp.size() > 1:
        # attention scores are scattered to TP ranks in head dimension.
        torch.distributed.all_reduce(attention_scores_sum.contiguous(), group=pg_collection.tp)

    # L1 normalize. Fully masked packed/varlen rows can have zero summed
    # attention mass; clamp the denominator so those rows stay finite and are
    # later zeroed by the row-valid loss mask.
    attention_scores_normalized = _normalize_indexer_teacher_target(
        attention_scores_sum, non_compressed_lse
    )
    # Free attention_scores_sum - no longer needed after normalization
    del attention_scores_sum

    # Backward through loss = kl_div * loss_coeff
    # where kl_div = kl_per_element.sum(dim=-1).mean()
    grad_kl_div = grad_loss * loss_coeff  # scalar

    if calculate_per_token_loss:
        grad_kl_per_row = grad_kl_div
    else:
        valid_row_count = (
            query_valid_rows.sum().to(
                dtype=torch.float32, device=attention_scores_normalized.device
            )
            if query_valid_rows is not None
            else torch.tensor(
                float(b * sq), dtype=torch.float32, device=attention_scores_normalized.device
            )
        ).clamp_min(1.0)
        grad_kl_per_row = grad_kl_div / valid_row_count  # scalar value for each real row

    # Backward through sum(dim=-1): broadcast back to [b, sq, sk]
    # Each element in a row contributes to the sum, so gradient is same for all
    grad_kl_per_element = grad_kl_per_row.view(1, 1, 1).expand(b, sq, sk)
    if query_valid_rows is not None:
        grad_kl_per_element = grad_kl_per_element * query_valid_rows.unsqueeze(-1).to(
            dtype=grad_kl_per_element.dtype
        )

    # For KL(target || softmax(logits)), the exact logit gradient is
    # predict * target.sum(-1) - target. Positive teacher rows are L1-normalized,
    # while a fully masked zero-mass row must have zero gradient.
    attention_target_mass = attention_scores_normalized.sum(dim=-1, keepdim=True)
    grad_index_scores_logits = (
        index_scores_softmax * attention_target_mass - attention_scores_normalized
    ) * grad_kl_per_element
    del index_scores_softmax, attention_scores_normalized

    # Zero out gradients for masked positions.
    if sparse_loss:
        # Also apply index mask - only topk positions are valid.
        del index_mask
        valid_mask = base_valid_mask & index_valid_mask  # [b, sq, sk]
        del index_valid_mask
    else:
        del index_mask
        valid_mask = base_valid_mask  # [b, sq, sk]
    del base_valid_mask
    if query_valid_rows is not None:
        valid_mask = valid_mask & query_valid_rows.unsqueeze(-1)

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

    # Backward through multiplication by weights (with optional ReLU).
    if use_relu:
        scores_for_weights = torch.relu(scores)
        relu_mask = scores > 0
    else:
        scores_for_weights = scores
        relu_mask = None
    del scores

    # ∂L/∂weights = grad * scores_for_weights (sum over sk)
    grad_weights = (grad_weighted_scores * scores_for_weights).sum(dim=-1)  # [sq, b, h]

    # ∂L/∂scores = grad * weights
    grad_scores = grad_weighted_scores * weights.unsqueeze(-1)  # [sq, b, h, sk]
    del grad_weighted_scores, scores_for_weights

    # Backward through ReLU (skip when use_relu=False)
    if use_relu:
        grad_scores = grad_scores * relu_mask.float()
        del relu_mask

    # Backward through einsum 'sbhd,tbd->sbht'
    # ∂L/∂q = einsum('sbht,tbd->sbhd', grad_scores, k)
    grad_q = torch.einsum('sbht,tbd->sbhd', grad_scores, k.float())  # [sq, b, h, d]
    # ∂L/∂k = einsum('sbht,sbhd->tbd', grad_scores, q)
    grad_k = torch.einsum('sbht,sbhd->tbd', grad_scores, q.float())  # [sk, b, d]
    del grad_scores

    return grad_q.to(q.dtype), grad_weights.to(weights.dtype), grad_k.to(k.dtype)


_FUSED_DSA_INDEXER_LOSS_INPUT_NAMES = (
    "q",
    "weights",
    "k",
    "query",
    "key",
    "softmax_scale",
    "topk",
    "loss_coeff",
    "mask",
    "sparse_loss",
    "pg_collection",
    "varlen_starts",
    "varlen_ends",
    "key_positions",
    "query_valid_rows",
    "calculate_per_token_loss",
    "use_relu",
    "non_compressed_lse",
)


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
        query_valid_rows=None,
        calculate_per_token_loss: bool = False,
        use_relu: bool = True,
        non_compressed_lse: torch.Tensor | None = None,
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
            query_valid_rows=query_valid_rows,
            calculate_per_token_loss=calculate_per_token_loss,
            use_relu=use_relu,
            non_compressed_lse=non_compressed_lse,
        )

        # Save for backward (recomputation strategy)
        saved_non_compressed_lse = (
            non_compressed_lse
            if non_compressed_lse is not None
            else q.new_empty(0, dtype=torch.float32)
        )
        ctx.save_for_backward(q, weights, k, query, key, topk_indices, saved_non_compressed_lse)
        ctx.has_non_compressed_lse = non_compressed_lse is not None
        ctx.softmax_scale = softmax_scale
        ctx.loss_coeff = loss_coeff
        ctx.sparse_loss = sparse_loss
        ctx.mask = mask
        ctx.pg_collection = pg_collection
        ctx.varlen_starts = varlen_starts
        ctx.varlen_ends = varlen_ends
        ctx.key_positions = key_positions
        ctx.query_valid_rows = query_valid_rows
        ctx.calculate_per_token_loss = calculate_per_token_loss
        ctx.use_relu = use_relu
        ctx.num_inputs = len(ctx.needs_input_grad)

        return topk_indices, loss

    @staticmethod
    def backward(ctx, grad_topk_indices, grad_loss):
        """
        Backward: Recompute what we need.
        """
        q, weights, k, query, key, topk_indices, saved_non_compressed_lse = ctx.saved_tensors
        non_compressed_lse = saved_non_compressed_lse if ctx.has_non_compressed_lse else None

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
            query_valid_rows=ctx.query_valid_rows,
            calculate_per_token_loss=ctx.calculate_per_token_loss,
            use_relu=ctx.use_relu,
            non_compressed_lse=non_compressed_lse,
        )

        grad_by_name = {
            "q": grad_q,
            "weights": grad_weights,
            "k": grad_k,
            # query and key are detached in forward, so return None for their gradients.
            "query": None,
            "key": None,
            "non_compressed_lse": None,
        }
        gradients = tuple(grad_by_name.get(name) for name in _FUSED_DSA_INDEXER_LOSS_INPUT_NAMES)
        return gradients[: ctx.num_inputs]


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

    query, query_was_thd = dsa_layout.ensure_sbhd(query, "query")
    key, _ = dsa_layout.ensure_sbhd(key, "key")
    value, _ = dsa_layout.ensure_sbhd(value, "value")

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

    row_mask, varlen_starts, varlen_ends, key_positions = dsa_masking.prepare_sparse_mask_context(
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

                # These tensors participate in autograd; reusing cached storage can
                # invalidate saved tensors before backward runs.
                m = torch.full(
                    (h_chunk, s_len), float("-inf"), dtype=torch.float32, device=query.device
                )
                l = torch.zeros((h_chunk, s_len), dtype=torch.float32, device=query.device)
                acc = torch.zeros((h_chunk, s_len, hnv), dtype=torch.float32, device=query.device)

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

                    valid_2d, mask_bias = dsa_masking.gather_sparse_topk_validity_and_bias(
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
                    m_new_for_exp = torch.where(
                        torch.isfinite(m_new), m_new, torch.zeros_like(m_new)
                    )
                    alpha = torch.exp(m - m_new_for_exp)
                    p = torch.exp(logits - m_new_for_exp.unsqueeze(-1))
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
