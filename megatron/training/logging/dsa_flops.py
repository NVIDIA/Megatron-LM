# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""DSA attention and lightning-indexer FLOPs accounting."""

import math

from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols, parse_hybrid_pattern
from megatron.core.ops.attention.dsa.modules import is_dsa_skip_topk_layer


def _dsa_sparse_core_scale(total_real_tokens, seqlen_squared_sum, dsa_indexer_topk):
    """Fraction of dense causal (query, key) pairs that DSA actually attends to.

    A DSA layer scores every past position with the indexer but runs attention
    only over the ``dsa_indexer_topk`` highest-scoring keys, so its core
    attention cost is ``sum_i(min(i, topk))`` pairs per sequence instead of the
    dense causal ``L^2 / 2``. Returns the ratio between the two, i.e. the
    factor the dense core-attention coefficient must be scaled by.

    The caller only has the batch aggregates ``sum_i(L_i)`` and
    ``sum_i(L_i ** 2)``, not the individual sequence lengths, so the ratio is
    evaluated at the length-weighted mean ``sum(L^2) / sum(L)``. For equal-length
    sequences, this aggregation introduces no additional approximation, but
    the pair counts still use the continuous approximation described below.
    For ragged batches, it weights toward the long sequences that dominate
    attention cost. Collapses to ``1.0`` when the sequences are
    no longer than ``topk``, where top-k selects everything and attention is
    dense.
    """
    if not dsa_indexer_topk or total_real_tokens <= 0 or seqlen_squared_sum <= 0:
        return 1.0
    mean_seqlen = seqlen_squared_sum / total_real_tokens
    # Average attended KV entries per query: ``min(i, topk)`` averaged over the
    # sequence, approximated as ``eff * (1 - eff / (2 * L))`` with
    # ``eff = min(topk, floor(L))``. The exact discrete average has ``eff - 1``
    # in the correction term; the O(1/L) difference is below the precision of
    # this estimate.
    eff = min(dsa_indexer_topk, math.floor(mean_seqlen))
    attended = eff * (1 - eff / (2 * mean_seqlen))
    # Dense causal attention averages ~``mean_seqlen / 2`` keys per query.
    return attended / (mean_seqlen / 2)


def _dsa_kpool_pairs(mean_seqlen, pool_size, max_pools=None):
    """Sum complete causal pools per query, optionally capped by selection.

    Interpolate adjacent integer lengths for a fractional length-weighted mean;
    individual document boundaries cannot be recovered from batch aggregates.
    """
    length = math.floor(mean_seqlen)
    pools, next_pools = length // pool_size, (length + 1) // pool_size
    if max_pools is not None:
        pools, next_pools = min(pools, max_pools), min(next_pools, max_pools)
    return (
        pool_size * pools * (pools - 1) / 2
        + pools * (length - pool_size * pools + 1)
        + (mean_seqlen - length) * next_pools
    )


def _dsa_kpool_attention_scale(
    total_tokens: float,
    seqlen_squared_sum: float,
    topk: int | None,
    pool_size: int,
    always_select_tail: bool,
) -> float:
    """Return the selected attention-pair fraction for pooled keys.

    Expand at most topk / pool_size complete pools to tokens and optionally
    append the incomplete tail. Normalize by discrete dense pairs so a dense
    selection retains the caller's existing L^2 / 2 FLOPs convention.
    """
    if topk is None or topk <= 0 or topk % pool_size:
        raise ValueError("KPool FLOPs require a positive top-k divisible by the pool size.")
    if total_tokens <= 0 or seqlen_squared_sum <= 0:
        return 0.0
    mean_seqlen = seqlen_squared_sum / total_tokens
    length = math.floor(mean_seqlen)
    fraction = mean_seqlen - length
    pools, tail = divmod(length, pool_size)
    attention_pairs = pool_size * _dsa_kpool_pairs(mean_seqlen, pool_size, topk // pool_size)
    if always_select_tail:
        attention_pairs += (
            pools * pool_size * (pool_size - 1) / 2
            + tail * (tail + 1) / 2
            + fraction * ((length + 1) % pool_size)
        )
    dense_pairs = length * (length + 1) / 2 + fraction * (length + 1)
    return attention_pairs / dense_pairs


def _dsa_indexer_flops(
    *,
    total_tokens,
    seqlen_squared_sum,
    hidden_size,
    q_lora_rank,
    n_heads,
    head_dim,
    num_indexer_layers,
    dsa_indexer_loss_enabled: bool,
    dsa_indexer_use_sparse_loss=False,
    dsa_indexer_topk=None,
    dsa_indexer_kpool=1,
):
    """DSA lightning-indexer FLOPs coefficients, fwd/bwd expansion included.

    Only actual indexer executions pay. Without KL loss, all work is forward
    only. With loss, detached projection inputs require fwd + wgrad (x2),
    while scoring requires dense fwd + dq + dk: x3 for dense KL, or
    ``1 + 2 * sparse_scale`` for sparse KL. This counts model FLOPs, excluding
    zero-padded work in the dense reference implementation of sparse backward.

    KPool is forward only: add its gate and complete-pool key reduction, and
    score complete pools instead of individual keys. Pool softmax, positional
    bias, top-k selection, teacher recomputation and KL operations are excluded.

    Returns ``(token_linear, core)`` including FMA x2 and backward factors;
    the caller multiplies by total tokens and sum(L^2), respectively.
    """
    if dsa_indexer_kpool > 1:
        if dsa_indexer_loss_enabled:
            raise ValueError("KPool FLOPs require indexer loss to be disabled, as in DSAttention.")
        if (
            dsa_indexer_topk is None
            or dsa_indexer_topk <= 0
            or dsa_indexer_topk % dsa_indexer_kpool
        ):
            raise ValueError("KPool FLOPs require a positive top-k divisible by the pool size.")
    if num_indexer_layers <= 0:
        return 0, 0
    if q_lora_rank is None:
        # Mirrors DSAIndexer's own fallback when the model has no q lora rank.
        q_lora_rank = hidden_size
    index_dim = n_heads * head_dim
    token_linear = num_indexer_layers * (
        q_lora_rank * index_dim  # wq_b
        + hidden_size * head_dim  # wk
        + hidden_size * n_heads  # weights_proj
    )
    # Scoring each query against every past token under a causal mask (/2).
    core = num_indexer_layers * index_dim / 2
    fma_expansion_factor = 2
    core_expansion: float
    if not dsa_indexer_loss_enabled:
        token_expansion, core_expansion = 1, 1
    else:
        # Projections: fwd + wgrad (input is detached, no dgrad).
        token_expansion = 2
        # Scoring: dense fwd + dq + dk, where dq/dk cover only the top-k pairs
        # under the sparse KL loss (see docstring).
        sparse_core_scale = (
            _dsa_sparse_core_scale(total_tokens, seqlen_squared_sum, dsa_indexer_topk)
            if dsa_indexer_use_sparse_loss
            else 1.0
        )
        core_expansion = 1 + 2 * sparse_core_scale
    token_linear = token_expansion * fma_expansion_factor * token_linear
    core = core_expansion * fma_expansion_factor * core
    if dsa_indexer_kpool > 1:
        indexer_scale = compression_scale = 0.0
        if total_tokens > 0 and seqlen_squared_sum > 0:
            mean_seqlen = seqlen_squared_sum / total_tokens
            length = math.floor(mean_seqlen)
            fraction = mean_seqlen - length
            pool_size = dsa_indexer_kpool
            pools, next_pools = length // pool_size, (length + 1) // pool_size
            dense_pairs = length * (length + 1) / 2 + fraction * (length + 1)
            indexer_scale = _dsa_kpool_pairs(mean_seqlen, pool_size) / dense_pairs
            compression_scale = pool_size * (pools + fraction * (next_pools - pools)) / mean_seqlen
        # Gate on every token; one FMA per key component in complete pools.
        token_linear += 2 * num_indexer_layers * head_dim * (hidden_size + compression_scale)
        core *= indexer_scale
    return token_linear, core


def _num_dsa_indexer_layers(num_layers, skip_topk_offset, topk_freq):
    """Count indexer executions in a block numbered from 1 to ``num_layers``."""
    return sum(
        1
        for layer_number in range(1, num_layers + 1)
        if not is_dsa_skip_topk_layer(layer_number, skip_topk_offset or 0, topk_freq or 1)
    )


def _num_dsa_indexer_layers_in_pattern(layer_types, dsa_symbol, skip_topk_offset, topk_freq):
    """Count hybrid DSA positions that compute their own index.

    The caller supplies one block with pipes stripped. Non-DSA positions
    advance the 1-based layer numbering used by DSAttention's skip predicate.
    """
    return sum(
        1
        for layer_number, symbol in enumerate(layer_types, start=1)
        if symbol == dsa_symbol
        and not is_dsa_skip_topk_layer(layer_number, skip_topk_offset or 0, topk_freq or 1)
    )


def dsa_layer_flops(
    total_tokens: float,
    seqlen_squared_sum: float,
    hidden_size: int,
    num_attention_heads: int,
    q_lora_rank: int | None,
    kv_lora_rank: int,
    qk_head_dim: int,
    qk_pos_emb_head_dim: int,
    v_head_dim: int,
    dsa_indexer_topk: int | None,
    dsa_indexer_kpool: int = 1,
    dsa_indexer_kpool_always_select_tail: bool = True,
) -> float:
    """Calculate forward FLOPs for one DSA attention layer, excluding its indexer.

    Like the other layer counters, the result includes the FMA factor of two
    and the batch token statistics, but no backward expansion or layer count.
    Callers multiply by the number of DSA layers and apply their common x3
    forward/backward expansion.

    Token-linear work includes the MLA projections and their norm estimates.
    Core attention uses the absorbed KV dimensions and top-k sparse pair scale.
    The indexer is accounted for separately by ``dsa_indexer_flops`` because
    its executed layer count and backward cost can differ from attention.
    With KPool, only complete pools are selected and expanded to original
    tokens; the tail flag controls additional incomplete-pool tokens.
    """
    if q_lora_rank is None:
        q_term = hidden_size * num_attention_heads * (qk_head_dim + qk_pos_emb_head_dim)
    else:
        q_term = q_lora_rank * (
            hidden_size + num_attention_heads * (qk_head_dim + qk_pos_emb_head_dim) + 1
        )
    fma_expansion_factor = 2
    token_linear = fma_expansion_factor * (
        ## q lora + rope + q norm
        q_term
        ## kv lora + rope + kv norm
        + kv_lora_rank * (hidden_size + num_attention_heads * (qk_head_dim + v_head_dim) + 1)
        + hidden_size * qk_pos_emb_head_dim
        ## o proj
        + (num_attention_heads * v_head_dim) * hidden_size
    )
    if dsa_indexer_kpool > 1:
        sparse_core_scale = _dsa_kpool_attention_scale(
            total_tokens,
            seqlen_squared_sum,
            dsa_indexer_topk,
            dsa_indexer_kpool,
            dsa_indexer_kpool_always_select_tail,
        )
    else:
        sparse_core_scale = _dsa_sparse_core_scale(
            total_tokens, seqlen_squared_sum, dsa_indexer_topk
        )
    core = (
        fma_expansion_factor
        * (
            num_attention_heads * (kv_lora_rank + qk_pos_emb_head_dim) / 2
            + num_attention_heads * kv_lora_rank / 2
        )
        * sparse_core_scale
    )
    return total_tokens * token_linear + seqlen_squared_sum * core


def dsa_indexer_flops(
    total_tokens: float,
    seqlen_squared_sum: float,
    hidden_size: int,
    q_lora_rank: int | None,
    dsa_indexer_n_heads: int,
    dsa_indexer_head_dim: int,
    dsa_indexer_topk: int | None,
    num_dsa_layers: int,
    dsa_indexer_loss_coeff: float | None = None,
    dsa_indexer_use_sparse_loss: bool = False,
    dsa_indexer_skip_topk_offset: int | None = 0,
    dsa_indexer_topk_freq: int | None = 1,
    hybrid_layer_pattern: str | None = None,
    mtp_num_layers: int | None = None,
    mtp_use_repeated_layer: bool = False,
    dsa_indexer_kpool: int = 1,
) -> float:
    """Calculate indexer training FLOPs across standard or hybrid DSA layers.

    Only layers computing their own top-k index contribute. Standard decoder
    and MTP blocks each number layers from 1. Repeated MTP executes the same
    layer 1 at every depth. Hybrid models use pattern positions across pipeline
    separators and restart numbering per MTP depth.

    The result includes the indexer's loss-dependent forward/backward factors
    and FMA factor. Add it after the model's common x3 expansion. Attention,
    MLPs, logits, and MTP norms/projections are accounted for by the caller.

    ``num_dsa_layers`` includes MTP and layers reusing another layer's index;
    zero returns zero FLOPs. ``mtp_num_layers`` is the standard MTP depth,
    already included in that total; hybrid MTP is derived from the pattern.
    Positive ``dsa_indexer_loss_coeff`` enables indexer backward work.
    ``dsa_indexer_kpool > 1`` pools keys and requires disabled indexer loss.
    """
    if num_dsa_layers == 0:
        return 0.0
    if hybrid_layer_pattern is None:
        mtp_num_layers = mtp_num_layers or 0
        num_indexer_layers = _num_dsa_indexer_layers(
            num_dsa_layers - mtp_num_layers, dsa_indexer_skip_topk_offset, dsa_indexer_topk_freq
        )
        # MTP numbering restarts at 1 in MultiTokenPredictionLayer. With
        # repeated MTP, that first layer computes an index at every depth.
        num_indexer_layers += (
            mtp_num_layers
            if mtp_use_repeated_layer
            else _num_dsa_indexer_layers(
                mtp_num_layers, dsa_indexer_skip_topk_offset, dsa_indexer_topk_freq
            )
        )
    else:
        # DSAttention's skip predicate runs on block-local layer numbers:
        # the main pattern numbers layers sequentially across pipes, and
        # each hybrid MTP depth restarts at 1 (``pp_layer_offset=0``).
        parsed_pattern = parse_hybrid_pattern(hybrid_layer_pattern)
        main_layer_types = (parsed_pattern.main_pattern or "").replace(Symbols.PIPE, "")
        num_indexer_layers = _num_dsa_indexer_layers_in_pattern(
            main_layer_types,
            Symbols.DS_ATTENTION,
            dsa_indexer_skip_topk_offset,
            dsa_indexer_topk_freq,
        )
        if parsed_pattern.mtp_pattern:
            num_indexer_layers += (
                parsed_pattern.mtp_num_depths
                * _num_dsa_indexer_layers_in_pattern(
                    parsed_pattern.mtp_pattern,
                    Symbols.DS_ATTENTION,
                    dsa_indexer_skip_topk_offset,
                    dsa_indexer_topk_freq,
                )
            )
    indexer_token_linear, indexer_core = _dsa_indexer_flops(
        total_tokens=total_tokens,
        seqlen_squared_sum=seqlen_squared_sum,
        hidden_size=hidden_size,
        q_lora_rank=q_lora_rank,
        n_heads=dsa_indexer_n_heads,
        head_dim=dsa_indexer_head_dim,
        num_indexer_layers=num_indexer_layers,
        dsa_indexer_loss_enabled=(dsa_indexer_loss_coeff or 0.0) > 0,
        dsa_indexer_use_sparse_loss=dsa_indexer_use_sparse_loss,
        dsa_indexer_topk=dsa_indexer_topk,
        dsa_indexer_kpool=dsa_indexer_kpool,
    )
    return total_tokens * indexer_token_linear + seqlen_squared_sum * indexer_core
