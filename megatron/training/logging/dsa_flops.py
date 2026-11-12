# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""DSA attention and lightning-indexer FLOPs accounting."""

from types import SimpleNamespace

from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols, parse_hybrid_pattern
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexer,
    DSAttention,
    _dsa_kpool_attention_scale,
    _dsa_sparse_core_scale,
    is_dsa_skip_topk_layer,
)


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
    """Compatibility coefficients from the indexer's own FLOP estimator."""
    config = SimpleNamespace(
        hidden_size=hidden_size,
        q_lora_rank=q_lora_rank,
        dsa_indexer_n_heads=n_heads,
        dsa_indexer_head_dim=head_dim,
        dsa_indexer_loss_coeff=int(dsa_indexer_loss_enabled),
        dsa_indexer_use_sparse_loss=dsa_indexer_use_sparse_loss,
        dsa_indexer_topk=dsa_indexer_topk,
        dsa_indexer_kpool=dsa_indexer_kpool,
    )
    return DSAIndexer._flops_coefficients(
        config,
        total_real_tokens_in_batch=total_tokens,
        seqlen_squared_sum_in_batch=seqlen_squared_sum,
        include_backward=True,
        num_indexer_layers=num_indexer_layers,
    )


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
    absorbed_mla: bool = True,
) -> float:
    """Compatibility wrapper for the layer-owned forward DSA estimate."""
    config = SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_head_dim=qk_head_dim,
        qk_pos_emb_head_dim=qk_pos_emb_head_dim,
        v_head_dim=v_head_dim,
        dsa_indexer_topk=dsa_indexer_topk,
        dsa_indexer_kpool=dsa_indexer_kpool,
        dsa_indexer_kpool_always_select_tail=dsa_indexer_kpool_always_select_tail,
    )
    return DSAttention.estimate_flops(
        config,
        total_real_tokens_in_batch=total_tokens,
        seqlen_squared_sum_in_batch=seqlen_squared_sum,
        absorbed_mla=absorbed_mla,
    )


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
