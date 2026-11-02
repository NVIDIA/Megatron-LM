# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Checks that a ``SelfAttention`` can run shared-prefix forest attention.

The forest attention itself lives in :mod:`megatron.core.ops.attention.shared_prefix`. A caller
that attaches a forest runs :func:`validate_shared_prefix_attention` for every attention layer
before any layer executes; the attention forward does not repeat these checks.
"""

from megatron.core.ops.attention.shared_prefix import _cp_kv_head_slices_for_destinations
from megatron.core.transformer.attention import SelfAttention


def _dense_core_attention_types() -> tuple:
    """Core attention classes that compute plain softmax dot-product attention."""
    from megatron.core.extensions.transformer_engine import HAVE_TE
    from megatron.core.transformer.dot_product_attention import DotProductAttention

    if not HAVE_TE:
        return (DotProductAttention,)
    from megatron.core.extensions.transformer_engine import TEDotProductAttention

    return (DotProductAttention, TEDotProductAttention)


def validate_shared_prefix_attention(attention, config, *, tp_size: int, cp_size: int) -> None:
    """Reject a ``SelfAttention`` that the forest attention cannot run on this TP/CP topology.

    A caller that attaches a forest runs this for every attention layer before any layer
    executes; the attention forward does not repeat it. ``tp_size`` and ``cp_size`` are the
    caller's group sizes.
    """
    # Other Attention subclasses, such as multi-latent attention, override forward and never
    # read the slot. SelfAttention (and any subclass that keeps its forward) does.
    if not isinstance(attention, SelfAttention):
        raise NotImplementedError(
            "shared-prefix fused attention supports only SelfAttention, got "
            f"{type(attention).__name__}"
        )
    # The forest replaces core attention outright, so only dense dot-product core attention can
    # be replaced exactly. Attention variants can keep SelfAttention's forward and plug in a
    # different core attention (DSA over GQA does), which the forest would silently bypass.
    if not isinstance(getattr(attention, "core_attention", None), _dense_core_attention_types()):
        raise NotImplementedError(
            "shared-prefix fused attention requires dot-product core attention, got "
            f"{type(getattr(attention, 'core_attention', None)).__name__}"
        )
    if attention.checkpoint_core_attention:
        raise NotImplementedError(
            'shared-prefix fused attention does not yet support selective '
            'core-attention recomputation'
        )
    if config.qk_clip or config.log_max_attention_logit:
        raise NotImplementedError(
            'shared-prefix fused attention does not yet produce QK-clipping/max-logit statistics'
        )
    # The forest branch replaces core attention, so it must not silently drop these settings.
    window_size = config.window_size
    if window_size is not None and tuple(window_size) != (-1, -1):
        raise NotImplementedError("shared-prefix fused attention does not support sliding windows")
    if config.softmax_type != "vanilla":
        raise NotImplementedError("shared-prefix fused attention supports only vanilla softmax")
    if config.attention_dropout != 0.0:
        raise NotImplementedError("shared-prefix fused attention requires zero attention dropout")
    if getattr(config, "attn_logit_softcapping", None) is not None:
        raise NotImplementedError(
            "shared-prefix fused attention does not support logit softcapping"
        )
    # Batch-invariant mode pins FlashAttention 3 or 4 with a fixed split schedule
    # (TransformerConfig requires flash_attention_version 3 or 4). The forest always runs
    # FlashAttention 2 passes merged by log-sum-exp, so it cannot reproduce those kernels.
    if getattr(config, "batch_invariant_mode", False):
        raise NotImplementedError(
            "shared-prefix fused attention does not support batch-invariant mode: it runs "
            "FlashAttention 2 passes whatever flash_attention_version selects"
        )
    if attention.pg_collection.tp.size() != tp_size:
        raise RuntimeError(
            f"shared-prefix attention TP helper size {attention.pg_collection.tp.size()} does "
            f"not match the caller's TP size {tp_size}"
        )
    if attention.pg_collection.cp.size() != cp_size:
        raise RuntimeError(
            f"shared-prefix attention CP helper size {attention.pg_collection.cp.size()} does "
            f"not match the caller's CP size {cp_size}"
        )
    if cp_size > 1:
        if config.num_attention_heads % tp_size:
            raise NotImplementedError(
                "shared-prefix attention requires query heads divisible by TP size"
            )
        query_heads = config.num_attention_heads // tp_size
        # This is the actual local K/V tensor width. When global KV heads are fewer than TP
        # ranks, SelfAttention replicates one KV head across the relevant TP ranks.
        kv_heads = attention.num_query_groups_per_partition
        _cp_kv_head_slices_for_destinations(query_heads, kv_heads, cp_size)
