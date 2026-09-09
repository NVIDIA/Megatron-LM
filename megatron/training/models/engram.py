# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Prepare Engram's optional Hybrid input provider from the training tokenizer."""

from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig


def engram_context_provider_spec(
    config: TransformerConfig, hybrid_layer_pattern: str | None, pad_id: int | None = None
) -> ModuleSpec | None:
    """Build the optional provider spec without accessing the tokenizer when disabled."""
    if not config.engram_layer_ids:
        return None

    from megatron.core.transformer.engram.hybrid_adapter import EngramHybridProvider
    from megatron.core.transformer.engram.tokenizer import (
        build_engram_tokenizer_lookup,
        get_engram_tokenizer_pad_id,
    )
    from megatron.training import get_tokenizer

    tokenizer = get_tokenizer()
    return ModuleSpec(
        module=EngramHybridProvider,
        params={
            "tokenizer_lookup": build_engram_tokenizer_lookup(tokenizer),
            "pad_id": get_engram_tokenizer_pad_id(tokenizer, pad_id),
            "hybrid_layer_pattern": hybrid_layer_pattern,
        },
    )
