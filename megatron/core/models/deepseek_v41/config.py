# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Configuration for V4.1 conditional n-gram memory."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EngramConfig:
    """Released n-gram table layout and tokenizer-compression contract."""

    layer_ids: list[int]
    """Zero-based backbone blocks receiving conditional memory."""

    num_embeddings: list[int]
    """Total fused-table rows per selected backbone block."""

    max_ngram_size: int
    """Largest n-gram order; all orders from two through this value are used."""

    vocab_size: int
    """Initial bucket-size budget for the distinct prime hash tables."""

    n_heads: int
    """Hash heads per n-gram order."""

    head_dim: int
    """Embedding channels returned by each hash head."""

    pad_token_id: int
    """Raw tokenizer ID used for missing causal history."""

    compressed_vocab_size: int
    """Exact number of normalized token identities used to derive hash multipliers."""
