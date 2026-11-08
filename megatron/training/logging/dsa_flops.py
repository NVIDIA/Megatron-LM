# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compatibility exports for DSA FLOP accounting now owned by Megatron Core."""

from megatron.core.models.hybrid.layers.dsa_flops import (
    _dsa_indexer_flops,
    _dsa_kpool_attention_scale,
    _dsa_kpool_pairs,
    _dsa_sparse_core_scale,
    _num_dsa_indexer_layers,
    _num_dsa_indexer_layers_in_pattern,
    dsa_indexer_flops,
    dsa_layer_flops,
)

__all__ = [
    "_dsa_indexer_flops",
    "_dsa_kpool_attention_scale",
    "_dsa_kpool_pairs",
    "_dsa_sparse_core_scale",
    "_num_dsa_indexer_layers",
    "_num_dsa_indexer_layers_in_pattern",
    "dsa_indexer_flops",
    "dsa_layer_flops",
]
