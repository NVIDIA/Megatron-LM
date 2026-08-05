# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.

from megatron.core.models.hybrid.hybrid_architecture import (
    HybridLayerPattern,
    HybridLayerSpec,
    PipelineSplit,
    ResolvedHybridArchitecture,
    flatten_hybrid_layer_pattern,
    resolve_hybrid_architecture,
)

__all__ = [
    "HybridLayerPattern",
    "HybridLayerSpec",
    "PipelineSplit",
    "ResolvedHybridArchitecture",
    "flatten_hybrid_layer_pattern",
    "resolve_hybrid_architecture",
]
