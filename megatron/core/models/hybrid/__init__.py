# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.

from megatron.core.models.hybrid.common_layer_config import CommonLayerConfig
from megatron.core.models.hybrid.layer_configs import (
    AttentionLayerConfig,
    CrossEntropyLayerConfig,
    DSALayerConfig,
    EmbeddingLayerConfig,
    GDNLayerConfig,
    LayerConfig,
    MambaLayerConfig,
    MLPLayerConfig,
    MoELayerConfig,
)
from megatron.core.models.hybrid.layer_pattern import (
    RECIPE_ENTRY_POINT,
    flatten_decoder_pattern,
    load_recipe,
)


def __getattr__(name):
    if name == "HybridModelConfig":
        from megatron.training.models.hybrid import HybridModelConfig

        return HybridModelConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AttentionLayerConfig",
    "CommonLayerConfig",
    "CrossEntropyLayerConfig",
    "DSALayerConfig",
    "EmbeddingLayerConfig",
    "GDNLayerConfig",
    "HybridModelConfig",
    "LayerConfig",
    "MambaLayerConfig",
    "MLPLayerConfig",
    "MoELayerConfig",
    "RECIPE_ENTRY_POINT",
    "load_recipe",
]
