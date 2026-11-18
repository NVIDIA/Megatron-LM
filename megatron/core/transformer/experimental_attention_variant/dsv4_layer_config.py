# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass

from megatron.core.transformer.transformer_config import MLATransformerConfig


class _CompactIndexerWorkspaceRegistry(dict):
    """Device-keyed runtime stores shared even before any layer has warmed up."""

    def __deepcopy__(self, memo):
        return self


def get_compact_indexer_workspace_registry(config):
    """Create the model-owned registry before per-layer config copies are made."""
    stores = getattr(config, "_compact_indexer_workspace_stores", None)
    if stores is None:
        stores = _CompactIndexerWorkspaceRegistry()
        # Runtime-only state is not a dataclass field or checkpoint/config payload.
        config._compact_indexer_workspace_stores = stores
    return stores


@dataclass(kw_only=True)
class CSALayerConfig(MLATransformerConfig):
    """Configuration for a Compressed Sparse Attention layer."""

    compress_ratio: int = 0
    """Token compression ratio for this attention layer."""

    @classmethod
    def from_config(cls, config):
        """Keep per-layer settings independent while sharing model-owned scratch space."""
        if config.dsa_compact_indexer_workspace_sharing:
            get_compact_indexer_workspace_registry(config)
        return super().from_config(config)
