# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from megatron.core.transformer.transformer_config import TransformerConfig


class MambaLayerConfig(TransformerConfig):
    """Configuration for a Mamba layer in a hybrid stack.

    Due to backwards-compatibility, this config's arguments are defined in TransformerConfig.
    """

    @classmethod
    def from_config(cls, config: TransformerConfig) -> "MambaLayerConfig":
        """Keep legacy Mamba headwise CP when no stack-layout policy was supplied."""
        layer_config = super().from_config(config)
        if not getattr(config, "_linear_cp_layout_explicit", True):
            # The historical linear_cp_mode knob controlled GDN/KDA. Mamba used
            # headwise CP; chunkwise GDP uses the new explicit contiguous layout.
            layer_config.linear_cp_mode = "headwise"
        return layer_config
