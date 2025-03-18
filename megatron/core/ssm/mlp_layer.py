# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from megatron.core.transformer import (
    TransformerConfig,
    TransformerLayer,
    TransformerLayerSubmodules,
)


class MLPLayer(TransformerLayer):
    """Drop-in replacement for TransformerLayer but initializes only an MLP via the spec."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: float = None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            hidden_dropout=hidden_dropout,
        )
