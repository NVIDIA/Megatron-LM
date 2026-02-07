# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import Optional

from megatron.core.process_groups_config import ProcessGroupCollection
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
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            hidden_dropout=hidden_dropout,
            pg_collection=pg_collection,
        )
