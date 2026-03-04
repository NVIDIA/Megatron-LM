# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""
Engram-enabled Transformer Layer.

Extends the standard TransformerLayer to inject an Engram module that runs
before the self-attention computation. The engram output is added as a residual
to the hidden states before they enter the standard attention + MLP pipeline.
"""

from __future__ import annotations

from typing import Any, Optional

from torch import Tensor

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)

from megatron.core.models.engram.engram_module import EngramConfig, EngramModule


class EngramTransformerLayer(TransformerLayer):
    """A transformer layer augmented with an Engram module.

    The Engram module is applied as a residual before the standard self-attention
    computation. The pre-computed engram embeddings must be set via the
    ``engram.precompute_embeddings()`` method before each forward pass (handled
    by ``EngramGPTModel``).

    For layers that are not in the engram_layer_ids list, this behaves identically
    to a standard TransformerLayer.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
        is_mtp_layer: bool = False,
        add_layer_offset: bool = True,
        pp_layer_offset: Optional[int] = None,
        engram_config: Optional[EngramConfig] = None,
        engram_vocab_size_across_layers: Optional[dict] = None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            hidden_dropout=hidden_dropout,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
            is_mtp_layer=is_mtp_layer,
            add_layer_offset=add_layer_offset,
            pp_layer_offset=pp_layer_offset,
        )

        self.engram: Optional[EngramModule] = None
        if (
            engram_config is not None
            and engram_vocab_size_across_layers is not None
            and self.layer_number in engram_config.engram_layer_ids
            and self.layer_number in engram_vocab_size_across_layers
        ):
            self.engram = EngramModule(
                layer_id=self.layer_number,
                hidden_size=config.hidden_size,
                engram_config=engram_config,
                vocab_size_for_layer=engram_vocab_size_across_layers[self.layer_number],
            )

    def forward(self, hidden_states: Tensor, *args: Any, **kwargs: Any):
        """Forward pass with optional Engram injection before attention.

        The Engram output is added as a residual to hidden_states before the
        standard TransformerLayer forward (layernorm → attention → MLP).
        """
        if self.engram is not None:
            engram_output = self.engram(hidden_states)
            hidden_states = engram_output + hidden_states

        return super().forward(hidden_states, *args, **kwargs)
