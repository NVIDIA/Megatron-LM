# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""An unregistered layer defined outside Core, shared by extension contract tests."""

from dataclasses import dataclass, field
from typing import ClassVar

import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig


class ExternalGainLayer(MegatronModule):
    """Trainable channel scaling, without inheriting any built-in layer family."""

    def __init__(
        self,
        config: TransformerConfig,
        *,
        pg_collection: ProcessGroupCollection,
        layer_number: int,
        pp_layer_offset: int = 0,
        is_mtp_layer: bool = False,
        mtp_layer_number: int | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(config)
        self.layer_number = layer_number
        self.pg_collection = pg_collection
        self.tp_group = pg_collection.tp
        self.is_mtp_layer = is_mtp_layer
        self.gain = torch.nn.Parameter(
            torch.full((config.hidden_size,), config.gain_init, dtype=config.params_dtype)
        )
        self.last_forward_inputs = None

    @classmethod
    def estimate_flops(
        cls,
        config: TransformerConfig,
        *,
        total_real_tokens_in_batch: float,
        seqlen_squared_sum_in_batch: float,
    ) -> float:
        return total_real_tokens_in_batch * config.hidden_size

    def forward(
        self,
        hidden_states,
        *,
        attention_mask=None,
        rotary_pos_emb=None,
        packed_seq_params=None,
        padding_mask=None,
        input_ids=None,
    ):
        self.last_forward_inputs = (
            attention_mask,
            rotary_pos_emb,
            packed_seq_params,
            padding_mask,
            input_ids,
        )
        return hidden_states * self.gain


@dataclass
class ExternalGainConfig(TransformerConfig):
    """A synthetic architecture with fields that Core does not know about."""

    layer_type: ClassVar[type[MegatronModule]] = ExternalGainLayer
    gain_init: float = 1.25
    notes: list[str] = field(default_factory=lambda: ["external"])
