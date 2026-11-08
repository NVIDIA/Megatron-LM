# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Config-owned construction and execution of layers defined outside Megatron Core."""

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import MISSING, dataclass, fields
from typing import ClassVar, Self, cast

import torch
from torch import Tensor

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass(frozen=True)
class HybridLayerBuildContext:
    """Runtime placement supplied to a config's ``build_layer`` method.

    This context is not part of the architecture config or checkpoint metadata.
    ``layer_number`` is one-based within the decoder, or within an MTP inner stack.
    ``mtp_layer_number`` identifies the enclosing MTP depth when applicable.
    """

    pg_collection: ProcessGroupCollection
    """Explicit process groups owned by the caller."""
    layer_number: int
    """Global physical layer number within this stack."""
    pp_layer_offset: int = 0
    """Number of preceding layers on other pipeline stages."""
    is_mtp_layer: bool = False
    """Whether this layer belongs to an MTP inner stack."""
    mtp_layer_number: int | None = None
    """Enclosing MTP depth's logging slot, when present."""
    name: str | None = None
    """Module name supplied by the enclosing model."""


@dataclass
class HybridLayerConfig(TransformerConfig, ABC):
    """Base config for a layer supplied by Bridge or another downstream package.

    Subclasses directly construct a :class:`HybridLayer`; no registration, symbol, or
    module specification is needed. The source config remains read-only: HybridModel
    clones each physical occurrence before construction, including repeated MTP layers.

    The initial execution contract supports ordinary residual streams, TP1/CP1, and
    eager or full-layer recomputation. HybridModel and HybridStack validate these
    integration limits; implementations validate their own requirements at construction.
    """

    uses_shared_rotary_embeddings: ClassVar[bool] = False
    """Request the model's shared rotary embeddings, with matching ``kv_channels``."""
    is_moe_layer: ClassVar[bool] = False
    """Use the standard MoE tracker and inherited MoE logging fields for this layer."""

    @classmethod
    def from_config(cls, config: TransformerConfig) -> Self:
        """Copy normalized settings and initialize newly introduced dataclass defaults.

        Like TransformerConfig.from_config, this avoids rerunning normalization. Unlike
        a plain state copy, converting a common config initializes subclass fields and
        default factories. Construct configs with required new fields directly instead.
        Validate layer-specific requirements in ``build_layer`` or the layer constructor,
        since copying a normalized config does not rerun ``__post_init__``.
        """
        if type(config) is cls:
            # Physical clones must preserve subclass slots as well as __dict__, without
            # rerunning normalization or replacing configured values with field defaults.
            return cast(Self, deepcopy(config))
        result = super().from_config(config)
        for config_field in fields(cls):
            if config_field.name in result.__dict__:
                continue
            if config_field.default is not MISSING:
                value = deepcopy(config_field.default)
            elif config_field.default_factory is not MISSING:
                value = config_field.default_factory()
            else:
                raise TypeError(
                    f"{cls.__name__}.from_config requires a value for {config_field.name!r}; "
                    "construct the layer config directly."
                )
            setattr(result, config_field.name, value)
        return result

    @abstractmethod
    def build_layer(self, context: HybridLayerBuildContext) -> "HybridLayer":
        """Construct a fresh module directly from this physical config and placement."""
        raise NotImplementedError

    def estimate_flops(
        self, *, total_real_tokens_in_batch: float, seqlen_squared_sum_in_batch: float
    ) -> float:
        """Return forward FLOPs for this layer over one global batch.

        This is a pure query: do not mutate the config or perform collectives. Required
        when using the standard training loop's FLOP accounting. The model
        estimator applies its forward/backward multiplier and counts all MTP depths.
        An unavailable estimate must not silently select a built-in architecture formula.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement estimate_flops for training FLOP accounting."
        )


class HybridLayer(MegatronModule, ABC):
    """Runtime contract for a config-constructed custom hybrid layer.

    Forward consumes and returns a hidden-state tensor with the same shape. It owns
    its complete layer computation, including any normalization and residual addition.
    Normal module calls are preserved so hooks and distributed wrappers can run.
    """

    def __init__(self, config: HybridLayerConfig, context: HybridLayerBuildContext) -> None:
        super().__init__(config)
        self.layer_number = context.layer_number
        self.pg_collection = context.pg_collection
        self.tp_group = context.pg_collection.tp
        self.is_mtp_layer = context.is_mtp_layer

    @abstractmethod
    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        rotary_pos_emb: Tensor | tuple[Tensor, Tensor] | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        padding_mask: Tensor | None = None,
        sequence_len_offset: Tensor | None = None,
        input_ids: Tensor | None = None,
    ) -> Tensor:
        """Apply the layer, preserving the input hidden-state shape and layout."""
        raise NotImplementedError

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Use caller-owned process groups for the standard Megatron checkpoint contract."""
        metadata = dict(metadata or {})
        if metadata.get("dp_cp_group") is None:
            metadata["dp_cp_group"] = self.pg_collection.dp_cp
        return super().sharded_state_dict(prefix, sharded_offsets, metadata)


def invoke_hybrid_layer(
    layer: HybridLayer,
    hidden_states: Tensor,
    *,
    attention_mask: Tensor | None = None,
    rotary_pos_emb: Tensor | tuple[Tensor, Tensor] | None = None,
    packed_seq_params: PackedSeqParams | None = None,
    padding_mask: Tensor | None = None,
    sequence_len_offset: Tensor | None = None,
    input_ids: Tensor | None = None,
    inference_context=None,
) -> Tensor:
    """Invoke the same custom-layer contract in eager and recomputed execution."""
    if inference_context is not None:
        raise ValueError("Custom hybrid layers do not support inference contexts yet.")
    output = layer(
        hidden_states,
        attention_mask=attention_mask,
        rotary_pos_emb=rotary_pos_emb,
        packed_seq_params=packed_seq_params,
        padding_mask=padding_mask,
        sequence_len_offset=sequence_len_offset,
        input_ids=input_ids,
    )
    if not isinstance(output, torch.Tensor) or output.shape != hidden_states.shape:
        raise ValueError("A custom hybrid layer must return a tensor with the input shape.")
    return output
