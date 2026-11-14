# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fuse token embeddings with previous-token hidden states for feedback passes.

Additive feedback adds a normalized, projected hidden state to each embedding.
Gated feedback uses the embedding to gate a projected hidden state, then normalizes
the product. Both preserve the shape of matching [sequence, batch, hidden_size] inputs.

Submodule specs select the normalization and projection implementations. The caller
controls repeated passes and shifts hidden states before passing them to these modules.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

import torch
from torch import nn

from megatron.core.ops.norm.torch_norm import LayerNormBuilder
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class AdditiveHiddenStateFusionSubmodules:
    """Norm builder and Megatron linear projection for additive feedback."""

    hidden_state_norm: LayerNormBuilder
    hidden_state_projection: Union[ModuleSpec, type]


@dataclass
class GatedHiddenStateFusionSubmodules:
    """Norm builders and Megatron linear projections for gated feedback."""

    token_embedding_norm: LayerNormBuilder
    output_norm: LayerNormBuilder
    hidden_state_projection: Union[ModuleSpec, type]
    gate_projection: Union[ModuleSpec, type]


class HiddenStateFusion(MegatronModule, ABC):
    """Combine embeddings and already-shifted hidden states into backbone inputs.

    Subclasses implement forward(token_embedding, previous_hidden), preserving the matching
    [sequence, batch, hidden_size] shape. Input preparation and pass execution belong
    to the caller; this base retains the supplied process group and parameter tags.

    ``config`` is the caller's model config for hidden size, normalization, and initialization.
    """

    def __init__(self, config: TransformerConfig, pg_collection) -> None:
        super().__init__(config)
        # MegatronModule.sharded_state_dict uses tp_group when it builds checkpoint entries.
        self.tp_group = pg_collection.tp

    def _build_projection(self, spec, init_method, *, bias=False):
        """Build a replicated Megatron linear returning (output, optional_bias)."""
        projection = build_module(
            spec,
            self.config.hidden_size,
            self.config.hidden_size,
            config=self.config,
            init_method=init_method,
            bias=bias,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        # Duplicated TELinear has no TP execution group; this group is for checkpoint metadata.
        projection._tp_group = self.tp_group
        # Several feedback passes share this projection; each backward must add to main_grad.
        projection.is_repeated_layer = True
        return projection

    @abstractmethod
    def forward(self, token_embedding: torch.Tensor, previous_hidden: torch.Tensor) -> torch.Tensor:
        """Fuse matching [sequence, batch, hidden_size] inputs without shifting them."""

    def _validate_inputs(self, token_embedding, previous_hidden):
        """Reject broadcasting or a hidden width incompatible with this fusion."""
        if (
            token_embedding.ndim != 3
            or token_embedding.shape != previous_hidden.shape
            or token_embedding.shape[-1] != self.config.hidden_size
        ):
            raise ValueError(
                "Feedback fusion requires matching [sequence, batch, hidden_size] inputs "
                f"with hidden_size={self.config.hidden_size}"
            )

    def tag_parameters(self):
        """Mark fusion parameters for optimizer selection.

        Callers can select these parameters with
        ``ParamKey(attr="is_hidden_state_feedback_parameter")`` to set LR overrides.
        The tag does not create optimizer groups. When Muon is enabled, matrix
        parameters use it and other parameters use the configured scalar optimizer.

        TODO [kwangjuna]: Add optimizer groups, schedules, and checkpoint migration
        in the training integration.
        """
        for parameter in self.parameters():
            parameter.is_hidden_state_feedback_parameter = True
            parameter.use_muon = parameter.ndim == 2


class AdditiveHiddenStateFusion(HiddenStateFusion):
    """E + Linear(Norm(H)), inspired by Recycler's additive hidden-state adapter.

    See ByteDance Seed, "Recycler: Cost-Free Latent Reasoning via Hidden State Recycling".
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: AdditiveHiddenStateFusionSubmodules,
        pg_collection,
        bias: bool = False,
    ) -> None:
        super().__init__(config, pg_collection)
        self.hidden_state_norm = submodules.hidden_state_norm(
            config=config, hidden_size=config.hidden_size, eps=config.layernorm_epsilon
        )
        self.hidden_state_projection = self._build_projection(
            submodules.hidden_state_projection, nn.init.zeros_, bias=bias
        )
        self.tag_parameters()

    def forward(self, token_embedding, previous_hidden):
        """Add the projected, normalized previous state to the embedding."""
        self._validate_inputs(token_embedding, previous_hidden)
        hidden = self.hidden_state_norm(previous_hidden).to(
            self.hidden_state_projection.weight.dtype
        )
        update, _ = self.hidden_state_projection(hidden)
        return token_embedding + update.to(token_embedding.dtype)


class GatedHiddenStateFusion(HiddenStateFusion):
    """OutputNorm(HiddenLinear(H) * sigmoid(GateLinear(InputNorm(E)))).

    Inspired by Wang et al., "Full-bandwidth Transformer".
    """

    def __init__(
        self, config: TransformerConfig, submodules: GatedHiddenStateFusionSubmodules, pg_collection
    ) -> None:
        super().__init__(config, pg_collection)
        self.token_embedding_norm = submodules.token_embedding_norm(
            config=config, hidden_size=config.hidden_size, eps=config.layernorm_epsilon
        )
        self.output_norm = submodules.output_norm(
            config=config, hidden_size=config.hidden_size, eps=config.layernorm_epsilon
        )
        self.hidden_state_projection = self._build_projection(
            submodules.hidden_state_projection, config.init_method
        )
        self.gate_projection = self._build_projection(
            submodules.gate_projection, config.init_method
        )
        self.tag_parameters()

    def forward(self, token_embedding, previous_hidden):
        """Gate the projected previous state by the embedding and normalize the product."""
        self._validate_inputs(token_embedding, previous_hidden)
        dtype = self.hidden_state_projection.weight.dtype
        value, _ = self.hidden_state_projection(previous_hidden.to(dtype))
        gate_logits, _ = self.gate_projection(self.token_embedding_norm(token_embedding).to(dtype))
        gate = torch.sigmoid(gate_logits)
        return self.output_norm(value * gate).to(token_embedding.dtype)
