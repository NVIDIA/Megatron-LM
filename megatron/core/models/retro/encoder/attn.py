# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from torch import Tensor

from megatron.core import InferenceParams
from megatron.core.models.retro.attn import BaseRetroCrossAttention
# from megatron.core.transformer.attention import CrossAttention, CrossAttentionSpec
# from megatron.core.transformer.custom_layers.transformer_engine import TENorm
# from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

# >>>
from lutil import pax
# <<<


class RetroEncoderCrossAttention(BaseRetroCrossAttention):

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states: Tensor = None,
        inference_params: InferenceParams = None,
        rotary_pos_emb: Tensor = None,
        retriever_input: Tensor = None,
        retriever_output: Tensor = None,
        retriever_attn_mask: Tensor = None,
    ):
        # hidden_states: [sq, b, h]

        attention_output_with_bias = self.attn( # super()(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            key_value_states=key_value_states,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
        )

        pax("attention_output_with_bias")

        assert isinstance(add_retriever, bool), "'add_retriever' must be defined."


class RetroEncoderBiasDropoutAdd(MegatronModule):

    def __init__(
        self,
        config: TransformerConfig,
        spec: ModuleSpec,
        # layer_number: int = 1,
        # attn_mask_type=AttnMaskType.padding,
        # **kwargs,
    ):
        super().__init__(config=config)
        self.spec = spec
        # pax("spec")


class RetroEncoderLayerNorm(MegatronModule):

    def __init__(
        self,
        config: TransformerConfig,
        spec: ModuleSpec,

        # hidden_size=self.config.hidden_size,
        # eps=self.config.layernorm_epsilon,
        # persist_layer_norm=self.config.persist_layer_norm,
        # sequence_parallel=self.config.sequence_parallel,
        # zero_centered_gamma=self.config.layernorm_zero_centered_gamma,
        # normalization=self.config.normalization,

        # hidden_size: int,
        # eps: float = 1e-5,
        # sequence_parallel: bool = False,
        # normalization: str = "LayerNorm",
        **kwargs,
    ):
        super().__init__(config=config)
        self.spec = spec

        self.norm = TENorm(
            config=config,
            # hidden_size=hidden_size,
            # eps=eps,
            # persist_layer_norm=config.persist_layer_norm,
            # sequence_parallel=sequence_parallel,
            # zero_centered_gamma=config.layernorm_zero_centered_gamma,
            # normalization=normalization,
            **kwargs,
        )

        # pax("config", "spec")
