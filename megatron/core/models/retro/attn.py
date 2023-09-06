# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from megatron.core.transformer.attention import CrossAttention
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

# >>>
from lutil import pax
# <<<


###########################################################################
# decoder
###########################################################################


# class RetroDecoderWithRetrieverCrossAttention(CrossAttention):
class RetroDecoderCrossAttention(CrossAttention):

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_params=None,
        rotary_pos_emb=None,
        add_retriever=None,
    ):
        # hidden_states: [sq, b, h]

        attention_output_with_bias = super()(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            key_value_states=key_value_states,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
        )

        pax("attention_output_with_bias")

        assert isinstance(add_retriever, bool), "'add_retriever' must be defined."


# class RetroDecoderWithRetrieverBiasDropoutAdd(MegatronModule):
class RetroDecoderBiasDropoutAdd(MegatronModule):

    def __init__(
        self,
        config: TransformerConfig,
        spec: ModuleSpec,
        # layer_number: int = 1,
        # attn_mask_type=AttnMaskType.padding,
        # **kwargs,
    ):
        super().__init__(config=config)

        pax("spec")


# class RetroDecoderWithRetrieverLayernorm(MegatronModule):
class RetroDecoderLayerNorm(MegatronModule):

    def __init__(
        self,
        config: TransformerConfig,
        spec: ModuleSpec,
    ):
        super().__init__(config=config)

        pax("spec")


###########################################################################
# encoder
###########################################################################


class RetroEncoderCrossAttention(CrossAttention):

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_params=None,
        rotary_pos_emb=None,
        add_retriever=None,
    ):
        # hidden_states: [sq, b, h]

        attention_output_with_bias = super()(
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

        pax("spec")


class RetroEncoderLayerNorm(MegatronModule):

    def __init__(
        self,
        config: TransformerConfig,
        spec: ModuleSpec,
    ):
        super().__init__(config=config)

        pax("spec")


# >>>
# eof
# <<<
