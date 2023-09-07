# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from megatron.core.transformer.attention import CrossAttention, CrossAttentionSpec
from megatron.core.transformer.custom_layers.transformer_engine import TENorm
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

# >>>
from lutil import pax
# <<<


class BaseRetroCrossAttention(MegatronModule):

    def __init__(
        self,
        config: TransformerConfig,
        spec: CrossAttentionSpec,
        layer_number: int = 1,
        attn_mask_type: AttnMaskType = AttnMaskType.padding,
        **kwargs,
    ):
        super().__init__(config=config)

        self.attn = CrossAttention(
            config=config,
            spec=spec,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            **kwargs,
        )

        self.retro_num_neighbors = config.retro_num_neighbors
        self.retro_chunk_length = config.retro_preprocess.retro_gpt_chunk_length
        self.retro_retrieved_length = config.retro_preprocess.retro_gpt_retrieved_length


###########################################################################
# decoder
###########################################################################


# class RetroDecoderWithRetrieverCrossAttention(CrossAttention):
# class RetroDecoderCrossAttention(CrossAttention):
# class RetroDecoderCrossAttention(MegatronModule):
class RetroDecoderCrossAttention(BaseRetroCrossAttention):

    # def __init__(
    #         self,
    #         config: TransformerConfig,
    #         spec: CrossAttentionSpec,
    #         layer_number: int,
    #         attn_mask_type: AttnMaskType,
    #         add_retriever: bool,
    #         **kwargs,
    # ):
    #     pax("spec")

    def __init__(
        self,
        config: TransformerConfig,
        spec: CrossAttentionSpec,
        layer_number: int = 1,
        attn_mask_type: AttnMaskType = AttnMaskType.padding,
        # add_retriever: bool = False,
        encoder: MegatronModule = None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            spec=spec,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            **kwargs,
        )

        self.encoder = encoder
        # self._encoder_key = 'encoder' # necessary?

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
        self.spec = spec
        # pax("config", "spec")


# class RetroDecoderWithRetrieverLayernorm(MegatronModule):
class RetroDecoderLayerNorm(MegatronModule):

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


###########################################################################
# encoder
###########################################################################


# class RetroEncoderCrossAttention(CrossAttention):
class RetroEncoderCrossAttention(BaseRetroCrossAttention):

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


# >>>
# eof
# <<<
