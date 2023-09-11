# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from functools import partial
import numpy as np
import torch
from torch import Tensor
from typing import Callable, Optional, Tuple

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.retro.attn import BaseRetroCrossAttention
from megatron.core.transformer import (
    ModuleSpec,
    TransformerBlockSpec,
    TransformerConfig,
)
from megatron.core.transformer.attention import CrossAttentionSpec
from megatron.core.transformer.custom_layers.transformer_engine import TENorm
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_block import TransformerBlock
# from megatron.core.transformer.transformer_config import TransformerConfig

# >>>
from lutil import pax
# <<<


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
        # encoder: MegatronModule = None,
        encoder_block_spec: TransformerBlockSpec = None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            spec=spec,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            **kwargs,
        )

        # pax("spec", "encoder_block_spec")

        if encoder_block_spec:
            self.encoder = TransformerBlock(
                config=config,
                spec=encoder_block_spec,
                pre_process=True,
                post_process=False,
            )
            # self._encoder_key = 'encoder' # necessary?
            # pax({
            #     "encoder" : self.encoder,
            #     "encoder / layers" : list(self.encoder.layers),
            # })
        else:
            self.encoder = None

    # def forward(
    #     self,
    #     hidden_states,
    #     attention_mask,
    #     key_value_states=None,
    #     inference_params=None,
    #     rotary_pos_emb=None,
    #     # add_retriever=None,
    #     retriever_input=None,
    #     retriever_output=None,
    #     retriever_attn_mask=None,
    # ):
    #     # hidden_states: [sq, b, h]

    #     pax(
    #         "hidden_states",
    #         "attention_mask",
    #         "key_value_states",
    #         "inference_params",
    #         "rotary_pos_emb",
    #         "retriever_input",
    #         "retriever_output",
    #         "retriever_attn_mask",
    #     )

    #     attention_output_with_bias = self.attn( # super()(
    #         hidden_states=hidden_states,
    #         attention_mask=attention_mask,
    #         key_value_states=key_value_states,
    #         # key_value_states=retriever_input,
    #         inference_params=inference_params,
    #         rotary_pos_emb=rotary_pos_emb,
    #     )

    #     pax("attention_output_with_bias")

    #     assert isinstance(add_retriever, bool), "'add_retriever' must be defined."
    # def forward(
    #     self,
    #     context=None,
    #     context_mask=None,
    #     layernorm_input=None,
    #     layernorm_output=None,
    #     inference_params=None,
    #     # rotary_pos_emb=None, # unsupported for retro.
    #     retriever_input=None,
    #     retriever_output=None,
    #     retriever_attn_mask=None,
    # ):
    #     # hidden_states: [sq, b, h]

    #     attention_output_with_bias = self.attn( # super()(
    #         hidden_states=hidden_states,
    #         attention_mask=attention_mask,
    #         key_value_states=key_value_states,
    #         # key_value_states=retriever_input,
    #         inference_params=inference_params,
    #         rotary_pos_emb=rotary_pos_emb,
    #     )
    # def forward(
    #     self,
    #     hidden_states,
    #     context=None,
    #     context_mask=None,
    #     inference_params=None,
    #     # rotary_pos_emb=None, # unsupported for retro.
    #     retriever_output=None,
    # ):
    #     # hidden_states: [sq, b, h]
    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_params=None,
        # rotary_pos_emb=None, # unsupported for retro.
        retriever_output=None,
    ):
        # hidden_states: [sq, b, h]

        # attention_output_with_bias = self.attn(
        #     hidden_states=hidden_states,
        #     attention_mask=attention_mask,
        #     key_value_states=key_value_states,
        #     # key_value_states=retriever_input,
        #     inference_params=inference_params,
        #     rotary_pos_emb=rotary_pos_emb,
        # )

        layernorm_output = hidden_states
        retriever_input = key_value_states
        retriever_attn_mask = attention_mask

        """Cross attention for Retro decoder.

        Notation:
            ns : Sequence length.
            bs : Batch size.
            d  : Hidden size.
            l  : Number of chunks per sample (i.e., seq_length/chunk_length).
            m  : Number of tokens per chunk.
            k  : Number of neighbors.
            r  : Number of retrieved tokens (neighbors + continuation).
        """

        ns, bs, d = layernorm_output.shape
        l = int(np.ceil(ns / self.retro_chunk_length))

        # pax("ns", "bs", "d", "l")

        # Retrieve neighbors.
        if self.encoder:
            first_ns = ns % self.retro_chunk_length
            if first_ns > 0:
                raise Exception("test this case.")
                first_chunk, rest_chunk = \
                    layernorm_output[:first_ns], layernorm_output[first_ns:]
                first_chunk = torch.nn.functional.pad(
                    first_chunk,
                    (0, 0, 0, 0, 0, self.retro_chunk_length - first_ns),
                    'constant',
                    0)
                chunked_output = \
                    torch.cat((first_chunk, rest_chunk), dim=0) # [l * m, bs, d]
            else:
                chunked_output = layernorm_output # [l * m, bs, d]
            chunked_output = chunked_output \
                .reshape(l, self.retro_chunk_length, bs, d) \
                .permute(1, 2, 0, 3) \
                .reshape(self.retro_chunk_length, bs * l, d) \
                .contiguous()

            # Get Encoder Output
            # retriever_output = self.encoder(
            #     hidden_states=retriever_input,
            #     attention_mask=retriever_attn_mask,
            #     retriever_output=chunked_output,
            #     retriever_attn_mask=retriever_attn_mask,
            #     inference_params=inference_params) # [r, k * bs * l , d]
            retriever_output = self.encoder(
                hidden_states=retriever_input,
                attention_mask=retriever_attn_mask,
                context=chunked_output,
                context_mask=None,
                inference_params=inference_params) # [r, k * bs * l , d]
            retriever_output = retriever_output.reshape(
                self.retro_retrieved_length * self.retro_num_neighbors, bs * l, d) # [r * k, bs * l, d]

            # pax("retriever_output")

        # Chunks.
        pad = (ns - 1) % self.retro_chunk_length
        attending_chunks = layernorm_output[pad:]
        padded_chunks = torch.nn.functional.pad(
            attending_chunks,
            (0, 0, 0, 0, 0, self.retro_chunk_length - 1),
            'constant', 0)
        padded_chunked_output = padded_chunks \
            .reshape(l, self.retro_chunk_length, bs, d) \
            .permute(1, 2, 0, 3)
        padded_chunked_output = padded_chunked_output.reshape(
            self.retro_chunk_length, bs * l, d).contiguous()

        # Encoder output.
        attention_output, attention_bias = \
            self.attn(padded_chunked_output,
                      None,
                      key_value_states=retriever_output)

        # # Residual connection.
        # if self.apply_residual_connection_post_layernorm:
        #     residual = layernorm_output
        # else:
        #     residual = layernorm_input

        # pax("attention_output", "attention_bias", "retriever_output")

        # return attention_output, attention_bias, retriever_output
        return {
            "ns" : ns,
            "bs" : bs,
            "d" : d,
            "l" : l,
            "pad" : pad,
            "attention_output" : attention_output,
            "attention_bias" : attention_bias,
            "retriever_output" : retriever_output,
        }


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
        self.retro_chunk_length = config.retro_preprocess.retro_gpt_chunk_length
        # pax("config", "spec")

    @classmethod
    def _forward(
        cls,
        # x_with_bias: Tuple[Tensor, Optional[Tensor]],
        x_with_bias: dict,
        residual: Tensor,
        prob: float,
        retro_chunk_length: int,
        bias_dropout_add: Callable,
    ) -> Tensor:

        # pax("x_with_bias")

        # attention_output, attention_bias = x_with_bias

        ns = x_with_bias["ns"]
        bs = x_with_bias["bs"]
        d = x_with_bias["d"]
        l = x_with_bias["l"]
        pad = x_with_bias["pad"]
        attention_output = x_with_bias["attention_output"]
        attention_bias = x_with_bias["attention_bias"]

        # pax("attention_output", "attention_bias")

        # Re-enable torch grad to enable fused optimization.
        with torch.enable_grad():
            x = bias_dropout_add(
                (attention_output,
                 None if attention_bias is None else attention_bias.expand_as(attention_output)),
                torch.zeros_like(attention_output),
                prob)
            # pax("retro_chunk_length", "x")
            x = x \
                .reshape(retro_chunk_length, bs, l, d) \
                .permute(2, 0, 1, 3) # [l, m, bs, d]
            x = x.reshape(retro_chunk_length * l, bs, d)
            x = torch.nn.functional.pad(
                x,
                (0, 0, 0, 0, pad, 0),
                'constant', 0)[:ns] # [ns, b, d]
            x = x + residual

        # pax("x")

        return x

    def forward(self, training, fused):
        return partial(
            self._forward,
            retro_chunk_length=self.retro_chunk_length,
            bias_dropout_add=get_bias_dropout_add(training, fused),
        )


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

    def forward(self, x):
        return self.norm(x)
