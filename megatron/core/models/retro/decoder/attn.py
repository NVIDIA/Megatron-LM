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

# >>>
from lutil import pax, tp
# <<<


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# from megatron.core.transformer.attention import CrossAttention
# class RetroDecoderCrossAttention_naive(CrossAttention):

#     def __init__(
#         self,
#         config: TransformerConfig,
#         spec: CrossAttentionSpec,
#         layer_number: int = 1,
#         attn_mask_type: AttnMaskType = AttnMaskType.padding,
#         **kwargs,
#     ):

#         super().__init__(
#             config=config,
#             spec=spec,
#             layer_number=layer_number,
#             # attn_mask_type=attn_mask_type,
#             # **kwargs,
#         )

#         # >>>
#         # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#         # print(self)
#         # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#         # # pax("config", "spec", "kwargs")
#         # pax("attn_mask_type")
#         # exit()
#         # <<<

#         self.norm = TENorm(
#             config=config,
#             # spec=spec,
#             hidden_size=self.config.hidden_size,
#             eps=self.config.layernorm_epsilon,
#             persist_layer_norm=self.config.persist_layer_norm,
#             sequence_parallel=self.config.sequence_parallel,
#             zero_centered_gamma=self.config.layernorm_zero_centered_gamma,
#             normalization=self.config.normalization,
#         )

#     def forward(
#         self,
#         hidden_states,
#         attention_mask,
#         key_value_states=None,
#         inference_params=None,
#         # rotary_pos_emb=None, # unsupported for retro.
#         # retriever_output=None, # set as key_value_states
#         **kwargs,
#     ):

#         # >>>
#         # return hidden_states
#         # return self.norm(hidden_states)
#         # <<<

#         # Encoder output.
#         # attention_output, attention_bias = \
#         attention_output_with_bias = \
#             super().forward(hidden_states=hidden_states,
#                             attention_mask=attention_mask, # None,
#                             key_value_states=key_value_states)

#         # # Re-enable torch grad to enable fused optimization.
#         bias_dropout_add_func = get_bias_dropout_add(
#             self.training,
#             self.config.bias_dropout_fusion)
#         # # with torch.enable_grad():
#         # layernorm_input = bias_dropout_add_func(
#         #     (attention_output,
#         #      None if attention_bias is None else attention_bias.expand_as(attention_output)),
#         #     torch.zeros_like(attention_output),
#         #     self.config.hidden_dropout)
#         # TODO: could we move `bias_dropout_add_exec_handler` itself
#         # inside the module provided in the `bias_dropout_add_spec` module?
#         # with self.bias_dropout_add_exec_handler():
#         residual = hidden_states
#         with torch.enable_grad():
#             layernorm_input = bias_dropout_add_func(
#                 attention_output_with_bias, residual, self.config.hidden_dropout
#             )

#         # Layer norm post the decoder attention
#         layernorm_output = self.norm(layernorm_input)

#         return layernorm_output


class RetroDecoderCrossAttention_naive(BaseRetroCrossAttention):

    def __init__(
        self,
        config: TransformerConfig,
        spec: CrossAttentionSpec,
        layer_number: int = 1,
        attn_mask_type: AttnMaskType = AttnMaskType.padding,
        **kwargs,
    ):

        super().__init__(
            config=config,
            spec=spec,
            layer_number=layer_number,
            # attn_mask_type=attn_mask_type,
            # **kwargs,
        )

        self.norm = TENorm(
            config=config,
            # spec=spec,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
            persist_layer_norm=self.config.persist_layer_norm,
            sequence_parallel=self.config.sequence_parallel,
            zero_centered_gamma=self.config.layernorm_zero_centered_gamma,
            normalization=self.config.normalization,
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_params=None,
        # rotary_pos_emb=None, # unsupported for retro.
        # retriever_output=None, # set as key_value_states
        **kwargs,
    ):
        # hidden_states: [sq, b, h]

        layernorm_output = hidden_states
        retriever_output = key_value_states

        # >>>
        # pax("retriever_output", "layernorm_output")
        # <<<

        ns, bs, d = layernorm_output.shape
        l = int(np.ceil(ns / self.retro_chunk_length))

        # Retrieve neighbors.
        # if self.layer_type == LayerType.retro_decoder_with_retriever:
        #     first_ns = ns % self.retro_chunk_length
        #     if first_ns > 0:
        #         raise Exception("test this case.")
        #         first_chunk, rest_chunk = \
        #             layernorm_output[:first_ns], layernorm_output[first_ns:]
        #         first_chunk = torch.nn.functional.pad(
        #             first_chunk,
        #             (0, 0, 0, 0, 0, self.retro_chunk_length - first_ns),
        #             'constant',
        #             0)
        #         chunked_output = \
        #             torch.cat((first_chunk, rest_chunk), dim=0) # [l * m, bs, d]
        #     else:
        #         chunked_output = layernorm_output # [l * m, bs, d]
        #     chunked_output = chunked_output \
        #         .reshape(l, self.retro_chunk_length, bs, d) \
        #         .permute(1, 2, 0, 3) \
        #         .reshape(self.retro_chunk_length, bs * l, d) \
        #         .contiguous()

        #     # Get Encoder Output
        #     # >>>
        #     # pax("layernorm_output")
        #     # pax("retriever_input", "retriever_attn_mask", "chunked_output")
        #     # <<<

        #     retriever_output = self.retriever(
        #         hidden_states=retriever_input,
        #         attention_mask=retriever_attn_mask,
        #         retriever_output=chunked_output,
        #         retriever_attn_mask=retriever_attn_mask,
        #         inference_params=inference_params) # [r, k * bs * l , d]
        #     retriever_output = retriever_output.reshape(
        #         self.retro_retrieved_length * self.retro_num_neighbors, bs * l, d) # [r * k, bs * l, d]

        #     # >>>
        #     # pax("retriever_output")
        #     # <<<

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
            self.attn(hidden_states=padded_chunked_output,
                      attention_mask=None,
                      key_value_states=retriever_output)

        # >>>
        # pax("attention_output", "attention_bias", "retriever_output")
        # <<<

        # Residual connection.
        # if self.apply_residual_connection_post_layernorm:
        residual = layernorm_output
        # else:
        #     residual = layernorm_input

        # Re-enable torch grad to enable fused optimization.
        bias_dropout_add_func = get_bias_dropout_add(
            self.training,
            self.config.bias_dropout_fusion)
        with torch.enable_grad():
            layernorm_input = bias_dropout_add_func(
                (attention_output,
                 None if attention_bias is None else attention_bias.expand_as(attention_output)),
                torch.zeros_like(attention_output),
                self.config.hidden_dropout)
            layernorm_input = layernorm_input \
                .reshape(self.retro_chunk_length, bs, l, d) \
                .permute(2, 0, 1, 3) # [l, m, bs, d]
            layernorm_input = layernorm_input.reshape(self.retro_chunk_length * l, bs, d)
            layernorm_input = torch.nn.functional.pad(
                layernorm_input,
                (0, 0, 0, 0, pad, 0),
                'constant', 0)[:ns] # [ns, b, d]
            layernorm_input = layernorm_input + residual

        # Layer norm post the decoder attention
        layernorm_output = self.norm(layernorm_input)

        # >>>
        # pax("retriever_output", "layernorm_output")
        # pax("layernorm_output")
        # <<<

        # return retriever_output, layernorm_input, layernorm_output
        return layernorm_output
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


class RetroDecoderCrossAttention(BaseRetroCrossAttention):

    def __init__(
        self,
        config: TransformerConfig,
        spec: CrossAttentionSpec,
        layer_number: int = 1,
        # attn_mask_type: AttnMaskType = AttnMaskType.padding,
        attn_mask_type: AttnMaskType = AttnMaskType.causal,
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

        # >>>
        # pax({"attn_mask_type": attn_mask_type})
        # <<<

        if encoder_block_spec:
            self.encoder = TransformerBlock(
                config=config,
                spec=encoder_block_spec,
                pre_process=True,
                post_process=False,
            )
            # self._encoder_key = 'encoder' # ... necessary?
        else:
            self.encoder = None

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_params=None,
        # rotary_pos_emb=None, # ... unsupported for retro.
        # retriever_output=None,
    ):
        # hidden_states: [sq, b, h]

        # >>>
        # pax("hidden_states", "key_value_states", {"attn_mask_type": self.attn_mask_type})
        # <<<

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

        ns, bs, d = hidden_states.shape
        l = int(np.ceil(ns / self.retro_chunk_length))

        # Retrieve neighbors.
        if self.encoder:
            first_ns = ns % self.retro_chunk_length
            if first_ns > 0:
                raise Exception("test this case.")
                first_chunk, rest_chunk = \
                    hidden_states[:first_ns], hidden_states[first_ns:]
                first_chunk = torch.nn.functional.pad(
                    first_chunk,
                    (0, 0, 0, 0, 0, self.retro_chunk_length - first_ns),
                    'constant',
                    0)
                chunked_output = \
                    torch.cat((first_chunk, rest_chunk), dim=0) # [l * m, bs, d]
            else:
                chunked_output = hidden_states # [l * m, bs, d]
            chunked_output = chunked_output \
                .reshape(l, self.retro_chunk_length, bs, d) \
                .permute(1, 2, 0, 3) \
                .reshape(self.retro_chunk_length, bs * l, d) \
                .contiguous()

            # Get Encoder Output
            # >>>
            pax("hidden_states")
            pax("key_value_states", "attention_mask", "chunked_output")
            # <<<

            key_value_states = self.encoder(
                hidden_states=key_value_states,
                attention_mask=attention_mask,
                context=chunked_output,
                context_mask=None,
                inference_params=inference_params) # [r, k * bs * l , d]
            key_value_states = key_value_states.reshape(
                self.retro_retrieved_length * self.retro_num_neighbors, bs * l, d) # [r * k, bs * l, d]

            # >>>
            pax("key_value_states")
            # <<<

        # Chunks.
        pad = (ns - 1) % self.retro_chunk_length
        attending_chunks = hidden_states[pad:]
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
                      key_value_states=key_value_states)

        # >>>
        # pax("attention_output", "attention_bias", "key_value_states")
        # <<<

        # Return dimensions for bias-dropout step.
        return {
            "ns" : ns,
            "bs" : bs,
            "d" : d,
            "l" : l,
            "pad" : pad,
            "attention_output" : attention_output,
            "attention_bias" : attention_bias,
            "context" : key_value_states,
        }


class RetroDecoderBiasDropoutAdd(MegatronModule):

    def __init__(
        self,
        config: TransformerConfig,
        spec: ModuleSpec,
        **kwargs,
    ):
        super().__init__(config=config)
        self.spec = spec
        self.retro_chunk_length = config.retro_preprocess.retro_gpt_chunk_length

    @classmethod
    def _forward(
        cls,
        x_with_bias: dict,
        residual: Tensor,
        prob: float,
        retro_chunk_length: int,
        bias_dropout_add: Callable,
    ) -> Tensor:

        ns = x_with_bias["ns"]
        bs = x_with_bias["bs"]
        d = x_with_bias["d"]
        l = x_with_bias["l"]
        pad = x_with_bias["pad"]
        attention_output = x_with_bias["attention_output"]
        attention_bias = x_with_bias["attention_bias"]

        # Re-enable torch grad to enable fused optimization.
        with torch.enable_grad():
            x = bias_dropout_add(
                (attention_output,
                 None if attention_bias is None else attention_bias.expand_as(attention_output)),
                torch.zeros_like(attention_output),
                prob)
            x = x \
                .reshape(retro_chunk_length, bs, l, d) \
                .permute(2, 0, 1, 3) # [l, m, bs, d]
            x = x.reshape(retro_chunk_length * l, bs, d)
            x = torch.nn.functional.pad(
                x,
                (0, 0, 0, 0, pad, 0),
                'constant', 0)[:ns] # [ns, b, d]
            x = x + residual

        return x

    def forward(self, training, fused):
        return partial(
            self._forward,
            retro_chunk_length=self.retro_chunk_length,
            bias_dropout_add=get_bias_dropout_add(training, fused),
        )


class RetroDecoderLayerNorm(MegatronModule):

    def __init__(
        self,
        config: TransformerConfig,
        spec: ModuleSpec,
        **kwargs,
    ):
        super().__init__(config=config)
        self.spec = spec
        self.norm = TENorm(config=config, **kwargs)

    def forward(self, x):
        return self.norm(x)
