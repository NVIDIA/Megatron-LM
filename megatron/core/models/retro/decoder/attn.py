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


class RetroDecoderCrossAttention(BaseRetroCrossAttention):

    def __init__(
        self,
        config: TransformerConfig,
        spec: CrossAttentionSpec,
        layer_number: int = 1,
        attn_mask_type: AttnMaskType = AttnMaskType.padding,
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
        retriever_output=None,
    ):
        # hidden_states: [sq, b, h]

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
            retriever_output = self.encoder(
                hidden_states=retriever_input,
                attention_mask=retriever_attn_mask,
                context=chunked_output,
                context_mask=None,
                inference_params=inference_params) # [r, k * bs * l , d]
            retriever_output = retriever_output.reshape(
                self.retro_retrieved_length * self.retro_num_neighbors, bs * l, d) # [r * k, bs * l, d]

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

        # Return dimensions for bias-dropout step.
        return {
            "ns" : ns,
            "bs" : bs,
            "d" : d,
            "l" : l,
            "pad" : pad,
            "attention_output" : attention_output,
            "attention_bias" : attention_bias,
            # "retriever_output" : retriever_output,
            "context" : retriever_output,
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
