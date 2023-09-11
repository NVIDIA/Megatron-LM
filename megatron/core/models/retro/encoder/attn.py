# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from functools import partial
import torch
from torch import Tensor
from typing import Callable, Optional, Tuple

from megatron.core import InferenceParams
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.retro.attn import BaseRetroCrossAttention
from megatron.core.transformer.custom_layers.transformer_engine import TENorm
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig


class RetroEncoderCrossAttention(BaseRetroCrossAttention):

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

        """Cross attention for Retro encoder.

        Notation:
            ns : Sequence length.
            bs : Batch size.
            d  : Hidden size.
            l  : Number of chunks per sample (i.e., seq_length/chunk_length).
            k  : Number of neighbors.
            r  : Number of retrieved tokens (neighbors + continuation).
        """

        ns, bs, d = hidden_states.shape # [r, bs * l * k, d]

        # Divide sequence dimension into chunks.
        chunked_outputs = hidden_states.reshape(self.retro_retrieved_length,
                                                -1,
                                                self.retro_num_neighbors,
                                                d)

        # Per-chunk attention.
        attention_output_tuples = []
        for k in range(self.retro_num_neighbors):

            # Attention.
            chunked_output = chunked_outputs[:,:,k].contiguous()
            attention_output, attention_bias = self.attn(
                hidden_states=chunked_output, # Q (neighbor embedding)
                attention_mask=None,
                key_value_states=key_value_states) # K, V (hidden act)

            # Residual connection.
            residual = chunked_output

            attention_output_tuples.append((attention_output,
                                            attention_bias,
                                            residual))

        return attention_output_tuples


class RetroEncoderBiasDropoutAdd(MegatronModule):

    def __init__(
        self,
        config: TransformerConfig,
        spec: ModuleSpec,
        **kwargs,
    ):
        super().__init__(config=config)
        self.spec = spec
        self.retro_num_neighbors = config.retro_num_neighbors

    @classmethod
    def _forward(
        cls,
        x_with_bias: Tuple[Tensor, Optional[Tensor]],
        residual: Tensor,
        prob: float,
        retro_num_neighbors: int,
        bias_dropout_add: Callable,
    ) -> Tensor:

        # Re-enable torch grad to enable fused optimization.
        with torch.enable_grad():
            outputs = [
                bias_dropout_add(
                    (attention_output,
                     None if attention_bias is None else attention_bias.expand_as(residual)),
                    residual,
                    prob,
                )
                for attention_output, attention_bias, residual in x_with_bias
            ]

        return outputs

    def forward(self, training, fused):
        return partial(
            self._forward,
            retro_num_neighbors=self.retro_num_neighbors,
            bias_dropout_add=get_bias_dropout_add(training, fused),
        )


class RetroEncoderLayerNorm(MegatronModule):

    def __init__(
        self,
        config: TransformerConfig,
        spec: ModuleSpec,
        **kwargs,
    ):
        super().__init__(config=config)
        self.spec = spec
        self.norm = TENorm(config=config, **kwargs)

    def forward(self, layernorm_inputs):

        layernorm_outputs = [ self.norm(inp) for inp in layernorm_inputs ]

        # Concatenate layer norms (to shape [r, k*bs*l, d]; see notation above).
        ns, _, d = layernorm_inputs[0].shape
        layernorm_output = torch.stack(layernorm_outputs, dim=1).reshape(ns,-1,d)

        return layernorm_output

