# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from functools import partial
import torch
from torch import Tensor
from typing import Callable, Optional, Tuple

from megatron.core import InferenceParams
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.retro.attn import BaseRetroCrossAttention
# from megatron.core.transformer.attention import CrossAttention, CrossAttentionSpec
from megatron.core.transformer.custom_layers.transformer_engine import TENorm
# from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

# >>>
from lutil import pax
# <<<


class RetroEncoderCrossAttention(BaseRetroCrossAttention):

    # def forward(
    #     self,
    #     hidden_states: Tensor,
    #     attention_mask: Tensor,
    #     key_value_states: Tensor = None,
    #     inference_params: InferenceParams = None,
    #     rotary_pos_emb: Tensor = None,
    #     retriever_input: Tensor = None,
    #     retriever_output: Tensor = None,
    #     retriever_attn_mask: Tensor = None,
    # ):
    #     # hidden_states: [sq, b, h]

    #     attention_output_with_bias = self.attn( # super()(
    #         hidden_states=hidden_states,
    #         attention_mask=attention_mask,
    #         key_value_states=key_value_states,
    #         inference_params=inference_params,
    #         rotary_pos_emb=rotary_pos_emb,
    #     )

    #     pax("attention_output_with_bias")

    #     assert isinstance(add_retriever, bool), "'add_retriever' must be defined."
    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_params=None,
        # rotary_pos_emb=None, # unsupported for retro.
        # retriever_output=None,
        **kwargs,
    ):
        # hidden_states: [sq, b, h]

        layernorm_output = hidden_states
        retriever_output = key_value_states

        """Cross attention for Retro encoder.

        Notation:
            ns : Sequence length.
            bs : Batch size.
            d  : Hidden size.
            l  : Number of chunks per sample (i.e., seq_length/chunk_length).
            k  : Number of neighbors.
            r  : Number of retrieved tokens (neighbors + continuation).
        """

        ns, bs, d = layernorm_output.shape # [r, bs * l * k, d]

        # pax("ns", "bs", "d")

        # Divide sequence dimension into chunks.
        chunked_outputs = layernorm_output.reshape(self.retro_retrieved_length,
                                                   -1,
                                                   self.retro_num_neighbors,
                                                   d)
        # chunked_outputs_before_layer_norm = \
        #     layernorm_input.reshape(self.retro_retrieved_length, -1,
        #                             self.retro_num_neighbors, d) # [r, bs*l, k, d]

        # Per-chunk attention.
        attention_output_tuples = []
        for k in range(self.retro_num_neighbors):

            # Attention.
            chunked_output = chunked_outputs[:,:,k].contiguous()
            attention_output, attention_bias = self.attn(
                hidden_states=chunked_output, # Q (neighbor embedding)
                attention_mask=None,
                key_value_states=retriever_output) # K, V (hidden act)

            # Residual connection.
            # if self.apply_residual_connection_post_layernorm:
            residual = chunked_output
            # else:
            #     residual = chunked_outputs_before_layer_norm[:,:,k]

            attention_output_tuples.append((attention_output,
                                            attention_bias,
                                            residual))

        # pax("attention_output_tuples")

        return attention_output_tuples


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

        # layernorm_inputs = []
        # layernorm_outputs = []
        # outputs = []
        # for k in range(retro_num_neighbors):

        #     # Re-enable torch grad to enable fused optimization.
        #     with torch.enable_grad():
        #         output = bias_dropout_add_func(
        #             attention_output,
        #             None if attention_bias is None else attention_bias.expand_as(residual),
        #             residual,
        #             self.hidden_dropout)
        #         outputs.append(output)

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

        # pax("x_with_bias", "outputs")

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

    def forward(self, layernorm_inputs):

        layernorm_outputs = [ self.norm(inp) for inp in layernorm_inputs ]

        # Concatenate layer norms.
        # layernorm_input : [r, k * bs * l, d]
        # layernorm_output : [r, k * bs * l, d]
        ns, _, d = layernorm_inputs[0].shape
        # layernorm_input = \
        #     torch.stack(layernorm_inputs, dim=1).reshape(ns, -1, d)
        layernorm_output = \
            torch.stack(layernorm_outputs, dim=1).reshape(ns, -1, d)

        # pax(
        #     "layernorm_inputs",
        #     "layernorm_outputs",
        #     # "layernorm_input",
        #     "layernorm_output",
        # )

        # return layernorm_input, layernorm_output
        return layernorm_output

