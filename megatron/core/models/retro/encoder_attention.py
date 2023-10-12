# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Retro's cross attention modules for the encoder block."""

from functools import partial
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

from megatron.core import InferenceParams
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.retro.base_attention import BaseRetroCrossAttention
from megatron.core.models.retro.config import RetroConfig
from megatron.core.transformer.custom_layers.transformer_engine import TENorm
from megatron.core.transformer.module import MegatronModule


class RetroEncoderCrossAttention(BaseRetroCrossAttention):

    """Retro encoder's cross attention operator.

    See this paper for more details: https://arxiv.org/abs/2112.04426.

    Neighboring chunks are retrieved from the chunk database, encoded, and
    used by the decoder layers for chunked cross attention.
    """

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states: Tensor = None,
        inference_params: InferenceParams = None,
        # rotary_pos_emb: Tensor = None, # unsupported for retro.
    ) -> Tensor:
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

        ns, bs, d = hidden_states.shape  # [r, bs * l * k, d]

        # Divide sequence dimension into chunks.
        chunked_outputs = hidden_states.reshape(
            self.retro_retrieved_length, -1, self.retro_num_neighbors, d
        )

        # Per-chunk attention.
        attention_output_tuples = []
        for k in range(self.retro_num_neighbors):

            # Attention.
            chunked_output = chunked_outputs[:, :, k].contiguous()
            attention_output, attention_bias = self.attn(
                hidden_states=chunked_output,  # Q (neighbor embedding)
                attention_mask=None,
                key_value_states=key_value_states,
            )  # K, V (hidden act)

            # Residual connection.
            residual = chunked_output

            attention_output_tuples.append((attention_output, attention_bias, residual))

        return attention_output_tuples


class RetroEncoderBiasDropoutAdd(MegatronModule):

    """Retro encoder's bias-dropout-add operator.

    This operator applies bias-dropout-add individually on each neighboring
    chunk that is retrieved from the chunk database.
    """

    def __init__(
        self, config: RetroConfig,
    ):
        super().__init__(config=config)
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
                    (
                        attention_output,
                        None if attention_bias is None else attention_bias.expand_as(residual),
                    ),
                    residual,
                    prob,
                )
                for attention_output, attention_bias, residual in x_with_bias
            ]

        # Concatenate outputs (to shape [r, k*bs*l, d]; see notation above).
        ns, _, d = outputs[0].shape
        output = torch.stack(outputs, dim=1).reshape(ns, -1, d)

        return output

    def forward(self, training: bool, fused: bool) -> Tensor:
        return partial(
            self._forward,
            retro_num_neighbors=self.retro_num_neighbors,
            bias_dropout_add=get_bias_dropout_add(training, fused),
        )


class RetroEncoderLayerNorm(MegatronModule):

    """Retro encoder's layernorm operator.

    This operator applies layernorm individually on each neighboring chunk that
    is retrieved from the chunk database, and then concatenates the chunks into
    a single tensor.
    """

    def __init__(
        self, config: RetroConfig, **kwargs,
    ):
        super().__init__(config=config)
        self.norm = TENorm(config=config, **kwargs)
        self.retro_num_neighbors = config.retro_num_neighbors

    def forward(self, input: Tensor) -> Tensor:

        # Split input into 'num_neighbors' tensors.
        chunk_size = input.shape[1] // self.retro_num_neighbors
        inputs = torch.split(input, chunk_size, dim=1)

        # Norm.
        outputs = [self.norm(inp.contiguous()) for inp in inputs]

        # Concatenate layer norms (to shape [r, k*bs*l, d]; see notation above).
        ns, _, d = inputs[0].shape
        output = torch.stack(outputs, dim=1).reshape(ns, -1, d)

        return output
