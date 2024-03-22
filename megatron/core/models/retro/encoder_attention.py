# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""Retro's cross attention modules for the encoder block."""

from functools import partial
from typing import Callable, List, Optional, Tuple, Type

import torch
from torch import Tensor

from megatron.core import InferenceParams
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.retro.base_attention import BaseRetroCrossAttention
from megatron.core.models.retro.config import RetroConfig
from megatron.core.models.retro.utils import get_all_true_mask
from megatron.core.transformer.module import MegatronModule


class RetroEncoderCrossAttention(BaseRetroCrossAttention):

    """Retro encoder's cross attention operator.

    See this paper for more details: https://arxiv.org/abs/2112.04426.
    Neighboring chunks are retrieved from the chunk database, encoded, and
    used by the decoder layers for chunked cross attention.

    Args:
        config (RetroConfig): Retro config.
        submodules (CrossAttentionSubmodules): Cross attention submodules.
        layer_number (int): Layer number within transformer block.
        attn_mask_type (AttnMaskType): Mask type ('causal' or 'padding').
    """

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states: Tensor = None,
        inference_params: InferenceParams = None,
        # rotary_pos_emb: Tensor = None, # unsupported for retro.
    ) -> List[Tuple[Tensor, Optional[Tensor], Tensor]]:
        """Cross attention for Retro encoder.

        Notation:
            ns : Sequence length.
            bs : Batch size.
            d  : Hidden size.
            l  : Number of chunks per sample (i.e., seq_length/chunk_length).
            k  : Number of neighbors.
            r  : Number of retrieved tokens (neighbors + continuation).

        Args:
            hidden_states (Tensor): Transformer layer hidden states.
            attention_mask (Tensor): Attention mask.
            key_value_states (Tensor): Neighbor embeddings.
            inference_params (InferenceParams): Inference params.

        Returns:
            List of tuples, where each tuple is (attention_output, attention_bias, residual).
        """

        # Input shape. [ r, bs*l*k, d ]
        ns, bs, d = hidden_states.shape

        # Reshape sequence into neighboring chunks.
        # - hidden_states:   [ r, bs*l*k, d ]
        # - chunked_outputs: [ r, bs*l, k, d ]
        chunked_outputs = hidden_states.reshape(
            self.retro_retrieved_length, -1, self.retro_num_neighbors, d
        )

        # flash attn: [ b, h, sq, sk ]
        # fused attn: [ b, 1, 1, sq ]
        chunked_output_mask = get_all_true_mask(
            size=(1, 1, chunked_outputs.shape[0], key_value_states.shape[0]),
            device=chunked_outputs.device,
        )

        # Per-chunk attention.
        attention_output_tuples = []
        for k in range(self.retro_num_neighbors):

            # Attend to current neighboring chunks.
            # - chunked_output:   [ r, bs*l, d ]
            # - key_value_states: [ m, bs*l, d ]
            # - attention_output: [ r, bs*l, d ]
            # - attention_bias:   [ d ]
            chunked_output = chunked_outputs[:, :, k].contiguous()
            attention_output, attention_bias = self.attn(
                hidden_states=chunked_output,  # Q (neighbor embedding)
                attention_mask=chunked_output_mask,
                key_value_states=key_value_states,  # K, V (hidden act)
            )

            # Residual connection. [ r, bs*l, d ]
            residual = chunked_output

            # Collect tensors.
            attention_output_tuples.append((attention_output, attention_bias, residual,))

        # Output. (List[Tuple[( [ r, bs*l, d ], [ d ] )]])
        return attention_output_tuples


class RetroEncoderBiasDropoutAdd(MegatronModule):

    """Retro encoder's bias-dropout-add operator.

    This operator applies bias-dropout-add individually on each neighboring
    chunk that is retrieved from the chunk database.

    Args:
        config (RetroConfig): Retro config.
    """

    def __init__(
        self, config: RetroConfig,
    ):
        super().__init__(config=config)
        self.retro_num_neighbors = config.retro_num_neighbors

    @classmethod
    def _forward(
        cls,
        x_with_bias: List[Tuple[Tensor, Optional[Tensor], Tensor]],
        residual: Tensor,
        prob: float,
        retro_num_neighbors: int,
        bias_dropout_add: Callable,
    ) -> Tensor:
        """Per-chunk bias-dropout-add.

        Args:
            x_with_bias (dict): Attention output and bias tuple.
            residual (Tensor): Transformer layer residual.
            prob (float): Dropout probability.
            retro_num_neighbors (int): Number of retrieved neighbor chunks (e.g., 2).
            bias_dropout_add (Callable): Bias-dropout-add function.

        Returns:
            Output of bias-dropout-add.
        """

        # Re-enable torch grad to enable fused optimization.
        with torch.enable_grad():

            # Per-neighbor bias-dropout-add.
            # - attention_output: [ r, bs*l, d ]
            # - attention_bias:   [ d ]
            # - residual:         [ r, bs*l, d ]
            # - output:           [ r, bs*l, d ]
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
        r, _, d = outputs[0].shape
        output = torch.stack(outputs, dim=1).reshape(r, -1, d)

        # Output. [ r, k*bs*l, d ]
        return output

    def forward(self, training: bool, fused: bool) -> partial:
        """Retro decoder bias-dropout-add.

        Args:
            training (bool): If training, then apply dropout.
            fused (bool): Fuse bias-dropout-add.

        Returns:
            A partial function for performing bias-dropout-add.
        """
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

    Args:
        config (RetroConfig): Retro config.
        submodules (Type): Layer norm class. (Named 'submodules' to fit external interface.)
    """

    def __init__(
        self, config: RetroConfig, submodules: Type, **kwargs: dict,
    ):
        super().__init__(config=config)
        norm_class = submodules
        self.norm = norm_class(config=config, **kwargs)
        self.retro_num_neighbors = config.retro_num_neighbors

    def forward(self, input: Tensor) -> Tensor:
        """Per-chunk layer norm.

        Args:
            input (Tensor): Input chunks, concatenated into a single tensor.
        
        Returns:
            Output of the layer norm.
        """

        # Input shape: [ r, k*bs*l, d ]. (see notation above in attention module)

        # Split input into 'num_neighbors' tensors.
        chunk_size = input.shape[1] // self.retro_num_neighbors
        inputs = torch.split(input, chunk_size, dim=1)

        # Norm.
        outputs = [self.norm(inp.contiguous()) for inp in inputs]

        # Concatenate layer norms (to shape [r, k*bs*l, d]; see notation above).
        r, _, d = inputs[0].shape
        output = torch.stack(outputs, dim=1).reshape(r, -1, d)

        # Output. [ r, k*bs*l, d ]
        return output
