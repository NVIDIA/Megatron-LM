# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import logging
import math
from typing import Callable, Optional

import torch
from torch import Tensor, nn

from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import deprecate_inference_params

logger = logging.getLogger(__name__)


__all__ = ['RelativePositionEmbedding']


class RelativePositionEmbedding(nn.Module):
    """Relative Position Embedding for language model.

    Args:

    """

    def __init__(
        self,
        bidirectional: bool,
        init_method: Callable,
        num_attention_heads: int,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
    ) -> None:
        super().__init__()

        self.bidirectional = bidirectional
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.relative_attention_bias = torch.nn.Embedding(
            self.relative_attention_num_buckets, num_attention_heads
        )
        init_method(self.relative_attention_bias.weight)

    def _relative_position_bucket(
        self, relative_position, bidirectional=True, num_buckets=32, max_distance=128
    ):
        """
        Adapted from HuggingFace T5 Model:
        https://github.com/huggingface/transformers/blob/329f5dbf97a5cb2473914c88c05aa3dcb242e19a/
        src/transformers/models/t5/modeling_t5.py#L397

        Translate relative position to a bucket number for relative attention.
        The relative position is defined as memory_position - query_position, i.e. the
        distance in tokens from the attending position to the attended-to position.
        If bidirectional=False, then positive relative positions are invalid. We use
        smaller buckets for small absolute relative_position and larger buckets for
        larger absolute relative_positions. All relative positions >=max_distance map
        to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the
        model has been trained on.

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position,
            containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger
        # bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def _compute_bias(self, query_length, key_length):
        """
        Adapted from HuggingFace T5 Model
        https://github.com/huggingface/transformers/blob/329f5dbf97a5cb2473914c88c05aa3dcb242e19a/
        src/transformers/models/t5/modeling_t5.py#L444C9-L444C21

        Compute binned relative position bias

        Args:
            query_length (int): The length of the query sequence
            (e.g., the input sequence in attention).
            key_length (int): The length of the key sequence
            (e.g., the sequence to compare against in attention).

        Returns:
            torch.Tensor: A tensor representing the relative position bias, with shape
            (1, num_heads, query_length, key_length).
        """
        device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]

        relative_position = memory_position - context_position  # shape(query_length,key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=self.bidirectional,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(
            relative_position_bucket
        )  # shape(query_length,key_length,num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(
            0
        )  # shape(1, num_heads,query_length,key_length)
        return values

    @staticmethod
    def get_relative_seq_len(
        inference_context: BaseInferenceContext,
        transformer: TransformerBlock,
        transformer_input: Tensor,
        transformer_config: TransformerConfig,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ) -> float:
        """Function to get the rotary sequence length.

        Args:
            inference_context (BaseInferenceContext): Used during Inference time
            transformer (TransformerBlock): The transformer block (decoder/encoder) used
                by the model
            transformer_input (Tensor): Input tensor to the transformer
            transformer_config (TransformerConfig): Transformer config used by the model

        Returns:
            float: The rotary sequence length
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)

        if inference_context is not None:
            relative_seq_len = inference_context.max_sequence_length
        else:
            if transformer.input_tensor is not None:
                relative_seq_len = transformer.input_tensor.size(0)
            else:
                relative_seq_len = transformer_input.size(0)

            if transformer_config.sequence_parallel:
                relative_seq_len *= transformer_config.tensor_model_parallel_size

        return relative_seq_len

    def forward(self, query_seq_length, key_seq_length):
        """
        Args:
        Returns:
        """
        return self._compute_bias(query_seq_length, key_seq_length)
