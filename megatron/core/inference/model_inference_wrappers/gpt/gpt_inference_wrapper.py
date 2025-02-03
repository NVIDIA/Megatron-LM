# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from typing import Any, Dict, Tuple

import torch

from megatron.core.inference.model_inference_wrappers.abstract_model_inference_wrapper import (
    AbstractModelInferenceWrapper,
)
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
from megatron.core.models.gpt import GPTModel


# pylint: disable=line-too-long
class GPTInferenceWrapper(AbstractModelInferenceWrapper):
    """Inference wrapper for GPT model"""

    def __init__(self, model: GPTModel, inference_wrapper_config: InferenceWrapperConfig):
        """Constructor for the model inference wrapper

        The wrapper prepares the model for inference, provides the required input data, and runs the forward pass

        Args:
            model (GPTModel): The GPT model (MCore or legacy)
            inference_wrapper_config (InferenceWrapperConfig): Has info like hidden size, vocab size etc
        """
        super().__init__(model, inference_wrapper_config)

    def prep_inference_input(self, prompts_tokens: torch.Tensor) -> Dict[str, Any]:
        """Prepares the inference input data.

        Args:
            prompts_tokens (torch.Tensor): A tensor of shape [batch_size, max_seq_len]

        Returns:
            A dict with all the inference input needed for the batch.
        """
        attention_mask, position_ids = self._build_attention_mask_and_position_ids(prompts_tokens)
        return {
            "tokens": prompts_tokens,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

    def _build_attention_mask_and_position_ids(
        self, prompts_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Builds the full attention mask and position ids for the input tokens

        Args:
            prompts_tokens (torch.Tensor): A tensor of shape [batch_size, max_seq_len]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The attention mask of shape [1, 1, max_seq_len, max_seq_len] and position ids of shape [batch_size, max_seq_len]
        """
        seq_length = prompts_tokens.size(1)
        attention_mask = torch.tril(
            torch.ones((1, seq_length, seq_length), device=prompts_tokens.device)
        ).view(1, 1, seq_length, seq_length)
        # Convert to boolean
        attention_mask = attention_mask < 0.5

        position_ids = (
            torch.arange(seq_length, dtype=torch.long, device=prompts_tokens.device)
            .unsqueeze(0)
            .expand_as(prompts_tokens)
        )

        return attention_mask, position_ids

    def get_batch_for_context_window(
        self,
        inference_input: Dict[str, Any],
        context_start_position: int,
        context_end_position: int,
    ) -> Dict[str, Any]:
        """Returns the inference data given context window

        This function gets called iteratively in a loop . Given the start and end context positions , it extracts the appropriate data.

        Args:
            inference_input (Dict[str, Any]): The inference input for the batch.
            context_start_position (int): Start of the context window. During the first inference step it is mostly 0
            context_end_position (int): End of the context window. During the last inference step it will mostly be the max generated sequence length.

        Returns:
            Dict[str, Any]: A dict of inputs that will be used by your model in the forward step
        """
        tokens = inference_input["tokens"]
        position_ids = inference_input["position_ids"]
        attention_mask = inference_input["attention_mask"]
        tokens2use = tokens[:, context_start_position:context_end_position]
        positions2use = position_ids[:, context_start_position:context_end_position]
        attention_mask2use = attention_mask[
            ..., context_start_position:context_end_position, :context_end_position
        ]
        return {
            "tokens": tokens2use,
            "position_ids": positions2use,
            "attention_mask": attention_mask2use,
        }
