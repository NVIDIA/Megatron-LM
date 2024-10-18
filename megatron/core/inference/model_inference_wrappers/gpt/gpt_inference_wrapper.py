# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from argparse import Namespace
from typing import List, Tuple

import torch

from megatron.core.inference.model_inference_wrappers.abstract_model_inference_wrapper import (
    AbstractModelInferenceWrapper,
)
from megatron.core.models.gpt import GPTModel


class GPTInferenceWrapper(AbstractModelInferenceWrapper):
    def __init__(self, model: GPTModel, args: Namespace):
        """Constructor for the model inference wrapper

        The wrapper prepares the model for inference, provides the required input data, and runs the forward pass

        Args:
            model (GPTModel): The GPT model (MCore or legacy)
            args (Namespace): The command line arguments that were passed
        """
        super().__init__(model, args)

    def prep_model_for_inference(self, prompts_tokens: torch.Tensor):
        """A utility function for preparing model for inference

        This function is called before the forward pass. It puts the model in eval mode, builds position ids, and creates attention masks so that required slices can be extracted during the forward pass.

        Args:
            prompts_tokens (torch.Tensor): A tensor of shape [batch_size, max_seq_len]
        """

        super().prep_model_for_inference(prompts_tokens=prompts_tokens)
        self.attention_mask, self.position_ids = self._build_attention_mask_and_position_ids(
            prompts_tokens
        )

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
        self, context_start_position: int, context_end_position: int
    ) -> List:
        """Returns the inference data given context window

        This function gets called iteratively in a loop . Given the start and end context positions , it extracts the appropriate data.

        Args:
            context_start_position (int): Start of the context window. During the first inference step it is mostly 0
            context_end_position (int): End of the context window. During the last inference step it will mostly be the max generated sequence length.

        Returns:
            List: A list of inputs that will be used by your model in the forward step
        """
        tokens2use = self.prompts_tokens[:, context_start_position:context_end_position]
        positions2use = self.position_ids[:, context_start_position:context_end_position]
        attention_mask2use = self.attention_mask[
            ..., context_start_position:context_end_position, :context_end_position
        ]
        data_at_step_idx = [tokens2use, positions2use, attention_mask2use]
        return data_at_step_idx
