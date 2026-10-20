# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from typing import Any, Dict, OrderedDict

import torch

from megatron.core.inference.inference_request import InferenceRequest
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.inference.utils import get_attention_mask


class EncoderDecoderTextGenerationController(TextGenerationController):
    """The text generation controller for encoder-decoder architecture

    This class inherits from TextGenerationController, adding features
    relating to encoder input encoder_prompt

    """

    def prep_inference_input(
        self,
        prompts_tokens: torch.Tensor,
        active_requests: OrderedDict[str, InferenceRequest],
        use_attention_mask: bool = False,
    ) -> Dict[str, Any]:
        """Preparing input data for inference, using respective wrapper's prep_inference_input method # pylint: disable=line-too-long

        Args:
            prompts_tokens (torch.Tensor): A tensor of shape [batch_size, max_sequence_length]
            active_requests (OrderedDict[str, InferenceRequest]): The input active requests
            use_attention_mask (bool): Whether to use an attention mask. Should be set to True only
                when exclusively doing prefill (no decode) with variable prompt lengths.

        Returns:
            A dict of the inference input for the current batch.
        """
        encoder_prompts = list(
            map(lambda request: request.encoder_prompt, active_requests.values())
        )

        inference_input = self.inference_wrapped_model.prep_inference_input(
            prompts_tokens, encoder_prompts, tokenizer=self.tokenizer
        )

        if use_attention_mask and (
            attention_mask := inference_input.get("attention_mask", None) is None
        ):
            inference_input["attention_mask"] = get_attention_mask(prompts_tokens.size(1))

        return inference_input
