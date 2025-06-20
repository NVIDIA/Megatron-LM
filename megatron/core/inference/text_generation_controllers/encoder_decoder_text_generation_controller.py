# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from typing import Any, Dict, OrderedDict

import torch

from megatron.core.inference.inference_request import InferenceRequest
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)


class EncoderDecoderTextGenerationController(TextGenerationController):
    """The text generation controller for encoder-decoder architecture

    This class inherits from TextGenerationController, adding features
    relating to encoder input encoder_prompt

    """

    def prep_inference_input(
        self, prompts_tokens: torch.Tensor, active_requests: OrderedDict[str, InferenceRequest]
    ) -> Dict[str, Any]:
        """Preparing input data for inference, using respective wrapper's prep_inference_input method # pylint: disable=line-too-long

        Args:
            prompts_tokens (torch.Tensor): A tensor of shape [batch_size, max_sequence_length]
            active_requests (OrderedDict[str, InferenceRequest]): The input active requests

        Returns:
            A dict of the inference input for the current batch.
        """
        encoder_prompts = list(
            map(lambda request: request.encoder_prompt, active_requests.values())
        )

        return self.inference_wrapped_model.prep_inference_input(
            prompts_tokens, encoder_prompts, tokenizer=self.tokenizer
        )
