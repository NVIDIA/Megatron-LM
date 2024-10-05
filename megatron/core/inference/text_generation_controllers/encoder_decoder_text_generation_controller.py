# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from typing import OrderedDict

import torch

from megatron.core.inference.inference_request import InferenceRequest
from megatron.core.inference.text_generation_controllers.simple_text_generation_controller import (
    SimpleTextGenerationController,
)


class EncoderDecoderTextGenerationController(SimpleTextGenerationController):
    """The text generation controller for encoder-decoder architecture

    This class ingherits from SimpleTextGenerationController, adding features
    relating to encoder input encoder_prompt

    """

    def prep_model_for_inference(
        self, prompts_tokens: torch.Tensor, active_requests: OrderedDict[int, InferenceRequest]
    ):
        """Preparing batch for inference, using respective wrapper's prep_model_for_inference method

        Args:
            prompts_tokens (torch.Tensor): A tensor of shape [batch_size, max_sequence_length]
            active_requests (OrderedDict[int, InferenceRequest]): The input active requests
        """
        encoder_prompts = list(
            map(lambda request: request.encoder_prompt, active_requests.values())
        )

        self.inference_wrapped_model.prep_model_for_inference(
            prompts_tokens=prompts_tokens, encoder_prompts=encoder_prompts, tokenizer=self.tokenizer
        )
