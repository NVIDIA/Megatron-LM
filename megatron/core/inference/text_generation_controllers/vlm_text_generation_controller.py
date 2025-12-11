# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from typing import OrderedDict

import torch

from megatron.core.inference.inference_request import InferenceRequest, VLMInferenceRequest
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.inference.utils import get_attention_mask


class VLMTextGenerationController(TextGenerationController):
    """The text generation controller for VLMs"""

    def prep_inference_input(
        self,
        prompts_tokens: torch.Tensor,
        active_requests: OrderedDict[str, InferenceRequest],
        use_attention_mask: bool = False,
    ):
        """Preparing input data for inference, using respective wrapper's prep_inference_input method # pylint: disable=line-too-long

        Currently only supports batch size 1 inference.

        Args:
            prompts_tokens (torch.Tensor): A tensor of shape [batch_size, max_sequence_length]
            active_requests (OrderedDict[str, InferenceRequest]): The input active requests
            use_attention_mask (bool): Whether to use an attention mask. Should be set to True only
                when exclusively doing prefill (no decode) with variable prompt lengths.
        """
        assert len(active_requests) == 1, f"VLM inference currently only supports batch size 1"

        request = list(active_requests.values())[0]

        assert isinstance(
            request, VLMInferenceRequest
        ), f"Found inference request of type {type(request)}, expected VLMInferenceRequest"

        inference_input = self.inference_wrapped_model.prep_inference_input(
            prompts_tokens,
            request.num_img_embeddings_per_tile,
            request.imgs,
            request.num_tiles,
            request.decoder_seq_length,
        )

        if use_attention_mask and (
            attention_mask := inference_input.get("attention_mask", None) is None
        ):
            inference_input["attention_mask"] = get_attention_mask(prompts_tokens.size(1))

        return inference_input
