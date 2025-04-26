# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from collections import deque
from typing import Any, Dict, List, Optional

import numpy
import torch

from megatron.core import tensor_parallel
from megatron.core.datasets.t5_dataset import T5MaskedWordPieceDataset
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.inference.model_inference_wrappers.abstract_model_inference_wrapper import (
    AbstractModelInferenceWrapper,
)
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
from megatron.core.models.T5 import T5Model
from megatron.core.utils import get_attr_wrapped_model


# pylint: disable=line-too-long
class T5InferenceWrapper(AbstractModelInferenceWrapper):
    """Inference wrapper for T5 model.

    The wrapper prepares the model for inference, provides the required input
    data, and runs the forward pass

    Args:
        model (T5Model): The T5 model (MCore or legacy)
        inference_wrapper_config (InferenceWrapperConfig): The command line arguments that were passed
        inference_context (BaseInferenceContext): Manages KV cache, and tracks
            sequence/token/batch offsets.
        use_local (bool): Whether  the T5 model's transformer impl
            is local (vs transformer_engine)
    """

    def __init__(
        self,
        model: T5Model,
        inference_wrapper_config: InferenceWrapperConfig,
        inference_context: Optional[BaseInferenceContext] = None,
        use_local: bool = False,
    ):
        super().__init__(model, inference_wrapper_config, inference_context)
        self.use_local = use_local

    def prep_inference_input(
        self,
        prompts_tokens: torch.Tensor,
        encoder_prompts: Optional[List[str]] = None,
        tokenizer: Any = None,
    ) -> Dict[str, Any]:
        """Prepares the inference input data.

        Args:
            prompts_tokens (torch.Tensor): A tensor of shape [batch_size, max_seq_len]
            encoder_prompts (dict): List of string of encoder input prompts
            tokenizer (_type_): Tokenizer used for tokenizing and detokenizing text

        Returns:
            A dict with all the inference input needed for the batch.
        """
        # get max_sequence_length
        max_sequence_length = get_attr_wrapped_model(self.model, "max_sequence_length")

        encoder_prompts_tokens_list = [
            self.tokenize_encoder_prompt(encoder_prompt, tokenizer)
            for encoder_prompt in encoder_prompts
        ]
        batch_encoder_prompts_tokens = self.pad_encoder_prompts_tokens(
            encoder_prompts_tokens_list, max_sequence_length, tokenizer
        )

        # create batch mask for encoder_prompt (self.batch_input_tokens) and
        # decoder_input (prompts_tokens), similar to megatron/core/datasets/t5_dataset.py
        decoder_prompts_tokens = prompts_tokens
        encoder_prompts_tokens = batch_encoder_prompts_tokens
        decoder_prompts_tokens_numpy = decoder_prompts_tokens.cpu().numpy()
        encoder_prompts_tokens_numpy = encoder_prompts_tokens.cpu().numpy()
        batch_mask_encoder = []
        batch_mask_decoder = []
        for i in range(len(prompts_tokens)):
            mask_encoder = encoder_prompts_tokens_numpy[i] == tokenizer.pad
            mask_decoder = decoder_prompts_tokens_numpy[i] == tokenizer.pad
            batch_mask_encoder.append(mask_encoder)
            batch_mask_decoder.append(mask_decoder)
        batch_mask_encoder = torch.tensor(numpy.array(batch_mask_encoder)).cuda()
        batch_mask_decoder = torch.tensor(numpy.array(batch_mask_decoder)).cuda()

        return {
            "encoder_tokens": encoder_prompts_tokens,
            "decoder_tokens": decoder_prompts_tokens,
            "encoder_mask": batch_mask_encoder,
            "decoder_mask": batch_mask_decoder,
        }

    def tokenize_encoder_prompt(self, encoder_prompt: str, tokenizer) -> torch.Tensor:
        """Utility to tokenize the encoder_prompt

        Args:
            encoder_prompt (str): The encoder_prompt
            tokenizer (_type_): Tokenizer used for tokenizing and detokenizing string

        Returns:
            torch.Tensor: Returns the tokenized prompt
        """

        # if there is the word "<mask>" in prompt, replacing it with special_additional_token,
        # similar to processing step in megatron/core/datasets/t5_dataset.py
        divided_encoder_prompt_list = encoder_prompt.split("<mask>")
        masks_count = len(divided_encoder_prompt_list) - 1
        sentinels = deque(tokenizer.additional_special_tokens_ids)

        encoder_prompt_tokens = []
        for divided_encoder_prompt in divided_encoder_prompt_list:
            divided_encoder_prompt_tokens = tokenizer.tokenize(divided_encoder_prompt)
            encoder_prompt_tokens.extend(divided_encoder_prompt_tokens)
            if masks_count > 0:
                sentinel = sentinels.popleft()
                encoder_prompt_tokens.extend([sentinel])
                masks_count -= 1

        return encoder_prompt_tokens

    def pad_encoder_prompts_tokens(
        self, encoder_prompts_tokens_list: List[List[int]], max_sequence_length: int, tokenizer
    ) -> torch.Tensor:
        """Method to pad input prompts

        Given a list of prompts, pad them all to uniform length

        Args:
            encoder_prompts_tokens_list (List[List[int]]): A list containing the
                encoder_input_tokens
            max_sequence_length (int): Maximum of the length of the encoder inputs tokens
            tokenizer (_type_): Tokenizer used for tokenizing and detokenizing text

        Returns:
            torch.Tensor: A torch tensor of shape [bs, max_sequence_length]
        """

        for encoder_prompt_tokens in encoder_prompts_tokens_list:
            padding_size = max_sequence_length - len(encoder_prompt_tokens)
            encoder_prompt_tokens.extend([tokenizer.pad] * padding_size)

        return torch.tensor(encoder_prompts_tokens_list).cuda()

    def get_batch_for_context_window(
        self,
        inference_input: Dict[str, Any],
        context_start_position: int,
        context_end_position: int,
    ) -> Dict[str, Any]:
        """Returns the inference data given context window

        This function gets called iteratively in a loop . Given the start and end context
        positions , it extracts the appropriate data.

        Args:
            inference_input (Dict[str, Any]): The inference input for the batch.
            context_start_position (int): Start of the context window. During
                the first inference step it is mostly 0
            context_end_position (int): End of the context window. During the
                last inference step it will mostly be the max generated sequence length.

        Returns:
            Dict: A dict of inputs that will be used by your model in the forward step
        """

        # T5 inference not yet support kv_cache
        encoder_tokens2use = inference_input["encoder_tokens"]
        decoder_tokens2use = inference_input["decoder_tokens"][:, :context_end_position]
        encoder_mask2use = inference_input["encoder_mask"]
        decoder_mask2use = inference_input["decoder_mask"][:, :context_end_position]

        # Configure attention mask based on different conditions
        # (e.g., transformer-impl, TE versions, TE backends)
        [encoder_mask2use, decoder_mask2use, encoder_decoder_mask2use] = (
            T5MaskedWordPieceDataset.config_attention_mask(
                encoder_tokens2use,
                decoder_tokens2use,
                encoder_mask2use,
                decoder_mask2use,
                self.use_local,
            )
        )

        return {
            "encoder_tokens": encoder_tokens2use,
            "decoder_tokens": decoder_tokens2use,
            "encoder_mask": encoder_mask2use,
            "decoder_mask": decoder_mask2use,
            "encoder_decoder_mask": encoder_decoder_mask2use,
        }

    def forward_pass_without_pipeline_parallel(
        self, inference_input: Dict[str, Any]
    ) -> torch.Tensor:
        """Utility to carry out simple forward pass for TP or no model parallel models

        Runs a very simple forward pass for model. Used  in the case of models without
        any parallelism or only tensor parallelism.

        Args:
            inference_input (Dict[str, Any]): A dict containg the inputs for the gpt
                model [tokens, position ids, attention mask]

        Returns:
            torch.Tensor: The output logits of shape [batch_size, seq_len, padded_vocab_size]
        """
        encoder_tokens = inference_input["encoder_tokens"]
        decoder_tokens = inference_input["decoder_tokens"]
        encoder_mask = inference_input["encoder_mask"]
        decoder_mask = inference_input["decoder_mask"]
        encoder_decoder_mask = inference_input["encoder_decoder_mask"]
        tokens = decoder_tokens

        # T5 inference not yet support kv_cache
        logits = self.model(
            encoder_tokens,
            decoder_tokens,
            encoder_mask,
            decoder_mask,
            encoder_decoder_mask,
            inference_context=None,
        )
        logits = tensor_parallel.gather_from_tensor_model_parallel_region(logits, self.tp_group)

        return logits
