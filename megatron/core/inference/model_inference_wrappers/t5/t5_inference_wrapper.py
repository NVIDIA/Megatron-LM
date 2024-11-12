# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from collections import deque
from typing import Any, List, Tuple

import numpy
import torch

from megatron.core import tensor_parallel
from megatron.core.datasets.t5_dataset import T5MaskedWordPieceDataset
from megatron.core.inference.model_inference_wrappers.abstract_model_inference_wrapper import (
    AbstractModelInferenceWrapper,
)
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
from megatron.core.models.T5 import T5Model


# pylint: disable=line-too-long
class T5InferenceWrapper(AbstractModelInferenceWrapper):
    """Constructor for the model inference wrapper

    The wrapper prepares the model for inference, provides the required input
    data, and runs the forward pass

    Args:
        model (T5Model): The T5 model (MCore or legacy)
        inference_wrapper_config (InferenceWrapperConfig): The command line arguments that were passed
    """

    def __init__(self, model: T5Model, inference_wrapper_config: InferenceWrapperConfig):
        super().__init__(model, inference_wrapper_config)

    def prep_model_for_inference(
        self, prompts_tokens: torch.Tensor, encoder_prompts: List[str] = None, tokenizer: Any = None
    ):
        """A utility function for preparing model for inference

        This function is called before the forward pass. It puts the model in eval mode, builds
        position ids, and creates attention masks so that required slices can be extracted during
        the forward pass.

        Args:
            prompts_tokens (torch.Tensor): A tensor of shape [batch_size, max_sequence_length]
            encoder_prompts (dict): List of string of encoder input prompts
            tokenizer (_type_): Tokenizer used for tokenizing and detokenizing text
        """

        super().prep_model_for_inference(prompts_tokens=prompts_tokens)

        encoder_prompts_tokens_list = [
            self.tokenize_encoder_prompt(encoder_prompt, tokenizer)
            for encoder_prompt in encoder_prompts
        ]
        self.batch_encoder_prompts_tokens = self.pad_encoder_prompts_tokens(
            encoder_prompts_tokens_list, self.model.max_sequence_length, tokenizer
        )

        # create batch mask for encoder_prompt (self.batch_input_tokens) and
        # decoder_input (self.prompts_tokens), similar to megatron/core/datasets/t5_dataset.py
        decoder_prompts_tokens = self.prompts_tokens.cpu().numpy()
        encoder_prompts_tokens = self.batch_encoder_prompts_tokens.cpu().numpy()
        self.batch_mask_encoder = []
        self.batch_mask_decoder = []
        self.batch_mask_encoder_decoder = []
        for i in range(len(self.prompts_tokens)):
            self.batch_mask_encoder.append(
                T5MaskedWordPieceDataset._make_attention_mask(
                    encoder_prompts_tokens[i], encoder_prompts_tokens[i]
                )
            )
            self.batch_mask_decoder.append(
                T5MaskedWordPieceDataset._make_attention_mask(
                    decoder_prompts_tokens[i], decoder_prompts_tokens[i]
                )
                * T5MaskedWordPieceDataset._make_history_mask(decoder_prompts_tokens[i])
            )
            self.batch_mask_encoder_decoder.append(
                T5MaskedWordPieceDataset._make_attention_mask(
                    decoder_prompts_tokens[i], encoder_prompts_tokens[i]
                )
            )
        self.batch_mask_encoder = torch.tensor(numpy.array(self.batch_mask_encoder)).cuda()
        self.batch_mask_decoder = torch.tensor(numpy.array(self.batch_mask_decoder)).cuda()
        self.batch_mask_encoder_decoder = torch.tensor(
            numpy.array(self.batch_mask_encoder_decoder)
        ).cuda()
        self.batch_mask_encoder = self.batch_mask_encoder < 0.5
        self.batch_mask_decoder = self.batch_mask_decoder < 0.5
        self.batch_mask_encoder_decoder = self.batch_mask_encoder_decoder < 0.5

    def tokenize_encoder_prompt(
        self, encoder_prompt: str, tokenizer
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        self, context_start_position: int, context_end_position: int
    ) -> List:
        """Returns the inference data given context window

        This function gets called iteratively in a loop . Given the start and end context
        positions , it extracts the appropriate data.

        Args:
            context_start_position (int): Start of the context window. During
                the first inference step it is mostly 0
            context_end_position (int): End of the context window. During the
                last inference step it will mostly be the max generated sequence length.

        Returns:
            List: A list of inputs that will be used by your model in the forward step
        """

        # rerun encoder every step
        # T5 inference not yet support kv_cache
        encoder_tokens2use = self.batch_encoder_prompts_tokens
        decoder_tokens2use = self.prompts_tokens[:, :context_end_position]
        encoder_mask2use = self.batch_mask_encoder
        decoder_mask2use = self.batch_mask_decoder[:, :context_end_position, :context_end_position]
        encoder_decoder_mask2use = self.batch_mask_encoder_decoder[:, :context_end_position, :]
        data_at_step_idx = [
            encoder_tokens2use,
            decoder_tokens2use,
            encoder_mask2use,
            decoder_mask2use,
            encoder_decoder_mask2use,
        ]

        return data_at_step_idx

    def forward_pass_without_pipeline_parallel(self, inference_input: List) -> torch.Tensor:
        """Utility to carry out simple forward pass for TP or no model parallel models

        Runs a very simple forward pass for model. Used  in the case of models without
        any parallelism or only tensor parallelism.

        Args:
            inference_input (List): A list containg the inputs for the gpt
                model [tokens, position ids, attention mask]

        Returns:
            torch.Tensor: The output logits of shape [batch_size, seq_len, padded_vocab_size]
        """
        [encoder_tokens, decoder_tokens, encoder_mask, decoder_mask, encoder_decoder_mask] = (
            inference_input
        )
        tokens = decoder_tokens

        # T5 inference not yet support kv_cache
        logits = self.model(
            encoder_tokens,
            decoder_tokens,
            encoder_mask,
            decoder_mask,
            encoder_decoder_mask,
            inference_params=None,
        )
        logits = tensor_parallel.gather_from_tensor_model_parallel_region(logits)

        return logits
