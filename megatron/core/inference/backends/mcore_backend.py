from typing import List

import torch

from megatron.core import parallel_state
from megatron.core.inference.backends.abstract_backend import AbstractBackend
from megatron.core.inference.common_inference_params import CommonInferenceParams
from megatron.core.inference.text_generation_strategies.simple_text_generation_strategy import (
    SimpleTextGenerationStrategy,
)


class MCoreBackend(AbstractBackend):
    def __init__(
        self, text_generation_strategy: SimpleTextGenerationStrategy, random_seed: int = None
    ):
        """The Megatron core backend constructor

        This is the backend that does a simple forward pass on the model. Supports any model that is callable (Accepts the inputs and outputs the tensor)

        Args:
            text_generation_strategy (SimpleTextGenerationStrategy): A text generation strategy that will be used to define how to preprocess prompts, generate outputs and detokenizer the output tokens.
            random_seed (int, optional): Use a random seed if you want dterministic results. Defaults to None.
        """

        self.text_generation_strategy = text_generation_strategy
        self.random_seed = random_seed

    def generate(self, prompts: List[str], common_inference_params: CommonInferenceParams) -> dict:
        """The megatron core inference backend generate function

        This backend returns the output generations as a dictionary. It returns the prompt tokens along with the generated tokens, the prompt plus the generated string and the output log probabilities if requested

        Args:
            prompts (List[str]): All the prompts (of a global batch size) as a list of strings
            common_inference_params (CommonInferenceParams): The inference parameters

        Returns:
            dict: The output dictionary containing the generated tokens, texts and log probs if required
        """

        # TODO :M core- get rng state tracker
        if self.random_seed:
            torch.random.manual_seed(self.random_seed)

        (
            prompts_tokens,
            prompts_lengths,
        ) = self.text_generation_strategy.tokenize_and_pad_input_prompts(
            prompts, common_inference_params.num_tokens_to_generate
        )

        (
            prompts_tokens_with_generations,
            required_sequence_lengths,
            output_log_probs,
        ) = self.text_generation_strategy.generate_output_tokens(
            prompts_tokens, prompts_lengths, common_inference_params
        )

        # Returns true for both if model is not PP (TODO: Maybe should move this into parallel state ?)
        model_is_not_pipeline_parallel = (
            parallel_state.is_pipeline_first_stage() and parallel_state.is_pipeline_last_stage()
        )

        # Returns the output in the first stage or in all GPUS for TP only models
        if model_is_not_pipeline_parallel or parallel_state.is_pipeline_first_stage():
            prompts_plus_generations_detokenized = self.text_generation_strategy.detokenize_generations(
                prompts_tokens_with_generations, required_sequence_lengths
            )

            return {
                'prompts_tokens_with_generations': prompts_tokens_with_generations,
                'prompts_plus_generations_detokenized': prompts_plus_generations_detokenized,
                'output_log_probs': output_log_probs,
            }

        else:
            return None
