from typing import Dict, List

import torch

from megatron.core.inference.common_inference_params import CommonInferenceParams
from megatron.core.inference.engines.abstract_engine import AbstractEngine
from megatron.core.inference.inference_request import InferenceRequest
from megatron.core.inference.scheduler import Scheduler
from megatron.core.inference.text_generation_controllers.simple_text_generation_controller import (
    SimpleTextGenerationController,
)


class MCoreEngine(AbstractEngine):
    def __init__(
        self,
        text_generation_controller: SimpleTextGenerationController,
        max_batch_size,
        random_seed: int = None,
    ):
        """The Megatron core backend constructor

        This is the backend that does a simple forward pass on the model. Supports any model that is callable (Accepts the inputs and outputs the tensor)

        Args:
            text_generation_controller (SimpleTextGenerationController): A text generation controller that will be used to define how to preprocess prompts, generate outputs and detokenizer the output tokens.
            max_batch_size : The maxinum number of requests to process at once
            random_seed (int, optional): Use a random seed if you want deterministic results. Defaults to None.
        """

        self.text_generation_controller = text_generation_controller
        self.random_seed = random_seed
        self.scheduler = Scheduler(max_batch_size=max_batch_size)

    def generate(self, prompts: List[str], common_inference_params: CommonInferenceParams) -> dict:
        """The megatron core inference backend generate function

        This backend returns the output generations as a dictionary. It returns the prompt tokens along with the generated tokens, the prompt plus the generated string and the output log probabilities if requested

        Args:
            prompts (List[str]): All the prompts as a list of strings
            common_inference_params (CommonInferenceParams): The inference parameters

        Returns:
            dict: The output dictionary containing the generated tokens, texts and log probs if required
        """
        # TODO :M core- get rng state tracker
        if self.random_seed:
            torch.random.manual_seed(self.random_seed)

        for prompt in prompts:
            prompt_tokens = self.text_generation_controller.tokenize_prompt(prompt)
            self.scheduler.add_request(
                prompt=prompt,
                prompt_tokens=prompt_tokens,
                inference_parameters=common_inference_params,
            )

        self.run_engine()

        result: List[InferenceRequest] = self.scheduler.completed_request_pool.values()
        return result

    def run_engine(self):
        """Main functionality to run inference

        We will keep running the engine , till we have requests in the queue. 

        Args:
            dynamic_generation (bool, optional): Set this to True, if you want to enable dynamic batching. Mainly used with an inference server. Defaults to False.
        """
        while self.scheduler.have_requests_pending():
            active_requests: Dict[int, InferenceRequest] = self.scheduler.active_request_pool.copy()
            result_dict: Dict[
                int, InferenceRequest
            ] = self.text_generation_controller.generate_all_output_tokens_static_batch(
                active_requests
            )

            self.scheduler.update_requests_pools(result_dict=result_dict)

        # TODO: Later for dynamic batching we will do something like this
        """ 
            if dynamic_batching:
                result_dict: Dict[
                    int, InferenceRequest
                ] = self.text_generation_controller.generate_output_tokens_one_step_dynamic_batch(
                    active_requests
                )
            self.scheduler.update_requests_pools(result_dict=result_dict)         
        """
