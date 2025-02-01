# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import asyncio
from collections import OrderedDict
from typing import AsyncGenerator, Dict, List, Optional, Union

import torch

from megatron.core.inference.async_stream import AsyncStream
from megatron.core.inference.engines.abstract_engine import AbstractEngine
from megatron.core.inference.inference_request import InferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.scheduler import Scheduler
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)


class MCoreEngine(AbstractEngine):
    """The Megatron core backend constructor

    This is the backend that does a simple forward pass on the model.
    Supports any model that is callable (Accepts the inputs and outputs the tensor)

    Args:
        text_generation_controller (TextGenerationController): A text generation
            controller that will be used to define how to preprocess prompts, generate
            outputs and detokenizer the output tokens.
        max_batch_size : The maxinum number of requests to process at once
        random_seed (int, optional): Use a random seed if you want deterministic
            results. Defaults to None.
    """

    def __init__(
        self,
        text_generation_controller: TextGenerationController,
        max_batch_size,
        random_seed: Optional[int] = None,
    ):
        self.text_generation_controller = text_generation_controller
        self.random_seed = random_seed
        self.scheduler = Scheduler(max_batch_size=max_batch_size)

    def add_request(
        self,
        prompt: str,
        add_BOS: bool = False,
        encoder_prompt: Optional[str] = None,
        inference_parameters: Optional[SamplingParams] = None,
        streaming: bool = False,
    ) -> str:
        """
        Adds a request to the scheduler and returns the request ID.

        Args:
            prompt (str): A prompt string
            add_BOS (bool): Whether to add BOS token to beginning of the prompt
            encoder_prompt (str): The encoder prompt string
            inference_parameters (SamplingParams): The inference parameters
            streaming (bool): Whether to stream incremental outputs for this request

        Returns:
            The newly created request ID.
        """

        prompt_tokens = self.text_generation_controller.tokenize_prompt(prompt, add_BOS)

        return self.scheduler.add_request(
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            encoder_prompt=encoder_prompt,
            inference_parameters=inference_parameters,
            streaming=streaming,
        )

    def get_stream_generator(
        self, request_id: str
    ) -> Union[AsyncGenerator[InferenceRequest, None], None]:
        """Returns the stream generator for the given request ID if it exists."""
        stream = self.scheduler.streams.get(request_id, None)
        if stream is not None:
            return stream.generator()
        return None

    def generate(
        self,
        prompts: List[str],
        add_BOS: bool = False,
        encoder_prompts: Optional[List[str]] = None,
        common_inference_params: Optional[SamplingParams] = None,
        sampling_params: Optional[SamplingParams] = None,
    ) -> List[InferenceRequest]:
        """The megatron core inference backend generate function

        This backend returns the output generations as a dictionary.
        It returns the prompt tokens along with the generated tokens, the prompt
        plus the generated string and the output log probabilities if requested

        Args:
            prompts (List[str]): All the prompts as a list of strings
            add_BOS (bool): Whether to add BOS token to beginning of prompts
            encoder_prompts (List[dict]): All the encoder prompts as a list of strings
            common_inference_params: Deprecated. Only used for backward compatibility with
            MCore <= 0.9.0. Use `sampling_params` going forward.
            sampling_params (SamplingParams): The request-level sampling parameters

        Returns:
            List[InferenceRequest]: The output is list of inference requests containing the
            generated tokens, texts and log probs if required
        """
        # TODO :M core- get rng state tracker

        if common_inference_params:
            sampling_params = common_inference_params
        if sampling_params is None:
            sampling_params = SamplingParams()

        if self.random_seed:
            torch.random.manual_seed(self.random_seed)

        request_ids: List[str] = []
        for i in range(len(prompts)):
            prompt = prompts[i]
            encoder_prompt = encoder_prompts[i] if encoder_prompts is not None else None
            request_id = self.add_request(
                prompt=prompt,
                add_BOS=add_BOS,
                encoder_prompt=encoder_prompt,
                inference_parameters=sampling_params,
            )
            request_ids.append(request_id)

        self.run_engine()

        result: List[InferenceRequest] = [
            self.scheduler.completed_request_pool[request_id] for request_id in request_ids
        ]
        return result

    def run_engine(self):
        """Main functionality to run inference

        Runs the engine until there are no requests in the queue.

        Args:
            dynamic_generation (bool, optional): Set this to True, if you want
                to enable dynamic batching. Mainly used with an inference server.
                Defaults to False.
        """
        while self.scheduler.have_requests_pending():
            active_requests: Dict[str, InferenceRequest] = self.scheduler.active_request_pool.copy()
            active_streams: Dict[str, AsyncStream] = OrderedDict()
            for request_id in active_requests:
                if (stream := self.scheduler.streams.get(request_id, None)) is not None:
                    assert isinstance(stream, AsyncStream), stream
                    active_streams[request_id] = stream
            result_dict: Dict[str, InferenceRequest] = (
                self.text_generation_controller.generate_all_output_tokens_static_batch(
                    active_requests, active_streams
                )
            )

            self.scheduler.update_requests_pools(result_dict=result_dict)

        # TODO: Later for dynamic batching we will do something like this
        """ 
            if dynamic_batching:
                result_dict: Dict[
                    str, InferenceRequest
                ] = self.text_generation_controller.generate_output_tokens_one_step_dynamic_batch(
                    active_requests
                )
            self.scheduler.update_requests_pools(result_dict=result_dict)         
        """

    def _wrapped_run_engine(self, cuda_device):
        """
        Explicitly sets the CUDA device before running the engine.

        This is to ensure that the CUDA device is correctly propagated when running
        in a new thread context.
        """
        torch.cuda.set_device(cuda_device)
        self.run_engine()

    async def run_engine_async(self):
        """Runs the engine asynchronously using asyncio"""
        loop = asyncio.get_running_loop()

        await loop.run_in_executor(None, self._wrapped_run_engine, torch.cuda.current_device())
