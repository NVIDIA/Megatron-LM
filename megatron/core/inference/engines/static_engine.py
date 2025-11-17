# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import warnings
from collections import OrderedDict
from typing import AsyncGenerator, Dict, List, Optional, Union

import torch

from megatron.core.inference.async_stream import AsyncStream
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines.abstract_engine import AbstractEngine
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
from megatron.core.inference.inference_request import InferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.scheduler import Scheduler
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)

try:
    from tqdm import tqdm

    HAVE_TQDM = True
except ImportError:
    from unittest.mock import MagicMock

    tqdm = MagicMock()
    HAVE_TQDM = False


# pylint: disable=line-too-long
class StaticInferenceEngine(AbstractEngine):
    """The Megatron core backend constructor

    This is the backend that does a simple forward pass on the model.
    Supports any model that is callable (Accepts the inputs and outputs the tensor)

    Args:
        text_generation_controller (TextGenerationController): A text generation
            controller that will be used to define how to preprocess prompts, generate
            outputs and detokenizer the output tokens.
        max_batch_size (int, optional): The maximum number of requests to process at once.
            Will be set from the InferenceWrapperConfig in `text_generation_controller` by
            default.
        random_seed (int, optional): Use a random seed if you want deterministic
            results. Defaults to None.
    """

    def __init__(
        self,
        text_generation_controller: TextGenerationController,
        max_batch_size: Optional[int] = None,
        random_seed: Optional[int] = None,
        legacy=False,
        buffer_size_gb: Optional[float] = 40,
    ):
        self.legacy = legacy
        if legacy:
            warnings.warn(
                "The static engine will be deprecated and removed in the future version of megatron-core. Switch to DynamicInferenceEngine."
            )
        else:
            warnings.warn(
                "`StaticInferenceEngine` will be deprecated in a future version of Megatron-core. "
                "Please directly use `DynamicInferenceEngine` instead. "
                "`StaticInferenceEngine` currently uses `DynamicInferenceEngine` under the hood.",
                DeprecationWarning,
            )

        inference_wrapper_config = (
            text_generation_controller.inference_wrapped_model.inference_wrapper_config
        )
        self.controller = text_generation_controller
        self.random_seed = random_seed or 1234

        inference_max_batch_size = inference_wrapper_config.inference_max_requests
        if max_batch_size is None:
            max_batch_size = inference_max_batch_size
        elif max_batch_size > inference_max_batch_size:
            warnings.warn(
                f"Engine `max_batch_size` ({max_batch_size}) > "
                f"`inference_max_requests` in `inference_wrapper_config` "
                f"({inference_max_batch_size}); setting `max_batch_size` to "
                f"{inference_max_batch_size}",
                UserWarning,
            )
            max_batch_size = inference_max_batch_size

        self.scheduler = Scheduler(max_batch_size=max_batch_size)

        # Store original context in case we need to fall back to legacy static engine
        original_context = text_generation_controller.inference_wrapped_model.inference_context

        try:
            if not legacy:
                dynamic_context = DynamicInferenceContext.from_config(
                    inference_config=inference_wrapper_config,
                    model=text_generation_controller.inference_wrapped_model.model,
                    max_batch_size=max_batch_size,
                    buffer_size_gb=buffer_size_gb,
                    num_cuda_graphs=1,
                )
                self.controller.inference_wrapped_model.inference_context = dynamic_context
                self.controller.inference_wrapped_model.prep_model_for_inference()

                self.dynamic_engine = DynamicInferenceEngine(
                    controller=self.controller,
                    random_seed=self.random_seed,
                    context=dynamic_context,
                    enable_cuda_graph=True,
                    static_sampling=True,
                )
        except Exception as e:
            # Get exception details for better debugging
            exception_msg = str(e) if str(e) else f"{type(e).__name__}: {repr(e)}"
            warnings.warn(
                f"Error initializing dynamic engine: {exception_msg} , using legacy static engine",
                UserWarning,
            )
            # Restore original context when falling back to legacy static engine
            self.controller.inference_wrapped_model.inference_context = original_context
            self.legacy = True

    def get_new_request_id(self) -> str:
        """Gets a new request id from the scheduler"""
        return self.scheduler.get_new_request_id()

    def add_request(
        self,
        prompt: Optional[str] = None,
        add_BOS: bool = False,
        encoder_prompt: Optional[str] = None,
        sampling_params: Optional[SamplingParams] = None,
        streaming: bool = False,
        inference_request: Optional[InferenceRequest] = None,
        *,
        inference_parameters: Optional[SamplingParams] = None,
    ) -> int:
        """
        Adds a request to the scheduler and returns the request ID.

        Args:
            prompt (str): A prompt string
            add_BOS (bool): Whether to add BOS token to beginning of the prompt
            encoder_prompt (str): The encoder prompt string
            sampling_params (SamplingParams): The inference parameters
            streaming (bool): Whether to stream incremental outputs for this request
            inference_request (InferenceRequest, optional): A fully constructed request.
                Defaults to None.
            inference_parameters (SamplingParams, optional): Deprecated and
                renamed to `SamplingParams`.

        Returns:
            The newly created request ID.
        """
        assert (
            prompt is not None or inference_request is not None
        ), f"At least one of `prompt` or `inference_request` must be specified"

        if sampling_params is None and inference_parameters is not None:
            warnings.warn(
                "`inference_parameters` has been renamed to `sampling_params`, "
                "and the previous name will be removed in Mcore v0.14."
            )
            sampling_params = inference_parameters

        if inference_request is None:
            # Support legacy single-arg tokenize_prompt mocks in tests.
            prompt_tokens = self.controller.tokenize_prompt(prompt, add_BOS)
        else:
            prompt_tokens = inference_request.prompt_tokens

        return self.scheduler.add_request(
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            encoder_prompt=encoder_prompt,
            sampling_params=sampling_params,
            streaming=streaming,
            inference_request=inference_request,
        )

    def get_stream_generator(
        self, request_id: int
    ) -> Union[AsyncGenerator[InferenceRequest, None], None]:
        """Returns the stream generator for the given request ID if it exists."""
        stream = self.scheduler.streams.get(request_id, None)
        if stream is not None:
            return stream.generator()
        return None

    @torch.inference_mode()
    def generate_using_dynamic_engine(
        self,
        prompts: Optional[List[str]] = None,
        add_BOS: bool = False,
        encoder_prompts: Optional[List[str]] = None,
        common_inference_params: Optional[SamplingParams] = None,
        sampling_params: Optional[SamplingParams] = None,
        inference_requests: Optional[List[InferenceRequest]] = None,
    ) -> List[InferenceRequest]:
        """Generate using dynamic engine

        Generate using dynamic engine.

        Args:
            prompts (List[str]): All the prompts as a list of strings
            add_BOS (bool): Whether to add BOS token to beginning of prompts
            encoder_prompts (List[dict]): All the encoder prompts as a list of strings
            common_inference_params: Deprecated. Only used for backward compatibility with
            MCore <= 0.9.0. Use `sampling_params` going forward.
            sampling_params (SamplingParams): The request-level sampling parameters
            inference_requests (List[InferenceRequest]): A pre-populated list of inference requests

        Returns:
            List[InferenceRequest]: The output is list of inference requests containing the
            generated tokens, texts and log probs if required
        """
        assert hasattr(self, 'dynamic_engine'), "Dynamic engine not initialized"
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # 'RuntimeError: There is no current event loop...'
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if common_inference_params:
            sampling_params = common_inference_params
        if prompts:
            if add_BOS:
                sampling_params.add_BOS = True
            return self.dynamic_engine.generate(prompts=prompts, sampling_params=sampling_params)
        elif inference_requests:
            prompts = [request.prompt for request in inference_requests]
            sampling_params = inference_requests[0].sampling_params
            if add_BOS:
                sampling_params.add_BOS = True
            return self.dynamic_engine.generate(prompts=prompts, sampling_params=sampling_params)

    def generate_using_legacy_static_engine(
        self,
        prompts: Optional[List[str]] = None,
        add_BOS: bool = False,
        encoder_prompts: Optional[List[str]] = None,
        common_inference_params: Optional[SamplingParams] = None,
        sampling_params: Optional[SamplingParams] = None,
        inference_requests: Optional[List[InferenceRequest]] = None,
    ) -> List[InferenceRequest]:
        """The megatron core inference backend generate function

        This backend returns the output generations as a list.
        Args:
            prompts (List[str]): All the prompts as a list of strings
            add_BOS (bool): Whether to add BOS token to beginning of prompts
            encoder_prompts (List[dict]): All the encoder prompts as a list of strings
            common_inference_params: Deprecated. Only used for backward compatibility with
            MCore <= 0.9.0. Use `sampling_params` going forward.
            sampling_params (SamplingParams): The request-level sampling parameters
            inference_requests (List[InferenceRequest]): A pre-populated list of inference requests

        Returns:
            List[InferenceRequest]: The output is list of inference requests containing the
            generated tokens, texts and log probs if required
        """
        request_ids: List[str] = []

        if self.random_seed:
            torch.random.manual_seed(self.random_seed)

        if inference_requests is None:
            assert prompts is not None

            if common_inference_params:
                sampling_params = common_inference_params

            for i in range(len(prompts)):
                prompt = prompts[i]
                encoder_prompt = encoder_prompts[i] if encoder_prompts is not None else None
                request_id = self.add_request(
                    prompt=prompt, encoder_prompt=encoder_prompt, sampling_params=sampling_params
                )
                request_ids.append(request_id)
        else:
            for inference_request in inference_requests:
                request_ids.append(inference_request.request_id)
                self.scheduler.add_request(inference_request=inference_request)

        self.run_engine()

        result: List[InferenceRequest] = [
            self.scheduler.completed_request_pool[request_id] for request_id in request_ids
        ]
        return result

    def generate(
        self,
        prompts: Optional[List[str]] = None,
        add_BOS: bool = False,
        encoder_prompts: Optional[List[str]] = None,
        common_inference_params: Optional[SamplingParams] = None,
        sampling_params: Optional[SamplingParams] = None,
        inference_requests: Optional[List[InferenceRequest]] = None,
    ) -> List[InferenceRequest]:
        """The megatron core inference backend generate function

        This uses dynamic engine if available, otherwise uses legacy static engine.

        Args:
            prompts (List[str]): All the prompts as a list of strings
            add_BOS (bool): Whether to add BOS token to beginning of prompts
            encoder_prompts (List[dict]): All the encoder prompts as a list of strings
            common_inference_params: Deprecated. Only used for backward compatibility with
            MCore <= 0.9.0. Use `sampling_params` going forward.
            sampling_params (SamplingParams): The request-level sampling parameters
            inference_requests (List[InferenceRequest]): A pre-populated list of inference requests

        Returns:
            List[InferenceRequest]: The output is list of inference requests containing the
            generated tokens, texts and log probs if required
        """
        # TODO :M core- get rng state tracker
        if not self.legacy:
            return self.generate_using_dynamic_engine(
                prompts=prompts,
                add_BOS=add_BOS,
                encoder_prompts=encoder_prompts,
                common_inference_params=common_inference_params,
                sampling_params=sampling_params,
                inference_requests=inference_requests,
            )
        else:
            return self.generate_using_legacy_static_engine(
                prompts=prompts,
                add_BOS=add_BOS,
                encoder_prompts=encoder_prompts,
                common_inference_params=common_inference_params,
                sampling_params=sampling_params,
                inference_requests=inference_requests,
            )

    def run_engine(self):
        """Main functionality to run inference

        Runs the engine until there are no requests in the queue.

        Args:
            dynamic_generation (bool, optional): Set this to True, if you want
                to enable dynamic batching. Mainly used with an inference server.
                Defaults to False.
        """

        if not HAVE_TQDM:
            raise ImportError(
                "tqdm is required for StaticInferenceEngine, "
                "please install it with `pip install tqdm`"
            )

        prev_num_requests_pending = self.scheduler.num_requests_pending()
        tbar = tqdm(desc="static requests", total=prev_num_requests_pending)
        while self.scheduler.have_requests_pending():
            active_requests: Dict[str, InferenceRequest] = self.scheduler.active_request_pool.copy()
            active_streams: Dict[str, AsyncStream] = OrderedDict()
            for request_id in active_requests:
                if (stream := self.scheduler.streams.get(request_id, None)) is not None:
                    assert isinstance(stream, AsyncStream), stream
                    active_streams[request_id] = stream
            result_dict: Dict[str, InferenceRequest] = (
                self.controller.generate_all_output_tokens_static_batch(
                    active_requests, active_streams
                )
            )

            self.scheduler.update_requests_pools(result_dict=result_dict)

            crnt_num_requests_pending = self.scheduler.num_requests_pending()
            tbar.update(prev_num_requests_pending - crnt_num_requests_pending)
            prev_num_requests_pending = crnt_num_requests_pending

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
