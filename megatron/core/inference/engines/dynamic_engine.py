# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import asyncio
import time
from collections import deque
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from megatron.core.inference.contexts.dynamic_context import (
    ChunkOverflowError,
    DynamicInferenceContext,
    MaxSequenceLengthOverflowError,
    RequestOverflowError,
    TokenOverflowError,
)
from megatron.core.inference.engines.abstract_engine import AbstractEngine
from megatron.core.inference.inference_request import DynamicInferenceRequest, Status
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.simple_text_generation_controller import (
    SimpleTextGenerationController,
)
from megatron.core.inference.utils import Counter
from megatron.core.transformer.cuda_graphs import create_cudagraphs


class DynamicInferenceEngine(AbstractEngine):
    """The dynamic inference engine.

    This engine allows requests of varying length to be dynamically added and
    removed in each inference step. In contrast to the static engine that has a
    set batch size and sequence length during the forward pass, each request in
    the dynamic engine can have different *current* prompt and output length at
    any given step, and the processing is restricted only by a max number of total
    tokens across all requests.

    Args:
        text_generation_controller (SimpleTextGenerationController): A text generation
            controller that will be used to define how to preprocess prompts, generate
            outputs and detokenizer the output tokens.
        inference_context (DynamicInferenceContext): Context for managing in-flight
            batching and a dynamic chunked KV cache (similar to paged attention).
        termination_id (int): Token ID to mark end-of-sequence.
        random_seed (Optional[int]): Use a random seed if you want deterministic
            results. Defaults to None.
    """

    def __init__(
        self,
        controller: SimpleTextGenerationController,
        context: DynamicInferenceContext,
        termination_id: int,
        enable_cuda_graph: bool,
        random_seed: Optional[int] = None,
    ):

        assert isinstance(controller, SimpleTextGenerationController)
        assert isinstance(context, DynamicInferenceContext)
        assert isinstance(termination_id, int)
        assert isinstance(random_seed, int)

        self.request_counter = Counter()
        self.controller = controller
        self.context = context
        self.termination_id = termination_id
        self.random_seed = random_seed
        self.finished_request_count = 0
        self.waiting_request_ids = deque()
        self.request_counter = Counter()
        self.requests: Dict[int, DynamicInferenceRequest] = {}
        self.request_completion_futures: Dict[int, asyncio.Future] = {}

        # Initialize the asyncio loop if it has not already been initialized.
        # TODO: Start the engine loop here.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as e:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        self._loop = loop
        self._cond = asyncio.Condition()

        # Capture cuda graph.
        self.enable_cuda_graph = enable_cuda_graph
        if enable_cuda_graph:

            # Initialize attention state.
            context.initialize_attention_state()
            assert context.is_decode_only(), "Decode-only required for cuda graph capture."

            # Get flat tokens, position ids.
            input_ids = context.current_input_ids()
            position_ids = context.current_position_ids()

            # Forward pass -> logits.
            with torch.inference_mode():
                logits = controller.inference_wrapped_model.run_one_forward_step(
                    {"tokens": input_ids, "position_ids": position_ids, "attention_mask": None}
                )
                create_cudagraphs()
                context.reset()  # todo: @lmcafee, remove if unnecessary.

    async def _notify_cond_for_new_request(self):
        """Helper function to notify condition variable when a new request is added."""
        async with self._cond:
            self._cond.notify_all()

    def has_unfinished_requests(self) -> bool:
        """Test if context contains unfinished requests."""
        return self.context.has_unfinished_requests() or len(self.waiting_request_ids) > 0

    def reset(self) -> None:
        """Reset by removing all requests and reset all state."""
        self.context.reset()
        self.waiting_request_ids.clear()
        self.finished_request_count = 0

    def add_request(
        self,
        request_id: int,
        prompt: Union[str, List[int], Tensor],
        num_tokens_to_generate: Optional[int] = None,
    ) -> asyncio.Future[DynamicInferenceRequest]:
        """Add request to inference context.

        Args:
            request_id (int): Unique ID of request.
            prompt (Union[str, Tensor]): Prompt as either a text string or token IDs.
            num_tokens_to_generate (Optional[int]): Number of output tokens to generate

        Return:
            Returns an asyncio `Future[DynamicInferenceRequest]` for the user to wait on.
        """

        # Tokenize prompt if text.
        if isinstance(prompt, str):
            # Tokenize prompt if text.
            tokens = torch.tensor(
                self.controller.tokenize_prompt(prompt),
                dtype=torch.int64,
                device=torch.cuda.current_device(),
            )
        elif isinstance(prompt, list):
            # Convert List[int] -> Tensor.
            tokens = torch.tensor(prompt, dtype=torch.int64, device=torch.cuda.current_device())
        elif isinstance(prompt, torch.Tensor):
            # Prompt already tokenized.
            assert prompt.dtype == torch.int64, prompt.dtype
            assert prompt.device == torch.device(
                f"cuda:{torch.cuda.current_device()}"
            ), prompt.device
            tokens = prompt

        else:
            raise Exception("specialize for <%s>." % type(prompt).__name__)

        self.requests[request_id] = DynamicInferenceRequest(
            request_id=request_id,
            prompt_tokens=tokens,
            sampling_params=SamplingParams(num_tokens_to_generate=num_tokens_to_generate),
        )
        try:
            # Add request to context.
            self.context.add_request(request_id, tokens, num_tokens_to_generate)
            self._loop.call_soon_threadsafe(
                asyncio.create_task, self._notify_cond_for_new_request()
            )
        except (TokenOverflowError, RequestOverflowError, ChunkOverflowError) as e:
            self.waiting_request_ids.append(request_id)
        except MaxSequenceLengthOverflowError as e:
            raise e

        # Create a new asyncio Future to notify the user when the request has completed.
        self.request_completion_futures[request_id] = asyncio.Future()
        return self.request_completion_futures[request_id]

    def post_process_requests(
        self, request_ids: torch.Tensor, finished_request_ids: torch.Tensor, sample: torch.Tensor
    ) -> List[DynamicInferenceRequest]:
        """
        Handles post-processing for requests after a step.

        Args:
            request_ids (torch.Tensor): A list of request_ids
            finished_request_ids (torch.Tensor): A list of finished request ids
            sample: (torch.Tensor): The newly generated tokens for each request

        Returns:
            A list of completed requests as `DynamicInferenceRequest` objects
        """
        finished_requests: List[DynamicInferenceRequest] = []
        finished_request_ids = set(finished_request_ids.tolist())
        self.finished_request_count += len(finished_request_ids)

        for request_id, token in zip(request_ids.tolist(), sample.tolist()):
            request: DynamicInferenceRequest = self.requests[request_id]
            request.generated_tokens.append(token)

            if request_id in finished_request_ids:
                request.generated_length = len(request.generated_tokens)
                request.status = Status.COMPLETED
                finished_request = self.requests.pop(request_id)
                finished_request.generated_length = len(finished_request.generated_tokens)
                finished_requests.append(finished_request)
                finished_request.generated_text = self.controller.tokenizer.detokenize(
                    finished_request.generated_tokens
                )
                self.request_completion_futures[request_id].set_result(finished_request)

        return finished_requests

    async def async_step(
        self, sampling_params: SamplingParams, *, verbose: Optional[bool] = False
    ) -> Tuple[List[DynamicInferenceRequest], float]:
        """Wrapper for controller.generate_output_tokens_dynamic_batch(), to
        match vLLM API.

        Uses `asyncio` for continuous generation which allows this
        method to sleep and wake up when new requests are available.
        """

        # Generate tokens.
        t = time.time()
        is_decode_only = self.context.is_decode_only()
        result = self.controller.generate_output_tokens_dynamic_batch(
            sampling_params, self.termination_id
        )
        step_time = time.time() - t

        finished_requests: List[DynamicInferenceRequest] = []

        if result is not None:
            request_ids, finished_request_ids, sample = result

            # TODO: Move this to a background thread?
            finished_requests.extend(
                self.post_process_requests(request_ids, finished_request_ids, sample)
            )

            # Schedule waiting requests
            # TODO: Move this to a background thread?
            for waiting_request_id in self.waiting_request_ids.copy():
                waiting_request: DynamicInferenceRequest = self.requests[waiting_request_id]
                try:
                    self.context.add_request(
                        waiting_request_id,
                        waiting_request.prompt_tokens,
                        waiting_request.sampling_params.num_tokens_to_generate,
                    )
                    self.waiting_request_ids.popleft()
                except Exception as e:
                    break

        # Print context state.
        if verbose:
            context = self.context
            mem = torch.cuda.memory_stats()
            print(
                "* step ... time: %.3f%s ... "
                "reqs: %d [ gtd %d, active %d, paused %d, finished %d ] ... "
                "mem: tensors %d, alloc %.1f gb, res %.1f gb."
                % (
                    step_time,
                    (
                        (" [decode + cuda graph %s]" % ("ON" if self.enable_cuda_graph else "OFF"))
                        if is_decode_only
                        else "[prefill]"
                    ),
                    context.total_request_count,
                    context.gtd_request_count,
                    context.total_request_count - context.paused_request_count,
                    context.paused_request_count,
                    self.finished_request_count,
                    mem["allocation.all.current"],
                    mem["allocated_bytes.all.current"] / (1024**3),
                    mem["reserved_bytes.all.current"] / (1024**3),
                )
            )

        return finished_requests, step_time

    def step(self, sampling_params: SamplingParams, *, verbose: Optional[bool] = False):
        """Synchronous wrapper for `self.async_step`."""
        return self._loop.run_until_complete(
            self.async_step(sampling_params=sampling_params, verbose=verbose)
        )

    def generate(
        self, prompts: List[str], sampling_params: Optional[SamplingParams] = SamplingParams()
    ) -> List[DynamicInferenceRequest]:
        """Generates completions for a static list of prompts."""

        for prompt in prompts:
            request_id = int(next(self.request_counter))
            _ = self.add_request(request_id, prompt, sampling_params.num_tokens_to_generate)

        finished_requests_list = []
        while self.has_unfinished_requests():
            finished_requests, step_time = self.step(sampling_params)
            finished_requests_list.extend(finished_requests)

        return finished_requests_list

    async def run_engine(self, sampling_params: SamplingParams, *, verbose: Optional[bool] = False):
        """Continually steps the engine asynchronously."""
        try:
            while True:
                # Wait until there are active requests before proceeding.
                async with self._cond:
                    await self._cond.wait_for(lambda: self.context.get_active_request_count() > 0)

                await self.async_step(sampling_params=sampling_params, verbose=verbose)
        except asyncio.CancelledError:
            pass
