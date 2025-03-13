# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import time
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor

from megatron.core.inference.contexts import DynamicInferenceContext
from megatron.core.inference.engines.abstract_engine import AbstractEngine
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.simple_text_generation_controller import (
    SimpleTextGenerationController,
)
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
        random_seed: int = None,
    ):

        assert isinstance(controller, SimpleTextGenerationController)
        assert isinstance(context, DynamicInferenceContext)
        assert isinstance(termination_id, int)
        assert isinstance(random_seed, int)

        self.controller = controller
        self.context = context
        self.termination_id = termination_id
        self.random_seed = random_seed
        self.finished_request_count = 0

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
            with torch.no_grad():
                logits = controller.inference_wrapped_model.run_one_forward_step(
                    {"tokens": input_ids, "position_ids": position_ids, "attention_mask": None}
                )
                create_cudagraphs()
                context.reset()  # todo: @lmcafee, remove if unnecessary.

    def has_unfinished_requests(self) -> bool:
        """Test if context contains unfinished requests."""
        return self.context.has_unfinished_requests()

    def reset(self) -> None:
        """Reset by removing all requests and reset all state."""
        self.context.reset()
        self.finished_request_count = 0

    def add_request(self, request_id: int, prompt: Union[str, List[int], Tensor]) -> None:
        """Add request to inference context.

        Args:
            request_id (int): Unique ID of request.
            prompt (Union[str, Tensor]): Prompt as either a text string or token IDs.

        Return:
            None.
        """

        # Tokenize prompt if text.
        if isinstance(prompt, str):
            tokens = torch.tensor(
                self.controller.tokenize_prompt(prompt),
                dtype=torch.int64,
                device=torch.cuda.current_device(),
            )

        # Convert List[int] -> Tensor.
        elif isinstance(prompt, list):
            tokens = torch.tensor(prompt, dtype=torch.int64, device=torch.cuda.current_device())

        # Prompt already tokenized.
        elif isinstance(prompt, torch.Tensor):
            assert prompt.dtype == torch.int64, prompt.dtype
            assert prompt.device == torch.device(
                f"cuda:{torch.cuda.current_device()}"
            ), prompt.device
            tokens = prompt

        else:
            raise Exception("specialize for <%s>." % type(prompt).__name__)

        # Add request to context.
        return self.context.add_request(request_id, tokens)

    def step(
        self, sampling_params: SamplingParams, *, verbose: Optional[bool] = False
    ) -> Optional[Tuple[Tensor, Tensor, Tensor]]:
        """Wrapper for controller.generate_output_tokens_dynamic_batch(), to
        match vLLM API.

        TODO: @lmcafee, use `asyncio` for continuous generation that allows this
        method to sleep and wake up when new requests are available.
        """

        # Generate tokens.
        t = time.time()
        is_decode_only = self.context.is_decode_only()
        result = self.controller.generate_output_tokens_dynamic_batch(
            sampling_params, self.termination_id
        )
        step_time = time.time() - t

        # Increment finished_request_count.
        if result is not None:
            self.finished_request_count += result[1].numel()

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

        return result, step_time

    def generate(self) -> dict:
        raise Exception("Dynamic engine does not support batch generation.")
