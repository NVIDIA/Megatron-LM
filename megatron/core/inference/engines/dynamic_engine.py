# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import logging
import multiprocessing
import os
import struct
import time
import warnings
from collections import deque
from datetime import datetime
from itertools import repeat
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.cuda.nvtx import range_pop, range_push

from megatron.core import parallel_state
from megatron.core.inference.contexts.dynamic_context import (
    DynamicInferenceContext,
    MaxSequenceLengthOverflowError,
    TokenOverflowError,
    WarmupEngineMode,
)
from megatron.core.inference.data_parallel_inference_coordinator import (
    DataParallelInferenceCoordinator,
)
from megatron.core.inference.engines.abstract_engine import AbstractEngine
from megatron.core.inference.headers import Headers
from megatron.core.inference.inference_request import DynamicInferenceRequest, Status
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.inference.utils import Counter
from megatron.core.utils import get_asyncio_loop

try:
    from tqdm import tqdm

    HAVE_TQDM = True
except:
    HAVE_TQDM = False

try:
    import zmq

    HAVE_ZMQ = True
except:
    HAVE_ZMQ = False

try:
    import msgpack

    HAVE_MSGPACK = True
except:
    HAVE_MSGPACK = False


def format_mem_bytes(mem_bytes):
    """Convert a byte count to a human-readable string in tb, gb, mb, kb, or bytes."""
    for power, suffix in [(4, "tb"), (3, "gb"), (2, "mb"), (1, "kb"), (0, "bytes")]:
        suffix_bytes = 1024**power
        if mem_bytes >= suffix_bytes:
            return "%.1f %s" % (mem_bytes / suffix_bytes, suffix)
    return "%d bytes" % mem_bytes


# pylint: disable=line-too-long
class DynamicInferenceEngine(AbstractEngine):
    """The dynamic inference engine.

    This engine allows requests of varying length to be dynamically added and
    removed in each inference step. In contrast to the static engine that has a
    set batch size and sequence length during the forward pass, each request in
    the dynamic engine can have different *current* prompt and output length at
    any given step, and the processing is restricted only by a max number of total
    tokens across all requests.

    Args:
        text_generation_controller (TextGenerationController): A text generation
            controller that will be used to define how to preprocess prompts, generate
            outputs and detokenizer the output tokens.
        inference_context (DynamicInferenceContext): Context for managing in-flight
            batching and a dynamic block-level KV cache (similar to paged attention).
        random_seed (Optional[int]): Use a random seed if you want deterministic
            results. Defaults to None.
        static_sampling (bool): If True, all requests are assumed to have the same
            sampling parameters. This avoids needing to loop through all requests and
            their sampling parameters every generation step, improving latency.
    """

    def __init__(
        self,
        controller: TextGenerationController,
        context: DynamicInferenceContext,
        enable_cuda_graph: Optional[bool] = None,
        random_seed: Optional[int] = None,
        *,
        track_paused_request_events: bool = False,
        enable_chunked_prefill: bool = True,
        static_sampling: bool = False,
    ):

        if enable_cuda_graph is not None:
            warnings.warn(
                "The `enable_cuda_graph` argument is deprecated and will be "
                "removed in `megatron-core 0.15`. `enable_cuda_graph` is now "
                "read directly from the transformer config object."
            )

        assert isinstance(
            controller, TextGenerationController
        ), f"controller must be a TextGenerationController, got {type(controller)}"
        assert isinstance(
            context, DynamicInferenceContext
        ), f"context must be a DynamicInferenceContext, got {type(context)}"
        assert isinstance(random_seed, int), f"random_seed must be an int, got {type(random_seed)}"

        self.request_counter = Counter()
        self.controller = controller
        self.context = context
        self.random_seed = random_seed
        self.track_paused_request_events = track_paused_request_events
        self.step_count = 0
        self.finished_request_count = 0
        self.waiting_request_ids = deque()
        self.failed_request_ids = []  # deque()
        self.request_counter = Counter()
        self.requests: Dict[int, DynamicInferenceRequest] = {}
        self.request_completion_futures: Dict[int, asyncio.Future] = {}
        self.step_start_event = torch.cuda.Event(enable_timing=True)
        self.step_end_event = torch.cuda.Event(enable_timing=True)
        self.paused = False
        self.stopped = False
        self.enable_chunked_prefill = enable_chunked_prefill
        self.static_sampling = static_sampling

        # Initialize the asyncio loop if it has not already been initialized.
        # TODO: Start the engine loop here.
        self._loop = get_asyncio_loop()
        self._cond = asyncio.Condition()

        # Capture cuda graph.
        self.capture_stats = None

        if enable_cuda_graph is not None:
            self.cuda_graph_impl = "local" if enable_cuda_graph else "none"
        else:
            self.cuda_graph_impl = controller.inference_wrapped_model.model.config.cuda_graph_impl

        if self.cuda_graph_impl == "local":
            self.create_cuda_graphs()

    def create_cuda_graphs(self, reset_context: bool = True):
        """Create cuda graphs.

        This method iterates the dynamic context's `cuda_graph_request_counts`
        to record and capture cuda graphs.

        Args:
            reset_context (bool): Whether to reset the context after building cuda graphs.
        """
        context = self.context
        controller = self.controller

        time_start = time.time()
        mem_stats_start = torch.cuda.memory_stats()

        logging.info(
            "> dynamic_engine.py: building cuda graphs for %d batch size(s): %s.",
            len(context.cuda_graph_token_counts),
            context.cuda_graph_token_counts,
        )
        for warmup_engine_mode in [WarmupEngineMode.DECODE, WarmupEngineMode.NON_DECODE]:
            # Iterate cuda graph dims.
            if (
                warmup_engine_mode == WarmupEngineMode.NON_DECODE
                and not context.non_decode_cuda_graphs
            ):
                continue
            tbar = enumerate(context.cuda_graph_token_counts)
            if HAVE_TQDM:
                tbar = tqdm(tbar, total=len(context.cuda_graph_token_counts))
            for tbar_idx, cuda_graph_token_count in tbar:
                if (
                    cuda_graph_token_count == 1
                    and warmup_engine_mode == WarmupEngineMode.NON_DECODE
                ):
                    # This case is not supported`` as we require atleast two
                    # tokens for a non-decode engine step.
                    continue

                # Initialize context.
                input_ids, position_ids = self.controller._dynamic_step_context_init(
                    num_warmup_tokens=cuda_graph_token_count, warmup_engine_mode=warmup_engine_mode
                )

                # Initialize attention state.
                assert (
                    cuda_graph_token_count == context.padded_active_token_count
                ), f"{cuda_graph_token_count} vs. {context.padded_active_token_count}."

                # Progress.
                mode_str = warmup_engine_mode.name.lower()
                tbar_str = f"cuda graph warmup - {mode_str}, d {cuda_graph_token_count}"
                if HAVE_TQDM:
                    tbar.set_description(tbar_str)
                else:
                    logging.info(f"{tbar_idx}/{len(context.cuda_graph_token_counts)}. {tbar_str}")

                # Forward pass -> logits.
                controller._dynamic_step_forward_logits(input_ids, position_ids)

                if reset_context:
                    with torch.inference_mode():
                        context.reset()  # todo: @lmcafee, remove if unnecessary.

        # Memory usage.
        time_end = time.time()
        mem_stats_end = torch.cuda.memory_stats()
        capture_stats = {
            "time": time_end - time_start,
            "allocated_bytes": (
                mem_stats_end["allocated_bytes.all.current"]
                - mem_stats_start["allocated_bytes.all.current"]
            ),
            "reserved_bytes": (
                mem_stats_end["reserved_bytes.all.current"]
                - mem_stats_start["reserved_bytes.all.current"]
            ),
        }
        logging.info(
            "> built cuda graph(s) in %.2f sec, with total memory usage: "
            "allocated %s, reserved %s.",
            capture_stats["time"],
            format_mem_bytes(capture_stats["allocated_bytes"]),
            format_mem_bytes(capture_stats["reserved_bytes"]),
        )

        self.capture_stats = capture_stats

    async def start_listening_to_data_parallel_coordinator(
        self, inference_coordinator_port: int, launch_inference_coordinator: bool = True
    ):
        """Initializes ZMQ communication to connect the engine with an inference coordinator.

        This asynchronous method sets up the distributed communication infrastructure
        that allows this inference engine to act as a worker under a central
        `InferenceCoordinator`. It configures different ZMQ socket patterns
        based on the rank's role within the distributed topology.

        The setup involves two primary roles within each data-parallel group:
        1.  **TP Coordinator (TP_rank=0, PP_rank=0)**: This rank connects directly
            to the central coordinator via a ZMQ `DEALER` socket. It receives
            requests and uses a ZMQ `PUB` (publisher) socket to broadcast them
            to all other ranks within its tensor-parallel (TP) group.
        2.  **TP Workers (all other ranks)**: These ranks use ZMQ `SUB` (subscriber)
            sockets to listen for requests broadcast by their local TP Coordinator.

        This architecture uses fast Inter-Process Communication (`ipc`) sockets for
        intra-node broadcasts within a TP group.

        Finally, after setting up the communication channels and ensuring all ranks
        are synchronized, this method starts the main engine processing loop
        (`self.run_engine`) as a background asyncio task.

        Args:
            inference_coordinator_port (int): The network port where the central
                `InferenceCoordinator` is or will be listening.
            launch_inference_coordinator (bool, optional): If True, the global rank 0
                process will spawn and manage the `InferenceCoordinator`
                process. Defaults to True.

        Note:
            The current implementation uses `ipc` sockets for broadcasting requests
            within a Tensor Parallel group, which limits each TP group to a single
            physical node. For example, if you have 8 GPUs per node, then this will only
            work with TP=[1,2,4,8]
        """

        assert HAVE_ZMQ, (
            "please install the pyzmq library to use InferenceCoordinator\n" "pip install pyzmq"
        )
        assert HAVE_MSGPACK, (
            "please install the messagepack library to use InferenceCoordinator\n"
            "pip install msgpack"
        )

        if launch_inference_coordinator and torch.distributed.get_rank() == 0:
            spawn_context = multiprocessing.get_context('spawn')
            coordinator_ready_event = spawn_context.Event()
            self.inference_coordinator_process = spawn_context.Process(
                target=DataParallelInferenceCoordinator.entrypoint,
                args=(
                    coordinator_ready_event,
                    inference_coordinator_port,
                    parallel_state.get_data_parallel_world_size(),
                ),
            )
            self.inference_coordinator_process.start()

        # Todo [Siddharth]: can we move this code to another file?
        self.zmq_context = zmq.Context()
        self.zmq_sockets = []  # keep track of all sockets created by this engine
        ip_address_of_dp_coordinator = os.getenv('MASTER_ADDR', '127.0.0.1')
        identity = f'tp-coord-{parallel_state.get_data_parallel_rank()}'
        if (
            parallel_state.get_tensor_model_parallel_rank() == 0
            and parallel_state.get_pipeline_model_parallel_rank() == 0
        ):
            # 1. Create dealer sockets where tp_rank = 0 and pp_rank = 0
            #    These will receive requests from an InferenceCoordinator.
            self.socket_for_receiving_requests = self.zmq_context.socket(zmq.DEALER)

            self.socket_for_receiving_requests.setsockopt(zmq.IDENTITY, identity.encode('utf-8'))
            self.socket_for_receiving_requests.connect(
                f"tcp://{ip_address_of_dp_coordinator}:{inference_coordinator_port}"
            )

            # send empty string. this is used to register with the coordinator.
            self.socket_for_receiving_requests.send(b"")

            # 2. Create a publisher socket. This is used to publish or broadcast
            #    requests within the tensor parallel group
            self.tensor_parallel_publisher_socket = self.zmq_context.socket(zmq.PUB)
            self.tensor_parallel_publisher_socket.bind(f"ipc:///tmp/{identity}-tp-bcast-socket-req")

            # 3. Create another publisher socket to broadcast the number of messages to receive.
            self.tensor_parallel_num_msgs_publisher_socket = self.zmq_context.socket(zmq.PUB)
            self.tensor_parallel_num_msgs_publisher_socket.bind(
                f"ipc:///tmp/{identity}-tp-bcast-socket-len"
            )
            self.zmq_sockets += [
                self.socket_for_receiving_requests,
                self.tensor_parallel_num_msgs_publisher_socket,
                self.tensor_parallel_publisher_socket,
            ]
        # All TP ranks subscribe to the two publisher sockets
        self.tensor_parallel_subscriber_socket = self.zmq_context.socket(zmq.SUB)
        self.tensor_parallel_subscriber_socket.connect(f"ipc:///tmp/{identity}-tp-bcast-socket-req")
        self.tensor_parallel_subscriber_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        self.tensor_parallel_num_msgs_subscriber_socket = self.zmq_context.socket(zmq.SUB)
        self.tensor_parallel_num_msgs_subscriber_socket.connect(
            f"ipc:///tmp/{identity}-tp-bcast-socket-len"
        )
        self.tensor_parallel_num_msgs_subscriber_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        self.zmq_sockets += [
            self.tensor_parallel_subscriber_socket,
            self.tensor_parallel_num_msgs_subscriber_socket,
        ]

        torch.distributed.barrier(parallel_state.get_tensor_model_parallel_group())

        if launch_inference_coordinator and torch.distributed.get_rank() == 0:
            coordinator_ready_event.wait()
            logging.info("Inference co-ordinator is ready to receive requests!")

        # Finally run the engine infinite loop
        self.engine_loop_task = asyncio.create_task(self.run_engine_with_coordinator())

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
        self.step_count = 0
        self.finished_request_count = 0

    def _add_request(
        self, request: DynamicInferenceRequest
    ) -> asyncio.Future[DynamicInferenceRequest]:

        request_id = request.request_id
        self.requests[request_id] = request
        if request.status is None:
            request.status = Status.ACTIVE_AND_GENERATING_TOKENS

        assert (
            request.sampling_params.num_tokens_to_generate is None
            or request.sampling_params.num_tokens_total is None
        )
        if request.sampling_params.num_tokens_total is not None:
            request.sampling_params.num_tokens_to_generate = (
                request.sampling_params.num_tokens_total - len(request.prompt_tokens)
            )
        if request.sampling_params.num_tokens_to_generate is None:
            request.sampling_params.num_tokens_to_generate = self.context.max_sequence_length - len(
                request.prompt_tokens
            )

        if (
            len(request.prompt_tokens) + request.sampling_params.num_tokens_to_generate
            > self.context.max_sequence_length
        ):
            request.status = Status.FAILED
            request.add_event_error_nontransient(MaxSequenceLengthOverflowError(request_id))

        if len(request.prompt_tokens) > self.context.max_tokens and not self.enable_chunked_prefill:
            request.status = Status.FAILED
            request.add_event_error_nontransient(TokenOverflowError(request_id))

        if request.status != Status.FAILED:
            self.waiting_request_ids.append(request_id)

        # Create a new asyncio Future to notify the user when the request has completed.
        self.request_completion_futures[request_id] = asyncio.Future()
        return self.request_completion_futures[request_id]

    def add_request(
        self,
        request_id: int,
        prompt: Union[str, List[int], Tensor],
        sampling_params: Optional[SamplingParams] = None,
    ) -> asyncio.Future[DynamicInferenceRequest]:
        """Add request to inference context.

        Args:
            request_id (int): Unique ID of request.
            prompt (Union[str, Tensor]): Prompt as either a text string or token IDs.
            sampling_params (Optional[SamplingParams]): Sampling parameters for the request.

        Return:
            Returns an asyncio `Future[DynamicInferenceRequest]` for the user to wait on.
        """

        prompt_str = None
        # Tokenize prompt if text.
        if isinstance(prompt, str):
            # Tokenize prompt if text. Support legacy single-arg mocks.
            prompt_str = prompt
            try:
                prompt_token_ids = self.controller.tokenize_prompt(prompt, sampling_params.add_BOS)
            except TypeError:
                prompt_token_ids = self.controller.tokenize_prompt(prompt)
            tokens = torch.tensor(
                prompt_token_ids, dtype=torch.int64, device=torch.cuda.current_device()
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

        # Initialize request.
        request = DynamicInferenceRequest(
            prompt=prompt_str,
            request_id=request_id,
            prompt_tokens=tokens,
            sampling_params=sampling_params,
        )

        # Add request.
        return self._add_request(request)

    def post_process_requests(
        self,
        request_ids: torch.Tensor,
        finished_request_ids: torch.Tensor,
        step_time: float,
        sample: torch.Tensor,
        log_probs: torch.Tensor,
    ) -> Tuple[List[DynamicInferenceRequest], List[DynamicInferenceRequest]]:
        """
        Handles post-processing for requests after a step.

        Args:
            request_ids (torch.Tensor): A list of request_ids
            finished_request_ids (torch.Tensor): A list of finished request ids
            step_time (float): The latency of the last step
            sample: (torch.Tensor): The newly generated tokens for each request
            log_probs: (List): Log probs for each request

        Returns:
            A list of active requests and completed requests as `DynamicInferenceRequest` objects
        """
        active_requests: List[DynamicInferenceRequest] = []
        finished_requests: List[DynamicInferenceRequest] = []
        finished_request_ids = set(finished_request_ids.tolist())
        self.finished_request_count += len(finished_request_ids)

        log_probs_iter = log_probs if log_probs else repeat(None)

        for request_id, token, request_log_probs in zip(
            request_ids.tolist(), sample.tolist(), log_probs_iter
        ):
            request: DynamicInferenceRequest = self.requests[request_id]
            if request_id != self.context.chunked_prefill_request_id:
                request.generated_tokens.append(token)
                if request.tpot is None:
                    request.tpot = []
                request.tpot.append(step_time)

                if request_log_probs is not None:
                    if not request.prompt_log_probs:
                        request.prompt_log_probs = []
                    if not request.generated_log_probs:
                        request.generated_log_probs = []
                    # If the request log probs span > 1 token we are in prefill
                    if len(request_log_probs) > 1:
                        request.prompt_log_probs.extend(request_log_probs)
                    else:
                        if (
                            # If it is a chunked prefill request
                            len(request.prompt_log_probs) > 0
                            # And we are missing the last token for prefill
                            and len(request.prompt_log_probs) < len(request.prompt_tokens)
                            # And we need to track full prefill
                            and not self.context.materialize_only_last_token_logits
                        ):
                            assert (
                                len(request.prompt_log_probs) == len(request.prompt_tokens) - 1
                            ), "Prompt log probs length is not equal to prompt tokens length - 1"
                            request.prompt_log_probs.extend(request_log_probs)
                        else:
                            request.generated_log_probs.extend(request_log_probs)

                if request_id in finished_request_ids:
                    request.generated_length = len(request.generated_tokens)
                    request.status = Status.COMPLETED
                    finished_request = self.requests.pop(request_id)
                    if finished_request.prompt is None:
                        finished_request.prompt = self.controller.tokenizer.detokenize(
                            finished_request.prompt_tokens.tolist()
                        )
                    finished_request.generated_length = len(finished_request.generated_tokens)
                    finished_requests.append(finished_request)
                    finished_request.generated_text = self.controller.tokenizer.detokenize(
                        finished_request.generated_tokens
                    )
                    self.request_completion_futures[request_id].set_result(finished_request)
                else:
                    active_requests.append(request)
            else:
                # The chunked prefill produces useless tokens
                # so we are not appending them to the generated tokens.
                # Additionally, chunked prefill request do not finish.
                # However, the log probs are still needed.
                if request_log_probs is not None:
                    if self.context.materialize_only_last_token_logits:
                        # Here we discard intermediate log probs
                        # as we only materialize the last token log probs
                        request.prompt_log_probs = []
                        request.generated_log_probs = []
                    else:
                        # Otherwise, we gather log probs for all tokens
                        if not request.prompt_log_probs:
                            request.prompt_log_probs = []
                        request.prompt_log_probs.extend(request_log_probs)
                        request.generated_log_probs = []
                    active_requests.append(request)

        return active_requests, finished_requests

    def schedule_waiting_requests(self):
        """Tries to schedule any requests in the waiting pool."""
        if self.enable_chunked_prefill:
            self.schedule_chunked_prefill()
        else:
            self.schedule_non_chunked_prefill()

    def schedule_non_chunked_prefill(self):
        """
        Perform the same original scheduling logic for non-chunked runs
        """
        while self.waiting_request_ids:
            req = self.requests[self.waiting_request_ids[0]]
            request_can_be_added, request_tokens_can_be_added, kv_cache_available = (
                self.context.check_availability(req, safe=True)
            )
            if request_can_be_added and request_tokens_can_be_added and kv_cache_available:
                self.context.add_request(req)
                self._loop.call_soon_threadsafe(
                    asyncio.create_task, self._notify_cond_for_new_request()
                )
                req.remaining_prompt_tokens = req.remaining_prompt_tokens.new_empty(0)
                req.add_event_add()
                self.waiting_request_ids.popleft()
            else:
                break

    def get_active_sampling_map(self) -> List[Tuple[SamplingParams, List[int]]]:
        """Gets a map of sampling methods to active requests indices in the context."""
        # Get all active request IDs.
        active_request_ids = self.context.request_ids[
            self.context.paused_request_count : self.context.total_request_count
        ].tolist()
        if self.static_sampling:
            return [(next(iter(self.requests.values())).sampling_params, active_request_ids)]

        # Get a map from request_id to context array index.
        context_id_map = {r: i for i, r in enumerate(active_request_ids)}

        # Create map of sampling methods to context array indices.
        sampling_map: List[Tuple[SamplingParams, List[int]]] = []
        for request_id, request in self.requests.items():
            if request_id not in context_id_map:
                continue
            context_id = context_id_map[request_id]
            sp = request.sampling_params

            # Look for a pre-existing group with these sampling parameters.
            for sampling, indices in sampling_map:
                if sampling == sp:
                    indices.append(context_id)
                    break
            # If no group exists, create a new one.
            else:
                sampling_map.append((sp, [context_id]))

        return sampling_map

    def schedule_chunked_prefill(self):
        """
        This function schedules chunked prefill requests.
        Invariant:
            - There are at most one chunked prefill request in the waiting pool,
                which should be the head
            - There are at most one chunked prefill request in the context,
                which should be the last active request
            - context.chunked_prefill_request_id == -1 if no chunked prefill request is scheduled,
                otherwise it is the request id of the chunked prefill request
            - For each request, finished_chunk_token_count is the number of tokens
                that have been prefilled for this request, non-zero means
                it is during a chunked prefill
            - For each request, remaining_prompt_tokens holds the **unprefilled** prompt tokens
        """
        can_schedule = True
        while self.waiting_request_ids and can_schedule:
            can_schedule = False
            req = self.requests[self.waiting_request_ids[0]]

            # is_continuing_chunked_prefill is True if we are scheduling next
            # chunk of a existing chunked prefill request
            is_continuing_chunked_prefill = self.context.chunked_prefill_request_id > 0

            # Use remaining prompt tokens for scheduling decisions
            remaining_len = len(req.remaining_prompt_tokens)
            token_fully_can_be_added = (
                self.context.active_token_count + remaining_len <= self.context.max_tokens
            )
            token_partially_can_be_added = self.context.active_token_count < self.context.max_tokens
            request_can_be_added, _, kv_cache_available = self.context.check_availability(
                req, safe=not is_continuing_chunked_prefill
            )
            request_can_be_added = is_continuing_chunked_prefill or request_can_be_added

            if request_can_be_added and kv_cache_available:
                if token_fully_can_be_added:
                    self.context.chunked_prefill_request_id = -1
                    self.context.add_request(req)
                    self._loop.call_soon_threadsafe(
                        asyncio.create_task, self._notify_cond_for_new_request()
                    )
                    req.remaining_prompt_tokens = req.remaining_prompt_tokens.new_empty(0)
                    req.add_event_add()
                    # Fully scheduled, so we remove from waiting pool
                    self.waiting_request_ids.popleft()
                    # Only this case we keep checking the rest of the waiting queue
                    can_schedule = True
                elif token_partially_can_be_added:
                    chunk_length = self.context.max_tokens - self.context.active_token_count
                    self.context.add_request(req, chunk_length=chunk_length)
                    self._loop.call_soon_threadsafe(
                        asyncio.create_task, self._notify_cond_for_new_request()
                    )
                    self.context.chunked_prefill_request_id = req.request_id
                    req.remaining_prompt_tokens = req.remaining_prompt_tokens[chunk_length:]
                    req.finished_chunk_token_count += chunk_length
                    # Still have tokens to prefill, so we break and keep the
                    # chunked prefill request at the head of the waiting queue
                    # Note that we do not need to continue check the queue, as the tokens are full

    async def async_step(
        self, *, verbose: Optional[bool] = False
    ) -> Tuple[List[DynamicInferenceRequest], List[DynamicInferenceRequest], float]:
        """
        Wrapper for controller.generate_output_tokens_dynamic_batch(), to
        match vLLM API. Uses `asyncio` for continuous generation which allows this
        method to sleep and wake up when new requests are available.

        Args:
            sampling_params (SamplingParams): The sampling parameters.
            verbose (bool): Whether to run in verbose mode.

        Returns:
            A tuple comprised of:
                1. Requests that ran in the last step and are still active.
                2. Requests that ran in the last step and have now finished.
                3. The step time in seconds.
        """
        # schedule requests
        self.schedule_waiting_requests()

        # Previous context state, for printing output below.
        prev_is_decode_only = self.context.is_decode_only()
        prev_total_request_count = self.context.total_request_count
        prev_paused_request_count = self.context.paused_request_count
        prev_active_token_count = self.context.active_token_count

        range_push("Prefill" if not prev_is_decode_only else "Decode")

        # Generate tokens.
        is_decode_only = self.context.is_decode_only()
        # save the is_decode_only AFTER scheduling, BEFORE update
        self.is_decode_only = is_decode_only
        self.step_start_event.record()
        sampling_map = self.get_active_sampling_map()
        result = await self.controller.async_generate_output_tokens_dynamic_batch(sampling_map)
        self.step_end_event.record()
        self.step_end_event.synchronize()
        step_time = self.step_start_event.elapsed_time(self.step_end_event) / 1e3

        # Increment finished_request_count.
        cuda_graph_request_count = None
        if result is not None:
            active_request_ids = result["active_request_ids"]
            newly_paused_request_ids = result["newly_paused_request_ids"]
            finished_request_ids = result["finished_request_ids"]
            sample = result["sample"]
            log_probs = result["log_probs"]
            cuda_graph_request_count = result["cuda_graph_request_count"]

            # Add paused events.
            if newly_paused_request_ids is not None and self.track_paused_request_events:
                newly_paused_request_ids = newly_paused_request_ids.tolist()
                [self.requests[i].add_event_pause() for i in newly_paused_request_ids]

            # Mark requests finished.
            [self.requests[i].add_event_finish() for i in finished_request_ids.tolist()]

            # Add finished events.
            (active_requests, finished_requests) = self.post_process_requests(
                active_request_ids, finished_request_ids, step_time, sample, log_probs
            )

        else:
            active_requests: List[DynamicInferenceRequest] = []
            finished_requests: List[DynamicInferenceRequest] = []

        # Failed requests.
        for failed_request_id in self.failed_request_ids:
            failed_request = self.requests.pop(failed_request_id)
            failed_request.status = Status.FAILED
            failed_request.add_event_fail()
            finished_requests.append(failed_request)
            self.request_completion_futures[failed_request_id].set_result(failed_request)
        self.failed_request_ids.clear()

        # Print context state.
        if verbose:
            context = self.context
            mem = torch.cuda.memory_stats()
            step_type = "decode" if is_decode_only else "non-decode"
            output_str = (
                "* step %d | %s ... time: %.3f%s ... "
                "reqs: %d [ gtd %d, active %d, paused %d, finished %d ] ... "
                "mem: tensors %d, alloc %.1f gb, res %.1f gb."
                % (
                    self.step_count,
                    datetime.now().strftime("%H:%M:%S"),
                    step_time,
                    (
                        " [%s + cuda graph %s]"
                        % (
                            step_type,
                            (
                                "DIM %d:%d"
                                % (context.padded_active_token_count, prev_active_token_count)
                                if self.context.using_cuda_graph_this_step()
                                else "OFF"
                            ),
                        )
                    ),
                    prev_total_request_count,
                    context.gtd_request_count,
                    prev_total_request_count - prev_paused_request_count,
                    prev_paused_request_count,
                    self.finished_request_count,
                    mem["allocation.all.current"],
                    mem["allocated_bytes.all.current"] / (1024**3),
                    mem["reserved_bytes.all.current"] / (1024**3),
                )
            )
            if prev_is_decode_only:
                output_str = f"\033[94m{output_str}\033[0m"
            logging.info(output_str)

        self.step_count += 1

        range_pop()
        return {
            "active_requests": active_requests,
            "finished_requests": finished_requests,
            "step_time": step_time,
            "cuda_graph_request_count": cuda_graph_request_count,
        }

    def step_modern(
        self, *, verbose: Optional[bool] = False
    ) -> Tuple[List[DynamicInferenceRequest], List[DynamicInferenceRequest], float]:
        """Synchronous wrapper for `self.async_step`."""
        return self._loop.run_until_complete(self.async_step(verbose=verbose))

    def step_legacy(
        self, sampling_params: SamplingParams, *, verbose: Optional[bool] = False
    ) -> Tuple[List[DynamicInferenceRequest], List[DynamicInferenceRequest], float]:
        """Synchronous wrapper for `self.async_step`."""
        warnings.warn(
            "`step_legacy()` is deprecated and will be removed in `megatron-core` "
            "0.16. Please use `step_modern()` going forward, which will eventually "
            "be renamed to `step()`."
        )
        result = self._loop.run_until_complete(
            self.async_step(sampling_params=sampling_params, verbose=verbose)
        )
        return (result["active_requests"], result["finished_requests"], result["step_time"])

    # For backwards compatibility, point `step()` to `step_legacy()`. Starting in
    # `megatron-core` 0.16, `step_modern()` will be renamed to `step()`.
    step = step_legacy

    def generate(
        self, prompts: List[str], sampling_params: Optional[SamplingParams] = SamplingParams()
    ) -> List[DynamicInferenceRequest]:
        """Generates completions for a static list of prompts."""

        for prompt in prompts:
            request_id = int(next(self.request_counter))
            _ = self.add_request(request_id, prompt, sampling_params)

        finished_requests_list = []
        while self.has_unfinished_requests():
            result = self.step_modern()
            finished_requests_list.extend(result["finished_requests"])

        # Ensure requests are returned in the same order they were passed in.
        finished_requests_list.sort(key=lambda x: x.request_id)

        return finished_requests_list

    def schedule_requests(self) -> int:
        """Drains the ZMQ socket for a batch of requests and adds them to the engine.

        This method is a collective and synchronous operation that must be called
        by all ranks in a Tensor Parallel (TP) group at the same time. It ensures
        that all ranks process the exact same batch of incoming requests and
        control signals.

        The synchronization works as follows:
        1.  The TP rank 0 drains all pending messages from its subscriber socket
            in a non-blocking manner.
        2.  TP rank 0 then broadcasts the number of messages it received to all other
            ranks in its TP group using a dedicated publisher socket.
        3.  The other TP ranks wait to receive this count, and then receive exactly
            that many messages from their subscriber sockets.

        Once all ranks have the same batch of messages, they are unpacked and
        processed. New requests are added to the engine's queue, and control
        signals (PAUSE, STOP, UNPAUSE) update the engine's internal state.

        Note:
            This function is synchronous and must be called collectively by all
            ranks in a TP group. It should not be launched in a separate coroutine
            to ensure all ranks execute it in lockstep before proceeding to the
            next engine step.

        Returns:
            int: The number of messages that were received and processed in this batch.
        """

        rank = parallel_state.get_tensor_model_parallel_rank()
        torch.cuda.nvtx.range_push("drain_zmq_socket")
        all_messages = []
        if rank == 0:
            while True:
                try:
                    # Receive messages in a non-blocking way.
                    all_messages.append(self.socket_for_receiving_requests.recv(flags=zmq.NOBLOCK))
                except zmq.Again:
                    # This exception is hit as soon as the socket is empty.
                    break
            messages_to_dequeue = len(all_messages)
            # First publish the number of messages to dequeue.
            # This is important because we want all tensor parallel ranks
            # to dequeue the same number of messages.
            self.tensor_parallel_num_msgs_publisher_socket.send(
                struct.pack('!i', messages_to_dequeue)
            )
            # Now publish the actual messages to all tensor parallel ranks
            for message in all_messages:
                self.tensor_parallel_publisher_socket.send(message)
        else:
            # First, receive the number of messages to dequeue from tp-rank 0
            messages_to_dequeue = struct.unpack(
                '!i', self.tensor_parallel_num_msgs_subscriber_socket.recv()
            )[0]
            # Now, dequeue the same number of messages from the subscriber socket.
            # Note that these receives are blocking, because the messages
            # are guaranteed to be available after the tp-rank 0 has sent them.
            for _ in range(messages_to_dequeue):
                all_messages.append(self.tensor_parallel_subscriber_socket.recv())

        torch.cuda.nvtx.range_pop()
        for message in all_messages:
            data = msgpack.unpackb(message, raw=False)
            header = Headers(data[0])
            if header == Headers.SUBMIT_REQUEST:
                request_id, prompt, sampling_params = data[1:]
                sampling_params = SamplingParams.deserialize(sampling_params)
                self.add_request(request_id, prompt, sampling_params)
            elif header == Headers.PAUSE:
                self.paused = True
            elif header == Headers.STOP:
                self.stopped = True
            elif header == Headers.UNPAUSE:
                self.paused = False

        return len(all_messages)

    def stop(self):
        """
        Stops the inference engine by terminating the inference coordinator process
        if it exists, and destroys the model parallel state.
        This method ensures that any running inference coordinator subprocess
        is properly terminated, and cleans up resources associated with
        model parallelism.
        """

        if hasattr(self, "inference_coordinator_process"):
            self.inference_coordinator_process.terminate()
        for socket in self.zmq_sockets:
            socket.close()
        self.zmq_context.term()
        parallel_state.destroy_model_parallel()

    async def run_engine(self, *, verbose: Optional[bool] = False):
        """Continually steps the engine asynchronously."""
        try:
            while True:
                # Wait until there are active requests before proceeding.
                async with self._cond:
                    await self._cond.wait_for(
                        lambda: self.context.get_active_request_count() > 0
                        or self.waiting_request_ids
                    )

                await self.async_step(verbose=verbose)
        except asyncio.CancelledError:
            pass

    async def run_engine_with_coordinator(self, *, verbose: Optional[bool] = False):
        """Continually steps the engine asynchronously."""
        try:
            while True:
                self.schedule_requests()
                if self.stopped:
                    self.stop()
                    return

                # for the cases below (engine is paused or no active requests),
                # do not use asyncio.sleep(0)
                # as tp-rank=0 will flood the num_messages publisher
                # with "0" repeatedly. This causes some packets to drop.
                # Instead be nice, and sleep
                # for a short time.
                # The minimum sleep time needed is ~100us i.e. the time
                # needed to send one message on an IPC socket. However
                # just to be safe, we use 20ms here.

                # todo [Siddharth]: Can this hardcoded sleep be avoided
                # with asyncio zmq sockets?
                if self.paused:
                    await asyncio.sleep(0.02)
                    continue

                if (
                    self.context.get_active_request_count() == 0
                    and len(self.waiting_request_ids) == 0
                ):
                    await asyncio.sleep(0.02)
                    continue

                engine_output = await self.async_step(verbose=verbose)

                is_tp0_and_pp0 = (
                    parallel_state.get_tensor_model_parallel_rank() == 0
                    and parallel_state.get_pipeline_model_parallel_rank() == 0
                )
                if (
                    is_tp0_and_pp0
                    and engine_output is not None
                    and engine_output["finished_requests"]
                ):
                    payload = msgpack.packb(
                        [
                            Headers.ENGINE_REPLY.value,
                            [r.serializable() for r in engine_output["finished_requests"]],
                        ],
                        use_bin_type=True,
                    )
                    self.socket_for_receiving_requests.send(payload)

        except asyncio.CancelledError:
            pass
