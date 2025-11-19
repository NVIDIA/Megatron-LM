# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import logging
import multiprocessing
import os
import socket
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
from megatron.core.inference.utils import Counter, await_process_event
from megatron.core.utils import get_asyncio_loop, trace_async_exceptions

try:
    from tqdm import tqdm

    HAVE_TQDM = True
except:
    HAVE_TQDM = False

try:
    import zmq
    import zmq.asyncio

    HAVE_ZMQ = True
except:
    HAVE_ZMQ = False

try:
    import msgpack

    HAVE_MSGPACK = True
except:
    HAVE_MSGPACK = False

try:
    import wandb

    HAVE_WANDB = True
except ImportError:
    HAVE_WANDB = False
    wandb = None


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
        inference_logging_step_interval (int): The step interval at which to log
        inference metrics to wandb. Defaults to 0, which means no logging.
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
        inference_logging_step_interval: int = 0,
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
        self.rank = torch.distributed.get_rank()

        self.inference_logging_step_interval = inference_logging_step_interval
        # Configure wandb to use separate step counter for inference metrics (only once)
        if self.inference_logging_step_interval > 0 and self.context.metrics_writer is not None:
            logging.info(
                f"\033[1;93m[INFERENCE]\033[0m "
                f"\033[1;95mLogging inference metrics to wandb (rank {self.rank})\033[0m"
            )
            if HAVE_WANDB and self.context.metrics_writer.__name__ == "wandb":
                # Make all inference/* metrics use inference_step as their x-axis
                # This allows inference and training to have independent step counters
                context.metrics_writer.define_metric(
                    "inference/*", step_metric="inference/inference_step"
                )
                # Initialize inference step offset by querying existing run history
                self.inference_step_offset = 0
                if wandb.run is not None:
                    api_run = wandb.Api().run(
                        f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}"
                    )
                    max_step = 0
                    for row in api_run.scan_history(keys=["inference/inference_step"]):
                        val = row.get("inference/inference_step")
                        if isinstance(val, (int, float)) and int(val) > max_step:
                            max_step = int(val)
                    self.inference_step_offset = int(max_step)

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

        config = controller.inference_wrapped_model.inference_wrapper_config
        moe_pad_experts = config.moe_pad_experts_for_cuda_graph_inference

        if moe_pad_experts and context.non_decode_cuda_graphs:
            context.non_decode_cuda_graphs = False
            if self.rank == 0:
                warnings.warn(
                    "MoE models do not support non-decode cuda graphs. "
                    "Forcing non_decode_cuda_graphs to False."
                )

        time_start = time.time()
        mem_stats_start = torch.cuda.memory_stats()

        logging.info(
            "> dynamic_engine.py: building cuda graphs for %d batch size(s): %s.",
            len(context.cuda_graph_token_counts),
            context.cuda_graph_token_counts,
        )
        for warmup_engine_mode in [WarmupEngineMode.DECODE, WarmupEngineMode.NON_DECODE]:
            # Check whether to skip non-decode graphs.
            if (
                warmup_engine_mode == WarmupEngineMode.NON_DECODE
                and not context.non_decode_cuda_graphs
            ):
                continue

            tbar = enumerate(context.cuda_graph_token_counts)
            if HAVE_TQDM:
                tbar = tqdm(tbar, total=len(context.cuda_graph_token_counts))

            # Iterate cuda graph dims.
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

    def start_listening_to_data_parallel_coordinator(
        self,
        inference_coordinator_port: int,
        launch_inference_coordinator: bool = True,
        *,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        microbatch_timeout: float = 0.001,
        steps_before_microbatch: int = 8,
    ):
        """Initializes ZMQ communication to connect the engine with an inference coordinator.

        This asynchronous method sets up the distributed communication infrastructure
        that allows this inference engine to act as a worker under a central
        `InferenceCoordinator`. It configures different ZMQ socket patterns
        based on the rank's role within the distributed topology.

        Note that this method must be called on all ranks, as it uses blocking torch broadcasts.

        The setup involves two primary roles within each data-parallel group:
        1.  **MP Coordinator (TP_rank=0, PP_rank=0)**: This rank connects directly
            to the central coordinator via a ZMQ `DEALER` socket. It receives
            requests and uses a ZMQ `PUB` (publisher) socket to broadcast them
            to all other ranks within its model-parallel (MP) group.
        2.  **MP Workers (all other ranks)**: These ranks use ZMQ `SUB` (subscriber)
            sockets to listen for requests broadcast by their local MP Coordinator.

        This architecture uses TCP sockets for both inter-node and intra-node broadcasts
        within an MP group.

        Finally, after setting up the communication channels and ensuring all ranks
        are synchronized, this method starts the main engine processing loop
        (`self.run_engine`) as a background asyncio task.

        Args:
            inference_coordinator_port (int): The network port where the central
                `InferenceCoordinator` is or will be listening.
            launch_inference_coordinator (bool, optional): If True, the global rank 0
                process will spawn and manage the `InferenceCoordinator`
                process. Defaults to True.
        """

        assert HAVE_ZMQ, (
            "please install the pyzmq library to use InferenceCoordinator\n" "pip install pyzmq"
        )
        assert HAVE_MSGPACK, (
            "please install the messagepack library to use InferenceCoordinator\n"
            "pip install msgpack"
        )
        loop = get_asyncio_loop(loop)
        self.use_coordinator = True
        self.launch_inference_coordinator = launch_inference_coordinator
        self.microbatch_timeout = microbatch_timeout
        self.steps_before_microbatch = steps_before_microbatch

        self.pending_microbatch = deque()
        self.microbatch_processing_event = asyncio.Event()
        self.microbatch_not_empty = asyncio.Event()
        self.coord_recv_lock = asyncio.Lock()
        self._send_awaitables = asyncio.Queue()

        self.zmq_context = zmq.asyncio.Context().instance()
        self.zmq_sockets = []  # keep track of all sockets created by this engine

        # Get world info.
        dp_group = parallel_state.get_data_parallel_group()
        dp_src = parallel_state.get_data_parallel_src_rank()
        dp_size = parallel_state.get_data_parallel_world_size()
        dp_rank = parallel_state.get_data_parallel_rank()

        mp_group = parallel_state.get_model_parallel_group()
        mp_src = parallel_state.get_model_parallel_src_rank()
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()

        self.is_mp_coordinator = tp_rank == 0 and pp_rank == 0
        self.is_dp_coordinator = (dp_rank == 0) and self.is_mp_coordinator

        # Get local IP.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as tmp_sock:
            tmp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            tmp_sock.connect(('<broadcast>', 0))
            local_ip = tmp_sock.getsockname()[0]
        del tmp_sock

        # Spawn a DP coordinator process and get the connection info.
        if launch_inference_coordinator and self.is_dp_coordinator:
            spawn_context = multiprocessing.get_context('spawn')
            self.coordinator_ready_event = spawn_context.Event()
            self.inference_coordinator_process = spawn_context.Process(
                target=DataParallelInferenceCoordinator.entrypoint,
                args=(
                    self.coordinator_ready_event,
                    inference_coordinator_port,
                    parallel_state.get_data_parallel_world_size(),
                ),
            )
            self.inference_coordinator_process.start()

        # Find available ports for MP and bind to them.
        if self.is_mp_coordinator:
            mp_req_sock = self.zmq_context.socket(zmq.PUB)
            mp_req_sock.bind_to_random_port(f"tcp://{local_ip}")
            mp_req_addr = [mp_req_sock.getsockopt_string(zmq.LAST_ENDPOINT)]

            mp_len_sock = self.zmq_context.socket(zmq.PUB)
            mp_len_sock.bind_to_random_port(f"tcp://{local_ip}")
            mp_len_addr = [mp_len_sock.getsockopt_string(zmq.LAST_ENDPOINT)]
        else:
            mp_req_addr = [None]
            mp_len_addr = [None]

        # Broadcast addresses to respective ranks.
        torch.distributed.broadcast_object_list(mp_req_addr, src=mp_src, group=mp_group)
        torch.distributed.broadcast_object_list(mp_len_addr, src=mp_src, group=mp_group)

        ip_address_of_dp_coordinator = os.getenv('MASTER_ADDR', '127.0.0.1')
        dp_addr = [f"tcp://{ip_address_of_dp_coordinator}:{inference_coordinator_port}"]
        identity = f'mp-coord-{dp_rank}'
        if self.is_mp_coordinator:
            # 1. Create dealer sockets where tp_rank = 0 and pp_rank = 0
            #    These will receive requests from an InferenceCoordinator.
            self.socket_for_receiving_requests = self.zmq_context.socket(zmq.DEALER)

            self.socket_for_receiving_requests.setsockopt(zmq.IDENTITY, identity.encode('utf-8'))
            self.socket_for_receiving_requests.connect(dp_addr[0])

            # send empty string. this is used to register with the coordinator.
            self._isend(self.socket_for_receiving_requests, Headers.CONNECT, b"")

            # 2. Create a publisher socket. This is used to publish or broadcast
            #    requests within the model parallel group
            self.model_parallel_publisher_socket = mp_req_sock

            self.zmq_sockets += [
                self.socket_for_receiving_requests,
                self.model_parallel_publisher_socket,
            ]
        # All MP ranks subscribe to the two publisher sockets
        self.model_parallel_subscriber_socket = self.zmq_context.socket(zmq.SUB)
        self.model_parallel_subscriber_socket.connect(mp_req_addr[0])
        self.model_parallel_subscriber_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        self.zmq_sockets += [self.model_parallel_subscriber_socket]

        print(f"[{self.rank}] Barriering after ZMQ setup...")
        torch.distributed.barrier(mp_group)
        print(f"[{self.rank}] ZMQ setup complete!")

        if launch_inference_coordinator and self.is_dp_coordinator:
            await await_process_event(coordinator_ready_event, self.inference_coordinator_process)
            logging.info("Inference co-ordinator is ready to receive requests!")

        # Finally run the engine infinite loop
        self.engine_loop_task = loop.create_task(self.run_engine_with_coordinator(loop=loop))

    @trace_async_exceptions
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
        if request.sampling_params.termination_id is None:
            try:
                eod = self.controller.tokenizer.eod
            except AttributeError:
                if self.rank == 0:
                    warnings.warn(
                        "Termination ID not specified, and tokenizer does not define eod."
                        "Defaulting to not using termination id."
                    )
                eod = -1
            request.sampling_params.termination_id = eod

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
        self.request_completion_futures[request_id] = self._loop.create_future()
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
                self.context.check_availability(req)
            )
            if request_can_be_added and request_tokens_can_be_added and kv_cache_available:
                self.context.add_request(req)
                self._loop.call_soon_threadsafe(
                    self._loop.create_task, self._notify_cond_for_new_request()
                )
                req.remaining_prompt_tokens = req.remaining_prompt_tokens.new_empty(0)
                req.add_event_add()
                self.waiting_request_ids.popleft()
            else:
                break

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
            is_continuing_chunked_prefill = self.context.chunked_prefill_request_id >= 0

            # Use remaining prompt tokens for scheduling decisions
            remaining_len = len(req.remaining_prompt_tokens)
            token_fully_can_be_added = (
                self.context.active_token_count + remaining_len <= self.context.max_tokens
            )
            token_partially_can_be_added = self.context.active_token_count < self.context.max_tokens
            request_can_be_added, _, kv_cache_available = self.context.check_availability(req)
            request_can_be_added = is_continuing_chunked_prefill or request_can_be_added

            if request_can_be_added and kv_cache_available:
                if token_fully_can_be_added:
                    self.context.chunked_prefill_request_id = -1
                    self.context.add_request(req)
                    self._loop.call_soon_threadsafe(
                        self._loop.create_task, self._notify_cond_for_new_request()
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
                        self._loop.create_task, self._notify_cond_for_new_request()
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
        result = await self.controller.async_generate_output_tokens_dynamic_batch()
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

        # Log KV cache utilization stats to W&B
        if (
            self.inference_logging_step_interval > 0
            and self.step_count > 0
            and self.step_count % self.inference_logging_step_interval == 0
            and self.context.metrics_writer is not None
        ):

            # Get KV cache utilization stats from dynamic context
            kv_stats = self.context.get_kvcache_utilization_stats()

            # Prepare metrics dictionary with all stats
            # Use 'inference/' prefix for all metrics to separate from training metrics
            metrics = {
                'inference/inference_step': int(self.inference_step_offset + int(self.step_count)),
                'inference/step_time_s': float(step_time),
                'inference/waiting_queue_len': int(len(self.waiting_request_ids)),
                'inference/total_requests_dict_size': int(len(self.requests)),
            }
            # Add KV stats with inference/ prefix
            # Convert utilization metrics from 0-1 range to 0-100 percentage range for better visualization
            for key, value in kv_stats.items():
                if 'utilization' in key:
                    # Convert to percentage (0-100) and group under kvcache_utilization
                    metrics[f'inference/{key}'] = float(value * 100.0)
                else:
                    metrics[f'inference/{key}'] = value

            if HAVE_WANDB and self.context.metrics_writer.__name__ == "wandb":
                self.context.metrics_writer.log(metrics, commit=True)
            else:
                raise ValueError(
                    f"Unsupported metrics writer type: {type(self.context.metrics_writer)}"
                )

        # Print context state.
        if verbose:
            context = self.context
            mem = torch.cuda.memory_stats()
            step_type = "decode" if is_decode_only else "non-decode"
            output_str = (
                "* step %d | %s ... time: %.3f%s ... "
                "reqs: a %d/%d, p %d/%d, w %d, f %d ... "
                "blocks: a %d/%d, p %d/%d ... "
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
                    prev_total_request_count - prev_paused_request_count,
                    context.block_allocator.active_count,
                    prev_paused_request_count,
                    context.block_allocator.paused_count,
                    len(self.waiting_request_ids),
                    self.finished_request_count,
                    context.block_allocator.get_active_used(),
                    context.block_allocator.active_count,
                    context.block_allocator.get_paused_used(),
                    context.block_allocator.paused_count,
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

        # Ensure requests are returned in the same order they were passed in
        finished_requests_list.sort(key=lambda x: x.request_id)

        return finished_requests_list

    async def schedule_requests(self):
        """Drains the ZMQ socket for a batch of requests and adds them to the engine.

        This method is a collective and synchronous operation that must be called
        by all ranks in a Model Parallel (MP) group at the same time. It ensures
        that all ranks process the exact same batch of incoming requests and
        control signals.

        The synchronization works as follows:
        1.  The MP rank 0 drains all pending messages from its subscriber socket
            in a non-blocking manner.
        2.  MP rank 0 then broadcasts the number of messages it received to all other
            ranks in its MP group using a dedicated publisher socket.
        3.  The other MP ranks wait to receive this count, and then receive exactly
            that many messages from their subscriber sockets.

        Once all ranks have the same batch of messages, they are unpacked and
        processed. New requests are added to the engine's queue, and control
        signals (PAUSE, STOP, UNPAUSE) update the engine's internal state.

        Note:
            This function is synchronous and must be called collectively by all
            ranks in a MP group. It should not be launched in a separate coroutine
            to ensure all ranks execute it in lockstep before proceeding to the
            next engine step.

        Returns:
            int: The number of messages that were received and processed in this batch.
        """
        # If the engine has active requests and is not at (step % N) == 0, return early.
        count_trigger = (self.step_count % self.steps_before_microbatch) == 0
        if self.engine.has_unfinished_requests() and (not count_trigger):
            return

        # Sleep for a small amount of time to allow communication to occur.
        if self.is_mp_coordinator:
            await asyncio.sleep(self.microbatch_timeout)

            # If there is no new data, stop stepping. Await new data.
            if not self.pending_microbatch and not self.engine.has_unfinished_requests():
                await self.microbatch_not_empty.wait()

        # MP coordinator sends a sync signal to all MP ranks.
        if self.is_mp_coordinator:
            async with self.coord_recv_lock:
                self._isend(self.model_parallel_publisher_socket, Headers.MICROBATCH_SYNC)
                # Reset the "new data" event; need to do this inside the lock.
                self.microbatch_not_empty.clear()

        # Await the receipt of the sync signal.
        await self.microbatch_processing_event.wait()

        while self.pending_microbatch:
            header, message = self.pending_microbatch.popleft()
            if header == Headers.SUBMIT_REQUEST:
                request_id, prompt, sampling_params = message
                sampling_params = SamplingParams.deserialize(sampling_params)
                self.add_request(request_id, prompt, sampling_params)
            elif header == Headers.PAUSE:
                self.paused = True
            elif header == Headers.STOP:
                self.stopped = True
            elif header == Headers.UNPAUSE:
                self.paused = False

        self.microbatch_processing_event.clear()

    @trace_async_exceptions
    async def _mp_coord_recv_task(self):
        """Listen for requests from the inference coordinator and forward them to the ranks."""
        while True:
            try:
                async with self.coord_recv_lock:
                    # Receive messages in a non-blocking way.
                    _, header, data = await self._irecv(
                        self.socket_for_receiving_requests, deserialize=False
                    )
                    # Now publish the actual messages to all tensor parallel ranks
                    self._isend(self.model_parallel_publisher_socket, header, data, serialize=False)
                    self.microbatch_not_empty.set()
            except asyncio.CancelledError:
                break

    @trace_async_exceptions
    async def _mp_rank_recv_task(self):
        """Listen for requests from the mp coordinator"""
        while True:
            try:
                _, header, message = await self._irecv(self.model_parallel_subscriber_socket)
                if header == Headers.MICROBATCH_SYNC:
                    # Set the microbatch processing signal.
                    self.microbatch_processing_event.set()
                else:
                    self.pending_microbatch.append((header, message))
            except asyncio.CancelledError:
                break

    @trace_async_exceptions
    async def _send_task(self):
        """Pop futures of sends out of a queue and await them.

        For explanation why this works, refer to the documentation for zmq.asyncio:
            'Returns a Future that resolves when sending is complete.'
        """
        while True:
            await (await self._send_awaitables.get())
            self._send_awaitables.task_done()

    def _isend(
        self,
        socket: zmq.asyncio.Socket,
        header: Headers,
        data: Optional[List] = None,
        serialize: bool = True,
    ) -> asyncio.Future:
        """
        Asynchronously send a signal to the inference coordinator.

        Args:
            socket (zmq.asyncio.Socket): The ZMQ socket to send the signal on.
            header (Headers): The signal header to send.
            data (Optional[List]): The data payload to send.
            serialize (bool): Whether to serialize the data using msgpack.
        """
        to_send = [header.value.to_bytes()]
        if data is not None:
            if serialize:
                data = msgpack.packb(data, use_bin_type=True)
            to_send.append(data)
        send_awaitable = socket.send_multipart(to_send)
        self._send_awaitables.put_nowait(send_awaitable)

    async def _irecv(
        self,
        socket: zmq.asyncio.Socket,
        socket_uses_identity: bool = False,
        deserialize: bool = True,
    ) -> Tuple[Optional[bytes], Headers, List | bytes | None]:
        """
        Asynchronously receive a signal from the inference coordinator.

        Returns:
            identity (Optional[bytes]): The source of the signal.
            header (Headers): The signal header received.
            data (List | bytes | None): The data payload received.
        """
        raw = await socket.recv_multipart()
        if socket_uses_identity:
            identity, header, *rest = raw
        else:
            header, *rest = raw
            identity = None

        header = Headers(int.from_bytes(header))
        data = rest[0] if rest else None

        if deserialize:
            message = msgpack.unpackb(data, raw=False) if data is not None else None
        else:
            message = data

        return identity, header, message

    def stop(self):
        """
        Stops the inference engine by terminating the inference coordinator process
        if it exists, and destroys the model parallel state.
        This method ensures that any running inference coordinator subprocess
        is properly terminated, and cleans up resources associated with
        model parallelism.
        """

        self.send_task.cancel()
        self.mp_rank_recv_task.cancel()
        if self.is_mp_coordinator:
            self.mp_coord_recv_task.cancel()
        if hasattr(self, "inference_coordinator_process"):
            self.inference_coordinator_process.terminate()
        for socket in self.zmq_sockets:
            socket.close()
        self.zmq_context.term()

    @trace_async_exceptions
    async def run_engine(
        self, *, loop: Optional[asyncio.AbstractEventLoop] = None, verbose: Optional[bool] = False
    ):
        """Continually steps the engine asynchronously."""
        self._loop = get_asyncio_loop(loop)
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

    @trace_async_exceptions
    async def run_engine_with_coordinator(
        self, *, loop: Optional[asyncio.AbstractEventLoop] = None, verbose: Optional[bool] = False
    ):
        """Continually steps the engine asynchronously."""
        self._loop = get_asyncio_loop(loop)
        self.send_task = loop.create_task(self._send_task())
        self.mp_rank_recv_task = loop.create_task(self._mp_rank_recv_task())

        if self.is_mp_coordinator:
            self.mp_coord_recv_task = loop.create_task(self._mp_coord_recv_task())

        if self.launch_inference_coordinator and self.is_dp_coordinator:
            print(f"[{self.rank}] Waiting for co-ordinator to be ready...")
            logging.info("Inference co-ordinator is ready to receive requests!")
            print(f"[{self.rank}] Co-ordinator is ready!")
            await await_process_event(self.coordinator_ready_event, self.inference_coordinator_process)

        try:
            while True:
                await self.schedule_requests()
                if self.stopped:
                    self.stop()
                    return

                engine_output = await self.async_step(verbose=verbose)

                if (
                    self.is_mp_coordinator
                    and engine_output is not None
                    and engine_output["finished_requests"]
                ):
                    self._isend(
                        self.socket_for_receiving_requests,
                        Headers.ENGINE_REPLY,
                        [r.serializable() for r in engine_output["finished_requests"]],
                    )

        except asyncio.CancelledError:
            pass
