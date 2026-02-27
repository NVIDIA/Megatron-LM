# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import concurrent.futures
import logging
import math
import multiprocessing
import socket
import struct
import time
import warnings
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from itertools import repeat
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.cuda.nvtx import range_pop, range_push

from megatron.core.inference.config import KVCacheManagementMode
from megatron.core.inference.contexts.dynamic_context import (
    DynamicInferenceContext,
    MaxSequenceLengthOverflowError,
    TokenOverflowError,
)
from megatron.core.inference.data_parallel_inference_coordinator import (
    DataParallelInferenceCoordinator,
)
from megatron.core.inference.engines.abstract_engine import AbstractEngine
from megatron.core.inference.headers import Headers, UnknownHeaderError
from megatron.core.inference.inference_request import (
    DynamicInferenceEvent,
    DynamicInferenceEventType,
    DynamicInferenceRequest,
    DynamicInferenceRequestRecord,
    Status,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.inference.utils import Counter, await_process_call
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.cuda_graphs import delete_cuda_graphs
from megatron.core.transformer.enums import CudaGraphScope
from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction
from megatron.core.utils import (
    deprecate_args,
    experimental_api,
    get_asyncio_loop,
    get_pg_rank,
    get_pg_size,
    get_pg_src_rank,
    internal_api,
    trace_async_exceptions,
)

from .async_zmq_communicator import AsyncZMQCommunicator

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

try:
    import wandb

    HAVE_WANDB = True
except ImportError:
    HAVE_WANDB = False
    wandb = None

try:
    import psutil

    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False

DEPRECATED_ARGS = [
    "enable_cuda_graph",
    "random_seed",
    "track_paused_request_events",
    "enable_chunked_prefill",
    "inference_logging_step_interval",
    "pg_collection",
]


class EngineSuspendedError(Exception):
    """Engine is currently suspended and not performing steps."""

    pass


def format_mem_bytes(mem_bytes):
    """Convert a byte count to a human-readable string in tb, gb, mb, kb, or bytes."""
    for power, suffix in [(4, "tb"), (3, "gb"), (2, "mb"), (1, "kb"), (0, "bytes")]:
        suffix_bytes = 1024**power
        if mem_bytes >= suffix_bytes:
            return "%.1f %s" % (mem_bytes / suffix_bytes, suffix)
    return "%d bytes" % mem_bytes


@dataclass(kw_only=True)
class RequestEntry:
    """Entry in the engine's `self.requests` dict."""

    record: DynamicInferenceRequestRecord
    future: asyncio.Future


# pylint: disable=line-too-long
@experimental_api
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
    """

    @deprecate_args(
        *DEPRECATED_ARGS,
        message="Argument `{name}` has been deprecated. Only pass `controller` and `context`",
    )
    def __init__(self, controller: TextGenerationController, context: DynamicInferenceContext):

        assert isinstance(
            controller, TextGenerationController
        ), f"controller must be a TextGenerationController, got {type(controller)}"
        assert isinstance(
            context, DynamicInferenceContext
        ), f"context must be a DynamicInferenceContext, got {type(context)}"

        model_config = controller.inference_wrapped_model.model.config
        inference_config = context.config

        if inference_config.pg_collection is not None:
            self.pg_collection = inference_config.pg_collection
        else:
            self.pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        # Initialization options.
        self.controller = controller
        self.context = context
        self.track_paused_request_events = inference_config.track_paused_request_events
        self.track_generated_token_events = inference_config.track_generated_token_events
        self.enable_chunked_prefill = inference_config.enable_chunked_prefill
        self.metrics_writer = inference_config.metrics_writer
        self.logging_step_interval = inference_config.logging_step_interval
        self.unified_memory_level = inference_config.unified_memory_level
        self.materialize_only_last_token_logits = (
            inference_config.materialize_only_last_token_logits
        )
        self.cuda_graph_impl = model_config.cuda_graph_impl
        self.cuda_graph_scope = model_config.cuda_graph_scope
        # Initialize engine.
        self.reset()

        # Set callback for getting stop word finished request IDs
        self.controller.set_stop_word_finished_ids_callback(
            self._get_and_clear_stop_word_finished_ids
        )

        # Configure wandb to use separate step counter for inference metrics (only once)
        if self.logging_step_interval > 0 and self.metrics_writer is not None:
            logging.info(
                f"\033[1;93m[INFERENCE]\033[0m "
                f"\033[1;95mLogging inference metrics to wandb (rank {self.rank})\033[0m"
            )
            if HAVE_WANDB and self.metrics_writer.__name__ == "wandb":
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

        # Create cuda graphs.
        self.create_cuda_graphs()

    def reset(self) -> None:
        """Reset by removing all requests and reset all state."""

        self.context.reset()

        # Request state.
        self.request_counter = Counter()
        self.finished_request_count = 0
        self.evicted_request_count = 0

        self.requests: Dict[int, RequestEntry] = {}
        self.waiting_request_ids = deque()
        self.failed_request_ids = []
        # Track requests that should stop due to stop words (detected in post_process_requests)
        self.stop_word_finished_request_ids: set[int] = set()
        # Track requests currently being finished due to stop words (to skip extra token)
        self.stop_word_being_finished_ids: set[int] = set()

        # Timing and logging variables.
        self.rank = torch.distributed.get_rank()
        self.step_start_event = torch.cuda.Event(enable_timing=True)
        self.step_end_event = torch.cuda.Event(enable_timing=True)
        self.capture_stats = None

        # Runtime state.
        self._loop = get_asyncio_loop(getattr(self, "_loop", None))
        self._cond = asyncio.Condition()
        self.running = asyncio.Event()
        self.paused = asyncio.Event()
        self.stopped = asyncio.Event()
        self.received_pause: bool = False
        self.received_stop: bool = False
        self.suspend_signal = False
        self.is_suspended = False
        self.resume_request_ids = None

        # Prefix caching coordination state.
        self._prefix_coordination_waits = 0

        # Coordinator state.
        self.use_coordinator = False

    def create_cuda_graphs(self, reset_context: bool = True):
        """Create cuda graphs.

        This method iterates the dynamic context's `cuda_graph_request_counts`
        to record and capture cuda graphs.

        Args:
            reset_context (bool): Whether to reset the context after building cuda graphs.
        """

        if self.cuda_graph_impl != "local":
            return

        if (
            CudaGraphScope.full_iteration in self.cuda_graph_scope
            and CudaGraphScope.full_iteration_inference not in self.cuda_graph_scope
        ):
            warnings.warn(
                "\n\n*** WARNING: 'full_iteration' CUDA graph scope used during inference! "
                "This will not create inference CUDA graphs. Use '--cuda-graph-scope=full_iteration_inference' instead. ***\n"
            )

        context = self.context
        controller = self.controller

        time_start = time.time()
        mem_stats_start = torch.cuda.memory_stats()

        logging.info("> dynamic_engine.py: building cuda graphs for ")
        for graph in context.cuda_graph_batch_dimensions_list:
            logging.info(graph)

        tbar = enumerate(context.cuda_graph_batch_dimensions_list)
        if HAVE_TQDM:
            tbar = tqdm(tbar, total=len(context.cuda_graph_batch_dimensions_list))
        for tbar_idx, cuda_graph_batch_dimension in tbar:
            input_ids, position_ids = self.controller._dynamic_step_context_init(
                construct_graph_dimensions=cuda_graph_batch_dimension
            )
            # Progress.
            tbar_str = f"cuda graph warmup - {cuda_graph_batch_dimension}"
            if HAVE_TQDM:
                tbar.set_description(tbar_str)
            else:
                logging.info(
                    f"{tbar_idx}/{len(context.cuda_graph_batch_dimensions_list)}. {tbar_str}"
                )

            # Enable routing recording during warmup if routing replay is enabled.
            # This ensures the record_indices copy operation is captured in the CUDA graph.
            model_config = controller.inference_wrapped_model.model.config
            if model_config.moe_enable_routing_replay:
                RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)

            # Forward pass -> logits.
            controller._dynamic_step_forward_logits(input_ids, position_ids)

            context.reset()

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

    @internal_api
    async def start_listening_to_data_parallel_coordinator(
        self,
        inference_coordinator_port: int | None = None,
        launch_inference_coordinator: bool = True,
        *,
        loop: Optional[asyncio.AbstractEventLoop] = None,
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
            inference_coordinator_port (int | None): The network port where the central
                `InferenceCoordinator` is or will be listening.
                If None, a random available port will be selected.
                If not None, the coordinator will attempt to bind to this port, but should it
                not succeed (e.g., if the port is already in use), it may bind to a different port.
                The actual port used is returned by this method.
            launch_inference_coordinator (bool, optional): If True, the global rank 0
                process will spawn and manage the `InferenceCoordinator`
                process. Defaults to True.

        Returns:
            inference_coordinator_addresss (str): The network address of the central
                `InferenceCoordinator`, which may not have the same port as what the user requested
                with `inference_coordinator_port`.
        """

        assert HAVE_ZMQ, (
            "please install the pyzmq library to use InferenceCoordinator\n" "pip install pyzmq"
        )
        assert HAVE_MSGPACK, (
            "please install the messagepack library to use InferenceCoordinator\n"
            "pip install msgpack"
        )

        self.zmq_context = zmq.Context().instance()
        self.zmq_sockets = []  # keep track of all sockets created by this engine

        # Get world info.
        dp_group = self.pg_collection.dp
        dp_src = get_pg_src_rank(dp_group)
        dp_size = get_pg_size(self.pg_collection.dp)
        dp_rank = get_pg_rank(self.pg_collection.dp)

        mp_group = self.pg_collection.mp
        mp_src = get_pg_src_rank(mp_group)
        tp_rank = get_pg_rank(self.pg_collection.tp)
        pp_rank = get_pg_rank(self.pg_collection.pp)

        self.is_mp_coordinator = tp_rank == 0 and pp_rank == 0
        self.is_dp_coordinator = (dp_rank == 0) and self.is_mp_coordinator

        local_ip = socket.gethostname()

        # Spawn a DP coordinator process and get the connection info.
        if launch_inference_coordinator and self.is_dp_coordinator:
            spawn_context = multiprocessing.get_context('spawn')
            deterministic_mode = torch.are_deterministic_algorithms_enabled()
            dp_pipe, dp_process_pipe = spawn_context.Pipe()
            coordinator_ready_event = spawn_context.Event()
            self.inference_coordinator_process = spawn_context.Process(
                target=DataParallelInferenceCoordinator.entrypoint,
                args=(
                    dp_process_pipe,
                    coordinator_ready_event,
                    get_pg_size(self.pg_collection.dp),
                    self.controller.tokenizer,
                    inference_coordinator_port,
                    deterministic_mode,
                ),
            )
            self.inference_coordinator_process.start()
            await await_process_call(dp_pipe.poll, self.inference_coordinator_process)
            dp_addr = dp_pipe.recv()
            dp_pipe.close()

            # Check if the port number is not inference_coordinator_port
            actual_port = int(dp_addr.rsplit(":", 1)[-1])
            if inference_coordinator_port != None and actual_port != inference_coordinator_port:
                logging.warning(
                    f"Requested InferenceCoordinator port {inference_coordinator_port} "
                    f"but got port {actual_port} instead. This happens if the request port "
                    f"is already in use."
                )
        elif not launch_inference_coordinator:
            dp_addr = f"tcp://{local_ip}:{inference_coordinator_port}"
        else:
            dp_addr = None

        # Find available ports for MP and bind to them.
        if self.is_mp_coordinator:
            mp_req_sock = self.zmq_context.socket(zmq.PUB)
            mp_req_sock.bind_to_random_port(f"tcp://{local_ip}")
            mp_req_addr = mp_req_sock.getsockopt_string(zmq.LAST_ENDPOINT)

            mp_len_sock = self.zmq_context.socket(zmq.PUB)
            mp_len_sock.bind_to_random_port(f"tcp://{local_ip}")
            mp_len_addr = mp_len_sock.getsockopt_string(zmq.LAST_ENDPOINT)
        else:
            mp_req_addr = None
            mp_len_addr = None

        # Broadcast addresses to respective ranks.
        bcast = [dp_addr]
        torch.distributed.broadcast_object_list(bcast, src=dp_src, group=dp_group)
        [dp_addr] = bcast
        bcast = [mp_req_addr, mp_len_addr]
        torch.distributed.broadcast_object_list(bcast, src=mp_src, group=mp_group)
        [mp_req_addr, mp_len_addr] = bcast

        identity = f'mp-coord-{dp_rank}'
        if self.is_mp_coordinator:
            # 1. Create dealer sockets where tp_rank = 0 and pp_rank = 0
            #    These will receive requests from an InferenceCoordinator.
            self.socket_for_receiving_requests = self.zmq_context.socket(zmq.DEALER)

            self.socket_for_receiving_requests.setsockopt(zmq.IDENTITY, identity.encode('utf-8'))
            self.socket_for_receiving_requests.connect(dp_addr)

            # send empty string. this is used to register with the coordinator.
            self.socket_for_receiving_requests.send(b"")

            # 2. Create a publisher socket. This is used to publish or broadcast
            #    requests within the model parallel group
            self.model_parallel_publisher_socket = mp_req_sock

            # 3. Create another publisher socket to broadcast the number of messages to receive.
            self.model_parallel_num_msgs_publisher_socket = mp_len_sock
            self.zmq_sockets += [
                self.socket_for_receiving_requests,
                self.model_parallel_num_msgs_publisher_socket,
                self.model_parallel_publisher_socket,
            ]
        # All MP ranks subscribe to the two publisher sockets
        self.model_parallel_subscriber_socket = self.zmq_context.socket(zmq.SUB)
        self.model_parallel_subscriber_socket.connect(mp_req_addr)
        self.model_parallel_subscriber_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        self.model_parallel_num_msgs_subscriber_socket = self.zmq_context.socket(zmq.SUB)
        self.model_parallel_num_msgs_subscriber_socket.connect(mp_len_addr)
        self.model_parallel_num_msgs_subscriber_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        self.zmq_sockets += [
            self.model_parallel_subscriber_socket,
            self.model_parallel_num_msgs_subscriber_socket,
        ]

        torch.distributed.barrier(mp_group)

        # initialize zmq-based EP communicator
        self.ep_rank = get_pg_rank(self.pg_collection.ep)
        self.ep_world_size = get_pg_size(self.pg_collection.ep)
        if self.ep_world_size > 1:
            self.expert_parallel_zmq_communicator = AsyncZMQCommunicator(
                self.zmq_context, process_group=self.pg_collection.ep
            )

        if launch_inference_coordinator and self.is_dp_coordinator:
            await await_process_call(
                coordinator_ready_event.wait, self.inference_coordinator_process
            )
            logging.info("Inference co-ordinator is ready to receive requests!")
            logging.info(f"Data parallel coordinator can be found at {dp_addr}")

        # Finally run the engine infinite loop.
        loop = get_asyncio_loop(loop)
        self.engine_loop_task = loop.create_task(self.run_engine_with_coordinator(loop=loop))

        return dp_addr

    @contextmanager
    @staticmethod
    def suspend_resume_ctx(key: str, *, unified_memory_level: int) -> None:
        """Context manager for of suspending and resuming the engine.

        This context manager records the time and memory usage when suspending
        and resuming the context. TODO(@lmcafee): add argument to optionally
        return nullcontext, to avoid overhead.

        Args:
            key (str): Key that identifies caller (e.g., 'suspend' or 'resume').

        Return:
            None.
        """

        try:

            start_mem = torch.cuda.memory_stats()
            start_time = time.time()
            range_push(f"{key}-inference-context")
            torch.cuda.synchronize()

            yield

        finally:

            range_pop()
            end_time = time.time()

            end_mem = torch.cuda.memory_stats()
            start_mem_alloc = start_mem["allocated_bytes.all.current"]
            end_mem_alloc = end_mem["allocated_bytes.all.current"]
            start_mem_res = start_mem["reserved_bytes.all.current"]
            end_mem_res = end_mem["reserved_bytes.all.current"]

            rank_str = torch.distributed.get_rank()
            dir_str = "deallocating" if end_mem_alloc <= start_mem_alloc else "allocating"
            relative_time_str = f"{end_time - start_time:.3f} sec"
            relative_mem_str = f"{abs(start_mem_alloc - end_mem_alloc) / 1024**3:.1f} gb"

            if HAVE_PSUTIL:
                process = psutil.Process()
                mem_info = process.memory_info()
                cpu_mem_str = f"{mem_info.rss / 1024**3:.1f} gb"
            else:
                cpu_mem_str = "--"

            total_mem_str = ", ".join(
                (
                    f"cpu: {cpu_mem_str}",
                    f"gpu: alloc {end_mem_alloc / 1024**3:.1f} gb",
                    f"res {end_mem_res / 1024**3:.1f} gb",
                )
            )
            logging.info(
                f"[rank {rank_str}] dynamic engine {key}, "
                f"unified {unified_memory_level}, "
                f"{dir_str} "
                f"{relative_mem_str} in {relative_time_str} ... "
                f"abs mem usage: {total_mem_str}"
            )

    def suspend(self):
        """Suspend engine by deallocating context's GPU state."""

        # Skip if already suspended, which can happen when using the inference
        # coordinator.
        if self.is_suspended:
            return
        self.is_suspended = True

        # Deallocate context tensors.
        with self.__class__.suspend_resume_ctx(
            "suspended", unified_memory_level=self.unified_memory_level
        ):
            self.context.deallocate_inference_state_buffers()

        if (
            self.context.kv_cache_management_mode != KVCacheManagementMode.PERSIST
            and not self.context.static_kv_memory_pointers
        ):
            delete_cuda_graphs()

        # Build the list of requests to re-add on resume.
        # All waiting requests are always included; active requests are included
        # only if they are marked for recompute (their KV cache will be gone).
        waiting_request_ids = list(self.waiting_request_ids)
        active_request_ids = set(self.requests.keys()) - set(waiting_request_ids)
        if self.context.kv_cache_management_mode == KVCacheManagementMode.RECOMPUTE:
            recompute_active_ids = active_request_ids
        else:
            recompute_active_ids = set()
        self.resume_request_ids = [*recompute_active_ids, *waiting_request_ids]
        self.waiting_request_ids.clear()

        # Checkpoint active requests that are marked for recompute.
        for request_id in recompute_active_ids:
            self.requests[request_id].record.checkpoint()

    def resume(self):
        """Resume engine by reallocating context's GPU state."""

        # Skip if not suspended, which can happen when using the inference
        # coordinator.
        if not self.is_suspended:
            return
        self.is_suspended = False

        # Resume.
        with self.__class__.suspend_resume_ctx(
            "resumed", unified_memory_level=self.unified_memory_level
        ):

            # Allocate context tensors.
            alloc_time = time.time()
            torch.cuda.synchronize()
            self.context.reinitialize_inference_state_buffers()
            torch.cuda.synchronize()
            alloc_time = time.time() - alloc_time

            capture_time = time.time()
            if (
                self.context.kv_cache_management_mode != KVCacheManagementMode.PERSIST
                and not self.context.static_kv_memory_pointers
            ):
                self.create_cuda_graphs()
            capture_time = time.time() - capture_time

            # Re-add requests saved during suspend.
            add_time = time.time()
            torch.cuda.synchronize()
            for request_id in self.resume_request_ids:
                self._add_request(self.get_request(request_id))
            torch.cuda.synchronize()
            add_time = time.time() - add_time

        # Print inner timing (must be outside context manager above for correct formatting).
        logging.info(
            "    > "
            + ", ".join(
                (
                    f"inner timing: alloc {alloc_time:.3f}",
                    f"add {add_time:.3f}",
                    f"capture {capture_time:.3f}.",
                )
            )
        )

        # Notify event loop.
        self._loop.call_soon_threadsafe(asyncio.create_task, self._notify_cond_for_new_request())

    @trace_async_exceptions
    async def _notify_cond_for_new_request(self):
        """Helper function to notify condition variable when a new request is added."""
        async with self._cond:
            self._cond.notify_all()

    def has_unfinished_requests(self) -> bool:
        """Test if context contains unfinished requests."""
        return self.context.has_unfinished_requests() or len(self.waiting_request_ids) > 0

    def get_request(self, request_id: int) -> DynamicInferenceRequest:
        """Get most recent request from a request record.

        Args:
            request_id (int): Request id.

        Returns:
            (DynamicInferenceRequest) The most recent request in the record.
        """
        return self.requests[request_id].record[-1]

    def _add_request(
        self, request: DynamicInferenceRequest
    ) -> asyncio.Future[DynamicInferenceRequest]:

        request_id = request.request_id

        # Add request to self.requests. If the engine has previously been
        # suspended, then the request may already exist.
        if request_id not in self.requests:
            self.requests[request_id] = RequestEntry(
                record=DynamicInferenceRequestRecord.from_request(request),
                future=self._loop.create_future(),
            )
            request.add_event_add_engine()  # Record when request enters engine

        if request.status is None:
            request.status = Status.ACTIVE_AND_GENERATING_TOKENS

        assert (
            request.sampling_params.num_tokens_to_generate is None
            or request.sampling_params.num_tokens_total is None
        )
        if request.sampling_params.top_n_logprobs > 0:
            assert (
                request.sampling_params.return_log_probs
            ), "top_n_logprobs requires sampling_params.return_log_probs to be True"
        if (
            request.sampling_params.return_log_probs
            and not request.sampling_params.skip_prompt_log_probs
        ):
            assert not self.materialize_only_last_token_logits, (
                "Prompt log probs cannot be calculated if only last token logits are materialized. "
                "Set materialize_only_last_token_logits to False in DynamicInferenceContext "
                "or skip_prompt_log_probs to True in SamplingParams."
            )

        if request.sampling_params.num_tokens_total is not None:
            request.sampling_params.num_tokens_to_generate = (
                request.sampling_params.num_tokens_total - len(request.prompt_tokens)
            )
            request.sampling_params.num_tokens_total = None
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
        ) or (request.sampling_params.num_tokens_to_generate < 0):
            request.status = Status.FAILED
            request.add_event_error_nontransient(MaxSequenceLengthOverflowError(request_id))

        if len(request.prompt_tokens) > self.context.max_tokens and not self.enable_chunked_prefill:
            request.status = Status.FAILED
            request.add_event_error_nontransient(TokenOverflowError(request_id))

        # Tokenize stop words if provided
        if request.sampling_params.stop_words:
            stop_word_ids = [
                self.controller.tokenize_prompt(stop_word, add_BOS=False)
                for stop_word in request.sampling_params.stop_words
            ]
            request.stop_word_ids = stop_word_ids

        if request.status != Status.FAILED:
            self.waiting_request_ids.append(request_id)
        else:
            self.failed_request_ids.append(request_id)
            if self.rank == 0:
                warnings.warn(
                    f"Request {request_id} failed to be added to the engine due to errors."
                )

        return self.requests[request_id].future

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
            request_id=request_id,
            prompt=prompt_str,
            prompt_tokens=tokens,
            sampling_params=sampling_params,
            block_size_tokens=self.context.block_size_tokens,
            enable_prefix_caching=self.context.enable_prefix_caching,
        )

        # Add request.
        return self._add_request(request)

    def post_process_requests(
        self,
        request_ids: torch.Tensor,
        finished_request_ids: torch.Tensor,
        evict_request_ids: torch.Tensor,
        step_time: float,
        sample: torch.Tensor,
        log_probs: torch.Tensor,
        top_n_logprobs: Optional[Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        routing_indices_per_request: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Tuple[List[DynamicInferenceRequest], List[DynamicInferenceRequest]]:
        """
        Handles post-processing for requests after a step.

        Args:
            request_ids (torch.Tensor): A list of request_ids
            finished_request_ids (torch.Tensor): A list of finished request ids
            evict_request_ids (torch.Tensor): A list of evicted request ids.
            step_time (float): The latency of the last step
            sample: (torch.Tensor): The newly generated tokens for each request
            log_probs: (List): Log probs for each request
            top_n_logprobs: (Dict): Top-n log probs for each request. Maps request_idx to
                list of (top_n_logprobs, top_n_indices) tuples.
            routing_indices_per_request: (Dict[int, Tensor]): MoE routing indices
                pre-mapped by request_id. Each value is a tensor of shape
                [num_tokens_this_step, num_layers, topk].

        Returns:
            A list of active requests and completed requests as `DynamicInferenceRequest` objects
        """
        active_request_ids: list[int] = []
        finished_request_ids = set(finished_request_ids.tolist())
        finished_request_records: list[DynamicInferenceRequestRecord] = []
        self.finished_request_count += len(finished_request_ids)
        if evict_request_ids is not None:
            self.evicted_request_count += evict_request_ids.numel()

        log_probs_iter = log_probs if log_probs else repeat(None)
        block_allocator = self.context.block_allocator

        # Pre-compute step-level block stats (before the per-request loop)
        if self.track_generated_token_events:
            blocks_allocated = block_allocator.total_count - block_allocator.total_avail
            if block_allocator.enable_prefix_caching:
                blocks_hashed_active = int((block_allocator.block_ref_counts > 0).sum().item())
                blocks_ref_count = block_allocator.block_ref_counts.sum().item()
            else:
                blocks_hashed_active = blocks_allocated
                blocks_ref_count = None

        for req_idx, (request_id, token, request_log_probs) in enumerate(
            zip(request_ids.tolist(), sample.tolist(), log_probs_iter)
        ):
            request: DynamicInferenceRequest = self.get_request(request_id)
            if request_id != self.context.chunked_prefill_request_id:
                # Skip appending token for requests being finished due to stop words
                # (they already have their final token from the previous step)
                if request_id not in self.stop_word_being_finished_ids:
                    is_first_token = len(request.generated_tokens) == 0
                    request.generated_tokens.append(token)
                    if self.track_generated_token_events:
                        if block_allocator.enable_prefix_caching:
                            event_generated_token = request.add_event_generated_token(
                                token,
                                blocks_total=block_allocator.total_count,
                                blocks_hashed_total=blocks_allocated,
                                blocks_hashed_active=blocks_hashed_active,
                                blocks_ref_count=blocks_ref_count,
                            )
                        else:
                            event_generated_token = request.add_event_generated_token(
                                token,
                                blocks_total=block_allocator.total_count,
                                blocks_hashed_total=blocks_allocated,
                                blocks_hashed_active=blocks_hashed_active,
                            )
                    if is_first_token:
                        if self.track_generated_token_events:
                            first_token_event = event_generated_token
                        else:
                            first_token_event = DynamicInferenceEvent(
                                type=DynamicInferenceEventType.GENERATED_TOKEN,
                                payload={"token_id": token},
                            )
                        request.ttft = (
                            first_token_event.timestamp - request.event_add_engine.timestamp
                        )
                    if request.tpot is None:
                        request.tpot = []
                    request.tpot.append(step_time)

                # Check for stop words (after token is appended)
                stop_word_hit = self._check_stop_words_for_request_post_append(request)

                if request_id in finished_request_ids:
                    # Request finished by normal means (termination_id, max_length, or stop word from previous step)
                    request.generated_length = len(request.generated_tokens)
                    request.status = Status.COMPLETED
                    request.add_event_finish()
                    finished_entry = self.requests.pop(request_id)
                    finished_request = finished_entry.record[-1]
                    finished_request.generated_length = len(finished_request.generated_tokens)
                    finished_request_records.append(finished_entry.record)
                    finished_entry.future.set_result(finished_entry.record)
                elif stop_word_hit:
                    # Stop word detected - mark for removal in next step's bookkeeping
                    # Don't pop yet; let the next step handle it properly via callback
                    self.stop_word_finished_request_ids.add(request_id)
                    active_request_ids.append(request_id)
                else:
                    active_request_ids.append(request_id)
            else:
                # The chunked prefill produces useless tokens
                # so we are not appending them to the generated tokens.
                # Additionally, chunked prefill request do not finish.
                active_request_ids.append(request_id)

            # Process log_probs if available (unified for both regular and chunked prefill)
            if request_log_probs is not None:
                # Initialize lists if they don't exist
                if not request.prompt_log_probs:
                    request.prompt_log_probs = []
                if not request.generated_log_probs:
                    request.generated_log_probs = []

                # For chunked prefill with materialize_only_last_token_logits, discard intermediate log probs
                if (
                    request_id == self.context.chunked_prefill_request_id
                    and self.materialize_only_last_token_logits
                ):
                    request.prompt_log_probs = []
                    request.generated_log_probs = []
                else:
                    prompt_length = len(request.prompt_tokens)
                    total_accumulated = len(request.prompt_log_probs) + len(
                        request.generated_log_probs
                    )

                    # Handle skip_prompt_log_probs during prefill
                    # If skip_prompt_log_probs is True and we have multiple log probs (prefill),
                    # only process the last one (first generated token)
                    if request.sampling_params.skip_prompt_log_probs and len(request_log_probs) > 1:
                        # Only append the last log prob (first generated token) to generated_log_probs
                        request.generated_log_probs.append(request_log_probs[-1])
                    else:
                        # Vectorized approach: calculate split point and use list slicing
                        if not request.sampling_params.skip_prompt_log_probs:
                            # Calculate how many log probs go to prompt vs generated
                            remaining_prompt_slots = max(0, prompt_length - 1 - total_accumulated)
                            split_idx = min(remaining_prompt_slots, len(request_log_probs))

                            # Batch extend instead of individual appends
                            if split_idx > 0:
                                request.prompt_log_probs.extend(request_log_probs[:split_idx])
                            if split_idx < len(request_log_probs):
                                request.generated_log_probs.extend(request_log_probs[split_idx:])
                        else:
                            # All log probs go to generated
                            request.generated_log_probs.extend(request_log_probs)

            # Process top_n_logprobs if available (unified for both regular and chunked prefill)
            if top_n_logprobs is not None and req_idx in top_n_logprobs:
                # Initialize lists if they don't exist
                if request.prompt_top_n_logprobs is None:
                    request.prompt_top_n_logprobs = []
                if request.generated_top_n_logprobs is None:
                    request.generated_top_n_logprobs = []

                top_n_data_list = top_n_logprobs[req_idx]
                prompt_length = len(request.prompt_tokens)

                # Process each token's top-n logprobs
                for top_n_values, top_n_indices in top_n_data_list:
                    logit_dict = {}
                    for logprob, logprob_index in zip(
                        top_n_values.cpu().tolist(), top_n_indices.cpu().tolist()
                    ):
                        key = self.controller.tokenizer.detokenize([logprob_index])
                        logit_dict[key] = logprob

                    # Simple decision: check total count accumulated so far
                    total_accumulated = len(request.prompt_top_n_logprobs) + len(
                        request.generated_top_n_logprobs
                    )

                    # If skip_prompt_log_probs is False and we haven't reached prompt end,
                    # append to prompt_top_n_logprobs. Otherwise append to generated_top_n_logprobs.
                    if (
                        not request.sampling_params.skip_prompt_log_probs
                        and total_accumulated < prompt_length - 1
                    ):
                        request.prompt_top_n_logprobs.append(logit_dict)
                    else:
                        request.generated_top_n_logprobs.append(logit_dict)

            # Process routing indices if available (keyed by request_id)
            # Each step's routing is a tensor of shape [num_tokens_this_step, num_layers, topk]
            # We concatenate along dim=0 to accumulate: [total_tokens, num_layers, topk]
            if (
                routing_indices_per_request is not None
                and request_id in routing_indices_per_request
            ):
                step_routing = routing_indices_per_request[
                    request_id
                ]  # [num_tokens, num_layers, topk]
                if request.routing_indices is None:
                    request.routing_indices = step_routing.clone()
                else:
                    request.routing_indices = torch.cat(
                        [request.routing_indices, step_routing], dim=0
                    )

        # Handle evicted requests.
        if evict_request_ids is not None and evict_request_ids.numel() > 0:

            evict_request_ids = evict_request_ids.tolist()

            # Insert into waiting_request_ids after any chunk prefill request.
            self.waiting_request_ids.extendleft(evict_request_ids)
            if self.context.chunked_prefill_request_id != -1:
                chunked_prefill_id = self.waiting_request_ids[len(evict_request_ids)]
                del self.waiting_request_ids[len(evict_request_ids)]
                self.waiting_request_ids.appendleft(chunked_prefill_id)

            # Checkpoint requests (i.e., prompt += generations) + add eviction event.
            for request_id in evict_request_ids:
                self.requests[request_id].record.checkpoint()
                self.get_request(request_id).add_event_evict()

        # Clear the stop word being finished set after processing
        self.stop_word_being_finished_ids.clear()

        return active_request_ids, finished_request_records

    def _get_and_clear_stop_word_finished_ids(self, active_request_ids: list[int]) -> set[int]:
        """Get and clear the set of request IDs that should be finished due to stop words.

        This callback is called from the controller during bookkeeping to get request IDs
        that were detected as hitting stop words in the previous step's post_process_requests.

        Args:
            active_request_ids: List of currently active request IDs.

        Returns:
            Set of request IDs from active_request_ids that should be marked as finished.
        """
        if not self.stop_word_finished_request_ids:
            return set()

        # Find which stop word finished IDs are in the current active requests
        result = self.stop_word_finished_request_ids & set(active_request_ids)
        # Move to "being finished" set so post_process_requests can skip the extra token
        self.stop_word_being_finished_ids = result
        # Clear the IDs that we're returning (they'll be marked as finished)
        self.stop_word_finished_request_ids -= result
        return result

    def _check_stop_words_for_request_post_append(self, request: DynamicInferenceRequest) -> bool:
        """Check if a request should stop due to stop words (after token is appended).

        This method is called from post_process_requests after the token has already
        been appended to request.generated_tokens.

        Args:
            request: The request to check.

        Returns:
            bool: True if the generated sequence ends with a stop word, False otherwise.
        """
        # Check if request has stop words configured
        if request.stop_word_ids is None or len(request.stop_word_ids) == 0:
            return False

        generated_tokens = request.generated_tokens

        # Check if the sequence ends with any stop word
        for stop_word_ids in request.stop_word_ids:
            stop_len = len(stop_word_ids)
            if len(generated_tokens) >= stop_len:
                # Check if the last stop_len tokens match the stop word
                if generated_tokens[-stop_len:] == stop_word_ids:
                    return True

        return False

    def get_prefix_coordination_metrics(self) -> dict:
        """Return prefix caching coordination metrics.

        Returns:
            Dict with coordination stats including the number of scheduling waits.
        """
        return {"waits": self._prefix_coordination_waits}

    def _compute_mamba_prefill_boundaries(
        self,
        req: DynamicInferenceRequest,
        num_matched_blocks: int,
    ) -> tuple:
        """Compute token boundaries for three-part Mamba prefill.

        Args:
            req: The inference request
            num_matched_blocks: Number of prefix blocks with cached Mamba state

        Returns:
            Tuple of (divergence_token, last_aligned_token, prompt_length)
        """
        block_size = self.context.block_size_tokens
        prompt_length = len(req.prompt_tokens)
        divergence_token = num_matched_blocks * block_size
        last_aligned_token = (prompt_length // block_size) * block_size
        return divergence_token, last_aligned_token, prompt_length

    def _find_mamba_divergence_block(self, matched_kv_blocks: list) -> int:
        """Find the last KV block that also has cached Mamba state.

        Args:
            matched_kv_blocks: List of KV block IDs that matched the prefix

        Returns:
            Number of blocks with valid Mamba state (0 if none)
        """
        for i in range(len(matched_kv_blocks) - 1, -1, -1):
            if self.context.has_mamba_state_for_block(matched_kv_blocks[i]):
                return i + 1
        return 0  # No Mamba state cached for any matched block

    def _store_mamba_states_for_completed_prefill(self):
        """Store Mamba state at block boundaries after prefill.

        Called after prefill forward pass completes. For each request that
        was part of this prefill, stores Mamba state at meaningful boundaries:
        the KV divergence block and the last-aligned block.

        Only stores state if:
        - The chunk ends at a mamba-meaningful boundary (KV divergence or last-aligned)
        - The request has at least one complete block
        - The block doesn't already have cached Mamba state
        """
        block_size = self.context.block_size_tokens

        for req_idx in range(self.context.paused_request_count,
                             self.context.total_request_count):
            request_id = self.context.request_ids[req_idx].item()
            req = self.get_request(request_id)

            # Calculate total tokens actually in context (processed through forward).
            # For continuing chunks, remaining_prompt_tokens still holds unprocessed
            # tokens, so subtract from total prompt length.
            total_prefilled = len(req.prompt_tokens) - len(req.remaining_prompt_tokens)

            # For hybrid mamba: only store when chunk ends at a meaningful boundary
            kv_divergence = getattr(req, '_kv_divergence_token', 0)
            last_aligned = getattr(req, '_mamba_last_aligned_token', 0)
            is_kv_divergence = (kv_divergence > 0 and total_prefilled == kv_divergence)
            is_last_aligned = (last_aligned > 0 and total_prefilled == last_aligned)
            if not (is_kv_divergence or is_last_aligned):
                continue

            # Find the last complete block index
            num_complete_blocks = total_prefilled // block_size
            if num_complete_blocks == 0:
                continue  # No complete blocks to store

            # Get the block ID of the last complete block
            last_complete_block_idx = num_complete_blocks - 1
            if last_complete_block_idx >= self.context.request_kv_block_counts[req_idx].item():
                continue  # Safety check

            block_id = self.context.request_to_kv_block_ids[req_idx, last_complete_block_idx].item()
            if block_id < 0:
                continue  # Invalid block ID

            # Store mamba state and register in mamba hash map
            if not self.context.has_mamba_state_for_block(block_id):
                self.context.store_mamba_state_for_block(block_id, req_idx)
                if req.precomputed_block_hashes and last_complete_block_idx < len(req.precomputed_block_hashes):
                    block_hash = req.precomputed_block_hashes[last_complete_block_idx]
                    self.context.block_allocator.register_mamba_block_hash(block_id, block_hash)

    def _get_mamba_chunk_limit(self, req) -> Optional[int]:
        """Return max chunk_length based on next mamba boundary, or None.

        Two boundaries matter for mamba state storage: KV divergence (where
        KV match ends) and last-aligned (last complete block in prompt).
        The engine must break chunks at these points so _store_mamba_states
        can detect and store state at the correct boundaries.
        """
        finished = req.finished_chunk_token_count
        kv_divergence = getattr(req, '_kv_divergence_token', 0)
        last_aligned = getattr(req, '_mamba_last_aligned_token', 0)

        if kv_divergence > 0 and finished < kv_divergence:
            return kv_divergence - finished
        elif last_aligned > 0 and finished < last_aligned:
            return last_aligned - finished
        else:
            return None

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
        # Mamba prefix caching requires chunked prefill for breaking at block boundaries
        assert not (self.context.is_hybrid_model and self.context.max_mamba_cache_slots > 0), \
            "Mamba prefix caching requires chunked prefill. Use schedule_chunked_prefill() instead."

        prefix_caching_enabled = self.context.enable_prefix_caching
        if prefix_caching_enabled:
            pending_block_hashes = set()
            pending_request_ids = []
        while self.waiting_request_ids:
            req = self.get_request(self.waiting_request_ids[0])

            # Check for conflicting block hashes.
            if prefix_caching_enabled:
                has_pending_hash = False
                for block_hash in req.precomputed_block_hashes:
                    if block_hash in pending_block_hashes:
                        has_pending_hash = True
                        break
                if has_pending_hash:
                    self._prefix_coordination_waits += 1
                    pending_request_ids.append(self.waiting_request_ids.popleft())
                    continue

            request_can_be_added, request_tokens_can_be_added, kv_cache_available = (
                self.context.check_availability(req)
            )
            if request_can_be_added and request_tokens_can_be_added and kv_cache_available:
                # Add these hashes to pending.
                if prefix_caching_enabled:
                    for block_hash in req.precomputed_block_hashes:
                        if block_hash not in self.context.block_allocator.kv_hash_to_block_id:
                            pending_block_hashes.add(block_hash)
                self.context.add_request(req)
                self._loop.call_soon_threadsafe(
                    self._loop.create_task, self._notify_cond_for_new_request()
                )
                req.remaining_prompt_tokens = req.remaining_prompt_tokens.new_empty(0)
                req.add_event_add_context()
                self.waiting_request_ids.popleft()
            else:
                break

        # Prepend pending request ids to waiting queue.
        if prefix_caching_enabled and pending_request_ids:
            self.waiting_request_ids.extendleft(reversed(pending_request_ids))

    def _reorder_waiting_for_mamba_priority(self):
        """Reorder waiting requests so those with restored Mamba states come first.

        This ensures the token layout groups batch-kernel requests before varlen
        requests in the prefill portion, which is required for correct Mamba state
        handling when multiple prefill requests have initial states.

        Skips the head of the queue if it's a continuing chunked prefill request.
        """
        is_continuing = self.context.chunked_prefill_request_id >= 0
        start_idx = 1 if is_continuing else 0

        if len(self.waiting_request_ids) - start_idx <= 1:
            return  # Nothing to reorder

        block_size = self.context.block_size_tokens
        with_mamba = []
        without_mamba = []

        for i in range(start_idx, len(self.waiting_request_ids)):
            req_id = self.waiting_request_ids[i]
            req = self.get_request(req_id)

            # Check if this request will have restored Mamba states
            num_blocks = math.ceil(len(req.prompt_tokens) / block_size)
            matched_blocks, _ = self.context._find_matching_prefix_blocks(req, 0, num_blocks)

            has_mamba = False
            if len(matched_blocks) > 0:
                num_mamba_matched = self._find_mamba_divergence_block(matched_blocks)
                if num_mamba_matched > 0:
                    has_mamba = True

            if has_mamba:
                with_mamba.append(req_id)
            else:
                without_mamba.append(req_id)

        # Rebuild the deque with reordered requests
        new_queue = deque()
        if is_continuing:
            new_queue.append(self.waiting_request_ids[0])
        new_queue.extend(with_mamba)
        new_queue.extend(without_mamba)
        self.waiting_request_ids = new_queue

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
        # Reorder new requests so those with restored Mamba states are scheduled
        # first, ensuring the batch-kernel token layout precedes varlen tokens.
        if self.context.is_hybrid_model and self.context.max_mamba_cache_slots > 0:
            self._reorder_waiting_for_mamba_priority()

        prefix_caching_enabled = self.context.enable_prefix_caching
        if prefix_caching_enabled:
            pending_block_hashes = set()
            pending_request_ids = []
        can_schedule = True
        while self.waiting_request_ids and can_schedule:
            can_schedule = False
            req = self.get_request(self.waiting_request_ids[0])

            # is_continuing_chunked_prefill is True if we are scheduling next
            # chunk of a existing chunked prefill request
            is_continuing_chunked_prefill = self.context.chunked_prefill_request_id >= 0

            # Check for conflicting block hashes.
            if prefix_caching_enabled and not is_continuing_chunked_prefill:
                has_pending_hash = False
                for block_hash in req.precomputed_block_hashes:
                    # pylint: disable-next=possibly-used-before-assignment
                    if block_hash in pending_block_hashes:
                        has_pending_hash = True
                        break
                if has_pending_hash:
                    self._prefix_coordination_waits += 1
                    pending_request_ids.append(  # pylint: disable=possibly-used-before-assignment
                        self.waiting_request_ids.popleft()
                    )
                    continue

            # For hybrid models with Mamba prefix caching: compute Mamba-aware boundaries
            if (self.context.is_hybrid_model and
                self.context.max_mamba_cache_slots > 0 and
                not is_continuing_chunked_prefill):
                # Find matching KV blocks
                num_hashes = len(req.precomputed_block_hashes) if req.precomputed_block_hashes else 0
                matched_blocks, _ = self.context._find_matching_prefix_blocks(req, 0, num_hashes)
                num_kv_matched = len(matched_blocks)

                # Find how many of those also have cached Mamba state
                if num_kv_matched > 0:
                    num_mamba_matched = self._find_mamba_divergence_block(matched_blocks)
                else:
                    num_mamba_matched = 0

                # Store for use in add_request() and chunk break enforcement
                req._mamba_num_matched_blocks = num_mamba_matched
                req._mamba_divergence_token, req._mamba_last_aligned_token, _ = \
                    self._compute_mamba_prefill_boundaries(req, num_mamba_matched)
                # KV divergence is only meaningful when mamba match is active and KV
                # match extends beyond it. Otherwise, add_request truncates the KV
                # match to 0, making the divergence point meaningless.
                if num_mamba_matched > 0 and num_kv_matched > num_mamba_matched:
                    req._kv_divergence_token = num_kv_matched * self.context.block_size_tokens
                else:
                    req._kv_divergence_token = 0

            # Use remaining prompt tokens for scheduling decisions
            remaining_len = len(req.remaining_prompt_tokens)
            token_fully_can_be_added = (
                self.context.active_token_count + remaining_len <= self.context.max_tokens
            )
            token_partially_can_be_added = self.context.active_token_count < self.context.max_tokens
            request_can_be_added, _, kv_cache_available = self.context.check_availability(req)
            request_can_be_added = is_continuing_chunked_prefill or request_can_be_added

            if request_can_be_added and kv_cache_available:
                # Compute mamba chunk limit for hybrid models
                mamba_limit = None
                if self.context.is_hybrid_model and self.context.max_mamba_cache_slots > 0:
                    mamba_limit = self._get_mamba_chunk_limit(req)

                # Check if mamba boundary requires a chunk break even when tokens fully fit
                mamba_forces_chunk = (mamba_limit is not None and remaining_len > mamba_limit)

                if token_fully_can_be_added and not mamba_forces_chunk:
                    # Add these hashes to pending.
                    if prefix_caching_enabled:
                        for block_hash in req.precomputed_block_hashes:
                            if block_hash not in self.context.block_allocator.kv_hash_to_block_id:
                                pending_block_hashes.add(block_hash)
                    self.context.chunked_prefill_request_id = -1
                    self.context.add_request(req)
                    self._loop.call_soon_threadsafe(
                        self._loop.create_task, self._notify_cond_for_new_request()
                    )
                    req.remaining_prompt_tokens = req.remaining_prompt_tokens.new_empty(0)
                    req.add_event_add_context()
                    # Fully scheduled, so we remove from waiting pool
                    self.waiting_request_ids.popleft()
                    # Only this case we keep checking the rest of the waiting queue
                    can_schedule = True
                elif token_partially_can_be_added or mamba_forces_chunk:
                    # Add these hashes to pending.
                    if prefix_caching_enabled:
                        for block_hash in req.precomputed_block_hashes:
                            if block_hash not in self.context.block_allocator.kv_hash_to_block_id:
                                pending_block_hashes.add(block_hash)
                    chunk_length = min(remaining_len, self.context.max_tokens - self.context.active_token_count)
                    if mamba_limit is not None:
                        chunk_length = min(chunk_length, mamba_limit)

                    # If this chunk would leave exactly 1 token for the final chunk, reduce this
                    # chunk by 1 so the final chunk has 2 tokens. This avoids the edge case where
                    # max_seqlen_q=1 which results in a bug with the Flash Attention kernel
                    # See https://github.com/Dao-AILab/flash-attention/issues/1537
                    if remaining_len - chunk_length == 1 and chunk_length > 1:
                        chunk_length -= 1

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

        # Prepend pending request ids to waiting queue.
        if prefix_caching_enabled and pending_request_ids:
            is_continuing_chunked_prefill = self.context.chunked_prefill_request_id >= 0
            if is_continuing_chunked_prefill:
                chunked_request_id = self.waiting_request_ids.popleft()
                self.waiting_request_ids.extendleft(reversed(pending_request_ids))
                self.waiting_request_ids.appendleft(chunked_request_id)
            else:
                self.waiting_request_ids.extendleft(reversed(pending_request_ids))

    async def async_forward(self) -> Tuple[Dict, Dict, float]:
        """Uses `asyncio` for continuous generation.
        Sleeps when no requests are available, until new requests have been added.

        Returns:
            A tuple comprised of:
                step_result (Optional[Dict]): The result of the step.
                context_state (Dict): A tuple consisting of the state of the context.
                is_decode_only, total/paused request count, active token count.
                step_time (float): How long this step took.
        """

        # If suspended, no stepping.
        if self.is_suspended:
            raise EngineSuspendedError(self.context.step_count)

        # schedule requests
        self.schedule_waiting_requests()

        # Saving pre-step state, for printing output below.
        is_decode_only = self.context.is_decode_only()
        pre_step_context_state = {
            "is_decode_only": is_decode_only,
            "max_requests": self.context.max_requests,
            "total_request_count": self.context.total_request_count,
            "paused_request_count": self.context.paused_request_count,
            "active_token_count": self.context.active_token_count,
        }

        # Generate tokens.
        range_push("Prefill" if not is_decode_only else "Decode")
        # TODO @TDE: Account for this line when overlapping forward and bookkeep.
        self.is_decode_only = is_decode_only

        self.step_start_event.record()
        result = await self.controller.async_generate_output_tokens_dynamic_batch()
        self.step_end_event.record()
        self.step_end_event.synchronize()
        step_time = self.step_start_event.elapsed_time(self.step_end_event) / 1e3
        self.context.step_count += 1

        # Store Mamba state for blocks that completed during this prefill
        if (not is_decode_only and self.context.enable_prefix_caching
                and self.context.is_hybrid_model and self.context.max_mamba_cache_slots > 0):
            self._store_mamba_states_for_completed_prefill()


        range_pop()

        if (
            self.logging_step_interval > 0
            and self.context.step_count > 0
            and self.context.step_count % self.logging_step_interval == 0
            and self.metrics_writer is not None
        ):
            kvcache_util_stats = self.context.get_kvcache_utilization_stats()
        else:
            kvcache_util_stats = None

        post_step_context_state = {
            "waiting_request_count": len(self.waiting_request_ids),
            "finished_request_count": self.finished_request_count,
            "evicted_request_count": self.evicted_request_count,
            "kv_stats": kvcache_util_stats,
            "padded_active_token_count": self.context.padded_active_token_count,
            "using_cuda_graph_this_step": self.context.using_cuda_graph_this_step(),
            "total_active_block_count": self.context.block_allocator.active_count,
            "total_paused_block_count": self.context.block_allocator.paused_count,
            "total_active_used_blocks": self.context.block_allocator.get_active_used(),
            "total_paused_used_blocks": self.context.block_allocator.get_paused_used(),
        }

        context_state = {**pre_step_context_state, **post_step_context_state}

        return result, context_state, step_time

    async def async_bookkeep(
        self, step_result: Optional[Dict], context_state: Dict, step_time: float
    ):
        """Uses `asyncio` for continuous bookkeeping.

        Args:
            step_result (Optional[Dict]): The result of the step.
            context_state (Dict): is_decode_only, total/paused request count, active token count.
            step_time (float): How long this step took.

        Returns:
            A dictionary containing:
                active_requests (List): Requests that ran in the last step and are still active.
                finished_requests (List): Requests that ran in the last step and have now finished.
                step_time (float): The step time in seconds.
                cuda_graph_request_count (int): The CUDA graph batch size matching this step.
        """
        # Increment finished_request_count.
        range_push("bookkeeping")
        cuda_graph_request_count = None

        if step_result is not None:
            active_request_ids = step_result["active_request_ids"]
            finished_request_ids = step_result["finished_request_ids"]
            newly_paused_request_ids = step_result.get("newly_paused_request_ids")
            evict_request_ids = step_result.get("evict_request_ids")
            sample = step_result["sample"]
            log_probs = step_result["log_probs"]
            top_n_logprobs = step_result.get("top_n_logprobs", None)
            routing_indices_per_request = step_result.get("routing_indices_per_request", None)
            cuda_graph_request_count = step_result["cuda_graph_request_count"]

            # Add paused events.
            if newly_paused_request_ids is not None and self.track_paused_request_events:
                newly_paused_request_ids = newly_paused_request_ids.tolist()
                [self.get_request(i).add_event_pause() for i in newly_paused_request_ids]

            # Process finished requests (adds FINISH events and returns records).
            (active_request_ids, finished_request_records) = self.post_process_requests(
                active_request_ids,
                finished_request_ids,
                evict_request_ids,
                step_time,
                sample,
                log_probs,
                top_n_logprobs,
                routing_indices_per_request,
            )

        else:
            active_request_ids: list[int] = []
            finished_request_records: list[DynamicInferenceRequestRecord] = []

        # Failed requests.
        for failed_request_id in self.failed_request_ids:
            failed_entry = self.requests.pop(failed_request_id)
            failed_request = failed_entry.record[-1]
            failed_request.status = Status.FAILED
            failed_request.add_event_fail()
            finished_request_records.append(failed_entry.record)
            failed_entry.future.set_result(failed_entry.record)
        self.failed_request_ids.clear()
        range_pop()

        # Detokenize all finished requests if not using
        # the coordinator. Otherwise, the coordinator will
        # overlap detokenization with the engine.
        if not self.use_coordinator:
            range_push("detokenization")
            for record in finished_request_records:
                for request in record.requests:
                    if request.prompt is None:
                        request.prompt = self.controller.tokenizer.detokenize(
                            request.prompt_tokens.tolist()
                        )
                    request.generated_text = self.controller.tokenizer.detokenize(
                        request.generated_tokens
                    )
            range_pop()

        # Handle necessary ZMQ DP coordinator communication.
        if self.use_coordinator and self.is_mp_coordinator and finished_request_records:
            range_push("coordinator_communication")
            payload = msgpack.packb(
                [Headers.ENGINE_REPLY.value, [r.serialize() for r in finished_request_records]],
                use_bin_type=True,
            )
            self.socket_for_receiving_requests.send(payload)
            range_pop()

        # Log KV cache utilization stats to W&B
        if context_state["kv_stats"] is not None:
            # Prepare metrics dictionary with all stats
            # Use 'inference/' prefix for all metrics to separate from training metrics
            metrics = {
                'inference/inference_step': int(
                    self.inference_step_offset + int(self.context.step_count)
                ),
                'inference/step_time_s': float(step_time),
                'inference/waiting_queue_len': int(len(self.waiting_request_ids)),
                'inference/total_requests_dict_size': int(len(self.requests)),
            }
            # Add KV stats with inference/ prefix
            # Convert utilization metrics from 0-1 range to 0-100 percentage range for better visualization
            for key, value in context_state["kv_stats"].items():
                if 'utilization' in key:
                    # Convert to percentage (0-100) and group under kvcache_utilization
                    metrics[f'inference/{key}'] = float(value * 100.0)
                else:
                    metrics[f'inference/{key}'] = value

            if HAVE_WANDB and self.metrics_writer.__name__ == "wandb":
                self.metrics_writer.log(metrics, commit=True)
            else:
                raise ValueError(f"Unsupported metrics writer type: {type(self.metrics_writer)}")

        # Print context state.
        if (
            self.logging_step_interval > 0
            and self.context.step_count % self.logging_step_interval == 0
        ):
            mem = torch.cuda.memory_stats()
            step_type = "decode" if context_state["is_decode_only"] else "non-decode"
            output_str = (
                "* rank %d | step %d | %s ... time: %.3f%s ... "
                "reqs: a %d/%d, p %d, w %d, f %d, e %d ... "
                "blocks: a %d/%d, p %d/%d ... "
                "mem: tensors %d, alloc %.1f gb, res %.1f gb."
                % (
                    self.rank,
                    self.context.step_count,
                    datetime.now().strftime("%H:%M:%S"),
                    step_time,
                    (
                        " [%s + real config %s + cuda graph %s]"
                        % (
                            step_type,
                            self.context.batch_dimensions,
                            (
                                "OFF"
                                if not self.context.using_cuda_graph_this_step()
                                else self.context.padded_batch_dimensions
                            ),
                        )
                    ),
                    context_state["total_request_count"] - context_state["paused_request_count"],
                    context_state["max_requests"],
                    context_state["paused_request_count"],
                    context_state["waiting_request_count"],
                    context_state["finished_request_count"],
                    context_state["evicted_request_count"],
                    context_state["total_active_used_blocks"],
                    context_state["total_active_block_count"],
                    context_state["total_paused_used_blocks"],
                    context_state["total_paused_block_count"],
                    mem["allocation.all.current"],
                    mem["allocated_bytes.all.current"] / (1024**3),
                    mem["reserved_bytes.all.current"] / (1024**3),
                )
            )
            if context_state["is_decode_only"]:
                output_str = f"\033[94m{output_str}\033[0m"
            logging.info(output_str)

        return {
            "active_request_ids": active_request_ids,
            "finished_request_records": finished_request_records,
            "step_time": step_time,
            "cuda_graph_request_count": cuda_graph_request_count,
        }

    async def async_step(
        self,
    ) -> Tuple[List[DynamicInferenceRequest], List[DynamicInferenceRequest], float]:
        """
        Wrapper for controller.generate_output_tokens_dynamic_batch(), to
        match vLLM API. Uses `asyncio` for continuous generation which allows this
        method to sleep and wake up when new requests are available.

        Returns:
            A tuple comprised of:
                1. Requests that ran in the last step and are still active.
                2. Requests that ran in the last step and have now finished.
                3. The step time in seconds.
        """
        last_step_data = await self.async_forward()
        ret = await self.async_bookkeep(*last_step_data)
        # Keep for compatibility with current test suite.
        return ret

    def _run_coroutine_sync(self, coro):
        """Run a coroutine synchronously, handling the case when already in an event loop.

        This method safely runs an async coroutine from synchronous code, even when
        called from within an already running event loop (e.g., when used with async
        frameworks like pytriton).
        """
        try:
            # Check if there's already a running event loop
            asyncio.get_running_loop()
            # We're inside a running loop - run in a separate thread
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        except RuntimeError:
            # No running loop - safe to use run_until_complete
            return self._loop.run_until_complete(coro)

    def step_modern(
        self,
    ) -> Tuple[List[DynamicInferenceRequest], List[DynamicInferenceRequest], float]:
        """Synchronous wrapper for `self.async_step`."""
        return self._run_coroutine_sync(self.async_step())

    def step_legacy(
        self, sampling_params: SamplingParams
    ) -> Tuple[List[DynamicInferenceRequest], List[DynamicInferenceRequest], float]:
        """Synchronous wrapper for `self.async_step`."""
        warnings.warn(
            "`step_legacy()` is deprecated and will be removed in `megatron-core` "
            "0.16. Please use `step_modern()` going forward, which will eventually "
            "be renamed to `step()`."
        )
        result = self._run_coroutine_sync(self.async_step())
        active_requests = [self.get_request(i) for i in result["active_request_ids"]]
        finished_requests = [r.merge() for r in result["finished_request_records"]]
        return active_requests, finished_requests, result["step_time"]

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

        finished_request_records_list = []
        while self.has_unfinished_requests():
            result = self.step_modern()
            finished_request_records_list.extend(result["finished_request_records"])

        # Ensure requests are returned in the same order they were passed in.
        finished_request_records_list.sort(key=lambda r: r.request_id)

        return finished_request_records_list

    def schedule_requests(self) -> int:
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
        signals (PAUSE, UNPAUSE, SUSPEND, RESUME, STOP) update the engine's
        internal state.

        Note:
            This function is synchronous and must be called collectively by all
            ranks in a MP group. It should not be launched in a separate coroutine
            to ensure all ranks execute it in lockstep before proceeding to the
            next engine step.

        Returns:
            int: The number of messages that were received and processed in this batch.
        """

        range_push("drain_zmq_socket")
        all_messages = []
        if self.is_mp_coordinator:
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
            self.model_parallel_num_msgs_publisher_socket.send(
                struct.pack('!i', messages_to_dequeue)
            )
            # Now publish the actual messages to all model parallel ranks
            if messages_to_dequeue > 0:
                self.model_parallel_publisher_socket.send_multipart(all_messages)
        else:
            # First, receive the number of messages to dequeue from mp-rank 0
            messages_to_dequeue = struct.unpack(
                '!i', self.model_parallel_num_msgs_subscriber_socket.recv()
            )[0]
            # Now, dequeue the same number of messages from the subscriber socket.
            # Note that these receives are blocking, because the messages
            # are guaranteed to be available after the tp-rank 0 has sent them.
            if messages_to_dequeue > 0:
                all_messages = self.model_parallel_subscriber_socket.recv_multipart()
            else:
                all_messages = []

        range_pop()
        for message in all_messages:
            data = msgpack.unpackb(message, raw=False)
            header = Headers(data[0])

            if self.received_stop:
                assert (
                    header == Headers.STOP_ACK
                ), "Engine is shutting down. No other messages allowed except STOP_ACK."

            if header == Headers.SUBMIT_REQUEST:
                request_id, prompt, sampling_params = data[1:]
                sampling_params = SamplingParams.deserialize(sampling_params)
                range_push("add_request")
                self.add_request(request_id, prompt, sampling_params)
                range_pop()
            elif header == Headers.PAUSE:
                # Pause thyself.
                self.received_pause = True
                self.running.clear()
                # Send PAUSE_ACK back to coordinator.
                if self.is_mp_coordinator:
                    payload = msgpack.packb([Headers.PAUSE_ACK.value], use_bin_type=True)
                    self.socket_for_receiving_requests.send(payload)
            elif header == Headers.STOP:
                # Stop thyself.
                self.received_stop = True
                self.running.clear()
                # Send STOP_ACK back to coordinator.
                if self.is_mp_coordinator:
                    payload = msgpack.packb([Headers.STOP_ACK.value], use_bin_type=True)
                    self.socket_for_receiving_requests.send(payload)
            elif header == Headers.PAUSE_ACK:
                self.paused.set()
                self.received_pause = False
            elif header == Headers.STOP_ACK:
                self.stopped.set()
                self.received_stop = False
            elif header == Headers.UNPAUSE:
                self.paused.clear()
                self.running.set()
            elif header == Headers.SUSPEND:
                self.suspend_signal = True
            elif header == Headers.RESUME:
                self.suspend_signal = False
            elif header == Headers.INCREMENT_STALENESS:
                waiting = set(self.waiting_request_ids)
                for request_id, entry in self.requests.items():
                    entry.record.increment_staleness(policy_only=request_id in waiting)
            elif header == Headers.STOP:
                self.received_stop = True
            else:
                raise UnknownHeaderError(header)

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
            self.inference_coordinator_process.join()
        for socket in self.zmq_sockets:
            socket.close()
        if hasattr(self, "expert_parallel_zmq_communicator"):
            self.expert_parallel_zmq_communicator.close()
        self.zmq_context.term()

    @trace_async_exceptions
    async def run_engine(self, *, loop: Optional[asyncio.AbstractEventLoop] = None):
        """Continually steps the engine asynchronously."""
        self._loop = get_asyncio_loop(loop)
        self.use_coordinator = False
        try:
            while True:
                # Wait until there are active requests before proceeding.
                async with self._cond:
                    await self._cond.wait_for(
                        lambda: (
                            not self.is_suspended
                            and (
                                self.context.get_active_request_count() > 0
                                or self.waiting_request_ids
                            )
                        )
                    )
                await self.async_step()
        except asyncio.CancelledError:
            pass

    async def _ep_group_has_work(self, local_work: int) -> bool:
        """Determines if there are some pending requests in the expert parallel group this
        rank is a part of.
        Args:
            local_work (int): The local work count for this rank. This is a sum of active
            and waiting requests.
        Returns:
            bool: True if there is some work in the EP group, False otherwise.
        """
        range_push("_ep_group_has_work")

        is_stopped = self.stopped.is_set() or self.received_stop
        is_paused = self.paused.is_set() or self.received_pause
        is_suspended = self.suspend_signal
        if is_stopped or is_paused or is_suspended:
            # Signals can be received asynchronously on EP ranks.
            # We do not want a rank to pause/stop/suspend prematurely if one of it's peers
            # is yet to receive the signal.
            # So this is an *attempt* to process the signal. This rank has received the signal
            # and passes 0 to the all-reduce. If any other rank in the EP group has not received the signal yet,
            # it will pass a non-zero value to the all-reduce, and hence the global work will be non-zero,
            # and we will defer processing the signal.
            # When all ranks receive the signal, global work will be zero, and we can process the signal safely.
            local_work = 0

        if self.ep_world_size > 1:
            # Perform all-reduce to get max global work across EP group.
            # Note that it is important to use a non-blocking asyncio-friendly all-reduce here.
            # The user may have other tasks running in the event loop that need to be serviced.
            # Do not using a torch.distributed blocking all-reduce here using nccl/gloo.
            # We have tried that and it blocks the event loop is megatron-rl.
            max_global_work = await self.expert_parallel_zmq_communicator.all_reduce_max(local_work)
        else:
            max_global_work = local_work

        range_pop()
        return max_global_work > 0

    @trace_async_exceptions
    async def run_engine_with_coordinator(
        self, *, loop: Optional[asyncio.AbstractEventLoop] = None
    ):
        """Continually steps the engine asynchronously."""
        self._loop = get_asyncio_loop(loop)
        self.use_coordinator = True
        try:
            while True:
                self.schedule_requests()

                # for the cases below (no active requests, or undergoing a state-change)
                # do not use asyncio.sleep(0)
                # as tp-rank=0 will flood the num_messages publisher
                # with "0" repeatedly. This causes some packets to drop.
                # Instead be nice, and sleep
                # for a short time.
                # The minimum sleep time needed is ~100us i.e. the time
                # needed to send one message on an IPC socket. However
                # just to be safe, we use 20ms here.

                local_pending_requests = self.context.get_active_request_count() + len(
                    self.waiting_request_ids
                )
                # 1. Check for work availability (Consensus Step)
                ep_group_has_work = await self._ep_group_has_work(local_pending_requests)

                # 2. Dummy Work Logic (Keep group alive if peers have work)
                if ep_group_has_work and local_pending_requests == 0:
                    # run dummy forward pass if EP group as a whole has work,
                    # but this rank does not have any work.
                    self.step_start_event.record()
                    self.controller.dummy_forward()
                    self.step_end_event.record()
                    self.step_end_event.synchronize()
                    self.context.step_count += 1
                    continue

                # 3. No work in EP group
                # We handle control signals (PAUSE/STOP/SUSPEND) only when
                # the entire EP group has received the signal. It is important to
                # not process these signals immediately upon receipt, because
                # other ranks in the EP group may not have received them yet.
                # If we exit prematurely, other ranks will deadlock at the all-to-all.
                # We use self._ep_group_has_work() to build consensus across the EP group
                # as to when it is safe to process these signals. The function returns False
                # when all ranks have received the signal.
                if not ep_group_has_work:
                    # Priority A: STOP
                    if self.stopped.is_set():
                        if self.rank == 0:
                            logging.info("Stopping engine.")
                        self.stop()
                        break

                    # Priority B: SUSPEND
                    if self.suspend_signal:
                        self.suspend()
                    else:
                        self.resume()

                    # Priority C: PAUSE or no work - nothing needs to be done
                    # To avoid flooding the TP publisher socket with packets,
                    # we sleep for 20 ms here.
                    # todo [Siddharth]: Can this hardcoded sleep be avoided
                    # with asyncio zmq sockets?
                    await asyncio.sleep(0.02)  # Yield to event loop
                    continue

                try:
                    await self.async_step()
                except EngineSuspendedError:
                    await asyncio.sleep(0.02)
                    continue

        except asyncio.CancelledError:
            pass
