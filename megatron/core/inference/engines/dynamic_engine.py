# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import concurrent.futures
import logging
import math
import multiprocessing
import socket
import time
import warnings
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from itertools import repeat
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from megatron.core.inference.batch_dimensions_utils import (
    CUDAGraphBatchDimensionBuilder,
    InferenceBatchDimensions,
)
from megatron.core.inference.config import AsyncScheduleMode, KVCacheManagementMode
from megatron.core.inference.contexts.dynamic_context import (
    BlockOverflowError,
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
    DynamicVLMInferenceRequest,
    FinishedRequestRecord,
    Status,
    resolve_multimodal_data_for_engine,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    DecodeOnly,
    DynamicBatchControllerStepResult,
    TextGenerationController,
)
from megatron.core.inference.utils import Counter, InferenceMode, await_process_call
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.cuda_graphs import CudaGraphManager, delete_cuda_graphs
from megatron.core.transformer.enums import InferenceCudaGraphScope
from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction
from megatron.core.utils import (
    deprecate_args,
    experimental_api,
    get_asyncio_loop,
    get_pg_rank,
    get_pg_size,
    get_pg_src_rank,
    internal_api,
    nvtx_range_pop,
    nvtx_range_push,
    round_up_to_nearest_multiple,
    trace_async_exceptions,
    unwrap_model,
)

from .async_zmq_communicator import AsyncZMQCommunicator, RankedPubSub

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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEPRECATED_ARGS = [
    "enable_cuda_graph",
    "random_seed",
    "track_paused_request_events",
    "enable_chunked_prefill",
    "inference_logging_step_interval",
    "pg_collection",
]


class EngineState(Enum):
    """State machine for the inference engine."""

    RUNNING = auto()  # Processing requests
    PAUSING = auto()  # PAUSE received; waiting for EP consensus + world barrier
    PAUSED = auto()  # Globally confirmed idle
    UNPAUSING = auto()  # UNPAUSE received; waiting for world barrier
    SUSPENDING = auto()  # SUSPEND received; offloading GPU; waiting for world barrier
    SUSPENDED = auto()  # GPU offloaded, all ranks confirmed
    RESUMING = auto()  # RESUME received; onloading GPU; waiting for world barrier
    RESUMED = auto()  # GPU onloaded, all ranks confirmed; cleared on next SUSPEND
    STOPPING = auto()  # STOP received; futures cancelled; waiting for world barrier
    STOPPED = auto()  # All ranks confirmed; teardown complete


class EngineSuspendedError(Exception):
    """Engine is currently suspended and not performing steps."""

    pass


def format_mem_bytes(mem_bytes):
    """Convert a byte count to a human-readable string in tb, gb, mb, kb, or bytes."""
    if mem_bytes < 0:
        return "-" + format_mem_bytes(-mem_bytes)
    for power, suffix in [(4, "tb"), (3, "gb"), (2, "mb"), (1, "kb"), (0, "bytes")]:
        suffix_bytes = 1024**power
        if mem_bytes >= suffix_bytes:
            return "%.1f %s" % (mem_bytes / suffix_bytes, suffix)
    return "%d bytes" % mem_bytes


def _get_decode_only_log_state(
    mode: AsyncScheduleMode, decode_only: DecodeOnly
) -> Tuple[str, Optional[bool]]:
    """Build the console transition label and color state for one inference step.

    Args:
        mode (AsyncScheduleMode): Active scheduling mode.
        decode_only (DecodeOnly): Decode-only state for the consumed and launched forwards.

    Returns:
        Tuple[str, Optional[bool]]: Current step label, including the previous
            step when it differs, and whether to use decode coloring.
    """
    if mode == AsyncScheduleMode.LEGACY:
        is_decode_only = bool(decode_only)
        return ("decode" if is_decode_only else "non-decode"), is_decode_only

    current_decode_only = (
        decode_only.launched if decode_only.launched is not None else decode_only.consumed
    )
    if current_decode_only is None:
        return "idle", None

    step_type = "decode" if current_decode_only else "non-decode"
    if (
        decode_only.consumed is not None
        and decode_only.launched is not None
        and decode_only.consumed != decode_only.launched
    ):
        previous_step_type = "decode" if decode_only.consumed else "non-decode"
        step_type = f"{step_type} (prev: {previous_step_type})"

    return step_type, current_decode_only


def _cuda_graph_mempool_bytes() -> Tuple[int, int]:
    """Return (reserved, allocated) bytes belonging to the global CUDA graph mempool.

    PyTorch's `torch.cuda.memory_stats()` reports process-wide totals that mix in
    every other allocation (KV cache, NCCL workspaces, layer scratch). To isolate
    growth caused by graph capture, we walk `torch.cuda.memory_snapshot()` and
    filter segments by their `segment_pool_id` against the graph pool handle.
    Returns (0, 0) if the pool hasn't been created yet.
    """
    pool_id = CudaGraphManager.global_mempool
    if pool_id is None:
        return 0, 0
    reserved = 0
    allocated = 0
    for seg in torch.cuda.memory_snapshot():
        seg_pool_id = (
            seg.get("segment_pool_id")
            or seg.get("private_pool_id")
            or seg.get("pool_id")
            or seg.get("pool")
        )
        if seg_pool_id == pool_id:
            reserved += seg.get("total_size", 0)
            allocated += seg.get("allocated_size", 0)
    return reserved, allocated


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

    # Map stable states to their corresponding asyncio events.
    _STATE_EVENTS = (
        EngineState.RUNNING,
        EngineState.PAUSED,
        EngineState.SUSPENDED,
        EngineState.RESUMED,
        EngineState.STOPPED,
    )

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

        self.num_speculative_tokens = inference_config.num_speculative_tokens
        self.materialize_only_last_token_logits = (
            inference_config.materialize_only_last_token_logits
        )

        assert self.num_speculative_tokens >= 0, "Number of speculative tokens must be non-negative"

        if self.num_speculative_tokens > 0:
            assert (
                model_config.mtp_use_repeated_layer
                or self.num_speculative_tokens <= model_config.mtp_num_layers
            ), f"Number of speculative tokens {self.num_speculative_tokens} must be less than or equal to number of MTP layers {model_config.mtp_num_layers}"
        self.track_paused_request_events = inference_config.track_paused_request_events
        self.track_generated_token_events = inference_config.track_generated_token_events
        self.enable_chunked_prefill = inference_config.enable_chunked_prefill
        self.cuda_graph_all_prefills = inference_config.cuda_graph_all_prefills
        self.metrics_writer = inference_config.metrics_writer
        self.logging_step_interval = inference_config.logging_step_interval
        self.unified_memory_level = inference_config.unified_memory_level
        self.use_synchronous_zmq_collectives = inference_config.use_synchronous_zmq_collectives
        self.disable_ep_consensus = inference_config.disable_ep_consensus
        self.ep_consensus_interval = inference_config.ep_consensus_interval
        self.cuda_graph_impl = model_config.cuda_graph_impl
        self.inference_cuda_graph_scope = model_config.inference_cuda_graph_scope
        self.cuda_graph_modules = model_config.cuda_graph_modules
        self._validate_async_sched_support_for_config()
        # Throw a cudagraph-admission warning if deferred for > max_sequence_length steps.
        # The floor value of 100 avoids warnings in test configs where max_sequence_length < 100.
        self._cg_admission_warn_after = max(100, self.context.max_sequence_length)
        self._initialize_disaggregation_state()
        # Initialize engine.
        self.reset()

        # Set callback for getting stop word finished request IDs
        self.controller.set_stop_word_finished_ids_callback(
            self._get_and_clear_stop_word_finished_ids
        )

        # Configure wandb to use separate step counter for inference metrics (only once)
        if self.logging_step_interval > 0 and self.metrics_writer is not None:
            logger.info(
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

        # Mark the inference engine as active. Cleared in `suspend()` and re-set in `resume()`.
        InferenceMode.set_active()

        # Create cuda graphs.
        self.create_cuda_graphs()

    def _initialize_disaggregation_state(self) -> None:
        """Hook overridden by the KV-handoff engine composition."""

    def _reset_pending_kv_imports(self) -> None:
        """Hook overridden by the KV-handoff engine composition."""

    @property
    def pending_kv_import_count(self) -> int:
        """Number of decode requests awaiting a KV import (none here)."""
        return 0

    @property
    def has_admittable_kv_import(self) -> bool:
        """Whether a completed KV import is eligible for admission (false here)."""
        return False

    def _poll_pending_kv_imports(self) -> int:
        return 0

    def _admit_pending_kv_imports(self) -> int:
        return 0

    def _setup_handoff_completion_tracking(self, hostname: str | None = None) -> None:
        """Hook overridden by the KV-handoff engine composition."""

    def _drain_handoff_completion_notifications(self) -> list[tuple[int, bool]]:
        """Hook overridden by the KV-handoff engine composition."""
        return []

    def _record_handoff_completion_notification(self, request_id: int, failed: bool) -> None:
        """Hook overridden by the KV-handoff engine composition."""
        self._raise_kv_handoff_not_enabled("KV handoff completion notification")

    def _prepare_handoff_metadata_batch(
        self, requests_and_state: list[tuple], decode_tokens_by_request: Dict[int, list[int]]
    ) -> dict:
        """Hook overridden by the KV-handoff engine composition."""
        if any(request.sampling_params.do_kv_handoff for request, *_ in requests_and_state):
            self._raise_kv_handoff_not_enabled("KV handoff completion")
        return {}

    def _capture_handoff_meta(self, request, prepared) -> None:
        self._raise_kv_handoff_not_enabled("KV handoff completion")

    def _release_pinned_handoff_blocks(self, block_ids: list) -> int:
        return 0

    def _release_pinned_handoff_ssm_slot(self, ssm_slot: int | None) -> None:
        return None

    def setup_kv_transfer(self, role: str, backend: str = "nixl") -> None:
        """Raising stub; the hand-off engine composition overrides it."""
        self._raise_kv_handoff_not_enabled("KV transfer setup")

    def push_handoff_kv(self, request_id: int, decode_metas: list) -> None:
        """Raising stub; the hand-off engine composition overrides it."""
        self._raise_kv_handoff_not_enabled("SEND_KV")

    def _poll_pending_kv_pushes(self) -> int:
        return 0

    @property
    def pending_kv_push_count(self) -> int:
        """Number of prefill sends awaiting completion (none here)."""
        return 0

    def add_request_with_kv_handoff(
        self, request_id, prompt, sampling_params, kv_meta, src_block_ids
    ) -> asyncio.Future[DynamicInferenceRequest]:
        """Raising stub; the hand-off engine composition overrides it."""
        self._raise_kv_handoff_not_enabled("SUBMIT_REQUEST_WITH_KV")

    def release_handoff_blocks(self, request_id: int) -> None:
        """Raising stub; the hand-off engine composition overrides it."""
        self._raise_kv_handoff_not_enabled("RELEASE_KV")

    @staticmethod
    def _raise_kv_handoff_not_enabled(operation: str) -> None:
        raise RuntimeError(
            f"{operation} requires KV handoff, but it is not enabled. "
            "Use DisaggDynamicInferenceEngine with KV transfer configured."
        )

    def reset(self) -> None:
        """Reset by removing all requests and reset all state."""

        self._reset_pending_kv_imports()
        self.context.reset()

        # Request state.
        self.request_counter = Counter()
        self.finished_request_count = 0
        self.evicted_request_count = 0

        self.requests: Dict[int, RequestEntry] = {}
        self.waiting_request_ids = deque()
        if hasattr(self, "_pinned_handoff_blocks"):
            self._pinned_handoff_blocks.clear()
            self._pinned_handoff_ssm_slots.clear()
        self.failed_request_ids = []
        # Generated token count already streamed for each request.
        self._partial_emit_lengths: Dict[int, int] = {}
        self._generation_epoch: Optional[int] = None
        self.local_metadata_ledger_enabled: bool = False
        self.local_metadata_ledger: dict[str, FinishedRequestRecord] = {}
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
        self.decode_only = DecodeOnly(consumed=None, launched=None)
        self._loop = get_asyncio_loop(getattr(self, "_loop", None))
        self._cond = asyncio.Condition()
        self._state_events = {k: asyncio.Event() for k in self._STATE_EVENTS}
        self.state = EngineState.RUNNING
        self._state_events[EngineState.RUNNING].set()
        self._pending_signals = deque()

        self.resume_request_ids = None

        # Speculative decoding acceptance tracking (per-position).
        # Each tensor has length num_speculative_tokens; index i tracks position i+1
        # (i.e. the i-th draft token proposed by the MTP head).
        self._spec_tokens_proposed_per_pos = torch.zeros(
            self.num_speculative_tokens, dtype=torch.int64
        )
        self._spec_tokens_accepted_per_pos = torch.zeros(
            self.num_speculative_tokens, dtype=torch.int64
        )
        self._spec_steps = 0

        # Prefix caching tracking.
        self._prefix_cache_hits = 0
        self._prefix_cache_blocks_matched = 0
        self._prefill_tokens_computed = 0
        self._prefill_tokens_skipped = 0
        self._prefix_coordination_waits = 0

        # Coordinator state.
        self.use_coordinator = False

    async def wait_until(self, state: EngineState):
        """Wait until the engine reaches the given state.

        Only stable states (RUNNING, PAUSED, SUSPENDED, RESUMED,
        STOPPED) are supported.  Transient states (PAUSING, SUSPENDING,
        RESUMING, STOPPING) are not directly waitable.
        """
        event = self._state_events.get(state)
        if event is None:
            raise ValueError(f"Cannot wait for transient state {state}")
        await event.wait()

    def create_cuda_graphs(self, reset_context: bool = True):
        """Create cuda graphs.

        This method iterates the dynamic context's `cuda_graph_request_counts`
        to record and capture cuda graphs.

        Args:
            reset_context (bool): Whether to reset the context after building cuda graphs.
        """

        if self.inference_cuda_graph_scope == InferenceCudaGraphScope.none:
            return

        if self.cuda_graph_impl != "local":
            return

        context = self.context
        controller = self.controller

        time_start = time.time()
        mem_stats_start = torch.cuda.memory_stats()

        # Snapshot of process-wide stats for the "total memory used by capture" summary.
        start_proc_reserved = mem_stats_start["reserved_bytes.all.current"]
        start_proc_alloc = mem_stats_start["allocated_bytes.all.current"]

        # Pool-scoped baselines for the per-iteration deltas.
        prev_pool_reserved, prev_pool_alloc = _cuda_graph_mempool_bytes()

        logger.info("> dynamic_engine.py: building cuda graphs for ")
        for graph in context.cuda_graph_batch_dimensions_list:
            logger.info(graph)

        # Enable inference dispatcher for EP during graph capture
        model_config = controller.inference_wrapped_model.model.config

        # Pre-size the GlobalMemoryBuffer sequence-parallel all-gather buffer ("mpu")
        # to the worst case BEFORE capturing graphs. get_tensor() is grow-only: in
        # training the shape is static so it settles before capture, but dynamic
        # inference issues forwards of varying token counts. A forward larger than
        # the capture-time size would reallocate (and free) the buffer whose address
        # a captured graph still writes to on replay, corrupting whatever later
        # reuses that freed block. Allocating the max size up front keeps the address
        # stable for the graph's lifetime. Only needed when sequence parallel is on
        # (otherwise the "mpu" all-gather path is not taken).
        if getattr(model_config, "sequence_parallel", False):
            from megatron.core.parallel_state import get_global_memory_buffer

            max_ag_numel = self.context.max_tokens * model_config.hidden_size
            get_global_memory_buffer().get_tensor((max_ag_numel,), model_config.params_dtype, "mpu")

        # MTP warmup preparation: capture MTP CUDA graphs alongside the
        # decoder graphs within the same loop rather than in a separate pass.
        unwrapped = unwrap_model(controller.inference_wrapped_model.model)
        mtp_warmup_enabled = (
            controller.num_mtp_depths > 0
            and (controller.num_speculative_tokens or 0) > 0
            and hasattr(unwrapped, 'mtp')
        )
        if mtp_warmup_enabled:
            tp_size = get_pg_size(controller.inference_wrapped_model.tp_group)
            sp_enabled = model_config.sequence_parallel and tp_size > 1
            mtp_pass_depth = not unwrapped.mtp.mtp_use_repeated_layer
            mtp_warmup_depths = range(controller.num_mtp_depths) if mtp_pass_depth else [None]
            mtp_seen_batch_sizes = set()

        tbar = enumerate(context.cuda_graph_batch_dimensions_list)
        if HAVE_TQDM:
            tbar = tqdm(tbar, total=len(context.cuda_graph_batch_dimensions_list))
        for tbar_idx, cuda_graph_batch_dimension in tbar:
            input_ids, position_ids, _ = self.controller._dynamic_step_context_init(
                construct_graph_dimensions=cuda_graph_batch_dimension
            )
            # Progress.
            tbar_str = f"cuda graph warmup - {cuda_graph_batch_dimension}"
            if HAVE_TQDM:
                tbar.set_description(tbar_str)
            else:
                logger.info(
                    f"{tbar_idx}/{len(context.cuda_graph_batch_dimensions_list)}. {tbar_str}"
                )

            # Enable routing recording during warmup if routing replay is enabled.
            # This ensures the record_indices copy operation is captured in the CUDA graph.
            if model_config.moe_enable_routing_replay:
                RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)

            # Forward pass -> logits.
            with torch.inference_mode():
                controller._dynamic_step_forward_logits(input_ids, position_ids)

                if controller._sampling_backend == "flashinfer":
                    if controller.num_speculative_tokens > 0:
                        controller._dynamic_step_sample_logits_and_verify_tokens(input_ids)
                    else:
                        controller._dynamic_step_sample_logits()

                # MTP CUDA graph warmup for this batch dimension.
                if mtp_warmup_enabled:
                    n = cuda_graph_batch_dimension.req_count
                    # pylint: disable-next=possibly-used-before-assignment
                    if sp_enabled:
                        n = round_up_to_nearest_multiple(n, tp_size)
                    # pylint: disable-next=possibly-used-before-assignment
                    if n > 0 and n not in mtp_seen_batch_sizes:
                        mtp_seen_batch_sizes.add(n)
                        device = torch.cuda.current_device()
                        batch_dim = n // tp_size if sp_enabled else n
                        # Use zeros (not empty) — garbage token IDs cause OOB embedding lookups during graph capture/replay.
                        for depth in mtp_warmup_depths:
                            unwrapped.compute_mtp_single_step(
                                hidden_states=torch.zeros(
                                    (batch_dim, 1, model_config.hidden_size),
                                    device=device,
                                    dtype=model_config.params_dtype,
                                ),
                                next_token_ids=torch.zeros((1, n), device=device, dtype=torch.long),
                                position_ids=torch.zeros((1, n), device=device, dtype=torch.int64),
                                depth=depth,
                                cache_key=("mtp", n, depth),
                            )

                context.reset()

            # Per-iteration memory accounting, scoped to the CUDA-graph mempool.
            # This isolates pool growth from process-wide scratch churn (KV cache,
            # NCCL workspaces, etc.) that pollutes `torch.cuda.memory_stats()`.
            pool_reserved, pool_alloc = _cuda_graph_mempool_bytes()
            logger.info(
                "  [graph %d/%d] %s | pool reserved=%s (Δiter=%s) " "pool allocated=%s (Δiter=%s)",
                tbar_idx + 1,
                len(context.cuda_graph_batch_dimensions_list),
                cuda_graph_batch_dimension,
                format_mem_bytes(pool_reserved),
                format_mem_bytes(pool_reserved - prev_pool_reserved),
                format_mem_bytes(pool_alloc),
                format_mem_bytes(pool_alloc - prev_pool_alloc),
            )
            prev_pool_reserved, prev_pool_alloc = pool_reserved, pool_alloc

        if mtp_warmup_enabled and mtp_seen_batch_sizes:
            logger.info("> MTP CUDA graph warmup: %d batch size(s)", len(mtp_seen_batch_sizes))

        # Memory usage.
        time_end = time.time()
        mem_stats_end = torch.cuda.memory_stats()
        final_pool_reserved, final_pool_alloc = _cuda_graph_mempool_bytes()
        capture_stats = {
            "time": time_end - time_start,
            "allocated_bytes": (mem_stats_end["allocated_bytes.all.current"] - start_proc_alloc),
            "reserved_bytes": (mem_stats_end["reserved_bytes.all.current"] - start_proc_reserved),
            "pool_reserved_bytes": final_pool_reserved,
            "pool_allocated_bytes": final_pool_alloc,
        }
        logger.info(
            "> built cuda graph(s) in %.2f sec. "
            "Mempool: reserved %s, allocated %s. "
            "Process-wide delta: allocated %s, reserved %s.",
            capture_stats["time"],
            format_mem_bytes(capture_stats["pool_reserved_bytes"]),
            format_mem_bytes(capture_stats["pool_allocated_bytes"]),
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
        hostname: str | None = None,
        coordinator_schedule_output_path: str | None = None,
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
            hostname (str | None): Hostname or IP address to use for ZMQ socket binding.
                If None, defaults to `socket.gethostname()`. Should be set to a routable
                address in multi-node settings where gethostname() may return 127.0.0.1.

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

        self.zmq_context = zmq.Context.instance()
        self.zmq_sockets = []  # keep track of all sockets created by this engine

        # Get world info.
        dp_group = self.pg_collection.dp
        dp_src = get_pg_src_rank(dp_group)
        dp_size = get_pg_size(self.pg_collection.dp)
        dp_rank = get_pg_rank(self.pg_collection.dp)

        mp_group = self.pg_collection.mp
        mp_src = get_pg_src_rank(mp_group)
        mp_size = get_pg_size(mp_group)
        mp_rank = get_pg_rank(mp_group)
        dp_replica_mp_request_broadcast = RankedPubSub(
            b"DynamicInferenceEngine.mp_request_broadcast:"
        )
        tp_rank = get_pg_rank(self.pg_collection.tp)
        pp_rank = get_pg_rank(self.pg_collection.pp)

        self.is_mp_coordinator = tp_rank == 0 and pp_rank == 0
        self.is_dp_coordinator = (dp_rank == 0) and self.is_mp_coordinator

        local_ip = hostname or socket.gethostname()

        # Spawn a DP coordinator process and get the connection info.
        if launch_inference_coordinator and self.is_dp_coordinator:
            spawn_context = multiprocessing.get_context('spawn')
            deterministic_mode = torch.are_deterministic_algorithms_enabled()
            dp_pipe, dp_process_pipe = spawn_context.Pipe()
            coordinator_ready_event = spawn_context.Event()
            self.inference_coordinator_process = spawn_context.Process(
                target=DataParallelInferenceCoordinator.entrypoint,
                kwargs={
                    "pipe_connection": dp_process_pipe,
                    "ready_event": coordinator_ready_event,
                    "data_parallel_size": get_pg_size(self.pg_collection.dp),
                    "tokenizer": self.controller.tokenizer,
                    "max_requests": self.context.max_requests,
                    "inference_coordinator_port": inference_coordinator_port,
                    "deterministic_mode": deterministic_mode,
                    "block_size_tokens": self.context.block_size_tokens,
                    "enable_prefix_caching": self.context.enable_prefix_caching,
                    "prefix_caching_coordinator_policy": self.context.prefix_caching_coordinator_policy,
                    "prefix_caching_routing_alpha": self.context.prefix_caching_routing_alpha,
                    "schedule_output_path": coordinator_schedule_output_path,
                    "hostname": hostname,
                },
            )
            self.inference_coordinator_process.start()
            await await_process_call(dp_pipe.poll, self.inference_coordinator_process)
            dp_addr = dp_pipe.recv()
            dp_pipe.close()

            # Check if the port number is not inference_coordinator_port
            actual_port = int(dp_addr.rsplit(":", 1)[-1])
            if inference_coordinator_port != None and actual_port != inference_coordinator_port:
                logger.warning(
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
            mp_req_sock = dp_replica_mp_request_broadcast.create_publisher(self.zmq_context)
            mp_req_sock.bind_to_random_port(f"tcp://{local_ip}")
            mp_req_addr = mp_req_sock.getsockopt_string(zmq.LAST_ENDPOINT)
        else:
            mp_req_addr = None

        # Broadcast addresses to respective ranks.
        bcast = [dp_addr]
        torch.distributed.broadcast_object_list(bcast, src=dp_src, group=dp_group)
        [dp_addr] = bcast
        bcast = [mp_req_addr]
        torch.distributed.broadcast_object_list(bcast, src=mp_src, group=mp_group)
        [mp_req_addr] = bcast

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
            self.zmq_sockets += [
                self.socket_for_receiving_requests,
                self.model_parallel_publisher_socket,
            ]
        # All MP ranks subscribe to the publisher socket
        self.model_parallel_subscriber_socket = dp_replica_mp_request_broadcast.create_subscriber(
            self.zmq_context, mp_req_addr, mp_rank
        )

        self.zmq_sockets += [self.model_parallel_subscriber_socket]

        if self.is_mp_coordinator:
            dp_replica_mp_request_broadcast.wait_for_subscribers(
                self.model_parallel_publisher_socket, range(mp_size)
            )

        self._setup_handoff_completion_tracking(hostname)

        torch.distributed.barrier(mp_group)

        # initialize zmq-based EP communicator
        self.ep_rank = get_pg_rank(self.pg_collection.ep)
        self.ep_world_size = get_pg_size(self.pg_collection.ep)
        self._ep_consensus_loop_counter = 0
        self._last_ep_consensus: tuple[int, bool] = (0, False)
        if self.ep_world_size > 1:
            self.expert_parallel_zmq_communicator = AsyncZMQCommunicator(
                self.zmq_context, process_group=self.pg_collection.ep, hostname=hostname
            )
            # Give the context a CPU-side MAX-reduction primitive so
            # match_graph_config() can avoid a per-step NCCL AllReduce kernel.
            if hasattr(self.context, "set_ep_zmq_communicator"):
                self.context.set_ep_zmq_communicator(self.expert_parallel_zmq_communicator)

        # initialize zmq-based world communicator for consensus barriers
        total_world_size = torch.distributed.get_world_size()
        if total_world_size > 1:
            self.world_zmq_communicator = AsyncZMQCommunicator(
                self.zmq_context, process_group=None, hostname=hostname
            )

        if launch_inference_coordinator and self.is_dp_coordinator:
            await await_process_call(
                coordinator_ready_event.wait, self.inference_coordinator_process
            )
            logger.info("Inference co-ordinator is ready to receive requests!")
            logger.info(f"Data parallel coordinator can be found at {dp_addr}")

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
            nvtx_range_push(f"{key}-inference-context")
            torch.cuda.synchronize()

            yield

        finally:

            nvtx_range_pop(f"{key}-inference-context")
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
            logger.info(
                f"[rank {rank_str}] dynamic engine {key}, "
                f"unified {unified_memory_level}, "
                f"{dir_str} "
                f"{relative_mem_str} in {relative_time_str} ... "
                f"abs mem usage: {total_mem_str}"
            )

    def suspend(self):
        """Suspend engine by deallocating context's GPU state."""

        # Skip if already suspended or in the process of suspending.
        if self.state in (EngineState.SUSPENDED, EngineState.SUSPENDING):
            return

        InferenceMode.unset_active()
        dynamo_helper = getattr(self.context, "dynamo_helper", None)
        if dynamo_helper is not None:
            dynamo_helper.discard_pending_kv_stored_events()

        # Deallocate context tensors.
        with self.__class__.suspend_resume_ctx(
            "suspended", unified_memory_level=self.unified_memory_level
        ):
            self.context.deallocate_inference_state_buffers()

        if (
            dynamo_helper is not None
            and self.context.kv_cache_management_mode == KVCacheManagementMode.RECOMPUTE
        ):
            # PERSIST and OFFLOAD restore the same cache contents on resume; only
            # RECOMPUTE invalidates the blocks previously advertised to Dynamo.
            dynamo_helper.notify_kv_cache_cleared()

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

            # Reset any partially prefilled requests so they recompute from the start
            for req_id in [*waiting_request_ids, *recompute_active_ids]:
                req = self.get_request(req_id)
                if req.finished_chunk_token_count > 0:
                    req.remaining_prompt_tokens = req.prompt_tokens
                    req.finished_chunk_token_count = 0
                    req.num_matched_prefix_blocks = 0

            # Reset the chunked prefill request id
            self.chunked_prefill_request_id = -1
        else:
            recompute_active_ids = set()
        self.resume_request_ids = [*recompute_active_ids, *waiting_request_ids]
        self.waiting_request_ids.clear()

        # Checkpoint active requests that are marked for recompute.
        for request_id in recompute_active_ids:
            self.requests[request_id].record.checkpoint()

        # If we are not using the inference coordinator, we need to manually handle state.
        if not self.use_coordinator:
            self.state = EngineState.SUSPENDED

    def resume(self):
        """Resume engine by reallocating context's GPU state."""

        # Skip if not suspended or in the process of suspending.
        if self.state not in (EngineState.SUSPENDED, EngineState.SUSPENDING):
            return

        InferenceMode.set_active()

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

            # Expire stale prefix-cache entries before any request is re-added
            # below, so none of them match state produced by the pre-suspend
            # weights. Engines driven by a coordinator get their epochs from
            # SET_GENERATION_EPOCH instead; only count the resume when that
            # signal is not in play, so a cycle is never counted twice.
            if self._generation_epoch is None:
                self.context.advance_prefix_cache_epoch()

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
                request = self.get_request(request_id)
                self._add_request(request)
                # Buffer reinit above wipes the context's per-request image
                # maps. Re-register them from the preserved VLM request fields
                # so the resumed request sees its own image_embeddings /
                # image_token_mask when the controller calls
                # current_image_token_mask on the next step, rather than
                # falling through to the text-only path.
                if isinstance(request, DynamicVLMInferenceRequest):
                    self.context.add_vlm_request_data(
                        request_id,
                        image_embeddings=request.image_embeddings,
                        image_token_mask=request.image_token_mask,
                    )

            # Ensure chunked prefill request remains at the head of the waiting queue
            if self.context.chunked_prefill_request_id != -1:
                if self.context.chunked_prefill_request_id in self.waiting_request_ids:
                    self.waiting_request_ids.remove(self.context.chunked_prefill_request_id)
                    self.waiting_request_ids.appendleft(self.context.chunked_prefill_request_id)

            torch.cuda.synchronize()
            add_time = time.time() - add_time

        # Print inner timing (must be outside context manager above for correct formatting).
        logger.info(
            "    > "
            + ", ".join(
                (
                    f"inner timing: alloc {alloc_time:.3f}",
                    f"add {add_time:.3f}",
                    f"capture {capture_time:.3f}.",
                )
            )
        )

        # If we are not using the inference coordinator, we need to manually handle state.
        if not self.use_coordinator:
            self.state = EngineState.RUNNING
            # Notify the condition variable that run_engine() waits on.
            self._loop.call_soon_threadsafe(
                asyncio.create_task, self._notify_cond_for_new_request()
            )

    @trace_async_exceptions
    async def _notify_cond_for_new_request(self):
        """Helper function to notify condition variable when a new request is added."""
        async with self._cond:
            self._cond.notify_all()

    def _send_request_records_to_coordinator(
        self, records: List[DynamicInferenceRequestRecord]
    ) -> None:
        """Send completed or failed request records from the MP coordinator."""

        merged_requests = [record.merge() for record in records]
        if self.local_metadata_ledger_enabled:
            # Failed requests are sent immediately but remain in the engine until the
            # next bookkeeping pass. Index only completed requests as they are dropped.
            for merged in merged_requests:
                if merged.status == Status.FAILED:
                    continue
                assert (
                    merged.uid not in self.local_metadata_ledger
                ), f"finished-request ledger: duplicate uid {merged.uid!r}"
                self.local_metadata_ledger[merged.uid] = FinishedRequestRecord.from_request(merged)
        payload = msgpack.packb(
            [Headers.ENGINE_REPLY.value, [request.serialize() for request in merged_requests]],
            use_bin_type=True,
        )
        self.socket_for_receiving_requests.send(payload)

    def _handle_failed_request(self, request_id: int):
        """Handle a failed request by sending the reply immediately.

        The request is added to failed_request_ids so that the next bookkeeping pass can return it.
        """
        request_entry = self.requests[request_id]
        request = request_entry.record[-1]

        if self.rank == 0:
            errors = [
                e.payload
                for e in request.events
                if e.type
                in (
                    DynamicInferenceEventType.ERROR_NONTRANSIENT,
                    DynamicInferenceEventType.ERROR_TRANSIENT,
                )
            ]
            errors_str = (
                "; ".join(f"{type(e).__name__}: {e}" for e in errors) if errors else "unknown error"
            )
            warnings.warn(
                f"Request {request_id} failed to be added to the engine ({errors_str}). "
                f"Prompt Tokens: {len(request.prompt_tokens)} "
                f"Tokens to generate: {request.sampling_params.num_tokens_to_generate} "
                f"Max sequence length: {self.context.max_sequence_length} "
                f"Chunked prefill enabled: {self.enable_chunked_prefill}"
            )

        request.status = Status.FAILED
        request.add_event_fail()
        self.failed_request_ids.append(request_id)

        # Send the reply immediately, because it may never get a chance to be sent again.
        if self.use_coordinator and self.is_mp_coordinator:
            self._send_request_records_to_coordinator([request_entry.record])
        elif not self.use_coordinator:
            if request.prompt is None:
                request.prompt = self.controller.tokenizer.detokenize(
                    request.prompt_tokens.tolist()
                )
            if request.generated_tokens:
                request.generated_text = self.controller.tokenizer.detokenize(
                    request.generated_tokens
                )
            else:
                request.generated_text = ""
        request_entry.future.set_result(request_entry.record)

    def _fail_submission(
        self, request_id: int, sampling_params: Optional[SamplingParams], exc: BaseException
    ) -> None:
        """Register a minimal failed request so a rejected admission still
        produces a client-visible failure reply.

        Called from the SUBMIT_REQUEST handler when image preprocessing or
        add_request raises. Registering a placeholder record with
        Status.FAILED lets ``_handle_failed_request`` publish the ENGINE_REPLY
        without leaving the client hanging or killing the engine loop.
        """
        if self.rank == 0:
            warnings.warn(
                f"Request {request_id} rejected before admission: " f"{type(exc).__name__}: {exc}"
            )
        # Empty prompt tokens are safe — the reply short-circuits at
        # Status.FAILED and the client sees the failure, not a completion.
        placeholder_request = DynamicInferenceRequest(
            request_id=request_id,
            prompt_tokens=torch.empty(0, dtype=torch.int64),
            sampling_params=(sampling_params or SamplingParams()),
        )
        placeholder_request.status = Status.FAILED
        self.requests[request_id] = RequestEntry(
            record=DynamicInferenceRequestRecord.from_request(placeholder_request),
            future=self._loop.create_future(),
        )
        self._handle_failed_request(request_id)

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

    def _validate_async_sched_support_for_config(self) -> None:
        """Validate config-level restrictions for async scheduling.

        Raises if the config does not support async scheduling.
        """
        mode = self.context.config.async_sched_mode
        if mode == AsyncScheduleMode.LEGACY:
            return
        if mode != AsyncScheduleMode.ASYNC:
            raise AssertionError(f"Unexpected async scheduling mode: {mode}")

        model_config = self.controller.inference_wrapped_model.model.config
        if self.num_speculative_tokens > self.controller.num_mtp_depths:
            raise ValueError("Async scheduling requires one MTP depth per speculative token.")
        if model_config.moe_enable_routing_replay:
            raise ValueError("Async scheduling does not support routing replay.")

    def _add_request(
        self, request: DynamicInferenceRequest
    ) -> asyncio.Future[DynamicInferenceRequest]:
        """Add a request to the engine.

        Args:
            request (DynamicInferenceRequest): Request to add.

        Returns:
            asyncio.Future[DynamicInferenceRequest]: Future completed when the request finishes.
        """

        request_id = request.request_id

        # Add request to self.requests. If the engine has previously been
        # suspended, then the request may already exist.
        if request_id not in self.requests:
            self.requests[request_id] = RequestEntry(
                record=DynamicInferenceRequestRecord.from_request(request),
                future=self._loop.create_future(),
            )
            request.add_event_add_engine()  # Record when request enters engine

            # Stamp new request with the current generation epoch.
            if self._generation_epoch is not None:
                epoch = self._generation_epoch
                request.policy_epoch = [(0, epoch)]
                request.kv_cache_epoch = [(0, epoch)]

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

        # Clamp large `num_tokens_to_generate` instead of rejecting the request.
        # This is included for compatibility with other frameworks.
        remaining_tokens = self.context.max_sequence_length - len(request.prompt_tokens)
        if request.sampling_params.num_tokens_to_generate < 0 or remaining_tokens < 0:
            request.status = Status.FAILED
            request.add_event_error_nontransient(MaxSequenceLengthOverflowError(request_id))
        elif request.sampling_params.num_tokens_to_generate > remaining_tokens:
            requested_tokens = request.sampling_params.num_tokens_to_generate
            request.sampling_params.num_tokens_to_generate = remaining_tokens
            if self.rank == 0:
                warnings.warn(
                    f"Request {request_id} requested num_tokens_to_generate={requested_tokens} "
                    f"which exceeds the maximum sequence length of the engine. "
                    f"Clamping num_tokens_to_generate to {remaining_tokens}."
                )

        if len(request.prompt_tokens) > self.context.max_tokens and not self.enable_chunked_prefill:
            request.status = Status.FAILED
            request.add_event_error_nontransient(TokenOverflowError(request_id))

        # Check that the shared KV pool has enough blocks for this request's stored tokens:
        # the prompt, all generated tokens but the last, and the final decode step's drafts.
        max_stored_tokens = len(request.prompt_tokens)
        if request.sampling_params.num_tokens_to_generate > 1:
            max_stored_tokens += (
                request.sampling_params.num_tokens_to_generate
                - 1
                + self.context.num_speculative_tokens
            )
        request_block_count = math.ceil(max_stored_tokens / self.context.block_size_tokens)
        usable_blocks = self.context.kv_block_allocator.pool_size - 1
        if request_block_count > usable_blocks:
            request.status = Status.FAILED
            request.add_event_error_nontransient(BlockOverflowError(request_id))

        # Tokenize stop words if provided
        if request.sampling_params.stop_words:
            stop_word_ids = [
                self.controller.tokenize_prompt(self.controller.tokenizer, stop_word, add_BOS=False)
                for stop_word in request.sampling_params.stop_words
            ]
            request.stop_word_ids = stop_word_ids

        if request.status != Status.FAILED:
            self.waiting_request_ids.append(request_id)
        else:
            self._handle_failed_request(request_id)

        return self.requests[request_id].future

    def add_request(
        self,
        request_id: int,
        prompt: Union[str, List[int], Tensor],
        sampling_params: Optional[SamplingParams] = None,
        precomputed_block_hashes: Optional[List[int]] = None,
        *,
        imgs: Optional[Tensor] = None,
        num_tiles: Optional[Tensor] = None,
        num_img_embeddings_per_tile: int = 0,
        imgs_sizes: Optional[Tensor] = None,
    ) -> asyncio.Future[DynamicInferenceRequest]:
        """Add request to inference context.

        Supports both text-only and multimodal requests. For text-only, call
        with just (request_id, prompt, sampling_params). For multimodal, also
        pass imgs and either (num_tiles + num_img_embeddings_per_tile) for static
        resolution or imgs_sizes for dynamic resolution.

        When multimodal kwargs are provided the method will:
        1. Expand image tokens in the prompt (replace <image> with padding).
        2. Run the vision encoder to produce image embeddings.
        3. Store the embeddings and mask in the context for later use by the
           controller's forward step.

        Args:
            request_id (int): Unique ID of request.
            prompt (Union[str, Tensor]): Prompt as either a text string or token IDs.
            sampling_params (Optional[SamplingParams]): Sampling parameters for the request.
            precomputed_block_hashes (Optional[List[int]]): Prefix-cache hashes already
                computed for the prompt's complete blocks. Values must match
                ``compute_block_hashes_batched(prompt_tokens, block_size_tokens)``.
            imgs (Optional[Tensor]): Image tensor [num_tiles, C, H, W] or
                [1, total_patches, patch_features] (or None).
            num_tiles (Optional[Tensor]): Number of tiles per image (1-D tensor, or None).
                Static resolution.
            num_img_embeddings_per_tile (int): Number of image embeddings per tile.
                Static resolution.
            imgs_sizes (Optional[Tensor]): Per-image sizes [N, 2] with [H, W].
                Dynamic resolution.

        Return:
            Returns an asyncio `Future[DynamicInferenceRequest]` for the user to wait on.
        """
        prompt_str = None
        # Tokenize prompt if text.
        if isinstance(prompt, str):
            # Tokenize prompt if text. Support legacy single-arg mocks.
            prompt_str = prompt
            try:
                prompt_token_ids = self.controller.tokenize_prompt(
                    self.controller.tokenizer, prompt, sampling_params.add_BOS
                )
            except TypeError:
                prompt_token_ids = self.controller.tokenize_prompt(
                    self.controller.tokenizer, prompt
                )
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

        if imgs is not None or num_tiles is not None or imgs_sizes is not None:
            request = self._build_vlm_request(
                request_id=request_id,
                prompt_str=prompt_str,
                tokens=tokens,
                sampling_params=sampling_params,
                imgs=imgs,
                num_tiles=num_tiles,
                num_img_embeddings_per_tile=num_img_embeddings_per_tile,
                imgs_sizes=imgs_sizes,
                precomputed_block_hashes=precomputed_block_hashes,
            )
            # _build_vlm_request has already registered the image embeddings
            # and token mask into the context (add_vlm_request_data). If
            # _add_request now rejects the request (oversized prompt, cache
            # exhaustion, ...), those tensors would linger in the context
            # dicts and leak GPU memory. Clean them up on failure.
            try:
                return self._add_request(request)
            except Exception:
                self.context.remove_vlm_request_data(request_id)
                raise
        else:
            request = DynamicInferenceRequest(
                request_id=request_id,
                prompt=prompt_str,
                prompt_tokens=tokens,
                sampling_params=sampling_params,
                block_size_tokens=self.context.block_size_tokens,
                enable_prefix_caching=self.context.enable_prefix_caching,
                precomputed_block_hashes=precomputed_block_hashes or [],
            )

        return self._add_request(request)

    def _resolve_image_token_id(self) -> Optional[int]:
        """Return the model's image token id, whichever wrapper level holds it.

        None when the model marks images with a negative sentinel instead of a
        real vocabulary entry (LLaVA's DEFAULT_IMAGE_TOKEN_INDEX), since such an
        id is no more decodable than the padding it would replace.
        """
        module = getattr(self.controller.inference_wrapped_model, "model", None)
        while module is not None:
            image_token_index = getattr(module, "image_token_index", None)
            if image_token_index is not None:
                image_token_index = int(image_token_index)
                return image_token_index if image_token_index >= 0 else None
            module = getattr(module, "module", None)
        return None

    def _build_vlm_request(
        self,
        *,
        request_id: int,
        prompt_str: Optional[str],
        tokens: Tensor,
        sampling_params: Optional[SamplingParams],
        imgs: Optional[Tensor],
        num_tiles: Optional[Tensor],
        num_img_embeddings_per_tile: int,
        imgs_sizes: Optional[Tensor],
        precomputed_block_hashes: Optional[List[int]] = None,
    ) -> DynamicVLMInferenceRequest:
        """Expand image tokens, run the vision encoder, register per-request
        image data on the context, and return a DynamicVLMInferenceRequest.
        """
        # PP>1 needs a non-first-stage embedding recv path (the wrapper's
        # _recv_only_vision_embeds TODO). Until that lands, only PP=1 is
        # correct: non-first stages would see None embeddings but a non-None
        # mask and silently skip image splicing.
        pp_group = self.controller.pp_group
        if pp_group is not None and torch.distributed.is_initialized():
            pp_world_size = torch.distributed.get_world_size(pp_group)
            if pp_world_size > 1:
                raise NotImplementedError(
                    "Dynamic VLM inference does not support pipeline parallel. "
                    "PP>1 requires the non-first-stage embedding recv path "
                    "which is not yet available upstream."
                )

        device = torch.cuda.current_device()
        if imgs is not None:
            imgs = imgs.to(device=device)
        if num_tiles is not None:
            num_tiles = num_tiles.to(device=device)
        if imgs_sizes is not None:
            imgs_sizes = imgs_sizes.to(device=device)

        is_dynamic_resolution = imgs_sizes is not None and imgs is not None
        # Dynamic-resolution requests derive their embedding count from
        # imgs_sizes downstream and don't need num_tiles.sum() at admission.
        # Static-tiling requests do; only pay the D2H sync on that path so
        # dynamic-res admissions stay sync-free here.
        if is_dynamic_resolution:
            total_num_tiles = 0
            num_img_embeddings = 0
            has_images = True
        else:
            total_num_tiles = int(num_tiles.sum().item()) if num_tiles is not None else 0
            num_img_embeddings = num_img_embeddings_per_tile * total_num_tiles
            has_images = num_img_embeddings > 0

        mask_tensor: Optional[Tensor] = None
        image_embeddings: Optional[Tensor] = None

        if has_images:
            token_list: List[List[int]] = [tokens.tolist()]
            expanded_tokens_list, mask_list = (
                self.controller.inference_wrapped_model.expand_image_tokens(
                    token_list, num_tiles=num_tiles, imgs_sizes=imgs_sizes
                )
            )
            # expand_image_tokens pads the embedding slots with -1, but the mask
            # below is what splices the embeddings in, so keep a real token id in
            # prompt_tokens where the model has one: they are echoed to HTTP
            # clients, detokenized for raw_text and hashed for prefix caching, and
            # none of those accept a negative id.
            expanded_tokens = expanded_tokens_list[0]
            image_token_id = self._resolve_image_token_id()
            if image_token_id is not None:
                expanded_tokens = [
                    image_token_id if token < 0 else token for token in expanded_tokens
                ]
            tokens = torch.tensor(expanded_tokens, dtype=torch.int64, device=device)
            mask_tensor = torch.tensor(
                [(-1 if v is None else int(v)) for v in mask_list[0]], device=device
            )

        # PP>1 is rejected above, so we're on the (only) stage that owns the
        # vision encoder — no is_pipeline_first_stage check needed here.
        if has_images and imgs is not None:
            with torch.inference_mode():
                image_embeddings = self.controller.inference_wrapped_model._forward_vision_encoder(
                    imgs, num_image_tiles=num_tiles, imgs_sizes=imgs_sizes
                )

        self.context.add_vlm_request_data(
            request_id, image_embeddings=image_embeddings, image_token_mask=mask_tensor
        )

        # Image-bearing requests: skip prefix caching. After image expansion,
        # two requests with the same text but different images produce
        # identical token sequences (runs of -1 pads), so KV block hashes
        # collide and the second request would serve completions conditioned
        # on the first request's image. Disabling caching at the request
        # level is a correctness fix; a follow-up could mix an image digest
        # into the block hash for cross-request reuse of identical (text,
        # image) pairs.
        request_has_images = has_images
        enable_prefix_caching = self.context.enable_prefix_caching and not request_has_images
        return DynamicVLMInferenceRequest(
            request_id=request_id,
            prompt=prompt_str,
            prompt_tokens=tokens,
            sampling_params=sampling_params,
            block_size_tokens=self.context.block_size_tokens,
            enable_prefix_caching=enable_prefix_caching,
            precomputed_block_hashes=precomputed_block_hashes or [],
            num_img_embeddings_per_tile=num_img_embeddings_per_tile,
            imgs=imgs,
            num_tiles=num_tiles,
            decoder_seq_length=0,
            image_embeddings=image_embeddings,
            image_token_mask=mask_tensor,
        )

    def post_process_requests(
        self,
        request_ids: torch.Tensor,
        finished_request_ids: torch.Tensor,
        evict_request_ids: torch.Tensor,
        step_time: float,
        sample: torch.Tensor,
        accepted_tokens: torch.Tensor,
        log_probs: torch.Tensor,
        consumed_chunked_prefill_request_id: int,
        top_n_logprobs: Optional[Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        pre_fwd_active_token_count: Optional[int] = None,
        pre_fwd_step_count: Optional[int] = None,
        finished_routing_block_ids: Optional[Dict[int, list[int]]] = None,
        finished_handoff_block_ids: Optional[Dict[int, list[int]]] = None,
        finished_handoff_ssm_slots: Optional[Dict[int, int]] = None,
        finished_handoff_decode_tokens: Optional[Dict[int, list[int]]] = None,
    ) -> Tuple[List[DynamicInferenceRequest], List[DynamicInferenceRequest]]:
        """
        Handles post-processing for requests after a step.

        Args:
            request_ids (torch.Tensor): A list of request_ids
            finished_request_ids (torch.Tensor): A list of finished request ids
            evict_request_ids (torch.Tensor): A list of evicted request ids.
            step_time (float): The latency of the last step
            sample: Tensor: The newly generated token for each request
            accepted_tokens: Tensor: The additional accepted tokens for each request
            log_probs: (List): Log probs for each request
            consumed_chunked_prefill_request_id (int): Chunked-prefill request ID
                associated with the consumed forward, or -1 if it had no partial chunk.
            top_n_logprobs: (Dict): Top-n log probs for each request. Maps request_idx to
                list of (top_n_logprobs, top_n_indices) tuples.
            pre_fwd_active_token_count (Optional[int]): Active token count for the
                consumed forward.
            pre_fwd_step_count (Optional[int]): Step count for the consumed forward.
            finished_routing_block_ids: (Dict[int, List[int]]): Block IDs for
                finished requests, saved before update_requests released them.
                Used for per-block routing reconstruction.
            finished_handoff_block_ids: Prompt KV block IDs retained for state handoff.
            finished_handoff_ssm_slots: Live SSM slots detached for state handoff.
            finished_handoff_decode_tokens: First sampled token and optional MTP proposals
                needed to resume directly from imported prefill state on decode.

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
        block_allocator = self.context.kv_block_allocator

        # Pre-compute step-level block stats (before the per-request loop)
        if self.track_generated_token_events:
            blocks_allocated = block_allocator.pool_size - block_allocator.pool_avail
            if block_allocator.enable_prefix_caching:
                blocks_hashed_active = int((block_allocator.block_ref_counts > 0).sum().item())
                blocks_ref_count = block_allocator.block_ref_counts.sum().item()
            else:
                blocks_hashed_active = blocks_allocated
                blocks_ref_count = None

        # When accepted_tokens is None (no speculative decoding), use repeat([]) to provide
        # empty lists for each request, so the zip produces the correct number of iterations
        accepted_tokens_iter = repeat([]) if accepted_tokens is None else accepted_tokens.tolist()

        if self.num_speculative_tokens > 0 and accepted_tokens is not None:
            self._spec_steps += 1

        # Convert the step's request IDs once, then batch-prepare handoff metadata for
        # finished requests using the KV blocks retained before context cleanup.
        request_id_list = request_ids.tolist()
        handoff_blocks_by_request = finished_handoff_block_ids or {}
        handoff_ssm_slots_by_request = finished_handoff_ssm_slots or {}
        prepared_handoff_metadata = self._prepare_handoff_metadata_batch(
            [
                (
                    self.get_request(request_id),
                    handoff_blocks_by_request.get(request_id, []),
                    handoff_ssm_slots_by_request.get(request_id),
                )
                for request_id in request_id_list
                if request_id in finished_request_ids
            ],
            finished_handoff_decode_tokens or {},
        )

        for req_idx, (request_id, tokens, accepted_tokens_list, request_log_probs) in enumerate(
            zip(request_id_list, sample.tolist(), accepted_tokens_iter, log_probs_iter)
        ):

            # Ensure tokens is always a list for consistent handling
            if not isinstance(tokens, list):
                tokens = [tokens]

            request: DynamicInferenceRequest = self.get_request(request_id)

            if self.num_speculative_tokens > 0:
                accepted_tokens = list(filter(lambda tok: tok != -1, accepted_tokens_list))

                # The order `accepted_tokens + tokens` is correct here.
                # `accepted_tokens` contains the sequence of
                # successfully verified draft tokens. `tokens` (from `sample`) is the
                # brand new token generated by the target model based on that accepted prefix.
                # Therefore, the newly sampled token must go at the end of the sequence.
                tokens = accepted_tokens + tokens

            num_stop_word_trim = 0
            is_prefill = len(request.generated_tokens) == 0
            if request_id != consumed_chunked_prefill_request_id:
                # Skip appending token for requests being finished due to stop words
                # (they already have their final token from the previous step)
                # If the request already has more tokens, then we only append as much as is necessary
                if (
                    len(request.generated_tokens) + len(tokens)
                    >= request.sampling_params.num_tokens_to_generate
                ):
                    keep = request.sampling_params.num_tokens_to_generate - len(
                        request.generated_tokens
                    )
                    num_tokens_before_trim = len(tokens)
                    tokens = tokens[:keep]
                    # Drop only the excess *trailing* log probs / top-n so the counts stay
                    # in sync. We must trim from the end, not the front: on a prefill step
                    # request_log_probs covers the whole prompt and is laid out as
                    # [<prompt log probs...>, <sampled token log prob>], so front-slicing
                    # (e.g. [:keep] with keep == 0 when num_tokens_to_generate == 0) would
                    # discard the prompt log probs that echo+logprobs requests need. In a
                    # decode step all entries are generated, so trailing == front-equivalent.
                    num_dropped = num_tokens_before_trim - len(tokens)
                    if num_dropped > 0:
                        if request_log_probs is not None:
                            request_log_probs = request_log_probs[:-num_dropped]
                        if top_n_logprobs is not None and req_idx in top_n_logprobs:
                            top_n_logprobs[req_idx] = top_n_logprobs[req_idx][:-num_dropped]
                if request_id not in self.stop_word_being_finished_ids:
                    is_first_token = len(request.generated_tokens) == 0
                    request.generated_tokens += tokens
                    first_token_event = None
                    if self.track_generated_token_events:
                        for token in tokens:
                            if block_allocator.enable_prefix_caching:
                                event = request.add_event_generated_token(
                                    token,
                                    blocks_total=block_allocator.pool_size,
                                    blocks_hashed_total=blocks_allocated,
                                    blocks_hashed_active=blocks_hashed_active,
                                    blocks_ref_count=blocks_ref_count,
                                    pre_fwd_active_token_count=pre_fwd_active_token_count,
                                    pre_fwd_step_count=pre_fwd_step_count,
                                )
                            else:
                                event = request.add_event_generated_token(
                                    token,
                                    blocks_total=block_allocator.pool_size,
                                    blocks_hashed_total=blocks_allocated,
                                    blocks_hashed_active=blocks_hashed_active,
                                    pre_fwd_active_token_count=pre_fwd_active_token_count,
                                    pre_fwd_step_count=pre_fwd_step_count,
                                )
                            if first_token_event is None:
                                first_token_event = event
                    if is_first_token and tokens:
                        if not self.track_generated_token_events:
                            first_token_event = DynamicInferenceEvent(
                                type=DynamicInferenceEventType.GENERATED_TOKEN,
                                payload={"token_id": tokens[0]},
                            )
                        request.ttft = (
                            first_token_event.timestamp - request.event_add_engine.timestamp
                        )
                    # TPOT is observability-only. step_time is 0.0 on
                    # non-logging steps (async_forward skips the event sync),
                    # so gate the update to keep the metric a truthful sparse
                    # sample instead of polluting it with zeros.
                    if step_time > 0 and tokens:
                        per_token_step_time = step_time / len(tokens)
                        request.tpot.extend([per_token_step_time] * len(tokens))

                # Check for stop words (after token is appended).
                # With speculative decoding, a stop word may end before the last
                # appended token. The check truncates generated_tokens in-place and
                # returns how many trailing tokens were removed so we can also trim
                # the corresponding log probs below.
                stop_word_hit, num_stop_word_trim = self._check_stop_words_for_request_post_append(
                    request
                )

                # Track per-position acceptance statistics for logging.
                # Skip prefill requests: MTP heads only propose speculative tokens
                # for decode requests, so counting prefill requests would inflate
                # the denominator and artificially deflate the acceptance rate.
                if (
                    not is_prefill
                    and len(request.generated_tokens) > 0
                    and self.num_speculative_tokens > 0
                ):
                    actual_proposed = max(0, self.num_speculative_tokens - num_stop_word_trim)
                    self._spec_tokens_proposed_per_pos[:actual_proposed] += 1
                    accepted_t = torch.tensor(accepted_tokens_list[:actual_proposed])
                    self._spec_tokens_accepted_per_pos[:actual_proposed] += (
                        accepted_t != -1
                    ).long()

                if request_id in finished_request_ids:
                    # Reconstruct routing from per-block storage before popping.
                    if (
                        finished_routing_block_ids
                        and request_id in finished_routing_block_ids
                        and len(self.requests[request_id].record.requests) == 1
                    ):
                        block_ids = finished_routing_block_ids[request_id]
                        total_tokens = len(request.prompt_tokens) + len(request.generated_tokens)
                        request.routing_indices = (
                            self.context.kv_block_allocator.reconstruct_routing_from_blocks(
                                block_ids, total_tokens - 1
                            )
                        )

                    # Request finished by normal means (termination_id, max_length, or stop word from previous step)
                    request.generated_length = len(request.generated_tokens)
                    request.status = Status.COMPLETED
                    request.add_event_finish()
                    # Keep handoff blocks only when the request needs them.
                    handoff_blocks = handoff_blocks_by_request.get(request_id, [])
                    if request.sampling_params.do_kv_handoff:
                        self._capture_handoff_meta(
                            request, prepared_handoff_metadata.get(request_id)
                        )
                    # A prefill-role engine may also serve regular requests; release the
                    # temporary state ownership when no handoff was requested.
                    elif handoff_blocks or request_id in handoff_ssm_slots_by_request:
                        self._release_pinned_handoff_blocks(handoff_blocks)
                        self._release_pinned_handoff_ssm_slot(
                            handoff_ssm_slots_by_request.get(request_id)
                        )
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

            # When a stop word was found mid-speculative-batch, trim log probs
            # and top_n_logprobs to match the truncated generated_tokens.
            if num_stop_word_trim > 0:
                if request_log_probs is not None:
                    request_log_probs = request_log_probs[:-num_stop_word_trim]
                if top_n_logprobs is not None and req_idx in top_n_logprobs:
                    top_n_logprobs[req_idx] = top_n_logprobs[req_idx][:-num_stop_word_trim]

            # Process log_probs if available (unified for both regular and chunked prefill)
            # Skip for requests being finished due to stop words — tokens are not
            # appended for these requests, so log probs must also be skipped to keep
            # the two lists in sync.
            if (
                request_log_probs is not None
                and request_id not in self.stop_word_being_finished_ids
            ):
                # Initialize lists if they don't exist
                if not request.prompt_log_probs:
                    request.prompt_log_probs = []
                if not request.generated_log_probs:
                    request.generated_log_probs = []

                is_chunked_prefill = request_id == consumed_chunked_prefill_request_id
                is_prefill = len(request.generated_log_probs) == 0

                if request.sampling_params.skip_prompt_log_probs:
                    # We only want decode log probs.
                    if is_chunked_prefill:
                        pass
                    elif is_prefill:
                        request.generated_log_probs.append(request_log_probs[-1])
                    else:
                        request.generated_log_probs.extend(request_log_probs)
                else:
                    # Split log probs between prompt and generated based on remaining prompt slots.
                    prompt_length = len(request.prompt_tokens)
                    total_accumulated = len(request.prompt_log_probs) + len(
                        request.generated_log_probs
                    )
                    remaining_prompt_slots = max(0, prompt_length - 1 - total_accumulated)
                    split_idx = min(remaining_prompt_slots, len(request_log_probs))

                    if split_idx > 0:
                        request.prompt_log_probs.extend(request_log_probs[:split_idx])
                    if split_idx < len(request_log_probs):
                        request.generated_log_probs.extend(request_log_probs[split_idx:])

            # Process top_n_logprobs if available (unified for both regular and chunked prefill)
            # Same stop-word guard as log probs above.
            if (
                top_n_logprobs is not None
                and req_idx in top_n_logprobs
                and request_id not in self.stop_word_being_finished_ids
            ):
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

        # Remove VLM data for finished requests from the context.
        for record in finished_request_records:
            req = record[-1]
            self.context.remove_vlm_request_data(req.request_id)

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

    def _check_stop_words_for_request_post_append(
        self, request: DynamicInferenceRequest
    ) -> Tuple[bool, int]:
        """Check if a request should stop due to stop words (after token is appended).

        This method is called from post_process_requests after the token has already
        been appended to request.generated_tokens. In the speculative decoding case,
        multiple tokens may have been appended at once. If a stop word is found in the
        middle of the speculative tokens, the trailing tokens after the stop word are
        truncated from generated_tokens.

        With speculative decoding, multiple tokens are appended at once. The stop word
        may end before the last appended token, leaving extra tokens that must be
        trimmed. When this happens, generated_tokens is truncated in-place and the
        number of trimmed tokens is returned so the caller can also trim log probs.

        Args:
            request: The request to check.

        Returns:
            Tuple of (stop_word_hit, num_tokens_trimmed):
                stop_word_hit: True if the generated sequence contains a stop word.
                num_tokens_trimmed: Number of tokens removed from the end of
                    generated_tokens (0 when the stop word is at the very end
                    or when no stop word was found).
        """
        if request.stop_word_ids is None or len(request.stop_word_ids) == 0:
            return False, 0

        generated_tokens = request.generated_tokens

        for stop_word_ids in request.stop_word_ids:
            stop_len = len(stop_word_ids)
            if len(generated_tokens) >= stop_len:
                # Check the last stop_len tokens shifting by 1 up to num_speculative_tokens.
                # Speculative decoding can append multiple tokens at once, so the stop
                # word might end at any position within the newly appended tokens.
                for i in range(self.num_speculative_tokens + 1):
                    end_idx = -i if i > 0 else None
                    if list(generated_tokens[-stop_len - i : end_idx]) == stop_word_ids:
                        trim = (
                            i if request.sampling_params.detokenize_stop_sequence else i + stop_len
                        )
                        if trim > 0:
                            request.generated_tokens = request.generated_tokens[:-trim]
                        return True, trim
        return False, 0

    def get_prefix_coordination_metrics(self) -> dict:
        """Return prefix caching coordination metrics.

        Returns:
            Dict with coordination stats including the number of scheduling waits.
        """
        return {"waits": self._prefix_coordination_waits}

    def _mamba_batch_invariant_prefill_chunk_length(
        self, req: DynamicInferenceRequest, capacity: int
    ) -> int:
        """Raw prefill length that computes an aligned chunk within `capacity`.

        Non-final calls must start and end at SSM chunk boundaries. The final
        prompt call may be shorter because it seeds the decode replay tail.
        """
        remaining = len(req.remaining_prompt_tokens)
        if capacity >= remaining:
            return remaining

        alignment = self.context.ssm_chunk_alignment
        computed_tokens = (capacity // alignment) * alignment
        if remaining - computed_tokens == 1:
            computed_tokens -= alignment
        if computed_tokens <= 0:
            return 0
        return computed_tokens

    def schedule_waiting_requests(self) -> None:
        """Try to schedule requests from the waiting pool."""
        # Keep track of which requests get scheduled.
        waiting_before = set(self.waiting_request_ids)
        if self.enable_chunked_prefill:
            self.schedule_chunked_prefill()
        else:
            self.schedule_non_chunked_prefill()
        waiting_after = set(self.waiting_request_ids)

        # Re-stamp kv_cache_epoch on requests that were just scheduled.
        if self._generation_epoch is not None:
            for request_id in waiting_before - waiting_after:
                req = self.get_request(request_id)
                if req.kv_cache_epoch is None:
                    req.kv_cache_epoch = [(0, self._generation_epoch)]

    def _can_schedule_non_chunked_prefill(self, req, *, record_cg_wait: bool) -> bool:
        """Return whether the queue-head request can be admitted now.

        Args:
            req: Queue-head inference request.
            record_cg_wait (bool): Whether a CUDA-graph miss should update the
                request's wait counter.

        Returns:
            bool: Whether all request, token, KV-cache, and CUDA-graph checks pass.
        """
        if not all(self.context.check_availability(req)):
            return False

        if not self._cg_admission_gating_active():
            return True

        candidate = InferenceBatchDimensions(
            token_count=self.context.active_token_count + len(req.remaining_prompt_tokens),
            prefill_req_count=self.context.num_prefill_requests + 1,
            decode_req_count=self.context.num_decode_requests,
        )
        if record_cg_wait:
            return self._cg_admission_check(req, candidate)
        return self._matches_cg_admission(candidate)

    def _can_schedule_chunked_prefill(self, req) -> bool:
        """Return whether the queue-head request can admit at least one prompt token.

        Args:
            req: Queue-head inference request.

        Returns:
            bool: Whether request, token, and KV-cache capacity permit a chunk.
        """
        request_can_be_added, _, kv_cache_available = self.context.check_availability(req)
        is_continuing_chunk = self.context.chunked_prefill_request_id == req.request_id
        token_capacity_available = self.context.active_token_count < self.context.max_tokens
        return (
            (is_continuing_chunk or request_can_be_added)
            and kv_cache_available
            and token_capacity_available
        )

    def _should_run_async_sched_overlap(self) -> bool:
        """Return whether this step should use overlap ordering.

        Returns:
            bool: Whether the next step can use overlap ordering.
        """
        # No-overlap also handles the first decode-only forward after prefill:
        # pending prefill output must be resolved before preparing its decode rows.
        # Paused requests and insufficient KV capacity likewise require complete
        # lifecycle bookkeeping before preparing the next batch.
        if not self.context.can_prepare_requests():
            return False
        if self.has_admittable_kv_import:
            return False
        if not self.waiting_request_ids:
            return True

        req = self.get_request(self.waiting_request_ids[0])
        if self.enable_chunked_prefill:
            return not self._can_schedule_chunked_prefill(req)
        return not self._can_schedule_non_chunked_prefill(req, record_cg_wait=False)

    def schedule_non_chunked_prefill(self) -> None:
        """Schedule non-chunked prefill requests."""
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

            if self._can_schedule_non_chunked_prefill(req, record_cg_wait=True):
                # Add these hashes to pending.
                if prefix_caching_enabled:
                    for block_hash in req.precomputed_block_hashes:
                        if block_hash not in self.context.kv_block_allocator.kv_hash_to_block_id:
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

    def _cg_admission_gating_active(self) -> bool:
        """Cudagraph-aware admission gating is active when --inference-cuda-graph-all-prefills
        is set, the engine has prefill/mixed CGs, and the batch-dim list is populated.

        All are required so legacy tests that exercise the scheduler without intending to run on
        captured graphs are unaffected. Gating is opt-in via `cuda_graph_all_prefills`.
        """
        return (
            self.cuda_graph_all_prefills
            and self.context.use_cuda_graphs_for_non_decode_steps
            and bool(self.context.cuda_graph_batch_dimensions_list)
        )

    def _find_cg_chunk_size(self, max_chunk_tokens: int) -> Optional[int]:
        """Return the largest chunk size <= max_chunk_tokens where batch matches a captured graph,
        or None if no graph covers any chunk in the budget.

        Walks the captured-CG list (sorted descending by token_count) and returns the first chunk
        that falls within budget and produces an applicable batch_dim under the engine's matching
        mode (strict for hybrid models). Callers must explicitly handle the None case by deferring
        the admission rather than scheduling eagerly.
        """
        active_tok = self.context.active_token_count
        active_p = self.context.num_prefill_requests
        active_d = self.context.num_decode_requests
        strict = self.context.is_hybrid_model

        for cg in self.context.cuda_graph_batch_dimensions_list:
            chunk = cg.token_count - active_tok
            if chunk < 1:
                continue
            if chunk > max_chunk_tokens:
                continue
            candidate = InferenceBatchDimensions(
                token_count=cg.token_count,
                prefill_req_count=active_p + 1,
                decode_req_count=active_d,
            )
            # candidate.token_count == cg.token_count, so the token-dimension check inside
            # is_applicable_for_batch_dim is always True here; this call filters on P/D compatibility only.
            if cg.is_applicable_for_batch_dim(candidate, strict=strict):
                return chunk

        return None

    def _register_cg_wait(self, req) -> None:
        """Track a deferred admission attempt and throw a starvation warning at the threshold.

        Decode is bounded by the number of decode steps.
        Persistent waits past `_cg_admission_warn_after` consecutive steps signal a problem.
        """
        req.cg_wait_iters += 1
        if req.cg_wait_iters % self._cg_admission_warn_after == 0:
            logger.warning(
                "request %d has been deferred by CG-aware admission for %d steps — "
                "possible starvation (strict=%s, active P=%d D=%d tok=%d)",
                req.request_id,
                req.cg_wait_iters,
                self.context.is_hybrid_model,
                self.context.num_prefill_requests,
                self.context.num_decode_requests,
                self.context.active_token_count,
            )

    def _cg_admission_check(self, req, candidate: InferenceBatchDimensions) -> bool:
        """Return True if the candidate batch shape matches a captured cudagraph.

        On miss, registers a wait + warning via `_register_cg_wait`. On hit, resets the counter.
        Caller is responsible for breaking the scheduler loop on False.
        Passes match_ep_token_counts=False so this local admission probe doesn't force a per-attempt
        NCCL all-reduce — the step-time matcher does its own EP sync.

        Args:
            req: Request whose CUDA-graph wait state should be updated.
            candidate (InferenceBatchDimensions): Candidate batch after admission.

        Returns:
            bool: Whether a compatible captured graph exists.
        """
        if self._matches_cg_admission(candidate):
            req.cg_wait_iters = 0
            return True
        self._register_cg_wait(req)
        return False

    def _matches_cg_admission(self, candidate: InferenceBatchDimensions) -> bool:
        """Return whether a candidate batch matches a captured CUDA graph.

        Args:
            candidate (InferenceBatchDimensions): Candidate batch after admission.

        Returns:
            bool: Whether a compatible captured graph exists.
        """
        matched = CUDAGraphBatchDimensionBuilder.match_graph_config(
            real_batch_dim=candidate,
            cuda_graph_batch_dimensions_list=self.context.cuda_graph_batch_dimensions_list,
            strict=self.context.is_hybrid_model,
            match_ep_token_counts=False,
        )
        return matched is not None

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
            batch_invariant_mamba_prefill = (
                self.context.batch_invariant_mode and self.context.is_hybrid_model
            )

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

            # Use remaining prompt tokens for scheduling decisions
            remaining_len = len(req.remaining_prompt_tokens)

            if self._can_schedule_chunked_prefill(req):
                # How many tokens we can admit this step.
                token_budget = self.context.max_tokens - self.context.active_token_count

                # Prefix-cache skip: on a request's first chunk, the tokens covered
                # by a cached prefix are reused rather than recomputed, so they do
                # NOT consume the compute budget. Extend this chunk's SPAN to cover
                # the entire skippable prefix plus up to `token_budget` newly computed
                # tokens. Without this the span is capped at the budget, forcing the
                # rest of a long cached prefix to be re-prefilled over many chunks
                # (latency then scales with prompt length instead of the delta).
                # add_request() only computes `effective = span - skip` tokens.
                prefix_skip = 0
                if prefix_caching_enabled and not is_continuing_chunked_prefill:
                    _, _, _, _, prefix_skip, _ = self.context._compute_prefix_match(
                        req, remaining_len
                    )
                    prefix_skip = min(prefix_skip, remaining_len - 1)  # keep >=1 token to run

                computed_budget = min(remaining_len - prefix_skip, token_budget)

                # Skip CG gating for the continuation of an in-flight chunked prefill:
                # the request is already mid-flight, deferring it would deadlock progress.
                if self._cg_admission_gating_active() and not is_continuing_chunked_prefill:
                    # Snap the COMPUTED chunk size to the largest captured-CG boundary
                    # within budget (skipped tokens don't affect the CG batch shape).
                    # Fall back to eager (computed_budget) if no CG shape covers it.
                    snapped_chunk = self._find_cg_chunk_size(computed_budget)
                    computed_chunk = snapped_chunk if snapped_chunk is not None else computed_budget
                    req.cg_wait_iters = 0
                else:
                    computed_chunk = computed_budget

                if batch_invariant_mamba_prefill:
                    prefill_chunk_length = self._mamba_batch_invariant_prefill_chunk_length(
                        req, computed_chunk
                    )
                    if prefill_chunk_length == 0:
                        can_schedule = False
                        break
                else:
                    prefill_chunk_length = prefix_skip + computed_chunk

                # Mamba prefix caching: keep chunk boundaries block-aligned.
                # compute_and_store_offsets() records a recurrent-state snapshot at a
                # KV-block boundary only when that boundary lands on a multiple of the
                # SSM chunk size measured FROM the start of the current prefill chunk
                # (it filters on `offset % mamba_chunk_size == 0`, where the chunk start
                # equals `finished_chunk_token_count` on continuation chunks). Block
                # boundaries are multiples of `block_size_tokens` (itself a multiple of
                # the SSM chunk size), so the filter only passes when
                # `finished_chunk_token_count` is block-aligned. If a chunk ends at an
                # arbitrary token offset, every candidate boundary in the following
                # chunks becomes unrecordable and the last-block snapshot that lets a
                # future request skip prefill is silently dropped. Stop a partial
                # (non-final) chunk short at the nearest lower block boundary so the
                # running `finished_chunk_token_count` stays block-aligned.
                if (
                    self.context.is_hybrid_model
                    and self.context.mamba_slot_allocator is not None
                    and prefill_chunk_length < remaining_len
                ):
                    block_size = self.context.block_size_tokens
                    chunk_end = req.finished_chunk_token_count + prefill_chunk_length
                    aligned_end = (chunk_end // block_size) * block_size
                    aligned_chunk_length = aligned_end - req.finished_chunk_token_count
                    # Only snap down when the aligned chunk still computes at least one
                    # token beyond the skipped prefix (a chunk whose budget is smaller
                    # than a block cannot be block-aligned; leave it unchanged).
                    if aligned_chunk_length > prefix_skip:
                        prefill_chunk_length = aligned_chunk_length

                # Flash-attn guard: if this chunk would leave exactly 1 token for the
                # final chunk, reduce by 1 (or defer if we only have 1 computed token).
                # See https://github.com/Dao-AILab/flash-attention/issues/1537
                # The -1 is safe after CG snapping: is_applicable_for_batch_dim matches on
                # cg.token_count >= real.token_count, so the snapped CG still covers token_count-1.
                if not batch_invariant_mamba_prefill and remaining_len - prefill_chunk_length == 1:
                    if computed_chunk > 1:
                        prefill_chunk_length -= 1
                    else:
                        can_schedule = False
                        break

                # add_request recomputes the skip for this exact chunk and applies a
                # ">= 2 computed tokens" clamp. When the chunk would compute fewer than
                # 2 tokens (tight budget late in a batched step, or a prompt that is
                # all-but-one cached) that clamp shrinks the skip and grows the computed
                # count by up to one block, which can exceed the token budget
                # (TokenOverflowError). Only then re-derive the exact effective length
                # add_request will use and defer on overflow (a later full-budget step
                # admits the request). For >= 2 computed tokens add_request computes
                # exactly this chunk, which already fits the budget.
                if prefix_skip > 0 and (prefill_chunk_length - prefix_skip) < 2:
                    _, _, _, _, _, actual_effective = self.context._compute_prefix_match(
                        req, prefill_chunk_length
                    )
                    if self.context.active_token_count + actual_effective > self.context.max_tokens:
                        can_schedule = False
                        break

                # Add hashes to pending set (prefix-caching bookkeeping).
                if prefix_caching_enabled:
                    for block_hash in req.precomputed_block_hashes:
                        if block_hash not in self.context.kv_block_allocator.kv_hash_to_block_id:
                            pending_block_hashes.add(block_hash)

                if prefill_chunk_length >= remaining_len:
                    self.context.chunked_prefill_request_id = -1
                    self.context.add_request(req)
                    self._loop.call_soon_threadsafe(
                        self._loop.create_task, self._notify_cond_for_new_request()
                    )
                    req.remaining_prompt_tokens = req.remaining_prompt_tokens.new_empty(0)
                    req.add_event_add_context()
                    self.waiting_request_ids.popleft()
                    can_schedule = True
                else:
                    # Partial admit: schedule this chunk and keep the request at the queue head.
                    self.context.add_request(req, prefill_chunk_length=prefill_chunk_length)
                    self._loop.call_soon_threadsafe(
                        self._loop.create_task, self._notify_cond_for_new_request()
                    )
                    self.context.chunked_prefill_request_id = req.request_id
                    req.remaining_prompt_tokens = req.remaining_prompt_tokens[prefill_chunk_length:]
                    req.finished_chunk_token_count += prefill_chunk_length

        # Prepend pending request ids to waiting queue.
        if prefix_caching_enabled and pending_request_ids:
            is_continuing_chunked_prefill = self.context.chunked_prefill_request_id >= 0
            if is_continuing_chunked_prefill:
                chunked_request_id = self.waiting_request_ids.popleft()
                self.waiting_request_ids.extendleft(reversed(pending_request_ids))
                self.waiting_request_ids.appendleft(chunked_request_id)
            else:
                self.waiting_request_ids.extendleft(reversed(pending_request_ids))

    async def async_forward(self) -> Tuple[Optional[Dict], Dict, float]:
        """Uses `asyncio` for continuous generation.
        Sleeps when no requests are available, until new requests have been added.

        Returns:
            A tuple comprised of:
                step_result (Optional[Dict]): The result of the step.
                context_state (Dict): Decode-only state, total/paused request
                    count, and active token count.
                step_time (float): How long this step took.
        """

        # If suspended, no stepping.
        if self.state in (EngineState.SUSPENDED, EngineState.SUSPENDING):
            raise EngineSuspendedError(self.context.step_count)

        # Discard registrations left by an interrupted prior step before this
        # step's scheduling queues new registrations.
        dynamo_helper = getattr(self.context, "dynamo_helper", None)
        if dynamo_helper is not None:
            dynamo_helper.discard_pending_kv_stored_events()

        mode = self.context.config.async_sched_mode
        if mode == AsyncScheduleMode.LEGACY:
            self.schedule_waiting_requests()
            step_nvtx_range = "Decode" if self.context.num_prefill_requests == 0 else "Prefill"
            controller_kwargs = {}
        elif mode == AsyncScheduleMode.ASYNC:
            run_async_overlap = self._should_run_async_sched_overlap()
            step_nvtx_range = "AsyncOverlap" if run_async_overlap else "AsyncNoOverlap"
            controller_kwargs = {
                "run_async_overlap": run_async_overlap,
                "schedule_waiting_requests": (
                    None if run_async_overlap else self.schedule_waiting_requests
                ),
            }
        else:
            raise AssertionError(f"Unexpected async scheduling mode: {mode}")

        # The print block (async_bookkeep) and metrics block both fire on this
        # condition after step_count is incremented. Predict it up-front so we
        # can skip the GPU-timing sync and the context_state dict builds that
        # only exist to feed those logging/metrics blocks.
        will_log_this_step = (
            self.logging_step_interval > 0
            and (self.context.step_count + 1) % self.logging_step_interval == 0
        )

        if will_log_this_step:
            pre_step_context_state = {
                "max_requests": self.context.max_requests,
                "total_request_count": self.context.total_request_count,
                "paused_request_count": self.context.paused_request_count,
                "active_token_count": self.context.active_token_count,
                "step_count": self.context.step_count,
            }
        else:
            # active_token_count and step_count are still consumed by
            # post_process_requests' pre_fwd_* args (for add_event_generated_token);
            # the other four fields are only read in the gated print block.
            pre_step_context_state = {
                "active_token_count": self.context.active_token_count,
                "step_count": self.context.step_count,
            }
        pre_step_context_state["chunked_prefill_request_id"] = (
            self.context.chunked_prefill_request_id
        )

        # Generate tokens.
        nvtx_range_push(step_nvtx_range)

        if will_log_this_step:
            self.step_start_event.record()
        controller_result: DynamicBatchControllerStepResult = (
            await self.controller.async_generate_output_tokens_dynamic_batch(**controller_kwargs)
        )
        self.decode_only = controller_result.decode_only
        pre_step_context_state["decode_only"] = self.decode_only
        result = controller_result.output
        if dynamo_helper is not None:
            dynamo_helper.publish_pending_kv_stored_events()
        if will_log_this_step:
            self.step_end_event.record()
            self.step_end_event.synchronize()
            step_time = self.step_start_event.elapsed_time(self.step_end_event) / 1e3
        else:
            step_time = 0.0
        self.context.step_count += 1
        self.context.prefix_cache_lru_clock += 1

        nvtx_range_pop(step_nvtx_range)

        if will_log_this_step:
            kvcache_util_stats = (
                self.context.get_kvcache_utilization_stats()
                if self.metrics_writer is not None
                else None
            )
            post_step_context_state = {
                "waiting_request_count": len(self.waiting_request_ids),
                "finished_request_count": self.finished_request_count,
                "evicted_request_count": self.evicted_request_count,
                "kv_stats": kvcache_util_stats,
                "usable_block_count": self.context.kv_block_allocator.pool_size - 1,
                "occupied_block_count": self.context.kv_block_allocator.get_total_used(),
                "allocatable_block_count": self.context.kv_block_allocator.get_allocatable_count(),
                "active_used_block_count": self.context.kv_block_allocator.get_active_used(),
                "paused_used_block_count": self.context.kv_block_allocator.get_paused_used(),
                "paused_block_budget": self.context.kv_block_allocator.paused_limit,
            }
            context_state = {**pre_step_context_state, **post_step_context_state}
        else:
            # Keep kv_stats=None so the metrics-block gate at `async_bookkeep`
            # (`if context_state["kv_stats"] is not None`) remains well-typed.
            context_state = {**pre_step_context_state, "kv_stats": None}

        return result, context_state, step_time

    def _try_send_streaming_partials(self) -> None:
        """Send pending token deltas to the inference coordinator."""
        partials: list = []
        emit_lengths: Dict[int, int] = {}
        for rid, entry in self.requests.items():
            request = entry.record[-1]
            if not getattr(request.sampling_params, "streaming", False):
                continue
            already = self._partial_emit_lengths.get(rid, 0)
            total = len(request.generated_tokens)
            stop_word_ids = getattr(request, "stop_word_ids", None)
            holdback = 0
            if stop_word_ids and not getattr(
                request.sampling_params, "detokenize_stop_sequence", False
            ):
                holdback = max(0, max(len(ids) for ids in stop_word_ids) - 1)
            emit_end = max(already, total - holdback)
            streaming_interval = getattr(request.sampling_params, "streaming_interval", 1)
            if emit_end - already >= streaming_interval:
                new_tokens = list(request.generated_tokens[already:emit_end])
                partial = {"request_id": rid, "new_tokens": new_tokens}
                if request.sampling_params.return_log_probs:
                    partial["new_log_probs"] = list(
                        (request.generated_log_probs or [])[already:emit_end]
                    )
                    partial["new_top_n_logprobs"] = list(
                        (getattr(request, "generated_top_n_logprobs", None) or [])[already:]
                    )
                    if already == 0 and not request.sampling_params.skip_prompt_log_probs:
                        partial["prompt_log_probs"] = list(
                            getattr(request, "prompt_log_probs", None) or []
                        )
                        partial["prompt_top_n_logprobs"] = list(
                            getattr(request, "prompt_top_n_logprobs", None) or []
                        )
                partials.append(partial)
                emit_lengths[rid] = emit_end

        if not partials:
            return

        payload = msgpack.packb([Headers.ENGINE_REPLY_PARTIAL.value, partials], use_bin_type=True)
        nvtx_range_push("coordinator_streaming")
        self.socket_for_receiving_requests.send(payload)
        nvtx_range_pop("coordinator_streaming")

        self._partial_emit_lengths.update(emit_lengths)

    async def async_bookkeep(
        self, step_result: Optional[Dict], context_state: Dict, step_time: float
    ):
        """Uses `asyncio` for continuous bookkeeping.

        Args:
            step_result (Optional[Dict]): The result of the step.
            context_state (Dict): Decode-only state, total/paused request count,
                and active token count.
            step_time (float): How long this step took.

        Returns:
            A dictionary containing:
                active_requests (List): Requests that ran in the last step and are still active.
                finished_requests (List): Requests that ran in the last step and have now finished.
                step_time (float): The step time in seconds.
                cuda_graph_request_count (int): The CUDA graph batch size matching this step.
        """
        # Increment finished_request_count.
        nvtx_range_push("bookkeeping")
        cuda_graph_request_count = None

        if step_result is not None:
            active_request_ids = step_result["active_request_ids"]
            finished_request_ids = step_result["finished_request_ids"]
            newly_paused_request_ids = step_result.get("newly_paused_request_ids")
            evict_request_ids = step_result.get("evict_request_ids")
            sample = step_result["sample"]
            accepted_tokens = step_result["accepted_tokens"]
            log_probs = step_result["log_probs"]
            top_n_logprobs = step_result.get("top_n_logprobs", None)
            finished_routing_block_ids = step_result.get("finished_routing_block_ids", None)
            finished_handoff_block_ids = step_result.get("finished_handoff_block_ids", None)
            finished_handoff_ssm_slots = step_result.get("finished_handoff_ssm_slots", None)
            finished_handoff_decode_tokens = step_result.get("finished_handoff_decode_tokens", None)
            cuda_graph_request_count = step_result["cuda_graph_request_count"]

            # Add paused events.
            if newly_paused_request_ids is not None and self.track_paused_request_events:
                newly_paused_request_ids = newly_paused_request_ids.tolist()
                [self.get_request(i).add_event_pause() for i in newly_paused_request_ids]

            # Process finished requests (adds FINISH events and returns records).
            active_request_ids, finished_request_records = self.post_process_requests(
                active_request_ids,
                finished_request_ids,
                evict_request_ids,
                step_time,
                sample,
                accepted_tokens,
                log_probs,
                consumed_chunked_prefill_request_id=context_state["chunked_prefill_request_id"],
                top_n_logprobs=top_n_logprobs,
                pre_fwd_active_token_count=context_state.get("active_token_count"),
                pre_fwd_step_count=context_state.get("step_count"),
                finished_routing_block_ids=finished_routing_block_ids,
                finished_handoff_block_ids=finished_handoff_block_ids,
                finished_handoff_ssm_slots=finished_handoff_ssm_slots,
                finished_handoff_decode_tokens=finished_handoff_decode_tokens,
            )

        else:
            active_request_ids: list[int] = []
            finished_request_records: list[DynamicInferenceRequestRecord] = []

        # Failed requests. Status and events were already set in _handle_failed_request;
        # here we just clean up the entry and include it in finished_request_records.
        for failed_request_id in self.failed_request_ids:
            failed_entry = self.requests.pop(failed_request_id)
            finished_request_records.append(failed_entry.record)
            assert (
                failed_entry.future.done()
            ), f"Failed request {failed_request_id} future has not been properly resolved."
        self.failed_request_ids.clear()

        nvtx_range_pop("bookkeeping")

        # Detokenize all finished requests if not using
        # the coordinator. Otherwise, the coordinator will
        # overlap detokenization with the engine.
        if not self.use_coordinator:
            nvtx_range_push("detokenization")
            for record in finished_request_records:
                for request in record.requests:
                    if request.prompt is None:
                        request.prompt = self.controller.detokenize(
                            self.controller.tokenizer,
                            request.prompt_tokens.tolist(),
                            remove_EOD=False,
                        )
                    request.generated_text = self.controller.detokenize(
                        self.controller.tokenizer,
                        request.generated_tokens,
                        remove_EOD=not request.sampling_params.detokenize_stop_sequence,
                    )
            nvtx_range_pop("detokenization")

        # Handle necessary ZMQ DP coordinator communication.
        # Failed request replies were already sent in _handle_failed_request,
        # so only send completed records here.
        if self.use_coordinator and self.is_mp_coordinator:
            records_to_send = [
                r for r in finished_request_records if r.requests[-1].status != Status.FAILED
            ]
            if records_to_send:
                nvtx_range_push("coordinator_communication")
                self._send_request_records_to_coordinator(records_to_send)
                nvtx_range_pop("coordinator_communication")

            # Stream newly generated tokens for active requests. Finished
            # requests were already popped from self.requests above, so their
            # emit lengths are dropped here rather than in the loop.
            for record in finished_request_records:
                self._partial_emit_lengths.pop(record.requests[-1].request_id, None)
            self._try_send_streaming_partials()

        # Drain prefix cache hit counters from context into engine accumulators.
        if self.context.enable_prefix_caching:
            self._prefix_cache_hits += self.context.prefix_cache_hits
            self._prefix_cache_blocks_matched += self.context.prefix_cache_blocks_matched
            self._prefill_tokens_computed += self.context.prefix_cache_prefill_computed_tokens
            self._prefill_tokens_skipped += self.context.prefix_cache_prefill_skipped_tokens
            self.context.prefix_cache_hits = 0
            self.context.prefix_cache_blocks_matched = 0
            self.context.prefix_cache_prefill_computed_tokens = 0
            self.context.prefix_cache_prefill_skipped_tokens = 0

        # Log KV cache utilization stats to W&B
        nvtx_range_push("wandb_logging")
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

            # Add speculative decoding acceptance metrics (aggregate + per-position).
            total_proposed = sum(self._spec_tokens_proposed_per_pos)
            total_accepted = sum(self._spec_tokens_accepted_per_pos)
            if self.num_speculative_tokens > 0 and total_proposed > 0:
                acceptance_rate = total_accepted / total_proposed
                metrics['inference/spec_decode_acceptance_rate'] = float(acceptance_rate * 100.0)
                metrics['inference/spec_decode_tokens_proposed'] = int(total_proposed)
                metrics['inference/spec_decode_tokens_accepted'] = int(total_accepted)
                metrics['inference/spec_decode_num_steps'] = int(self._spec_steps)
                for pos in range(self.num_speculative_tokens):
                    if self._spec_tokens_proposed_per_pos[pos] > 0:
                        pos_rate = (
                            self._spec_tokens_accepted_per_pos[pos]
                            / self._spec_tokens_proposed_per_pos[pos]
                        )
                        metrics[f'inference/spec_decode_acceptance_rate_pos{pos + 1}'] = float(
                            pos_rate * 100.0
                        )

            # Add prefix caching metrics.
            if self.context.enable_prefix_caching and self._prefix_cache_hits > 0:
                metrics['inference/prefix_cache_hits'] = int(self._prefix_cache_hits)
                metrics['inference/prefix_cache_blocks_matched'] = int(
                    self._prefix_cache_blocks_matched
                )

            if HAVE_WANDB and self.metrics_writer.__name__ == "wandb":
                self.metrics_writer.log(metrics, commit=True)
            else:
                raise ValueError(f"Unsupported metrics writer type: {type(self.metrics_writer)}")
        nvtx_range_pop("wandb_logging")

        # Print context state.
        nvtx_range_push("console_logging")
        if (
            self.logging_step_interval > 0
            and self.context.step_count % self.logging_step_interval == 0
        ):
            nvtx_range_push("cuda_memory_stats")
            mem = torch.cuda.memory_stats()
            nvtx_range_pop("cuda_memory_stats")
            decode_only = context_state["decode_only"]
            step_type, color_decode_only = _get_decode_only_log_state(
                self.context.config.async_sched_mode, decode_only
            )
            output_str = (
                "* rank %d | step %d | %s ... time: %.3f ms%s ... "
                "reqs: a %d/%d, p %d, w %d, f %d, e %d ... "
                "blocks: occupied %d/%d, allocatable %d, active-used %d, "
                "paused-used %d/%d ... "
                "mem: tensors %d, alloc %.1f gb, res %.1f gb."
                % (
                    self.rank,
                    self.context.step_count,
                    datetime.now().strftime("%H:%M:%S"),
                    step_time * 1000,
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
                    context_state["occupied_block_count"],
                    context_state["usable_block_count"],
                    context_state["allocatable_block_count"],
                    context_state["active_used_block_count"],
                    context_state["paused_used_block_count"],
                    context_state["paused_block_budget"],
                    mem["allocation.all.current"],
                    mem["allocated_bytes.all.current"] / (1024**3),
                    mem["reserved_bytes.all.current"] / (1024**3),
                )
            )
            total_proposed = sum(self._spec_tokens_proposed_per_pos)
            total_accepted = sum(self._spec_tokens_accepted_per_pos)
            if self.num_speculative_tokens > 0 and total_proposed > 0:
                spec_rate = total_accepted / total_proposed * 100.0
                per_pos_rates = []
                for pos in range(self.num_speculative_tokens):
                    if self._spec_tokens_proposed_per_pos[pos] > 0:
                        pos_rate = (
                            self._spec_tokens_accepted_per_pos[pos]
                            / self._spec_tokens_proposed_per_pos[pos]
                            * 100.0
                        )
                        per_pos_rates.append("t%d=%.1f%%" % (pos + 1, pos_rate))
                output_str += " ... spec (cumul): accept %.1f%% (%d/%d in %d steps) [%s]" % (
                    spec_rate,
                    total_accepted,
                    total_proposed,
                    self._spec_steps,
                    ", ".join(per_pos_rates),
                )
            if self.context.enable_prefix_caching and self._prefix_cache_hits > 0:
                output_str += " ... prefix cache (cumul): %d hits, %d blocks matched" % (
                    self._prefix_cache_hits,
                    self._prefix_cache_blocks_matched,
                )
            if self.context.enable_prefix_caching:
                # Prefill compute actually saved by prefix caching (cumulative).
                # computed = prompt tokens run through the model; skipped = prompt
                # tokens whose prefill was reused from cache. If skipped% stays high
                # while per-step latency grows, the growth is attention over the
                # growing KV context, NOT re-prefilling skipped tokens.
                _computed = self._prefill_tokens_computed
                _skipped = self._prefill_tokens_skipped
                _total = _computed + _skipped
                output_str += " ... prefill (cumul): computed %d, skipped %d (%.1f%% skipped)" % (
                    _computed,
                    _skipped,
                    (100.0 * _skipped / _total) if _total > 0 else 0.0,
                )
                # Current cache occupancy (utilization). A Mamba durable-slot count
                # near its max indicates the cache is saturating and will start
                # LRU-evicting cached prefixes (hybrid models can only skip prefill
                # where Mamba state is still cached).
                kv_alloc = self.context.kv_block_allocator
                output_str += " ... prefix cache util: KV %d/%d blocks cached (%d evictable)" % (
                    len(kv_alloc.kv_hash_to_block_id),
                    kv_alloc.pool_size,
                    int(kv_alloc.get_evictable_block_count()),
                )
                msa = self.context.mamba_slot_allocator
                if msa is not None:
                    output_str += ", mamba %d/%d durable slots" % (
                        msa.max_slots - msa.free_count,
                        msa.max_slots,
                    )
            if color_decode_only:
                output_str = f"\033[94m{output_str}\033[0m"
            logger.info(output_str)

        nvtx_range_pop("console_logging")

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

        nvtx_range_push("drain_zmq_socket")
        all_messages = []
        if self.is_mp_coordinator:
            all_messages.extend(
                msgpack.packb(
                    [Headers.KV_HANDOFF_COMPLETE.value, request_id, failed], use_bin_type=True
                )
                for request_id, failed in self._drain_handoff_completion_notifications()
            )
            while True:
                try:
                    # Receive messages in a non-blocking way.
                    all_messages.append(self.socket_for_receiving_requests.recv(flags=zmq.NOBLOCK))
                except zmq.Again:
                    # This exception is hit as soon as the socket is empty.
                    break
            self.model_parallel_publisher_socket.send_multipart(
                [bytes([Headers.TP_BROADCAST.value])] + all_messages
            )
        else:
            frames = self.model_parallel_subscriber_socket.recv_multipart()
            all_messages = frames[1:]

        nvtx_range_pop("drain_zmq_socket")

        # First pass: add requests.
        # Control signals are queued for the second pass.
        new_generation_epoch = None
        for message in all_messages:
            data = msgpack.unpackb(message, raw=False)
            header = Headers(data[0])
            if header == Headers.SUBMIT_REQUEST:
                # Payload is [request_id, prompt, sampling_params, multi_modal_data].
                fields = data[1:]
                if len(fields) == 3:
                    request_id, prompt, sampling_params = fields
                    multi_modal_data = None
                else:
                    request_id, prompt, sampling_params, multi_modal_data = fields[:4]
                sampling_params = SamplingParams.deserialize(sampling_params)
                nvtx_range_push("add_request")
                # TODO(perf): image preprocessing (PIL decode / resize /
                # normalize / patchify) runs synchronously on the engine step
                # loop, adding directly to inter-token latency for every
                # in-flight request. Move off the engine thread — either via a
                # bounded ThreadPoolExecutor here or, better, on the
                # server/coordinator side before the ZMQ hop so the engine
                # receives ready tensors.
                try:
                    if multi_modal_data is None:
                        # Skip the config-attribute lookup for text-only
                        # requests so test fixtures (DummyContext) without an
                        # image_preprocessing_config don't AttributeError on
                        # every SUBMIT_REQUEST and desync the ranks.
                        vlm_kwargs = {}
                    else:
                        vlm_kwargs = resolve_multimodal_data_for_engine(
                            multi_modal_data,
                            image_preprocessing_config=(
                                self.context.config.image_preprocessing_config
                            ),
                        )
                    if vlm_kwargs:
                        self.add_request(request_id, prompt, sampling_params, **vlm_kwargs)
                    else:
                        self.add_request(request_id, prompt, sampling_params)
                except Exception as error:  # pylint: disable=broad-except
                    self._fail_submission(request_id, sampling_params, error)
                nvtx_range_pop("add_request")
            elif header == Headers.SUBMIT_REQUEST_WITH_KV:
                # Decode-side KV import.
                request_id, prompt, sampling_params, kv_meta, src_block_ids = data[1:]
                sampling_params = SamplingParams.deserialize(sampling_params)
                nvtx_range_push("add_request_with_kv_handoff")
                self.add_request_with_kv_handoff(
                    request_id, prompt, sampling_params, kv_meta, src_block_ids
                )
                nvtx_range_pop("add_request_with_kv_handoff")
            elif header == Headers.RELEASE_KV:
                # Coordinator-broadcast release. Unknown request ids are no-ops.
                self.release_handoff_blocks(int(data[1]))
            elif header == Headers.SEND_KV:
                # Push transport: send a pinned hand-off's KV to the decode
                # instance the coordinator picked.
                self.push_handoff_kv(int(data[1]), data[2])
            elif header == Headers.KV_HANDOFF_COMPLETE:
                self._record_handoff_completion_notification(int(data[1]), bool(data[2]))
            elif header == Headers.ABORT_REQUEST:
                request_id = int(data[1])
                entry = self.requests.get(request_id)
                if entry is not None:
                    request = entry.record[-1]
                    # Force active requests to finish on the next step.
                    request.sampling_params.num_tokens_to_generate = len(request.generated_tokens)
                    active_ids = self.context.request_ids[: self.context.total_request_count]
                    matches = torch.where(active_ids == request_id)[0]
                    if matches.numel() > 0:
                        assert matches.numel() == 1
                        idx = int(matches[0].item())
                        self.context.request_output_lengths[idx] = (
                            self.context.request_kv_length_offsets[idx]
                            + self.context.request_query_lengths[idx]
                        )
            elif header == Headers.SET_GENERATION_EPOCH:
                new_generation_epoch = data[1]
            elif header == Headers.START_CUDA_PROFILER:
                # Side-effect, not a state transition: apply immediately on every
                # rank so an outer nsys --capture-range=cudaProfilerApi starts here.
                torch.cuda.cudart().cudaProfilerStart()
            elif header == Headers.STOP_CUDA_PROFILER:
                torch.cuda.cudart().cudaProfilerStop()
            else:
                # Control signal: queue for second pass.
                self._pending_signals.append(message)

        self._poll_pending_kv_imports()
        self._poll_pending_kv_pushes()

        if new_generation_epoch is not None:
            self._generation_epoch = new_generation_epoch
            # Stamp all active requests with the new epoch.
            # Each field stores a sparse list of (start_token_index, epoch) boundaries.
            for entry in self.requests.values():
                request = entry.record[-1]
                total = len(request.prompt_tokens) + len(request.generated_tokens)
                if total > 0:
                    boundary = (total - 1, new_generation_epoch)
                    if request.policy_epoch is None:
                        request.policy_epoch = [(0, new_generation_epoch)]
                    else:
                        request.policy_epoch.append(boundary)
                    if request.kv_cache_epoch is None:
                        request.kv_cache_epoch = [(0, new_generation_epoch)]
                    else:
                        request.kv_cache_epoch.append(boundary)
            # New weights invalidate cached state produced by the old ones; drop
            # whatever has now outlived its bounded-staleness lease.
            self.context.set_prefix_cache_epoch(new_generation_epoch)

        # Second pass: apply at most one control signal (the engine loop
        # processes one state transition per iteration).
        if self._pending_signals:
            message = self._pending_signals.popleft()
            data = msgpack.unpackb(message, raw=False)
            header = Headers(data[0])

            if header == Headers.PAUSE:
                if self.state == EngineState.RUNNING:
                    self.state = EngineState.PAUSING
                    self._state_events[EngineState.RUNNING].clear()
                # Any other state can safely ignore PAUSE.

            elif header == Headers.UNPAUSE:
                assert self.state == EngineState.PAUSED, f"Received UNPAUSE in state {self.state}"
                self.state = EngineState.UNPAUSING

            elif header == Headers.SUSPEND:
                assert self.state == EngineState.PAUSED, f"Received SUSPEND in state {self.state}"
                self._state_events[EngineState.RESUMED].clear()
                self.suspend()
                self.state = EngineState.SUSPENDING

            elif header == Headers.RESUME:
                assert self.state == EngineState.SUSPENDED, f"Received RESUME in state {self.state}"
                self._state_events[EngineState.SUSPENDED].clear()
                self.resume()
                self.state = EngineState.RESUMING

            elif header == Headers.STOP:
                assert self.state in (
                    EngineState.PAUSED,
                    EngineState.SUSPENDED,
                ), f"Received STOP in state {self.state}"
                if self.state == EngineState.SUSPENDED:
                    self._state_events[EngineState.SUSPENDED].clear()
                self.state = EngineState.STOPPING

            else:
                raise UnknownHeaderError(header)

        return len(all_messages)

    async def shutdown(self):
        """Shut down the engine and clean up ZMQ resources.

        Called from the engine loop's finally block after the loop exits.
        """
        self.state = EngineState.STOPPED

        # Cleanup the request futures.
        for entry in self.requests.values():
            if not entry.future.done():
                entry.future.cancel()

        # ZMQ cleanup; designed to be idempotent.
        sock = getattr(self, 'socket_for_receiving_requests', None)
        if sock is not None and not sock.closed:
            try:
                sock.send(msgpack.packb([Headers.DISCONNECT.value], use_bin_type=True))
            except Exception:
                pass
        for socket in getattr(self, 'zmq_sockets', []):
            socket.close(linger=0)
        if hasattr(self, 'zmq_sockets'):
            self.zmq_sockets.clear()
        if hasattr(self, "expert_parallel_zmq_communicator"):
            self.expert_parallel_zmq_communicator.close()
        if hasattr(self, "world_zmq_communicator"):
            self.world_zmq_communicator.close()
        if not self.zmq_context.closed:
            self.zmq_context.term()

        # Set the stopped state at the very end.
        self._state_events[EngineState.STOPPED].set()

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
                            self.state not in (EngineState.SUSPENDED, EngineState.SUSPENDING)
                            and (
                                self.context.get_active_request_count() > 0
                                or self.waiting_request_ids
                                or self.pending_kv_import_count > 0
                                or self.pending_kv_push_count > 0
                            )
                        )
                    )
                self._poll_pending_kv_imports()
                self._poll_pending_kv_pushes()
                if (
                    self.context.get_active_request_count() > 0
                    or self.waiting_request_ids
                    or self.has_admittable_kv_import
                ):
                    await self.async_step()
                else:
                    # Reached when there is no model work to step but a handoff transfer is
                    # still pending. Handles expose polling rather than an async completion
                    # callback, so yield briefly to avoid busy-spinning while keeping decode
                    # admission responsive when the transfer completes.
                    await asyncio.sleep(0.001)
        except asyncio.CancelledError:
            pass

    async def _ep_establish_consensus(
        self, local_work: int, signal_consensus: bool
    ) -> tuple[int, bool]:
        """EP all-reduce to share work counts and pause consensus.

        All-reduces two integers at once:
        - local_work: actual pending request count (always >= 0).
        - consensus flag: -1 if this rank wants to pause, 0 otherwise.

        Using max for both:
        - max(work) > 0 means at least one EP peer has real work.
        - max(consensus) == -1 means ALL peers signaled -1 (all PAUSING).
          Any RUNNING peer contributes 0, pulling the max to 0.

        Args:
            local_work: Pending request count for this rank.
            signal_consensus: True if this rank is ready to pause.
        Returns:
            (global_work, all_pausing): max work across EP, and whether
            all peers signaled consensus.
        """
        nvtx_range_push("_ep_establish_consensus")

        consensus_val = -1 if signal_consensus else 0

        # Signals can be received asynchronously on EP ranks.
        # We do not want a rank to pause prematurely if its peers have yet to receive the signal.
        # So this is an *attempt* to process the signal. This rank has received the signal
        # and passes -1 to the all-reduce. If any other rank in the EP group has not received
        # the signal yet, it will pass a zero value to the all-reduce, hence the global consensus
        # will be zero and we will defer processing the signal.
        # When all ranks receive the signal, global consensus will be -1 and we can process.

        if self.ep_world_size > 1:
            # Note that it is important to use a non-blocking asyncio-friendly all-reduce here.
            # The user may have other tasks running in the event loop that need to be serviced.
            # Do not using a torch.distributed blocking all-reduce here using nccl/gloo.
            # We have tried that and it blocks the event loop in megatron-rl.
            global_work, global_consensus = (
                await self.expert_parallel_zmq_communicator.all_reduce_max(
                    local_work, consensus_val, async_op=(not self.use_synchronous_zmq_collectives)
                )
            )
        else:
            global_work, global_consensus = local_work, consensus_val

        nvtx_range_pop("_ep_establish_consensus")
        return global_work, global_consensus == -1

    async def _world_barrier(self):
        """World-wide ZMQ all-reduce barrier for global rank consensus.

        Used for all state transitions that require global synchronization:
        PAUSING → PAUSED, UNPAUSING → RUNNING, SUSPENDING → SUSPENDED,
        RESUMING → PAUSED, and STOPPING → STOPPED.

        No-op when world_size == 1 (communicator is not created).
        """
        nvtx_range_push("world_barrier")
        if hasattr(self, 'world_zmq_communicator'):
            await self.world_zmq_communicator.all_reduce_max(
                1, async_op=(not self.use_synchronous_zmq_collectives)
            )
        nvtx_range_pop("world_barrier")

    @trace_async_exceptions
    async def run_engine_with_coordinator(
        self, *, loop: Optional[asyncio.AbstractEventLoop] = None
    ):
        """Continually steps the engine asynchronously.

        State-dependent behavior:
        - RUNNING: EP all-reduce to check for work, then step or idle.
        - PAUSING: EP all-reduce to reach consensus, then world barrier.
        - PAUSED / SUSPENDED: Idle-sleep, wait for signals via schedule_requests().
        - UNPAUSING / SUSPENDING / RESUMING / STOPPING: World barrier, then transition.
        - STOPPED: Teardown and exit.
        """
        self._loop = get_asyncio_loop(loop)
        self.use_coordinator = True

        try:
            while True:
                self.schedule_requests()

                if self.state in (EngineState.RUNNING, EngineState.PAUSING):
                    local_schedulable = (
                        self.context.get_active_request_count()
                        + len(self.waiting_request_ids)
                        + int(self.has_admittable_kv_import)
                    )
                    local_pending_imports = self.pending_kv_import_count
                    if self.disable_ep_consensus:
                        # Skip the EP consensus all-reduce; act on local state only.
                        # NOTE: even with no consensus we must still participate in EP
                        # collectives (NCCL all-to-all, etc.) every iteration. A peer with
                        # real work will block at its all-to-all kernel waiting for this
                        # rank, so when there is no local work we run dummy_forward()
                        # rather than sleeping. Sleeping here would deadlock EP > 1.
                        if self.state == EngineState.PAUSING:
                            await self._world_barrier()
                            self.state = EngineState.PAUSED
                            self._state_events[EngineState.PAUSED].set()
                        elif local_schedulable > 0:
                            await self.async_step()
                        elif self.ep_world_size == 1 and local_pending_imports > 0:
                            # No model work is ready; poll the network transfer without
                            # spending a dummy forward while waiting for decode admission.
                            await asyncio.sleep(0.001)
                        else:
                            self.step_start_event.record()
                            nvtx_range_push("EP-dummy-forward")
                            self.controller.dummy_forward()
                            self.step_end_event.record()
                            self.step_end_event.synchronize()
                            nvtx_range_pop("EP-dummy-forward")
                            self.context.step_count += 1
                            self.context.prefix_cache_lru_clock += 1
                            # The consensus path yields via _ep_establish_consensus;
                            # without it we must still let other coroutines (signal
                            # delivery, request scheduling) run between steps.
                            await asyncio.sleep(0)
                        continue
                    global_work_from_last_consensus, _ = self._last_ep_consensus
                    if (
                        global_work_from_last_consensus == 0
                        or self._ep_consensus_loop_counter % self.ep_consensus_interval == 0
                    ):
                        # selectively enter ep_establish_consensus if
                        # 1. there is no global work -> engine is idle. At any step in the future
                        #    one of the ranks can receive work. So we should be eagerly checking for that
                        # 2. it has been 20 steps since we last established consensus, and that consensus
                        #    had some work.
                        # In the worst case, this delays pausing by 20 steps which is around
                        # 200-400 milliseconds.
                        self._last_ep_consensus = await self._ep_establish_consensus(
                            local_schedulable, signal_consensus=(self.state == EngineState.PAUSING)
                        )
                    global_work, all_pausing = self._last_ep_consensus
                    self._ep_consensus_loop_counter += 1

                    if all_pausing:
                        # All EP peers are PAUSING: pause immediately.
                        await self._world_barrier()
                        self.state = EngineState.PAUSED
                        self._state_events[EngineState.PAUSED].set()
                    elif global_work > 0:
                        # At least one EP peer has work: all must participate.
                        if local_schedulable > 0:
                            await self.async_step()
                        else:
                            # Dummy forward to participate in the EP collective.
                            self.step_start_event.record()
                            nvtx_range_push("EP-dummy-forward")
                            self.controller.dummy_forward()
                            self.step_end_event.record()
                            self.step_end_event.synchronize()
                            nvtx_range_pop("EP-dummy-forward")
                            self.context.step_count += 1
                            self.context.prefix_cache_lru_clock += 1
                    else:
                        # No work, but not all pausing: idle.
                        await asyncio.sleep(0.001 if local_pending_imports > 0 else 0.02)

                elif self.state == EngineState.PAUSED:
                    await asyncio.sleep(0.02)

                elif self.state == EngineState.UNPAUSING:
                    await self._world_barrier()
                    self.state = EngineState.RUNNING
                    self._state_events[EngineState.PAUSED].clear()
                    self._state_events[EngineState.RUNNING].set()
                    # The cache from the PAUSING phase still has all_pausing=True;
                    # without this reset the next RUNNING iteration would skip
                    # consensus, read the stale flag, and immediately re-pause.
                    self._last_ep_consensus = (0, False)

                elif self.state == EngineState.SUSPENDING:
                    await self._world_barrier()
                    self.state = EngineState.SUSPENDED
                    self._state_events[EngineState.SUSPENDED].set()

                elif self.state == EngineState.SUSPENDED:
                    await asyncio.sleep(0.02)

                elif self.state == EngineState.RESUMING:
                    await self._world_barrier()
                    self.state = EngineState.PAUSED
                    self._state_events[EngineState.RESUMED].set()

                elif self.state == EngineState.STOPPING:
                    await self._world_barrier()
                    if self.rank == 0:
                        logger.info("Stopping engine.")
                    break

        finally:
            await self.shutdown()
