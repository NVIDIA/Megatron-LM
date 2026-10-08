# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import logging
import time
from collections import OrderedDict, deque
from contextlib import contextmanager
from typing import Dict, Optional, Tuple

import torch

from megatron.core.inference.config import AsyncScheduleMode, KVCacheManagementMode
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines.abstract_engine import AbstractEngine
from megatron.core.inference.inference_request import (
    DynamicInferenceRequest,
    DynamicVLMInferenceRequest,
    FinishedRequestRecord,
    RequestPayloadStager,
    RequestPromptPreparer,
)
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    DecodeOnly,
    TextGenerationController,
)
from megatron.core.inference.utils import Counter, InferenceMode
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.cuda_graphs import CudaGraphManager, delete_cuda_graphs
from megatron.core.transformer.enums import InferenceCudaGraphScope
from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction
from megatron.core.utils import (
    deprecate_args,
    experimental_api,
    get_asyncio_loop,
    get_pg_size,
    nvtx_range_pop,
    nvtx_range_push,
    round_up_to_nearest_multiple,
    unwrap_model,
)

from .coordinator import CoordinatorMixin
from .loop import (
    DynamicInferenceEngineStepResult,
    EngineLoopMixin,
    EngineState,
    EngineSuspendedError,
)
from .multimodal import MultimodalRequestMixin, _VisionCacheEntry
from .post_process import StepPostProcessingMixin
from .requests import RequestEntry, RequestIntakeMixin, _weight_scoped_salt

try:
    from tqdm import tqdm

    HAVE_TQDM = True
except:
    HAVE_TQDM = False

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

__all__ = [
    "DynamicInferenceEngine",
    "DynamicInferenceEngineStepResult",
    "EngineState",
    "EngineSuspendedError",
    "RequestEntry",
]

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


def format_mem_bytes(mem_bytes):
    """Convert a byte count to a human-readable string in tb, gb, mb, kb, or bytes."""
    if mem_bytes < 0:
        return "-" + format_mem_bytes(-mem_bytes)
    for power, suffix in [(4, "tb"), (3, "gb"), (2, "mb"), (1, "kb"), (0, "bytes")]:
        suffix_bytes = 1024**power
        if mem_bytes >= suffix_bytes:
            return "%.1f %s" % (mem_bytes / suffix_bytes, suffix)
    return "%d bytes" % mem_bytes


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


# pylint: disable=line-too-long
@experimental_api
class DynamicInferenceEngine(
    EngineLoopMixin,
    RequestIntakeMixin,
    MultimodalRequestMixin,
    StepPostProcessingMixin,
    CoordinatorMixin,
    AbstractEngine,
):
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
            output tokens, and apply token-level generation policy.
        inference_context (DynamicInferenceContext): Context for managing in-flight
            batching and a dynamic block-level KV cache (similar to paged attention).
    """

    # Class-level default so the attribute is always readable: engines are built
    # without running __init__ in places (tests, and anything constructing via
    # object.__new__), and admitting a request reads this. An int is immutable, so
    # the += in resume() still rebinds onto the instance.
    _weight_epoch: int = 0

    # Defaults for instances created without __init__ (tests build the engine via __new__).
    payload_stager: Optional[RequestPayloadStager] = None
    prompt_preparer: Optional[RequestPromptPreparer] = None

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
        self.vision_embedding_cache_max_bytes = int(
            getattr(inference_config, "vision_embedding_cache_max_bytes", 0)
        )
        self.allow_stale_multimodal_embeddings = bool(
            getattr(inference_config, "allow_stale_multimodal_embeddings", False)
        )
        if self.vision_embedding_cache_max_bytes < 0:
            raise ValueError("vision_embedding_cache_max_bytes must be non-negative.")
        self._vision_embedding_cache: OrderedDict[str, _VisionCacheEntry] = OrderedDict()
        self._vision_embedding_cache_bytes = 0
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

        # Payload offload: with a stager attached, each completed request's per-token payload
        # (log probs, MoE routing indices, token ids) is handed to stage() and dropped from the
        # reply instead of riding the RESTful API. Consumer-owned, so it survives reset().
        self.payload_stager: Optional[RequestPayloadStager] = None
        self.prompt_preparer: Optional[RequestPromptPreparer] = None

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
        """Reset per-run state; the caller must first drain all requests."""

        initialize_runtime_state = not hasattr(self, "_state_events")
        if not initialize_runtime_state and self.state not in (
            EngineState.RUNNING,
            EngineState.PAUSED,
        ):
            raise RuntimeError(
                "A drained engine can only be reset while RUNNING or PAUSED; "
                f"got {self.state.name}."
            )

        if not initialize_runtime_state and self.requests:
            raise RuntimeError(
                "The engine must drain all requests before reset; "
                f"got {len(self.requests)} outstanding request(s)."
            )

        use_coordinator = getattr(self, "use_coordinator", False)
        self._reset_pending_kv_imports()
        self.clear_vision_embedding_cache()
        self.context.reset()
        self.controller._async_sched_logits.clear()

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
        # Weight generation counter, bumped on every resume. Used only to salt
        # block hashes so the prefix cache cannot serve KV computed by earlier
        # weights. Deliberately separate from `_generation_epoch`, which is driven
        # by the SET_GENERATION_EPOCH control message and stamps per-request
        # reporting fields.
        self._weight_epoch: int = 0
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
        if initialize_runtime_state:
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

        # Coordinator mode and its long-lived runtime objects survive a drained
        # reset; replacing them can strand waiters on the old asyncio objects.
        self.use_coordinator = use_coordinator

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

                        # KV-aware MTP graph: when the MTP KV cache is enabled, ALSO capture a graph
                        # that includes the draft-attention KV append+attend (the cache-free graph
                        # above does not), under a distinct ("mtp_kv", n, depth) key that the real
                        # spec-decode path replays. The EP dummy path keeps the cache-free graph.
                        # Uses synthetic scratch metadata (all dummy_block_idx / position 0); graph
                        # replay overwrites it from gpu_view each step, so only shapes/bounds count.
                        if context.enable_mtp_kv_cache:
                            context.mtp_metadata.begin_decode_for_capture(n)
                            for depth in mtp_warmup_depths:
                                context._mtp_setup_decode_step()
                                unwrapped.compute_mtp_single_step(
                                    hidden_states=torch.zeros(
                                        (batch_dim, 1, model_config.hidden_size),
                                        device=device,
                                        dtype=model_config.params_dtype,
                                    ),
                                    next_token_ids=torch.zeros(
                                        (1, n), device=device, dtype=torch.long
                                    ),
                                    position_ids=torch.zeros(
                                        (1, n), device=device, dtype=torch.int64
                                    ),
                                    depth=depth,
                                    mtp_inference_context=context,
                                    cache_key=("mtp_kv", n, depth),
                                )
                                context.mtp_metadata.advance_decode_step()
                            context.mtp_metadata.end_forward()

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

        # A suspend may be followed by an in-place weight refit. Invalidate
        # shared and request-local projected media unless explicitly permitted.
        self._invalidate_vision_state()
        InferenceMode.unset_active()
        dynamo_helper = getattr(self.context, "dynamo_helper", None)
        if dynamo_helper is not None:
            dynamo_helper.discard_pending_kv_stored_events()

        # RECOMPUTE must preserve the current active row order, which maps sampling
        # RNG draws and request metadata to IDs. Paused rows precede active rows and
        # resume from the right (LIFO). Snapshot both before releasing their tensors.
        if (
            self.context.kv_cache_management_mode == KVCacheManagementMode.RECOMPUTE
            and self.requests
        ):
            resident_request_ids = self.context.request_ids[
                : self.context.total_request_count
            ].tolist()
            resident_paused_count = self.context.paused_request_count
        else:
            resident_request_ids = []
            resident_paused_count = 0

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
        waiting_request_id_set = set(waiting_request_ids)
        if self.context.kv_cache_management_mode == KVCacheManagementMode.RECOMPUTE:
            self.controller._async_sched_logits.clear()
            ordered_resident_ids = [
                *resident_request_ids[resident_paused_count:],
                *reversed(resident_request_ids[:resident_paused_count]),
            ]
            recompute_resident_ids = [
                request_id
                for request_id in ordered_resident_ids
                if request_id not in waiting_request_id_set
            ]

            # Reset any partially prefilled requests so they recompute from the start
            for req_id in [*waiting_request_ids, *recompute_resident_ids]:
                req = self.get_request(req_id)
                if req.finished_chunk_token_count > 0 and not req.generated_tokens:
                    req.remaining_prompt_tokens = req.prompt_tokens
                    req.finished_chunk_token_count = 0
                    # The restarted prefill recomputes these positions, so its
                    # scores must replace rather than follow the partial scores.
                    req.prompt_log_probs = None
                    req.prompt_top_n_logprobs = None
                    req.num_matched_prefix_blocks = 0

            # Reset the chunked prefill request id
            self.chunked_prefill_request_id = -1
        else:
            recompute_resident_ids = []
        self.resume_request_ids = [*recompute_resident_ids, *waiting_request_ids]
        self.waiting_request_ids.clear()

        # Checkpoint resident requests that are marked for recompute.
        for request_id in recompute_resident_ids:
            self.requests[request_id].record.checkpoint()

        # If we are not using the inference coordinator, we need to manually handle state.
        if not self.use_coordinator:
            self.state = EngineState.SUSPENDED

    def resume(self):
        """Resume engine by reallocating context's GPU state."""

        # Skip if not suspended or in the process of suspending.
        if self.state not in (EngineState.SUSPENDED, EngineState.SUSPENDING):
            return

        # A suspend/resume cycle is how a weight refit is staged, so treat resume
        # as a new weight generation and re-salt the prefix cache. Requests can
        # be constructed before this bump -- those still waiting at suspend, and
        # those submitted while paused or suspended, since the coordinator loop
        # keeps admitting SUBMITs -- so re-salt every request that holds no KV
        # yet: all of its blocks will be computed by the new weights. A request
        # already part-way through prefill or decode keeps its original salt, so
        # it republishes under its original generation and is unmatchable by
        # new arrivals.
        self._weight_epoch += 1
        self._resalt_unprefilled_requests()

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

            capture_time = time.time()
            if (
                self.context.kv_cache_management_mode != KVCacheManagementMode.PERSIST
                and not self.context.static_kv_memory_pointers
            ):
                self.create_cuda_graphs()
            capture_time = time.time() - capture_time

            # Request records outlive the context buffers. Refresh every stale
            # VLM record, including PERSIST/OFFLOAD requests that need not be
            # re-added below, so no later checkpoint can carry old projections.
            for entry in self.requests.values():
                request = entry.record[-1]
                if isinstance(request, DynamicVLMInferenceRequest):
                    self._refresh_vlm_request_data(request)

            # Re-add requests saved during suspend.
            add_time = time.time()
            torch.cuda.synchronize()
            for request_id in self.resume_request_ids:
                request = self.get_request(request_id)
                self._add_request(request, is_resume=True)
                if isinstance(request, DynamicVLMInferenceRequest):
                    # Buffer reinitialization wipes the context maps. Restore
                    # either the freshly recomputed state or the explicitly
                    # permitted retained state.
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

    def _resalt_unprefilled_requests(self) -> None:
        """Re-salt requests that hold no KV yet under the current weight epoch.

        Such a request has its whole prompt computed by the weights that serve
        it after resume, so it must hash like a request admitted now. A request
        with any prefill or decode progress keeps its salt: its leading blocks
        were computed by the previous weights.
        """
        for entry in self.requests.values():
            request = entry.record[-1]
            if (
                request.finished_chunk_token_count > 0
                or request.generated_tokens
                or request.num_cached_tokens > 0
                or request.num_matched_prefix_blocks > 0
                or request.request_id == self.context.chunked_prefill_request_id
            ):
                continue
            # Same scoping as add_request: VLM requests are image-bearing, so
            # they are salted by their media identity as well.
            media_cache_key = (
                request.media_cache_key if isinstance(request, DynamicVLMInferenceRequest) else None
            )
            request.resalt_block_hashes(_weight_scoped_salt(self._weight_epoch, media_cache_key))

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
