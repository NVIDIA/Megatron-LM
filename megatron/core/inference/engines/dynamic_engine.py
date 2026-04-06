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
from enum import Enum, auto
from itertools import repeat
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.cuda.nvtx import range_pop, range_push

from megatron.core.inference.config import KVCacheManagementMode
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
    Status,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.inference.utils import (
    Counter,
    await_process_call,
    set_inference_cuda_graphed_iteration_for_ep_inference,
    unset_inference_cuda_graphed_iteration_for_ep_inference,
)
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
                self.num_speculative_tokens <= self.controller.num_mtp_heads
            ), f"Number of speculative tokens {self.num_speculative_tokens} must be less than or equal to number of MTP heads {self.controller.num_mtp_heads}"
            assert (
                not self.materialize_only_last_token_logits
            ), "materialize_only_last_token_logits must be False when num_speculative_tokens > 0"

        self.track_paused_request_events = inference_config.track_paused_request_events
        self.track_generated_token_events = inference_config.track_generated_token_events
        self.enable_chunked_prefill = inference_config.enable_chunked_prefill
        self.metrics_writer = inference_config.metrics_writer
        self.logging_step_interval = inference_config.logging_step_interval
        self.unified_memory_level = inference_config.unified_memory_level
        self.use_synchronous_zmq_collectives = inference_config.use_synchronous_zmq_collectives
        self.cuda_graph_impl = model_config.cuda_graph_impl
        self.cuda_graph_scope = model_config.cuda_graph_scope
        self.autotune = inference_config.autotune
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

        # Auto-tune parameters before building CUDA graphs.
        if self.autotune:
            self._autotune_and_rebuild()

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
        self._generation_epoch: Optional[int] = None
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
        self._state_events = {k: asyncio.Event() for k in self._STATE_EVENTS}
        self.state = EngineState.RUNNING
        self._state_events[EngineState.RUNNING].set()
        self._pending_signals = deque()

        self.resume_request_ids = None

        # Speculative decoding acceptance tracking.
        self._spec_tokens_proposed = 0
        self._spec_tokens_accepted = 0
        self._spec_steps = 0

        # Prefix caching tracking.
        self._prefix_cache_hits = 0
        self._prefix_cache_blocks_matched = 0
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

        logging.info(
            "> dynamic_engine.py: building cuda graphs for %d batch dimensions. "
            "Context: max_requests=%d, max_tokens=%d, max_seq_len=%d, "
            "block_size_tokens=%d, total_blocks=%d, active_blocks=%d, "
            "block_size_bytes=%d, is_hybrid=%s",
            len(context.cuda_graph_batch_dimensions_list),
            context.max_requests,
            context.max_tokens,
            context.max_sequence_length,
            context.block_size_tokens,
            context.kv_block_allocator.total_count,
            context.kv_block_allocator.active_count,
            context.block_size_bytes,
            context.is_hybrid_model,
        )
        for graph in context.cuda_graph_batch_dimensions_list:
            logging.info(graph)

        # Enable inference dispatcher for EP during graph capture
        model_config = controller.inference_wrapped_model.model.config
        is_inference_optimized_ep = (
            model_config.transformer_impl == "inference_optimized"
            and model_config.expert_model_parallel_size > 1
        )
        if is_inference_optimized_ep:
            unwrapped_model = controller.inference_wrapped_model.model
            set_inference_cuda_graphed_iteration_for_ep_inference(unwrapped_model)

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

        # Disable inference dispatcher after graph capture
        if is_inference_optimized_ep:
            unset_inference_cuda_graphed_iteration_for_ep_inference(unwrapped_model)

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

    @staticmethod
    def _force_free_context_tensors(context):
        """Free GPU memory held by a context's TMS regions.

        Pauses all TMS tags (freeing physical pages) and clears TMS's
        internal MemPool entries so that subsequent allocations with the
        same tags start from a clean state.

        Non-TMS tensors are left alone — they'll be garbage collected
        when all references to the context are dropped.
        """
        import gc

        tms_tags = getattr(context, '_tms_tags', None) or []
        if tms_tags:
            try:
                from torch_memory_saver import torch_memory_saver as tms

                for tag in tms_tags:
                    tms.pause(tag)

                # Clear TMS's private MemPools so their virtual address
                # reservations are released for new allocations.
                mem_pools = tms._impl._mem_pools
                for tag in tms_tags:
                    for cpu_backup in (True, False):
                        mem_pools.pop((tag, cpu_backup), None)
            except (ImportError, AttributeError, KeyError):
                pass

        gc.collect()
        torch.cuda.empty_cache()

    def _autotune_and_rebuild(self, profiling_context_fraction: float = 0.5):
        """Profile activation memory on the existing context, then rebuild
        with tuned parameters before CUDA graphs are created.

        Args:
            profiling_context_fraction: Fraction of free GPU memory (after
                workspace warmup) to use for the profiling context state.
                The rest is left for activation memory during profiling
                forward passes. Defaults to 0.5 (50/50 split).
        """
        from dataclasses import replace

        from megatron.core.inference.autotune import AutotuneProfile, compute_optimal_params

        model_config = self.controller.inference_wrapped_model.model.config
        old_config = self.context.config
        controller = self.controller

        gpu_total = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).total_memory

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        # --- Step 1: Tiny context to measure workspace cost ---
        # The first forward pass allocates one-time workspace buffers
        # (cuBLAS, MoE expert weights). We need to measure this before
        # we can size the real profiling context.
        self._force_free_context_tensors(self.context)
        torch.cuda.empty_cache()

        tp_size = max(getattr(model_config, 'tensor_model_parallel_size', 1), 1)
        alignment = max(tp_size, DynamicInferenceContext.REQUEST_ROUNDER)
        tiny_max_requests = max(alignment, (16 // alignment) * alignment)
        tiny_buffer_gb = 0.5  # minimal

        tiny_config = replace(
            old_config,
            buffer_size_gb=tiny_buffer_gb,
            max_requests=tiny_max_requests,
            max_tokens=None,
            autotune=False,
            static_kv_memory_pointers=False,
            kv_cache_management_mode=KVCacheManagementMode.PERSIST,
        )
        self.context = DynamicInferenceContext(model_config, tiny_config)
        controller.inference_wrapped_model.inference_context = self.context
        controller._reinit_for_context()
        self.reset()

        # Run one forward pass to trigger workspace allocation.
        from megatron.core.inference.inference_request import DynamicInferenceRequest
        from megatron.core.inference.sampling_params import SamplingParams

        warmup_ok = False
        try:
            dummy = [DynamicInferenceRequest(
                request_id=0,
                prompt_tokens=torch.tensor([0], device=torch.cuda.current_device()),
                sampling_params=SamplingParams(num_tokens_to_generate=1, termination_id=-1),
            )]
            self.context.add_dummy_requests_parallel(dummy)
            self.context.initialize_attention_state()
            input_ids, position_ids = self.context.current_input_and_position_ids()
            with torch.inference_mode():
                controller._dynamic_step_forward_logits(input_ids, position_ids)
            warmup_ok = True
        except Exception as e:
            if rank == 0:
                logging.info("Autotune: workspace warmup failed: %r", e)
        self.context.reset()

        # Barrier: if workspace warmup failed on any rank, abort autotune.
        if torch.distributed.is_initialized():
            ok = torch.tensor(
                [1 if warmup_ok else 0], dtype=torch.int32,
                device=torch.cuda.current_device(),
            )
            torch.distributed.all_reduce(ok, op=torch.distributed.ReduceOp.MIN)
            if ok.item() == 0:
                logging.warning("Autotune: workspace warmup failed on at least one rank. "
                                "Skipping autotune.")
                return

        # Measure free memory after workspace is allocated.
        free_after_workspace, _ = torch.cuda.mem_get_info()
        if rank == 0:
            logging.info(
                "Autotune: workspace warmup done. Free memory: %.1f GB",
                free_after_workspace / (1024 ** 3),
            )

        # --- Step 2: Properly-sized profiling context ---
        # Now we know how much memory workspace takes. Free the tiny context
        # and create one sized from actual free memory.
        # Read per-request cost from the tiny context before freeing it.
        per_req_bytes = self.context.block_size_bytes
        if self.context.is_hybrid_model:
            conv_bytes = (
                math.prod(self.context.mamba_conv_states_shape)
                * self.context.mamba_conv_states_dtype.itemsize
                * self.context.num_mamba_layers
            )
            ssm_bytes = (
                math.prod(self.context.mamba_ssm_states_shape)
                * self.context.mamba_ssm_states_dtype.itemsize
                * self.context.num_mamba_layers
            )
            per_req_bytes += conv_bytes + ssm_bytes

        self._force_free_context_tensors(self.context)
        torch.cuda.empty_cache()

        free_for_profiling, _ = torch.cuda.mem_get_info()
        context_budget = int(free_for_profiling * profiling_context_fraction)
        profiling_max_requests = max(alignment, context_budget // max(per_req_bytes, 1))
        profiling_max_requests = (profiling_max_requests // alignment) * alignment

        # All ranks must use the same profiling_max_requests to avoid NCCL
        # deadlocks in MoE/TP all-to-all communication during forward passes.
        if torch.distributed.is_initialized():
            sync_tensor = torch.tensor(
                [profiling_max_requests], dtype=torch.int64,
                device=torch.cuda.current_device(),
            )
            torch.distributed.all_reduce(sync_tensor, op=torch.distributed.ReduceOp.MIN)
            profiling_max_requests = int(sync_tensor.item())
            profiling_max_requests = (profiling_max_requests // alignment) * alignment
            profiling_max_requests = max(alignment, profiling_max_requests)

        profiling_buffer_gb = max(0.1, (profiling_max_requests * per_req_bytes * 1.05) / (1024 ** 3))

        profiling_config = replace(
            old_config,
            buffer_size_gb=profiling_buffer_gb,
            max_requests=profiling_max_requests,
            max_tokens=None,
            autotune=False,
            static_kv_memory_pointers=False,
            kv_cache_management_mode=KVCacheManagementMode.PERSIST,
        )
        self.context = DynamicInferenceContext(model_config, profiling_config)
        controller.inference_wrapped_model.inference_context = self.context
        controller._reinit_for_context()
        self.reset()

        context = self.context

        free_after_profiling_ctx, _ = torch.cuda.mem_get_info()
        if rank == 0:
            logging.info(
                "Autotune: profiling context created (max_requests=%d, buffer=%.2f GB). "
                "Free memory: %.1f GB",
                context.max_requests,
                profiling_buffer_gb,
                free_after_profiling_ctx / (1024 ** 3),
            )

        # Compute mamba memory per request from the context.
        mamba_memory_per_request = 0
        if context.is_hybrid_model:
            conv_bytes = (
                math.prod(context.mamba_conv_states_shape)
                * context.mamba_conv_states_dtype.itemsize
                * context.num_mamba_layers
            )
            ssm_bytes = (
                math.prod(context.mamba_ssm_states_shape)
                * context.mamba_ssm_states_dtype.itemsize
                * context.num_mamba_layers
            )
            mamba_memory_per_request = conv_bytes + ssm_bytes

        mem_after_model = gpu_total - free_after_profiling_ctx

        if rank == 0:
            logging.info(
                "Autotune: estimated model+runtime %.1f GB (GPU total %.1f GB)",
                mem_after_model / (1024 ** 3),
                gpu_total / (1024 ** 3),
        )

        profile = AutotuneProfile(
            gpu_total_bytes=gpu_total,
            memory_after_model_load_bytes=mem_after_model,
            block_size_bytes=context.block_size_bytes,
            mamba_memory_per_request=mamba_memory_per_request,
            max_kv_block_count=context.max_kv_block_count,
            per_request_bytes=(
                DynamicInferenceContext.PER_REQUEST_SCALAR_BYTES
                + DynamicInferenceContext.PER_REQUEST_METADATA_BYTES
                + context.max_kv_block_count * 4
            ),
            per_token_bytes=DynamicInferenceContext.PER_TOKEN_METADATA_BYTES,
        )

        tp_size = max(getattr(model_config, 'tensor_model_parallel_size', 1), 1)

        # --- Profile on the existing context ---
        # Generate token counts: geometric spread from 1 up to max we can fit.
        token_counts = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        t = 512
        while t <= context.max_requests:
            token_counts.append(t)
            t += max(256, t // 4)
        # Include the largest valid decode size.
        if token_counts[-1] != context.max_requests:
            token_counts.append(context.max_requests)
        token_counts = sorted(set(tc for tc in token_counts if tc <= context.max_requests))

        if rank == 0:
            logging.info(
                "Autotune: profiling %d token counts (max=%d) on existing context",
                len(token_counts),
                token_counts[-1] if token_counts else 0,
            )

        for tc in token_counts:
            # Add dummy decode requests and initialize attention state in
            # eager mode. We don't pass construct_graph_dimensions because
            # no CUDA graphs exist yet during profiling.
            try:
                from megatron.core.inference.inference_request import (
                    DynamicInferenceRequest,
                )
                from megatron.core.inference.sampling_params import SamplingParams

                dummy_requests = []
                for i in range(tc):
                    req = DynamicInferenceRequest(
                        request_id=i,
                        prompt_tokens=torch.tensor([0], device=torch.cuda.current_device()),
                        sampling_params=SamplingParams(
                            num_tokens_to_generate=1,
                            termination_id=-1,
                        ),
                    )
                    dummy_requests.append(req)
                context.add_dummy_requests_parallel(dummy_requests)
                context.initialize_attention_state()
                input_ids, position_ids = context.current_input_and_position_ids()
                init_ok = True
            except Exception as e:
                if rank == 0:
                    logging.info("Autotune: context_init failed at tc=%d: %r", tc, e)
                context.reset()
                init_ok = False

            # All ranks must agree before running the forward pass —
            # if any rank failed context_init, all must skip to avoid
            # NCCL deadlocks in MoE/TP collective ops.
            if torch.distributed.is_initialized():
                ready = torch.tensor(
                    [1 if init_ok else 0], dtype=torch.int32,
                    device=torch.cuda.current_device(),
                )
                torch.distributed.all_reduce(ready, op=torch.distributed.ReduceOp.MIN)
                if ready.item() == 0:
                    context.reset()
                    continue
            elif not init_ok:
                continue

            try:
                baseline_bytes = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                with torch.inference_mode():
                    controller._dynamic_step_forward_logits(input_ids, position_ids)
                end_event.record()
                torch.cuda.synchronize()

                peak_bytes = torch.cuda.max_memory_allocated()
                activation_bytes = peak_bytes - baseline_bytes
                elapsed_ms = start_event.elapsed_time(end_event)
                profile.add_sample(tc, activation_bytes, elapsed_ms)
            except Exception as e:
                if rank == 0:
                    logging.info("Autotune: forward failed at tc=%d: %r", tc, e)
                torch.cuda.empty_cache()

            context.reset()

        # --- Prefill profiling ---
        # Measure activation memory for prefill (1 request with long prompt).
        # Prefill uses full attention over the prompt, so activation per token
        # is higher than decode. The solver needs the worst case.
        prefill_token_counts = []
        t = context.max_requests  # start above decode range
        max_prefill = context.max_tokens
        while t <= max_prefill:
            prefill_token_counts.append(t)
            t += max(256, t // 4)
        if prefill_token_counts and prefill_token_counts[-1] != max_prefill:
            prefill_token_counts.append(max_prefill)

        if rank == 0:
            logging.info(
                "Autotune: prefill profiling — %d token counts (up to %d)",
                len(prefill_token_counts),
                prefill_token_counts[-1] if prefill_token_counts else 0,
            )

        for tc in prefill_token_counts:
            init_ok = False
            try:
                # 1 request with tc prompt tokens.
                dummy = [DynamicInferenceRequest(
                    request_id=0,
                    prompt_tokens=torch.zeros(tc, dtype=torch.long,
                                              device=torch.cuda.current_device()),
                    sampling_params=SamplingParams(
                        num_tokens_to_generate=1,
                        termination_id=-1,
                    ),
                )]
                context.add_dummy_requests_parallel(dummy, count_as_prefill=True)
                context.initialize_attention_state()
                input_ids, position_ids = context.current_input_and_position_ids()
                init_ok = True
            except Exception as e:
                if rank == 0:
                    logging.info("Autotune: prefill init failed at tc=%d: %r", tc, e)
                context.reset()

            if torch.distributed.is_initialized():
                ready = torch.tensor(
                    [1 if init_ok else 0], dtype=torch.int32,
                    device=torch.cuda.current_device(),
                )
                torch.distributed.all_reduce(ready, op=torch.distributed.ReduceOp.MIN)
                if ready.item() == 0:
                    context.reset()
                    continue
            elif not init_ok:
                continue

            try:
                baseline_bytes = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                with torch.inference_mode():
                    controller._dynamic_step_forward_logits(input_ids, position_ids)
                end_event.record()
                torch.cuda.synchronize()

                peak_bytes = torch.cuda.max_memory_allocated()
                activation_bytes = peak_bytes - baseline_bytes
                elapsed_ms = start_event.elapsed_time(end_event)
                profile.add_sample(tc, activation_bytes, elapsed_ms)
            except Exception as e:
                if rank == 0:
                    logging.info("Autotune: prefill forward failed at tc=%d: %r", tc, e)
                torch.cuda.empty_cache()

            context.reset()

        if rank == 0:
            logging.info("Autotune: collected %d total profiling samples "
                         "(%d decode + prefill)", len(profile.token_counts),
                         len(profile.token_counts))

        # Re-measure model+runtime memory AFTER profiling. The first forward
        # pass allocates one-time workspace (cuBLAS, MoE expert weights) that
        # persists. The solver must account for this.
        self._force_free_context_tensors(self.context)
        torch.cuda.empty_cache()
        free_after_profiling, _ = torch.cuda.mem_get_info()
        profile.memory_after_model_load_bytes = gpu_total - free_after_profiling
        if rank == 0:
            logging.info(
                "Autotune: after profiling — model+runtime+workspace = %.1f GB "
                "(free %.1f GB)",
                profile.memory_after_model_load_bytes / (1024 ** 3),
                free_after_profiling / (1024 ** 3),
            )

        # --- Compute optimal parameters ---
        new_max_requests, new_max_tokens, new_buffer_size_gb = compute_optimal_params(
            profile, tp_size=tp_size,
            request_rounder=DynamicInferenceContext.REQUEST_ROUNDER,
        )

        # All ranks must use the same tuned parameters to avoid NCCL
        # deadlocks during CUDA graph build and runtime (MoE/TP all-to-all).
        if torch.distributed.is_initialized():
            sync_tensor = torch.tensor(
                [new_max_requests, new_max_tokens],
                dtype=torch.int64,
                device=torch.cuda.current_device(),
            )
            torch.distributed.all_reduce(sync_tensor, op=torch.distributed.ReduceOp.MIN)
            new_max_requests = int(sync_tensor[0].item())
            new_max_tokens = int(sync_tensor[1].item())
            # Recompute buffer_size_gb from the synchronized max_requests.
            kv_bytes = (new_max_requests + 1) * profile.block_size_bytes
            mamba_bytes = new_max_requests * profile.mamba_memory_per_request
            new_buffer_size_gb = (kv_bytes + mamba_bytes) / (1024 ** 3)

        # --- Rebuild context with tuned parameters ---
        if rank == 0:
            logging.info(
                "Autotune: rebuilding context with max_requests=%d, max_tokens=%d, "
                "buffer_size_gb=%.2f",
                new_max_requests,
                new_max_tokens,
                new_buffer_size_gb,
            )

        # Context already freed above when re-measuring memory.
        tuned_config = replace(
            old_config,
            max_requests=new_max_requests,
            max_tokens=new_max_tokens,
            buffer_size_gb=new_buffer_size_gb,
            autotune=False,
        )
        self.context = DynamicInferenceContext(model_config, tuned_config)
        controller.inference_wrapped_model.inference_context = self.context
        controller._reinit_for_context()
        self.autotune = False
        self.reset()

        # CUDA graphs are built by the caller (create_cuda_graphs) after we return.

        if rank == 0:
            logging.info(
                "\n"
                "========== Autotune Complete ==========\n"
            "  max_requests:    %d\n"
            "  max_tokens:      %d\n"
            "  buffer_size_gb:  %.2f\n"
            "\n"
            "To reproduce without autotune, replace\n"
            "  --inference-dynamic-batching-autotune\n"
            "with:\n"
            "  --inference-dynamic-batching-max-requests %d \\\n"
            "  --inference-dynamic-batching-max-tokens %d \\\n"
            "  --inference-dynamic-batching-buffer-size-gb %.2f\n"
            "=======================================",
            new_max_requests,
            new_max_tokens,
            new_buffer_size_gb,
            new_max_requests,
            new_max_tokens,
            new_buffer_size_gb,
        )

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
                self.zmq_context, process_group=self.pg_collection.ep, hostname=hostname
            )

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

        # Skip if already suspended or in the process of suspending.
        if self.state in (EngineState.SUSPENDED, EngineState.SUSPENDING):
            return

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

            # Reset any partially prefilled requests so they recompute from the start
            for req_id in [*waiting_request_ids, *recompute_active_ids]:
                req = self.get_request(req_id)
                if req.finished_chunk_token_count > 0:
                    req.remaining_prompt_tokens = req.prompt_tokens
                    req.finished_chunk_token_count = 0

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

            # Ensure chunked prefill request remains at the head of the waiting queue
            if self.context.chunked_prefill_request_id != -1:
                if self.context.chunked_prefill_request_id in self.waiting_request_ids:
                    self.waiting_request_ids.remove(self.context.chunked_prefill_request_id)
                    self.waiting_request_ids.appendleft(self.context.chunked_prefill_request_id)

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

    def _handle_failed_request(self, request_id: int):
        """Handle a failed request by sending the reply immediately.

        The request is added to failed_request_ids so that the next bookkeeping pass can return it.
        """
        request_entry = self.requests[request_id]
        request = request_entry.record[-1]

        if self.rank == 0:
            warnings.warn(
                f"Request {request_id} failed to be added to the engine due to errors. "
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
            payload = msgpack.packb(
                [Headers.ENGINE_REPLY.value, [request_entry.record.merge().serialize()]],
                use_bin_type=True,
            )
            self.socket_for_receiving_requests.send(payload)
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

        if (
            len(request.prompt_tokens) + request.sampling_params.num_tokens_to_generate
            > self.context.max_sequence_length
        ) or (request.sampling_params.num_tokens_to_generate < 0):
            request.status = Status.FAILED
            request.add_event_error_nontransient(MaxSequenceLengthOverflowError(request_id))

        if len(request.prompt_tokens) > self.context.max_tokens and not self.enable_chunked_prefill:
            request.status = Status.FAILED
            request.add_event_error_nontransient(TokenOverflowError(request_id))

        # Check that the KV cache has enough blocks for this request's max sequence length.
        max_request_tokens = (
            len(request.prompt_tokens) + request.sampling_params.num_tokens_to_generate
        )
        request_block_count = math.ceil(max_request_tokens / self.context.block_size_tokens)
        total_blocks = self.context.kv_block_allocator.total_count - 1  # -1 for dummy block
        if request_block_count > total_blocks:
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
        accepted_tokens: torch.Tensor,
        log_probs: torch.Tensor,
        top_n_logprobs: Optional[Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        routing_indices_per_request: Optional[Dict[int, torch.Tensor]] = None,
        pre_fwd_active_token_count: Optional[int] = None,
        pre_fwd_step_count: Optional[int] = None,
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
        block_allocator = self.context.kv_block_allocator

        # Pre-compute step-level block stats (before the per-request loop)
        if self.track_generated_token_events:
            blocks_allocated = block_allocator.total_count - block_allocator.total_avail
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

        for req_idx, (request_id, tokens, accepted_tokens_list, request_log_probs) in enumerate(
            zip(request_ids.tolist(), sample.tolist(), accepted_tokens_iter, log_probs_iter)
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
            if request_id != self.context.chunked_prefill_request_id:
                # Skip appending token for requests being finished due to stop words
                # (they already have their final token from the previous step)
                # If the request already has more tokens, then we only append as much as is necessary
                if (
                    len(request.generated_tokens) + len(tokens)
                    >= request.sampling_params.num_tokens_to_generate
                ):
                    tokens = tokens[
                        : request.sampling_params.num_tokens_to_generate
                        - len(request.generated_tokens)
                    ]
                if request_id not in self.stop_word_being_finished_ids:
                    is_first_token = len(request.generated_tokens) == 0
                    request.generated_tokens += tokens
                    first_token_event = None
                    if self.track_generated_token_events:
                        for token in tokens:
                            if block_allocator.enable_prefix_caching:
                                event = request.add_event_generated_token(
                                    token,
                                    blocks_total=block_allocator.total_count,
                                    blocks_hashed_total=blocks_allocated,
                                    blocks_hashed_active=blocks_hashed_active,
                                    blocks_ref_count=blocks_ref_count,
                                    pre_fwd_active_token_count=pre_fwd_active_token_count,
                                    pre_fwd_step_count=pre_fwd_step_count,
                                )
                            else:
                                event = request.add_event_generated_token(
                                    token,
                                    blocks_total=block_allocator.total_count,
                                    blocks_hashed_total=blocks_allocated,
                                    blocks_hashed_active=blocks_hashed_active,
                                    pre_fwd_active_token_count=pre_fwd_active_token_count,
                                    pre_fwd_step_count=pre_fwd_step_count,
                                )
                            if first_token_event is None:
                                first_token_event = event
                    if is_first_token:
                        if not self.track_generated_token_events:
                            first_token_event = DynamicInferenceEvent(
                                type=DynamicInferenceEventType.GENERATED_TOKEN,
                                payload={"token_id": tokens[0]},
                            )
                        request.ttft = (
                            first_token_event.timestamp - request.event_add_engine.timestamp
                        )
                    if request.tpot is None:
                        request.tpot = []
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

                # Track acceptance statistics for logging.
                if len(request.generated_tokens) > 0 and self.num_speculative_tokens > 0:
                    actual_proposed = max(0, self.num_speculative_tokens - num_stop_word_trim)
                    actual_accepted = max(0, len(accepted_tokens) - num_stop_word_trim)

                    self._spec_tokens_proposed += actual_proposed
                    self._spec_tokens_accepted += actual_accepted

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

            # When a stop word was found mid-speculative-batch, trim log probs
            # and top_n_logprobs to match the truncated generated_tokens.
            if num_stop_word_trim > 0:
                if request_log_probs is not None:
                    request_log_probs = request_log_probs[:-num_stop_word_trim]
                if top_n_logprobs is not None and req_idx in top_n_logprobs:
                    top_n_logprobs[req_idx] = top_n_logprobs[req_idx][:-num_stop_word_trim]

            # Process log_probs if available (unified for both regular and chunked prefill)
            if request_log_probs is not None:
                # Initialize lists if they don't exist
                if not request.prompt_log_probs:
                    request.prompt_log_probs = []
                if not request.generated_log_probs:
                    request.generated_log_probs = []

                is_chunked_prefill = request_id == self.context.chunked_prefill_request_id
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

    def _find_mamba_match_count(self, req: DynamicInferenceRequest) -> int:
        """Find farthest block with cached Mamba state by iterating from the end.

        Not all blocks have Mamba state cached in mamba_hash_to_block_id,
        only divergence and last-aligned blocks do. Iterating from the end
        finds the farthest block with cached state, which is the only one
        needed for restore since Mamba state is cumulative.
        """
        if not req.precomputed_block_hashes:
            return 0
        mamba_map = self.context.mamba_slot_allocator.hash_to_block_id
        for i in range(len(req.precomputed_block_hashes) - 1, -1, -1):
            if req.precomputed_block_hashes[i] in mamba_map:
                return i + 1
        return 0

    def schedule_waiting_requests(self):
        """Tries to schedule any requests in the waiting pool."""
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

    def schedule_non_chunked_prefill(self):
        """
        Perform the same original scheduling logic for non-chunked runs
        """
        prefix_caching_enabled = self.context.enable_prefix_caching
        mamba_caching_enabled = (
            prefix_caching_enabled
            and self.context.is_hybrid_model
            and self.context.mamba_slot_allocator is not None
        )
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

            # Find Mamba prefix match before check_availability (sets skip count)
            if mamba_caching_enabled:
                req._mamba_num_matched_blocks = self._find_mamba_match_count(req)

            request_can_be_added, request_tokens_can_be_added, kv_cache_available = (
                self.context.check_availability(req)
            )
            if request_can_be_added and request_tokens_can_be_added and kv_cache_available:
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
        mamba_caching_enabled = (
            prefix_caching_enabled
            and self.context.is_hybrid_model
            and self.context.mamba_slot_allocator is not None
        )
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

            # Find Mamba prefix match for non-continuing requests
            if mamba_caching_enabled and not is_continuing_chunked_prefill:
                req._mamba_num_matched_blocks = self._find_mamba_match_count(req)

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
                    # Add these hashes to pending.
                    if prefix_caching_enabled:
                        for block_hash in req.precomputed_block_hashes:
                            if (
                                block_hash
                                not in self.context.kv_block_allocator.kv_hash_to_block_id
                            ):
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
                elif token_partially_can_be_added:
                    # Add these hashes to pending.
                    if prefix_caching_enabled:
                        for block_hash in req.precomputed_block_hashes:
                            if (
                                block_hash
                                not in self.context.kv_block_allocator.kv_hash_to_block_id
                            ):
                                pending_block_hashes.add(block_hash)
                    prefill_chunk_length = self.context.max_tokens - self.context.active_token_count

                    # If this chunk would leave exactly 1 token for the final chunk, reduce
                    # this chunk by 1 or skip scheduling so the final chunk has 2 tokens.
                    # This avoids the edge case where max_seqlen_q=1 which results in a bug
                    # with the Flash Attention kernel.
                    # See https://github.com/Dao-AILab/flash-attention/issues/1537
                    if remaining_len - prefill_chunk_length == 1:
                        if prefill_chunk_length > 1:
                            prefill_chunk_length -= 1
                        else:
                            # We only have space for 1 token, but remaining is 2.
                            # Delay scheduling to avoid leaving exactly 1 token for the final chunk.
                            can_schedule = False
                            break

                    self.context.add_request(req, prefill_chunk_length=prefill_chunk_length)
                    self._loop.call_soon_threadsafe(
                        self._loop.create_task, self._notify_cond_for_new_request()
                    )
                    self.context.chunked_prefill_request_id = req.request_id
                    req.remaining_prompt_tokens = req.remaining_prompt_tokens[prefill_chunk_length:]
                    req.finished_chunk_token_count += prefill_chunk_length
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
        if self.state in (EngineState.SUSPENDED, EngineState.SUSPENDING):
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
            "step_count": self.context.step_count,
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
        self.context.prefix_cache_lru_clock += 1

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
            "total_active_block_count": self.context.kv_block_allocator.active_count,
            "total_paused_block_count": self.context.kv_block_allocator.paused_count,
            "total_active_used_blocks": self.context.kv_block_allocator.get_active_used(),
            "total_paused_used_blocks": self.context.kv_block_allocator.get_paused_used(),
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
            accepted_tokens = step_result["accepted_tokens"]
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
                accepted_tokens,
                log_probs,
                top_n_logprobs,
                routing_indices_per_request,
                pre_fwd_active_token_count=context_state.get("active_token_count"),
                pre_fwd_step_count=context_state.get("step_count"),
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

        range_pop()

        # Detokenize all finished requests if not using
        # the coordinator. Otherwise, the coordinator will
        # overlap detokenization with the engine.
        if not self.use_coordinator:
            range_push("detokenization")
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
            range_pop()

        # Handle necessary ZMQ DP coordinator communication.
        # Failed request replies were already sent in _handle_failed_request,
        # so only send completed records here.
        if self.use_coordinator and self.is_mp_coordinator:
            records_to_send = [
                r for r in finished_request_records if r.requests[-1].status != Status.FAILED
            ]
            if records_to_send:
                range_push("coordinator_communication")
                payload = msgpack.packb(
                    [Headers.ENGINE_REPLY.value, [r.merge().serialize() for r in records_to_send]],
                    use_bin_type=True,
                )
                self.socket_for_receiving_requests.send(payload)
                range_pop()

        # Drain prefix cache hit counters from context into engine accumulators.
        if self.context.enable_prefix_caching:
            self._prefix_cache_hits += self.context.prefix_cache_hits
            self._prefix_cache_blocks_matched += self.context.prefix_cache_blocks_matched
            self.context.prefix_cache_hits = 0
            self.context.prefix_cache_blocks_matched = 0

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

            # Add speculative decoding acceptance metrics.
            if self.num_speculative_tokens > 0 and self._spec_tokens_proposed > 0:
                acceptance_rate = self._spec_tokens_accepted / self._spec_tokens_proposed
                metrics['inference/spec_decode_acceptance_rate'] = float(acceptance_rate * 100.0)
                metrics['inference/spec_decode_tokens_proposed'] = int(self._spec_tokens_proposed)
                metrics['inference/spec_decode_tokens_accepted'] = int(self._spec_tokens_accepted)
                metrics['inference/spec_decode_num_steps'] = int(self._spec_steps)

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

        # Print context state.
        if (
            self.logging_step_interval > 0
            and self.context.step_count % self.logging_step_interval == 0
        ):
            mem = torch.cuda.memory_stats()
            step_type = "decode" if context_state["is_decode_only"] else "non-decode"
            output_str = (
                "* rank %d | step %d | %s ... time: %.3f ms%s ... "
                "reqs: a %d/%d, p %d, w %d, f %d, e %d ... "
                "blocks: a %d/%d, p %d/%d ... "
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
                    context_state["total_active_used_blocks"],
                    context_state["total_active_block_count"],
                    context_state["total_paused_used_blocks"],
                    context_state["total_paused_block_count"],
                    mem["allocation.all.current"],
                    mem["allocated_bytes.all.current"] / (1024**3),
                    mem["reserved_bytes.all.current"] / (1024**3),
                )
            )
            if self.num_speculative_tokens > 0 and self._spec_tokens_proposed > 0:
                spec_rate = self._spec_tokens_accepted / self._spec_tokens_proposed * 100.0
                output_str += " ... spec: accept %.1f%% (%d/%d in %d steps)" % (
                    spec_rate,
                    self._spec_tokens_accepted,
                    self._spec_tokens_proposed,
                    self._spec_steps,
                )
            if self.context.enable_prefix_caching and self._prefix_cache_hits > 0:
                output_str += " ... prefix cache: %d hits, %d blocks matched" % (
                    self._prefix_cache_hits,
                    self._prefix_cache_blocks_matched,
                )
            if context_state["is_decode_only"]:
                output_str = f"\033[94m{output_str}\033[0m"
            logging.info(output_str)

            # Reset speculative decoding accumulators after both wandb and console logging.
            if self.num_speculative_tokens > 0:
                self._spec_tokens_proposed = 0
                self._spec_tokens_accepted = 0
                self._spec_steps = 0

            # Reset prefix caching accumulators after both wandb and console logging.
            if self.context.enable_prefix_caching:
                self._prefix_cache_hits = 0
                self._prefix_cache_blocks_matched = 0

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

        # First pass: add requests.
        # Control signals are queued for the second pass.
        new_generation_epoch = None
        for message in all_messages:
            data = msgpack.unpackb(message, raw=False)
            header = Headers(data[0])
            if header == Headers.SUBMIT_REQUEST:
                request_id, prompt, sampling_params = data[1:]
                sampling_params = SamplingParams.deserialize(sampling_params)
                range_push("add_request")
                self.add_request(request_id, prompt, sampling_params)
                range_pop()
            elif header == Headers.SET_GENERATION_EPOCH:
                new_generation_epoch = data[1]
            else:
                # Control signal: queue for second pass.
                self._pending_signals.append(message)

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
                            )
                        )
                    )
                await self.async_step()
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
        range_push("_ep_establish_consensus")

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

        range_pop()
        return global_work, global_consensus == -1

    async def _world_barrier(self):
        """World-wide ZMQ all-reduce barrier for global rank consensus.

        Used for all state transitions that require global synchronization:
        PAUSING → PAUSED, UNPAUSING → RUNNING, SUSPENDING → SUSPENDED,
        RESUMING → PAUSED, and STOPPING → STOPPED.

        No-op when world_size == 1 (communicator is not created).
        """
        range_push("world_barrier")
        if hasattr(self, 'world_zmq_communicator'):
            await self.world_zmq_communicator.all_reduce_max(
                1, async_op=(not self.use_synchronous_zmq_collectives)
            )
        range_pop()

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
                    local_pending = self.context.get_active_request_count() + len(
                        self.waiting_request_ids
                    )
                    global_work, all_pausing = await self._ep_establish_consensus(
                        local_pending, signal_consensus=(self.state == EngineState.PAUSING)
                    )

                    if all_pausing:
                        # All EP peers are PAUSING: pause immediately.
                        await self._world_barrier()
                        self.state = EngineState.PAUSED
                        self._state_events[EngineState.PAUSED].set()
                    elif global_work > 0:
                        # At least one EP peer has work: all must participate.
                        if local_pending > 0:
                            await self.async_step()
                        else:
                            # Dummy forward to participate in the EP collective.
                            self.step_start_event.record()
                            self.controller.dummy_forward()
                            self.step_end_event.record()
                            self.step_end_event.synchronize()
                            self.context.step_count += 1
                            self.context.prefix_cache_lru_clock += 1
                    else:
                        # No work, but not all pausing: idle.
                        await asyncio.sleep(0.02)

                elif self.state == EngineState.PAUSED:
                    await asyncio.sleep(0.02)

                elif self.state == EngineState.UNPAUSING:
                    await self._world_barrier()
                    self.state = EngineState.RUNNING
                    self._state_events[EngineState.PAUSED].clear()
                    self._state_events[EngineState.RUNNING].set()

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
                        logging.info("Stopping engine.")
                    break

        finally:
            await self.shutdown()
