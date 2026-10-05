# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import concurrent.futures
import logging
import warnings
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, TypedDict

import torch

from megatron.core.inference.config import AsyncScheduleMode
from megatron.core.inference.inference_request import DynamicInferenceRequest, Status
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    DecodeOnly,
    DynamicBatchControllerStepResult,
)
from megatron.core.utils import (
    get_asyncio_loop,
    nvtx_range_pop,
    nvtx_range_push,
    trace_async_exceptions,
)

try:
    import wandb  # pylint: disable=unused-import

    HAVE_WANDB = True
except ImportError:
    HAVE_WANDB = False
    wandb = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


class DynamicInferenceEngineStepResult(TypedDict):
    """Result returned by modern dynamic-engine step APIs."""

    active_request_ids: list[int]
    finished_requests: list[DynamicInferenceRequest]
    step_time: float
    cuda_graph_request_count: int | None


# pylint: disable=line-too-long
class EngineLoopMixin:
    """Step loop, shutdown, and run loops for `DynamicInferenceEngine`."""

    # Map stable states to their corresponding asyncio events.
    _STATE_EVENTS = (
        EngineState.RUNNING,
        EngineState.PAUSED,
        EngineState.SUSPENDED,
        EngineState.RESUMED,
        EngineState.STOPPED,
    )

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

    @trace_async_exceptions
    async def _notify_cond_for_new_request(self):
        """Helper function to notify condition variable when a new request is added."""
        async with self._cond:
            self._cond.notify_all()

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

    async def async_bookkeep(
        self, step_result: Optional[Dict], context_state: Dict, step_time: float
    ) -> DynamicInferenceEngineStepResult:
        """Uses `asyncio` for continuous bookkeeping.

        Args:
            step_result (Optional[Dict]): The result of the step.
            context_state (Dict): Decode-only state, total/paused request count,
                and active token count.
            step_time (float): How long this step took.

        Returns:
            A dictionary containing:
                active_request_ids (List): IDs that ran in the last step and remain active.
                finished_requests (List): Flat, text-unfinalized requests that finished.
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

            # Process finished requests after applying all final-step metadata.
            active_request_ids, finished_requests = self.post_process_requests(
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
            finished_requests: list[DynamicInferenceRequest] = []

        # Failed requests. Take the current queue snapshot so a later admission
        # cannot be erased by this bookkeeping pass.
        finished_requests.extend(self._collect_failed_requests())

        nvtx_range_pop("bookkeeping")

        # Handle necessary ZMQ DP coordinator communication.
        # Failed request replies were already sent in _handle_failed_request.
        if self.use_coordinator and self.is_mp_coordinator:
            requests_to_send = [
                request for request in finished_requests if request.status != Status.FAILED
            ]
            if requests_to_send:
                nvtx_range_push("coordinator_communication")
                self._send_requests_to_coordinator(requests_to_send)
                nvtx_range_pop("coordinator_communication")

            # Stream newly generated tokens for active requests. Finished
            # requests were already popped from self.requests above, so their
            # emit lengths are dropped here rather than in the loop.
            for request in finished_requests:
                self._partial_emit_lengths.pop(request.request_id, None)
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
            "finished_requests": finished_requests,
            "step_time": step_time,
            "cuda_graph_request_count": cuda_graph_request_count,
        }

    async def async_step(self) -> DynamicInferenceEngineStepResult:
        """
        Wrapper for controller.generate_output_tokens_dynamic_batch(), to
        match vLLM API. Uses `asyncio` for continuous generation which allows this
        method to sleep and wake up when new requests are available.

        Returns:
            Active request IDs, finished requests, and step metadata.
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

    def step_modern(self) -> DynamicInferenceEngineStepResult:
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
        return active_requests, result["finished_requests"], result["step_time"]

    # For backwards compatibility, point `step()` to `step_legacy()`. Starting in
    # `megatron-core` 0.16, `step_modern()` will be renamed to `step()`.
    step = step_legacy

    def generate(
        self, prompts: List[str], sampling_params: Optional[SamplingParams] = SamplingParams()
    ) -> List[DynamicInferenceRequest]:
        """Generate token-complete, text-unfinalized requests for a prompt batch."""

        request_futures = []
        submitted_request_ids = set()
        for prompt in prompts:
            request_id = int(next(self.request_counter))
            submitted_request_ids.add(request_id)
            request_futures.append(self.add_request(request_id, prompt, sampling_params))

        # Admission failures resolve their futures synchronously. Remove only
        # this call's failed entries, and do not run an empty model step solely
        # to make bookkeeping observe them.
        self._collect_failed_requests(submitted_request_ids)
        while any(not future.done() for future in request_futures):
            self.step_modern()

        finished_requests = [future.result() for future in request_futures]

        # Ensure requests are returned in the same order they were passed in.
        finished_requests.sort(key=lambda request: request.request_id)

        return finished_requests

    async def shutdown(self):
        """Shut down the engine and clean up ZMQ resources.

        Called from the engine loop's finally block after the loop exits.
        """
        self.state = EngineState.STOPPED

        # Cleanup the request futures.
        for entry in self.requests.values():
            if not entry.future.done():
                entry.future.cancel()

        await self._close_coordinator_links()

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
