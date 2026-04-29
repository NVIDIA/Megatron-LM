# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Ordered dynamic-step retirement for async-overlap inference."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Deque, Dict, Optional

import torch

from megatron.core.inference.headers import Headers
from megatron.core.inference.inference_request import DynamicInferenceRequestRecord, Status
from megatron.core.utils import nvtx_range_pop, nvtx_range_push

if TYPE_CHECKING:
    from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine


@dataclass(frozen=True)
class StepRetirementWorkItem:
    """One ordered step waiting for public output retirement."""

    step_result: Optional[Dict]
    context_state: Dict
    step_time: float


try:
    import msgpack

    HAVE_MSGPACK = True
except ImportError:
    HAVE_MSGPACK = False
    msgpack = None

try:
    import wandb  # pylint: disable=unused-import

    HAVE_WANDB = True
except ImportError:
    HAVE_WANDB = False


class StepRetirementService:
    """Retire dynamic steps in order.

    Queue depth one submits and drains synchronously through DynamicAsyncPipeline.
    Later queue-depth commits can enqueue outputs here and drain them independently
    while preserving step order.
    """

    def __init__(self, engine: "DynamicInferenceEngine"):
        self.engine = engine
        self._last_retired_step_id = -1
        self._pending_retirements: Deque[StepRetirementWorkItem] = deque()
        self.max_retirement_backlog = max(1, int(getattr(engine, "async_overlap_queue_depth", 1)))
        self._update_backlog_counters()

    @property
    def pending_count(self) -> int:
        """Number of step outputs waiting for ordered retirement."""
        return len(self._pending_retirements)

    def enqueue_step(
        self, step_result: Optional[Dict], context_state: Dict, step_time: float
    ) -> None:
        """Queue one step for ordered retirement, enforcing the backlog limit."""
        if self.pending_count >= self.max_retirement_backlog:
            raise RuntimeError(
                "Step retirement backlog is full: "
                f"{self.pending_count}/{self.max_retirement_backlog}"
            )
        self._pending_retirements.append(
            StepRetirementWorkItem(
                step_result=step_result,
                context_state=context_state,
                step_time=step_time,
            )
        )
        self._update_backlog_counters()

    def drain_next(self) -> Optional[Dict]:
        """Retire the next queued step, if any."""
        if not self._pending_retirements:
            return None
        item = self._pending_retirements.popleft()
        self._update_backlog_counters()
        return self._retire_step_now(item.step_result, item.context_state, item.step_time)

    def drain_pending(self) -> None:
        """Synchronously retire every queued output in order."""
        while self._pending_retirements:
            self.drain_next()

    def submit_step(
        self, step_result: Optional[Dict], context_state: Dict, step_time: float
    ) -> Dict:
        """Queue and immediately retire one step in queue-depth-one mode."""
        self.enqueue_step(step_result, context_state, step_time)
        result = self.drain_next()
        assert result is not None
        return result

    def drain_for_shutdown(self) -> None:
        """Drain in-flight retirement work before shutdown.

        Queue depth one has no detached retirement work, but an interrupted step
        can leave an open journal entry that must not survive shutdown.
        """
        self.drain_pending()
        self.engine.context.rollback_all_open_step_journals(reason="shutdown_drain")

    def drain_for_suspend(self) -> None:
        """Drain in-flight retirement work before suspend.

        Queue depth one has no detached retirement work, but an interrupted step
        can leave an open journal entry that must not survive suspend.
        """
        self.drain_pending()
        self.engine.context.rollback_all_open_step_journals(reason="suspend_drain")

    def drain_for_request_reuse(self, request_id: int) -> None:
        """Drain work that could still reference a request ID before reuse.

        Queue depth one retires each step before the next request can reuse an ID.
        """
        del request_id
        self.drain_pending()

    def retire_step(
        self, step_result: Optional[Dict], context_state: Dict, step_time: float
    ) -> Dict:
        """Compatibility wrapper for queueing and retiring one dynamic step."""
        return self.submit_step(step_result, context_state, step_time)

    def _retire_step_now(
        self, step_result: Optional[Dict], context_state: Dict, step_time: float
    ) -> Dict:
        """Retire one dynamic step and return the engine-visible bookkeeping result."""
        engine = self.engine
        step_id = context_state.get("dynamic_step_id", engine._current_dynamic_step_id)
        if step_id <= self._last_retired_step_id:
            raise RuntimeError(
                f"Dynamic step {step_id} retired after {self._last_retired_step_id}; "
                "retirement must remain ordered"
            )
        self._last_retired_step_id = step_id

        output_retirement_range = engine._step_nvtx_label("output_retirement", step_id)
        nvtx_range_push(output_retirement_range)
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
            if newly_paused_request_ids is not None and engine.track_paused_request_events:
                newly_paused_request_ids = newly_paused_request_ids.tolist()
                [engine.get_request(i).add_event_pause() for i in newly_paused_request_ids]

            # Process finished requests (adds FINISH events and returns records).
            post_process_range = engine._step_nvtx_label("post_process_requests", step_id)
            nvtx_range_push(post_process_range)
            (active_request_ids, finished_request_records) = engine.post_process_requests(
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
            nvtx_range_pop(post_process_range)

        else:
            active_request_ids: list[int] = []
            finished_request_records: list[DynamicInferenceRequestRecord] = []

        # Failed requests. Status and events were already set in _handle_failed_request;
        # here we just clean up the entry and include it in finished_request_records.
        for failed_request_id in engine.failed_request_ids:
            failed_entry = engine.requests.pop(failed_request_id)
            finished_request_records.append(failed_entry.record)
            assert (
                failed_entry.future.done()
            ), f"Failed request {failed_request_id} future has not been properly resolved."
        engine.failed_request_ids.clear()

        nvtx_range_pop(output_retirement_range)
        engine.context.commit_step_journal(step_id)

        # Detokenize all finished requests if not using the coordinator. Otherwise, the
        # coordinator overlaps detokenization with the engine.
        if not engine.use_coordinator:
            detokenization_range = engine._step_nvtx_label("detokenization", step_id)
            nvtx_range_push(detokenization_range)
            for record in finished_request_records:
                for request in record.requests:
                    if request.prompt is None:
                        request.prompt = engine.controller.detokenize(
                            engine.controller.tokenizer,
                            request.prompt_tokens.tolist(),
                            remove_EOD=False,
                        )
                    request.generated_text = engine.controller.detokenize(
                        engine.controller.tokenizer,
                        request.generated_tokens,
                        remove_EOD=not request.sampling_params.detokenize_stop_sequence,
                    )
            nvtx_range_pop(detokenization_range)

        # Handle necessary ZMQ DP coordinator communication. Failed request replies were
        # already sent in _handle_failed_request, so only send completed records here.
        if engine.use_coordinator and engine.is_mp_coordinator:
            records_to_send = [
                r for r in finished_request_records if r.requests[-1].status != Status.FAILED
            ]
            if records_to_send:
                if not HAVE_MSGPACK:
                    raise ImportError("msgpack is required to send coordinator replies")
                coordinator_range = engine._step_nvtx_label("coordinator_send", step_id)
                nvtx_range_push(coordinator_range)
                payload = msgpack.packb(
                    [
                        Headers.ENGINE_REPLY.value,
                        [r.merge().serialize() for r in records_to_send],
                    ],
                    use_bin_type=True,
                )
                engine.socket_for_receiving_requests.send(payload)
                nvtx_range_pop(coordinator_range)

        self._drain_prefix_cache_counters()
        self._write_metrics(context_state, step_time)
        self._log_step(context_state, step_time)

        return {
            "active_request_ids": active_request_ids,
            "finished_request_records": finished_request_records,
            "step_time": step_time,
            "cuda_graph_request_count": cuda_graph_request_count,
        }

    def _update_backlog_counters(self) -> None:
        counters = getattr(self.engine, "async_overlap_debug_counters", None)
        if counters is None:
            return
        setattr(counters, "retirement_backlog", self.pending_count)
        setattr(counters, "max_retirement_backlog", self.max_retirement_backlog)

    def _drain_prefix_cache_counters(self) -> None:
        engine = self.engine
        if engine.context.enable_prefix_caching:
            engine._prefix_cache_hits += engine.context.prefix_cache_hits
            engine._prefix_cache_blocks_matched += engine.context.prefix_cache_blocks_matched
            engine.context.prefix_cache_hits = 0
            engine.context.prefix_cache_blocks_matched = 0

    def _write_metrics(self, context_state: Dict, step_time: float) -> None:
        engine = self.engine
        if context_state["kv_stats"] is None:
            return

        metrics = {
            'inference/inference_step': int(
                engine.inference_step_offset + int(engine.context.step_count)
            ),
            'inference/step_time_s': float(step_time),
            'inference/waiting_queue_len': int(len(engine.waiting_request_ids)),
            'inference/total_requests_dict_size': int(len(engine.requests)),
        }
        for key, value in context_state["kv_stats"].items():
            if 'utilization' in key:
                metrics[f'inference/{key}'] = float(value * 100.0)
            else:
                metrics[f'inference/{key}'] = value

        if engine.num_speculative_tokens > 0 and engine._spec_tokens_proposed > 0:
            acceptance_rate = engine._spec_tokens_accepted / engine._spec_tokens_proposed
            metrics['inference/spec_decode_acceptance_rate'] = float(acceptance_rate * 100.0)
            metrics['inference/spec_decode_tokens_proposed'] = int(engine._spec_tokens_proposed)
            metrics['inference/spec_decode_tokens_accepted'] = int(engine._spec_tokens_accepted)
            metrics['inference/spec_decode_num_steps'] = int(engine._spec_steps)

        if engine.context.enable_prefix_caching and engine._prefix_cache_hits > 0:
            metrics['inference/prefix_cache_hits'] = int(engine._prefix_cache_hits)
            metrics['inference/prefix_cache_blocks_matched'] = int(
                engine._prefix_cache_blocks_matched
            )

        if HAVE_WANDB and engine.metrics_writer.__name__ == "wandb":
            engine.metrics_writer.log(metrics, commit=True)
        else:
            raise ValueError(f"Unsupported metrics writer type: {type(engine.metrics_writer)}")

    def _log_step(self, context_state: Dict, step_time: float) -> None:
        engine = self.engine
        if not (
            engine.logging_step_interval > 0
            and engine.context.step_count % engine.logging_step_interval == 0
        ):
            return

        mem = torch.cuda.memory_stats()
        step_type = "decode" if context_state["is_decode_only"] else "non-decode"
        output_str = (
            "* rank %d | step %d | %s ... time: %.3f ms%s ... "
            "reqs: a %d/%d, p %d, w %d, f %d, e %d ... "
            "blocks: a %d/%d, p %d/%d ... "
            "mem: tensors %d, alloc %.1f gb, res %.1f gb."
            % (
                engine.rank,
                engine.context.step_count,
                datetime.now().strftime("%H:%M:%S"),
                step_time * 1000,
                (
                    " [%s + real config %s + cuda graph %s]"
                    % (
                        step_type,
                        engine.context.batch_dimensions,
                        (
                            "OFF"
                            if not engine.context.using_cuda_graph_this_step()
                            else engine.context.padded_batch_dimensions
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
        if engine.num_speculative_tokens > 0 and engine._spec_tokens_proposed > 0:
            spec_rate = engine._spec_tokens_accepted / engine._spec_tokens_proposed * 100.0
            output_str += " ... spec: accept %.1f%% (%d/%d in %d steps)" % (
                spec_rate,
                engine._spec_tokens_accepted,
                engine._spec_tokens_proposed,
                engine._spec_steps,
            )
        if engine.context.enable_prefix_caching and engine._prefix_cache_hits > 0:
            output_str += " ... prefix cache: %d hits, %d blocks matched" % (
                engine._prefix_cache_hits,
                engine._prefix_cache_blocks_matched,
            )
        if context_state["is_decode_only"]:
            output_str = f"\033[94m{output_str}\033[0m"
        logging.info(output_str)

        # Reset speculative decoding accumulators after both wandb and console logging.
        if engine.num_speculative_tokens > 0:
            engine._spec_tokens_proposed = 0
            engine._spec_tokens_accepted = 0
            engine._spec_steps = 0

        # Reset prefix caching accumulators after both wandb and console logging.
        if engine.context.enable_prefix_caching:
            engine._prefix_cache_hits = 0
            engine._prefix_cache_blocks_matched = 0
