# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Engine-side lifecycle for disaggregated prefill/decode KV-cache handoff."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict

import torch

from megatron.core.inference.disaggregation.handoff_completion_tracker import (
    HandoffCompletionTracker,
)
from megatron.core.inference.disaggregation.pending_handoff_imports import (
    DeferredKvHandoff,
    PendingKvImport,
)
from megatron.core.inference.disaggregation.transfer_backends.base import (
    construct_kv_transfer_backend_class,
)
from megatron.core.inference.disaggregation.utils import (
    drop_transfer_prefix_blocks,
    transfer_block_count,
)
from megatron.core.inference.inference_request import compute_block_hashes_batched
from megatron.core.utils import get_pg_rank, get_pg_size

if TYPE_CHECKING:
    from megatron.core.inference.inference_request import DynamicInferenceRequest
    from megatron.core.inference.sampling_params import SamplingParams


@dataclass(frozen=True)
class _PreparedHandoffMetadata:
    """Per-request metadata assembled once for a completed prefill batch."""

    local_blocks: list[int]  # Complete blocks owned and pinned by this PP stage.
    kv_meta: Any


class InferenceStateHandoffMixin:
    """Optional KV-cache handoff behavior composed into the dynamic engine."""

    def _initialize_disaggregation_state(self) -> None:
        """Initialize state without importing or constructing a transfer backend."""

        self._pinned_handoff_blocks: Dict[int, list] = {}  # Request ID -> pinned KV block IDs.
        self._kv_transfer_agent = None
        self._kv_peer_metas = None  # KV descriptors for this PP stage's TP ranks.
        self._pp_kv_peer_metas = None  # Each PP stage's set of TP KV descriptors.
        self._deferred_kv_handoffs = deque()
        self._pending_kv_imports = deque()
        self._handoff_completion_tracker: HandoffCompletionTracker | None = None
        self._handoff_completion_notifications: dict[int, bool] = {}  # Request ID -> failed.
        # Retain handoff blocks until the admitted request acquires its own references.
        self._handoff_import_owners: Dict[int, list[int]] = {}  # Request ID -> local KV blocks.
        self._pending_kv_pushes: list = []

    def _setup_handoff_completion_tracking(self, hostname: str | None = None) -> None:
        """Create the CPU path used to aggregate model-parallel transfer completion."""

        self._handoff_completion_tracker = HandoffCompletionTracker(
            self.zmq_context, self.pg_collection.mp, hostname
        )
        self.zmq_sockets.extend(self._handoff_completion_tracker.sockets)

    def _drain_handoff_completion_notifications(self) -> list[tuple[int, bool]]:
        """Collect decisions that the existing MP schedule broadcast must distribute.

        Side: decode MP coordinator; pull and push transport paths.
        """

        if self._handoff_completion_tracker is None:
            return []
        return self._handoff_completion_tracker.drain_completed()

    def _record_handoff_completion_notification(self, request_id: int, failed: bool) -> None:
        """Record the coordinator's shared admission decision on every MP rank.

        Side: decode engine; pull and push transport paths.
        """

        self._handoff_completion_notifications[request_id] = failed

    @property
    def pending_kv_import_count(self) -> int:
        """Number of decode requests waiting for capacity or transfer completion.

        Side: decode engine; pull and push transport paths.
        """

        return len(self._deferred_kv_handoffs) + len(self._pending_kv_imports)

    @property
    def pending_kv_push_count(self) -> int:
        """Number of prefill sends waiting for transport completion.

        Side: prefill engine; push transport path only.
        """

        return len(self._pending_kv_pushes)

    def _reset_pending_kv_imports(self) -> None:
        """Drain and release pending handoff transfers before an engine reset."""

        unsafe_pushes = []
        if not hasattr(self, "_pending_kv_pushes"):
            self._pending_kv_pushes = []
        for request_id, handles in self._pending_kv_pushes:
            if not self._wait_for_transfer_handles(*handles):
                unsafe_pushes.append((request_id, handles))
        self._pending_kv_pushes = unsafe_pushes

        if not hasattr(self, "_pending_kv_imports"):
            self._pending_kv_imports = deque()
        if not hasattr(self, "_deferred_kv_handoffs"):
            self._deferred_kv_handoffs = deque()
        while self._deferred_kv_handoffs:
            handoff = self._deferred_kv_handoffs.popleft()
            if not handoff.future.done():
                handoff.future.cancel()
        unsafe = deque()
        while self._pending_kv_imports:
            pending = self._pending_kv_imports.popleft()
            safe_to_release = pending.destinations_safe and self._wait_for_transfer_handles(
                *self._pending_transfer_handles(pending)
            )
            if safe_to_release:
                self._release_pending_kv_import(pending)
                if not pending.future.done():
                    pending.future.cancel()
            else:
                unsafe.append(pending)
        self._pending_kv_imports = unsafe
        if unsafe_pushes or unsafe:
            raise RuntimeError(
                "Cannot reset while KV handoff transfers may still access cache storage"
            )
        if not hasattr(self, "_handoff_import_owners"):
            self._handoff_import_owners = {}
        for request_id in list(self._handoff_import_owners):
            self._release_handoff_import_owner(request_id)
        self._handoff_completion_notifications.clear()

    def _release_handoff_import_owner(self, request_id: int) -> bool:
        """Release blocks retained until a decode request enters the context.

        Side: decode engine; pull and push transport paths.
        """

        local_blocks = self._handoff_import_owners.pop(request_id, None)
        if not local_blocks:
            return False
        block_tensor = torch.tensor(local_blocks, dtype=torch.int32, device="cpu")
        self.context.kv_block_allocator.release_memory_blocks(block_tensor)
        logging.debug(
            "DISAGG_DECODE_IMPORT_OWNER_RELEASE request_id=%d blocks=%d",
            request_id,
            len(local_blocks),
        )
        return True

    def schedule_waiting_requests(self) -> None:
        """Release imported-block ownership after requests acquire context blocks.

        Side: the ownership-release behavior is decode-only and applies to pull and push paths.
        """

        waiting_before = set(self.waiting_request_ids)
        owned_waiting = {
            request_id: self.get_request(request_id).finished_chunk_token_count
            for request_id in self._handoff_import_owners.keys() & waiting_before
        }
        try:
            super().schedule_waiting_requests()
        finally:
            waiting_after = set(self.waiting_request_ids)
            for request_id, previous_chunk_tokens in owned_waiting.items():
                request_started = request_id not in waiting_after
                if not request_started:
                    request_started = (
                        self.get_request(request_id).finished_chunk_token_count
                        > previous_chunk_tokens
                    )
                if request_started:
                    self._release_handoff_import_owner(request_id)

    def setup_kv_transfer(self, role: str, backend: str = "nixl") -> None:
        """Bring up the KV transfer agents for this engine.

        This method must be called collectively by every model-parallel rank in
        the engine because each rank participates in PP and TP metadata gathers.

        Args:
            role: "prefill" or "decode"; used to name the local transfer agent.
            backend: transfer backend name, resolved through the explicit
                registry ("nixl"; "nccl" selects the two-sided push family).
        """
        if self.context.is_hybrid_model:
            raise RuntimeError(
                "Hybrid models require recurrent-state handoff in addition to KV-cache "
                "handoff; this engine only has KV transfer support."
            )

        backend_cls = construct_kv_transfer_backend_class(backend)

        # Prefill output blocks stay pinned until the peer finishes reading
        # them. Decode requests consume imports but do not produce another
        # handoff, so pinning their completed blocks would retain cache state
        # without a downstream release acknowledgement.
        allocator = self.context.kv_block_allocator
        assert allocator.enable_prefix_caching, (
            "KV handoff requires prefix caching on both prefill and decode "
            "engines (--inference-dynamic-batching-prefix-caching)."
        )
        allocator.enable_handoff_pinning = role == "prefill"

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        # TP topology, so a peer at a different TP can re-shard our KV heads.
        # KV heads under GQA == num_query_groups (falls back to attention heads).
        model_config = self.controller.inference_wrapped_model.model.config
        num_kv_heads_global = model_config.num_query_groups or model_config.num_attention_heads
        tp_size = get_pg_size(self.pg_collection.tp)
        tp_rank = get_pg_rank(self.pg_collection.tp)

        # Compute this PP rank's global attention-layer range.
        pp_size = get_pg_size(self.pg_collection.pp)
        pp_rank = get_pg_rank(self.pg_collection.pp)
        local_num_layers = self.context.num_attention_layers

        if pp_size > 1 and torch.distributed.is_initialized():
            layer_counts: list = [None] * pp_size
            torch.distributed.all_gather_object(
                layer_counts, local_num_layers, group=self.pg_collection.pp
            )
            layer_start = sum(layer_counts[:pp_rank])
            num_layers_global = sum(layer_counts)
        else:
            layer_start = 0
            num_layers_global = local_num_layers
        layer_end = layer_start + local_num_layers

        self._kv_transfer_agent = backend_cls(
            agent_name=f"{role}-rank{rank}",
            memory_buffer=self.context.memory_buffer,
            expected_num_blocks=self.context.kv_block_allocator.pool_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            num_kv_heads_global=num_kv_heads_global,
            heads_per_partition=self.context.num_attention_heads_per_partition,
            head_dim=self.context.hidden_size_per_attention_head,
            tokens_per_block=self.context.block_size_tokens,
            global_rank=rank,
            pp_size=pp_size,
            pp_rank=pp_rank,
            num_layers_global=num_layers_global,
            layer_start=layer_start,
            layer_end=layer_end,
        )

        # Cache static peer metadata for every request.
        self._kv_peer_metas = self._kv_transfer_agent.export_meta()
        if torch.distributed.is_initialized() and tp_size > 1:
            gathered: list = [None] * tp_size
            torch.distributed.all_gather_object(
                gathered, self._kv_peer_metas, group=self.pg_collection.tp
            )
            self._kv_peer_metas = gathered

        # Transport descriptors are static. Gather them across pipeline stages
        # once instead of serializing them with every completed request.
        self._pp_kv_peer_metas = [self._kv_peer_metas]
        if torch.distributed.is_initialized() and pp_size > 1:
            self._pp_kv_peer_metas = [None] * pp_size
            torch.distributed.all_gather_object(
                self._pp_kv_peer_metas, self._kv_peer_metas, group=self.pg_collection.pp
            )

    def push_handoff_kv(self, request_id: int, decode_metas: list) -> None:
        """Push a pinned hand-off's KV to the decode instance described by
        `decode_metas` (two-sided transports only).

        The decode posted its matching receives when SUBMIT_REQUEST_WITH_KV
        arrived; the sends are reaped asynchronously and the pins stay until
        the coordinator's RELEASE_KV.

        Side: prefill engine; push transport path only.
        """
        block_ids = self._pinned_handoff_blocks.get(request_id)
        if not block_ids:
            logging.warning(
                "SEND_KV for request %d with no pinned hand-off blocks; skipping", request_id
            )
            return
        kv_peer = {"tp_metas": list(decode_metas)}
        handles = [self._kv_transfer_agent.begin_push_blocks(kv_peer, block_ids)]
        self._pending_kv_pushes.append((request_id, handles))
        logging.info("DISAGG_PREFILL_PUSH request_id=%d blocks=%d", request_id, len(block_ids))

    def _poll_pending_kv_pushes(self) -> int:
        """Reap completed push sends; unfinished ones stay pending.

        Side: prefill engine; push transport path only.
        """
        if not self._pending_kv_pushes:
            return 0
        remaining = []
        reaped = 0
        for request_id, handles in self._pending_kv_pushes:
            if all(h.poll() for h in handles):
                reaped += 1
            else:
                remaining.append((request_id, handles))
        self._pending_kv_pushes = remaining
        return reaped

    def _prepare_handoff_metadata_batch(
        self, requests_and_blocks: list[tuple["DynamicInferenceRequest", list[int]]]
    ) -> dict[int, _PreparedHandoffMetadata]:
        """Assemble metadata for all handoffs completed by one engine step.

        Side: prefill engine; pull and push transport paths.
        """

        handoffs = [
            (request, list(block_ids))
            for request, block_ids in requests_and_blocks
            if request.sampling_params.do_kv_handoff
        ]
        if not handoffs:
            return {}
        if self._kv_peer_metas is None:
            raise RuntimeError("KV handoff requested before transfer setup")

        pp_size = get_pg_size(self.pg_collection.pp)
        static_pp_metas = self._pp_kv_peer_metas or [self._kv_peer_metas]
        if len(static_pp_metas) != pp_size:
            raise RuntimeError(
                f"Expected static metadata for {pp_size} pipeline stages, "
                f"got {len(static_pp_metas)}"
            )

        prepared = {}
        for request, block_ids in handoffs:
            # Prefix-cache entries must be immutable. The last prompt block is
            # still writable when the prompt is not block-aligned, so transfer
            # only complete blocks and recompute the bounded tail on decode.
            num_complete = len(request.prompt_tokens) // self.context.block_size_tokens
            complete_blocks = block_ids[:num_complete]
            dropped_blocks = block_ids[num_complete:]
            if dropped_blocks:
                self._release_pinned_handoff_blocks(dropped_blocks)

            if pp_size > 1:
                # Dynamic batching is mirrored across model-parallel ranks: they
                # receive the same request stream, use a synchronized KV pool size,
                # and perform allocations, cache registration, and releases in the
                # same order. Physical block IDs are therefore identical on every
                # pipeline stage, just as they are across the TP ranks described by
                # each stage's static metadata.
                kv_meta: Any = {
                    "pp_metas": [
                        {"tp_metas": stage_meta, "block_ids": complete_blocks}
                        for stage_meta in static_pp_metas
                    ]
                }
            else:
                kv_meta = self._kv_peer_metas
            prepared[request.request_id] = _PreparedHandoffMetadata(
                local_blocks=complete_blocks, kv_meta=kv_meta
            )
        return prepared

    def _capture_handoff_meta(
        self,
        request: "DynamicInferenceRequest",
        block_ids: list,
        prepared: _PreparedHandoffMetadata | None = None,
    ) -> None:
        """Attach prepared transfer metadata and retain the request's blocks.

        Side: prefill engine; pull and push transport paths.
        """

        rid = request.request_id
        if prepared is None:
            prepared = self._prepare_handoff_metadata_batch([(request, block_ids)])[rid]
        block_ids = prepared.local_blocks
        kv_meta = prepared.kv_meta

        if not block_ids:
            logging.warning(
                "DISAGG_PREFILL_HANDOFF request_id=%d has no complete prompt blocks; "
                "decode will recompute the prompt tail",
                rid,
            )

        self._pinned_handoff_blocks[rid] = list(block_ids)

        if isinstance(kv_meta, list):
            kv_meta = {"tp_metas": kv_meta}
        else:
            kv_meta = dict(kv_meta)

        request.disaggregated_params = {
            "request_id": rid,
            "block_ids": block_ids,
            "kv_meta": kv_meta,
        }
        logging.info("DISAGG_PREFILL_HANDOFF request_id=%d pinned_blocks=%d", rid, len(block_ids))

    def release_handoff_blocks(self, request_id: int) -> None:
        """Release blocks pinned by a previous do_kv_handoff completion.

        Side: prefill engine; pull and push transport paths.
        """
        block_ids = self._pinned_handoff_blocks.pop(request_id, None)
        if not block_ids:
            return
        released = self._release_pinned_handoff_blocks(block_ids)
        logging.info(
            "DISAGG_PREFILL_RELEASE request_id=%d released_blocks=%d", request_id, released
        )

    def _release_pinned_handoff_blocks(self, block_ids: list) -> int:
        """Release this request's ownership of its pinned handoff blocks.

        Side: prefill engine; pull and push transport paths.
        """
        if not block_ids:
            return 0
        allocator = self.context.kv_block_allocator
        allocator.release_memory_blocks(torch.tensor(block_ids, dtype=torch.int32, device="cpu"))
        return len(block_ids)

    def add_request_with_kv_handoff(
        self,
        request_id: int,
        prompt: list,
        sampling_params: "SamplingParams",
        kv_meta: dict,
        src_block_ids: list,
    ) -> "asyncio.Future[DynamicInferenceRequest]":
        """Start or capacity-queue a KV import and return its completion future.

        Side: decode engine; pull and push transport paths. A pull backend starts
        the read here, while a push backend posts the matching receive.
        """
        allocator = self.context.kv_block_allocator
        if not allocator.enable_prefix_caching:
            raise RuntimeError(
                "add_request_with_kv_handoff requires "
                "--inference-dynamic-batching-prefix-caching on the decode engine; "
                "the prefill-skip path uses the prefix-cache match logic."
            )

        if self._kv_transfer_agent is None:
            raise RuntimeError("KV handoff received without a transfer backend")

        prompt_tensor = torch.tensor(prompt, dtype=torch.int64)
        hashes = compute_block_hashes_batched(prompt_tensor, self.context.block_size_tokens)
        num_blocks = transfer_block_count(kv_meta, src_block_ids)
        future = self._loop.create_future()
        handoff = DeferredKvHandoff(
            request_id=request_id,
            prompt=prompt,
            sampling_params=sampling_params,
            kv_meta=kv_meta,
            src_block_ids=list(src_block_ids),
            hashes=hashes,
            num_blocks=num_blocks,
            future=future,
        )
        if self._deferred_kv_handoffs or not self._try_start_kv_handoff_import(handoff):
            self._deferred_kv_handoffs.append(handoff)
            logging.debug(
                "DISAGG_DECODE_CAPACITY_QUEUE request_id=%d queued=%d",
                request_id,
                len(self._deferred_kv_handoffs),
            )
        return future

    def _try_start_kv_handoff_import(self, handoff: DeferredKvHandoff) -> bool:
        """Start one capacity-safe import, or return false without mutation.

        Side: decode engine; pull and push transport paths.
        """

        allocator = self.context.kv_block_allocator
        cached_blocks = []
        if not self._kv_transfer_agent.is_push:
            cached_blocks = self._find_cached_handoff_prefix(handoff.hashes, handoff.num_blocks)

        num_blocks_to_import = handoff.num_blocks - len(cached_blocks)
        if not self._handoff_capacity_available(num_blocks_to_import, cached_blocks):
            return False

        allocator.retain_memory_blocks(cached_blocks)
        local_blocks_tensor = allocator.allocate_memory_blocks(num_blocks_to_import)
        if local_blocks_tensor is None:
            if cached_blocks:
                allocator.release_memory_blocks(
                    torch.tensor(cached_blocks, dtype=torch.int32, device="cpu")
                )
            raise RuntimeError("KV allocator capacity changed during handoff admission")
        imported_blocks = [int(block) for block in local_blocks_tensor.tolist()]
        local_blocks = cached_blocks + imported_blocks

        handle = None
        start_error = None
        try:
            transfer_meta, transfer_src_blocks = drop_transfer_prefix_blocks(
                handoff.kv_meta, handoff.src_block_ids, len(cached_blocks)
            )

            handle = self._kv_transfer_agent.begin_pull_blocks(
                transfer_meta, transfer_src_blocks, imported_blocks
            )
        except Exception as exc:
            start_error = exc

        pending = PendingKvImport(
            request_id=handoff.request_id,
            prompt=handoff.prompt,
            sampling_params=handoff.sampling_params,
            local_blocks=local_blocks,
            hashes=handoff.hashes,
            cached_prefix_block_count=len(cached_blocks),
            handle=handle,
            future=handoff.future,
            local_error=start_error,
            destinations_safe=(
                start_error is None or getattr(start_error, "transfer_destinations_safe", True)
            ),
        )
        self._pending_kv_imports.append(pending)
        logging.debug(
            "DISAGG_DECODE_PULL_SUBMIT request_id=%d prompt_tokens=%d "
            "cached_blocks=%d imported_blocks=%d pending_imports=%d",
            handoff.request_id,
            len(handoff.prompt),
            len(cached_blocks),
            len(imported_blocks),
            len(self._pending_kv_imports),
        )
        self._loop.call_soon_threadsafe(self._loop.create_task, self._notify_cond_for_new_request())
        return True

    def _find_cached_handoff_prefix(self, hashes: list[int], num_blocks: int) -> list[int]:
        """Find the contiguous handoff prefix already cached on decode.

        Side: decode engine; pull transport path only.
        """

        allocator = self.context.kv_block_allocator
        cached_blocks = []
        for block_hash in hashes[:num_blocks]:
            block_id = allocator.kv_hash_to_block_id.get(block_hash)
            if block_id is None:
                break
            cached_blocks.append(int(block_id))
        return cached_blocks

    def _handoff_capacity_available(self, num_blocks: int, cached_blocks: list[int]) -> bool:
        """Check capacity in the rank-local mirror of model-parallel allocator state.

        Side: decode engine; pull and push transport paths.
        """

        allocator = self.context.kv_block_allocator
        potential_matched_count = 0
        if cached_blocks:
            block_tensor = torch.tensor(cached_blocks, dtype=torch.int32, device="cpu")
            potential_matched_count = int((allocator.block_ref_counts[block_tensor] == 0).sum())
        # Model-parallel ranks process the same request and allocator operations
        # in lockstep, so capacity must remain mirrored without a per-request collective.
        return allocator.is_memory_available(
            num_blocks, potential_matched_count=potential_matched_count
        )

    def _drain_deferred_kv_handoffs(self) -> int:
        """Start queued handoffs in FIFO order while the queue head fits.

        Side: decode engine; pull and push transport paths.
        """

        started = 0
        while self._deferred_kv_handoffs:
            handoff = self._deferred_kv_handoffs[0]
            if not self._try_start_kv_handoff_import(handoff):
                break
            self._deferred_kv_handoffs.popleft()
            started += 1
        return started

    @staticmethod
    def _pending_transfer_handles(pending: PendingKvImport) -> list:
        """Return this decode import's active transfer handles.

        Side: decode engine; pull and push transport paths.
        """

        return [pending.handle] if pending.handle is not None else []

    def _finalize_kv_handoff_import(self, pending: PendingKvImport) -> None:
        """Register transferred blocks and admit the decode request.

        Side: decode engine; pull and push transport paths.
        """

        allocator = self.context.kv_block_allocator
        local_blocks = pending.local_blocks
        cached_prefix_block_count = pending.cached_prefix_block_count
        registration_end = min(len(local_blocks), len(pending.hashes))
        num_hashes_to_register = registration_end - cached_prefix_block_count

        if pending.request_id in self._handoff_import_owners:
            raise RuntimeError(f"Duplicate decode handoff request ID {pending.request_id}")

        if num_hashes_to_register > 0:
            # The imported suffix extends any retained local prefix. Preserve
            # that predecessor link in the allocator's parent-aware LRU forest.
            parent_hashes = [
                pending.hashes[block_idx - 1] if block_idx > 0 else 0
                for block_idx in range(cached_prefix_block_count, registration_end)
            ]
            allocator.register_kv_block_hashes(
                local_blocks[cached_prefix_block_count:registration_end],
                pending.hashes[cached_prefix_block_count:registration_end],
                parent_hashes=parent_hashes,
            )

        logging.debug(
            "DISAGG_DECODE_IMPORT request_id=%d prompt_tokens=%d "
            "cached_blocks=%d imported_blocks=%d hashes_registered=%d pending_imports=%d",
            pending.request_id,
            len(pending.prompt),
            cached_prefix_block_count,
            len(local_blocks) - cached_prefix_block_count,
            num_hashes_to_register,
            len(self._pending_kv_imports),
        )

        self._handoff_import_owners[pending.request_id] = list(local_blocks)
        pending.local_blocks = []
        try:
            request_future = self.add_request(
                pending.request_id,
                pending.prompt,
                pending.sampling_params,
                precomputed_block_hashes=(
                    pending.hashes[:registration_end] if registration_end > 0 else None
                ),
            )
        except Exception:
            self._release_handoff_import_owner(pending.request_id)
            raise
        if pending.request_id not in self.waiting_request_ids:
            self._release_handoff_import_owner(pending.request_id)

        def _relay_result(src: asyncio.Future) -> None:
            """Release import ownership and forward decode completion to the handoff future."""

            self._release_handoff_import_owner(pending.request_id)
            if pending.future.done():
                return
            if src.cancelled():
                pending.future.cancel()
                return
            exc = src.exception()
            if exc is not None:
                pending.future.set_exception(exc)
            else:
                pending.future.set_result(src.result())

        request_future.add_done_callback(_relay_result)

    def _release_pending_kv_import(self, pending: PendingKvImport) -> None:
        """Release storage owned by an unadmitted decode import.

        Side: decode engine; pull and push transport paths.
        """

        owner_released = self._release_handoff_import_owner(pending.request_id)
        if pending.local_blocks and not owner_released:
            block_tensor = torch.tensor(pending.local_blocks, dtype=torch.int32, device="cpu")
            self.context.kv_block_allocator.release_memory_blocks(block_tensor)

    @staticmethod
    def _wait_for_transfer_handles(*handles) -> bool:
        """Wait for known handles; return false if any may still be active."""

        safe_to_release = True
        for handle in handles:
            if handle is None:
                continue
            try:
                handle.wait()
            except TimeoutError:
                safe_to_release = False
            except Exception:
                # NIXL reports transfer errors only after all segments belonging
                # to the handle have reached a terminal state.
                pass
        return safe_to_release

    def _report_completed_kv_imports(self) -> None:
        """Report locally terminal imports without synchronizing the compute ranks.

        Side: decode engine; pull and push transport paths.
        """

        for pending in self._pending_kv_imports:
            request_id = pending.request_id
            if pending.terminal_state_reported:
                continue
            failed = pending.local_error is not None
            if not failed:
                try:
                    if not all(handle.poll() for handle in self._pending_transfer_handles(pending)):
                        continue
                except Exception as exc:
                    pending.local_error = exc
                    failed = True

            if (
                self._handoff_completion_tracker is None
                or self._handoff_completion_tracker.world_size == 1
            ):
                if get_pg_size(self.pg_collection.mp) != 1:
                    raise RuntimeError(
                        "Model-parallel KV handoff requires coordinator completion tracking"
                    )
                self._handoff_completion_notifications[request_id] = failed
            else:
                self._handoff_completion_tracker.report(request_id, failed)
            pending.terminal_state_reported = True

    def _poll_pending_kv_imports(self) -> int:
        """Finalize decode imports after all model-parallel ranks report completion.

        Side: decode engine; pull and push transport paths.
        """

        self._drain_deferred_kv_handoffs()
        if not self._pending_kv_imports:
            return 0
        self._report_completed_kv_imports()
        ready = 0
        remaining = deque()
        while self._pending_kv_imports:
            pending = self._pending_kv_imports.popleft()
            failed = self._handoff_completion_notifications.pop(pending.request_id, None)
            if failed is None:
                remaining.append(pending)
                continue
            try:
                if failed:
                    raise pending.local_error or RuntimeError(
                        "KV handoff transfer failed on a model-parallel peer"
                    )
                self._finalize_kv_handoff_import(pending)
                ready += 1
            except Exception as exc:
                # A peer can fail before posting its half of a two-sided transfer.
                # Do not wait on this rank's unmatched operation; quarantine its
                # destination unless its own handle already reached a terminal state.
                safe_to_release = pending.destinations_safe and pending.terminal_state_reported
                safe_to_release = safe_to_release and self._wait_for_transfer_handles(
                    *self._pending_transfer_handles(pending)
                )
                if safe_to_release:
                    self._release_pending_kv_import(pending)
                else:
                    remaining.append(pending)
                    logging.error(
                        "Quarantining request %d cache storage after an incomplete handoff",
                        pending.request_id,
                    )
                if not pending.future.done():
                    pending.future.set_exception(exc)
                logging.exception("DISAGG_DECODE_PULL_FAILED request_id=%d", pending.request_id)
                remaining.extend(self._pending_kv_imports)
                self._pending_kv_imports = remaining
                raise
        self._pending_kv_imports = remaining
        if ready:
            self._loop.call_soon_threadsafe(
                self._loop.create_task, self._notify_cond_for_new_request()
            )
        return ready
