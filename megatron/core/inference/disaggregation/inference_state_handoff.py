# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Engine-side lifecycle for disaggregated prefill/decode KV-cache handoff."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import TYPE_CHECKING, Any, Dict

import torch

from megatron.core.inference.disaggregation.pending_handoff_imports import PendingKvImport
from megatron.core.inference.disaggregation.transfer_backends.base import (
    construct_kv_transfer_backend_class,
)
from megatron.core.inference.disaggregation.utils import (
    drop_transfer_prefix_blocks,
    transfer_block_count,
)
from megatron.core.utils import get_pg_rank, get_pg_size

if TYPE_CHECKING:
    from megatron.core.inference.inference_request import DynamicInferenceRequest
    from megatron.core.inference.sampling_params import SamplingParams


class InferenceStateHandoffMixin:
    """Optional KV-cache handoff behavior composed into the dynamic engine."""

    def _initialize_disaggregation_state(self) -> None:
        """Initialize state without importing or constructing a transfer backend."""

        self._pinned_handoff_blocks: Dict[int, list] = {}
        self._kv_transfer_agent = None
        self._kv_peer_metas = None
        self._pending_kv_imports = deque()
        self._handoff_import_owners: Dict[int, list[int]] = {}
        self._pending_kv_pushes: list = []

    @property
    def pending_kv_import_count(self) -> int:
        """Number of decode requests waiting for capacity or transfer completion."""

        return len(self._pending_kv_imports)

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
        unsafe = deque()
        while self._pending_kv_imports:
            pending = self._pending_kv_imports.popleft()
            if self._wait_for_transfer_handles(*self._pending_transfer_handles(pending)):
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

    def _release_handoff_import_owner(self, request_id: int) -> bool:
        """Release blocks retained until a decode request enters the context."""

        owners = getattr(self, "_handoff_import_owners", None)
        local_blocks = owners.pop(request_id, None) if owners is not None else None
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
        """Release imported-block ownership after requests acquire context blocks."""

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

        Args:
            role: "prefill" or "decode"; used to name the local transfer agent.
            backend: transfer backend name, resolved through the explicit
                registry ("nixl"; "nccl" selects the two-sided push family).
        """
        if getattr(self.context, "is_hybrid_model", False):
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

    def push_handoff_kv(self, request_id: int, decode_metas: list) -> None:
        """Push a pinned hand-off's KV to the decode instance described by
        `decode_metas` (two-sided transports only).

        The decode posted its matching receives when SUBMIT_REQUEST_WITH_KV
        arrived; the sends are reaped asynchronously and the pins stay until
        the coordinator's RELEASE_KV."""
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
        """Reap completed push sends; unfinished ones stay pending."""
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

    def _capture_handoff_meta(self, request: "DynamicInferenceRequest", block_ids: list) -> None:
        """Attach transfer metadata and retain the request's pinned blocks."""
        rid = request.request_id
        if not block_ids:
            logging.warning(
                "DISAGG_PREFILL_HANDOFF request_id=%d had no snapshot blocks "
                "(controller missed the slot?); decode peer will receive empty handoff",
                rid,
            )
            return

        self._pinned_handoff_blocks[rid] = list(block_ids)

        if self._kv_peer_metas is None:
            raise RuntimeError("KV handoff requested before transfer setup")
        local_kv: Any = self._kv_peer_metas

        pp_size = get_pg_size(self.pg_collection.pp)
        if pp_size > 1 and torch.distributed.is_initialized():
            local_entry = {"kv_meta": local_kv, "block_ids": list(block_ids)}
            gathered: list = [None] * pp_size
            torch.distributed.all_gather_object(gathered, local_entry, group=self.pg_collection.pp)
            kv_meta: Any = {
                "pp_metas": [
                    {"tp_metas": e["kv_meta"], "block_ids": e["block_ids"]} for e in gathered
                ]
            }
            top_block_ids: Any = gathered[0]["block_ids"]
        else:
            kv_meta = local_kv
            top_block_ids = block_ids

        if isinstance(kv_meta, list):
            kv_meta = {"tp_metas": kv_meta}
        else:
            kv_meta = dict(kv_meta)

        request.disaggregated_params = {
            "request_id": rid,
            "block_ids": top_block_ids,
            "kv_meta": kv_meta,
        }
        logging.info("DISAGG_PREFILL_HANDOFF request_id=%d pinned_blocks=%d", rid, len(block_ids))

    def release_handoff_blocks(self, request_id: int) -> None:
        """Release blocks pinned by a previous do_kv_handoff completion."""
        block_ids = self._pinned_handoff_blocks.pop(request_id, None)
        if not block_ids:
            return
        released = self._release_pinned_handoff_blocks(block_ids)
        logging.info(
            "DISAGG_PREFILL_RELEASE request_id=%d released_blocks=%d", request_id, released
        )

    def _release_pinned_handoff_blocks(self, block_ids: list) -> int:
        """Release this request's ownership of its pinned handoff blocks."""
        allocator = self.context.kv_block_allocator
        return allocator.release_pinned_memory_blocks(block_ids)

    def add_request_with_kv_handoff(
        self,
        request_id: int,
        prompt: list,
        sampling_params: "SamplingParams",
        kv_meta: dict,
        src_block_ids: list,
    ) -> "asyncio.Future[DynamicInferenceRequest]":
        """Start a KV-cache pull and return the request completion future."""
        from megatron.core.inference.inference_request import compute_block_hashes_batched

        allocator = self.context.kv_block_allocator
        if not allocator.enable_prefix_caching:
            raise RuntimeError(
                "add_request_with_kv_handoff requires --enable-prefix-caching on the "
                "decode engine; the prefill-skip path uses the prefix-cache match logic."
            )

        if self._kv_transfer_agent is None:
            raise RuntimeError("KV handoff received without a transfer backend")

        prompt_tensor = torch.tensor(prompt, dtype=torch.int64)
        hashes = compute_block_hashes_batched(
            prompt_tensor, self.context.block_size_tokens, include_partial=True
        )
        num_blocks = transfer_block_count(kv_meta, src_block_ids)
        future = self._loop.create_future()
        self._start_kv_handoff_import(
            request_id=request_id,
            prompt=prompt,
            sampling_params=sampling_params,
            kv_meta=kv_meta,
            src_block_ids=src_block_ids,
            hashes=hashes,
            num_blocks=num_blocks,
            future=future,
        )
        return future

    def _start_kv_handoff_import(
        self,
        request_id: int,
        prompt: list,
        sampling_params: "SamplingParams",
        kv_meta: dict,
        src_block_ids: list,
        hashes: list[int],
        num_blocks: int,
        future: asyncio.Future,
    ) -> None:
        """Reserve destination blocks and start one KV-cache transfer."""

        allocator = self.context.kv_block_allocator
        cached_blocks = []
        if not getattr(self._kv_transfer_agent, "is_push", False):
            cached_blocks = self._retain_cached_handoff_prefix(hashes, num_blocks)

        num_blocks_to_import = num_blocks - len(cached_blocks)
        local_blocks_tensor = allocator.allocate_memory_blocks(num_blocks_to_import)
        if local_blocks_tensor is None:
            if cached_blocks:
                allocator.release_memory_blocks(
                    torch.tensor(cached_blocks, dtype=torch.int32, device="cpu")
                )
            raise RuntimeError(
                f"add_request_with_kv_handoff: OOM allocating {num_blocks_to_import} blocks"
            )
        imported_blocks = [int(block) for block in local_blocks_tensor.tolist()]
        local_blocks = cached_blocks + imported_blocks
        owned_blocks_tensor = torch.tensor(local_blocks, dtype=torch.int32, device="cpu")

        handle = None
        try:
            transfer_meta, transfer_src_blocks = drop_transfer_prefix_blocks(
                kv_meta, src_block_ids, len(cached_blocks)
            )

            handle = self._kv_transfer_agent.begin_pull_blocks(
                transfer_meta, transfer_src_blocks, imported_blocks
            )
        except Exception as exc:
            safe_to_release = getattr(exc, "transfer_destinations_safe", True)
            safe_to_release &= self._wait_for_transfer_handles(handle)
            if safe_to_release:
                allocator.release_memory_blocks(owned_blocks_tensor)
            else:
                logging.error(
                    "Quarantining KV blocks after a timed-out handoff submission: %s", local_blocks
                )
            raise

        pending = PendingKvImport(
            request_id=request_id,
            prompt=prompt,
            sampling_params=sampling_params,
            local_blocks=local_blocks,
            hashes=hashes,
            hashes_to_register=max(0, min(num_blocks, len(hashes)) - len(cached_blocks)),
            hash_registration_start=len(cached_blocks),
            handle=handle,
            future=future,
        )
        self._pending_kv_imports.append(pending)
        logging.debug(
            "DISAGG_DECODE_PULL_SUBMIT request_id=%d prompt_tokens=%d "
            "cached_blocks=%d imported_blocks=%d pending_imports=%d",
            request_id,
            len(prompt),
            len(cached_blocks),
            len(imported_blocks),
            len(self._pending_kv_imports),
        )
        self._loop.call_soon_threadsafe(self._loop.create_task, self._notify_cond_for_new_request())

    def _retain_cached_handoff_prefix(self, hashes: list[int], num_blocks: int) -> list[int]:
        """Retain the contiguous handoff prefix already cached on decode."""

        allocator = self.context.kv_block_allocator
        cached_blocks = []
        for block_hash in hashes[:num_blocks]:
            block_id = allocator.kv_hash_to_block_id.get(block_hash)
            if block_id is None:
                break
            cached_blocks.append(int(block_id))

        if cached_blocks:
            block_tensor = torch.tensor(cached_blocks, dtype=torch.int32, device="cpu")
            allocator.block_ref_counts[block_tensor] += 1
            allocator.update_timestamps(block_tensor)
        return cached_blocks

    @staticmethod
    def _pending_transfer_handles(pending: PendingKvImport) -> list:
        return [pending.handle] if pending.handle is not None else []

    def _finalize_kv_handoff_import(self, pending: PendingKvImport) -> None:
        allocator = self.context.kv_block_allocator
        n = pending.hashes_to_register
        start = pending.hash_registration_start
        end = start + n
        local_blocks = pending.local_blocks

        if pending.request_id in self._handoff_import_owners:
            raise RuntimeError(f"Duplicate decode handoff request ID {pending.request_id}")

        if n > 0:
            # The imported suffix extends any retained local prefix. Preserve
            # that predecessor link in the allocator's parent-aware LRU forest.
            parent_hashes = [
                pending.hashes[block_idx - 1] if block_idx > 0 else 0
                for block_idx in range(start, end)
            ]
            allocator.register_kv_block_hashes(
                local_blocks[start:end], pending.hashes[start:end], parent_hashes=parent_hashes
            )

        logging.debug(
            "DISAGG_DECODE_IMPORT request_id=%d prompt_tokens=%d "
            "cached_blocks=%d imported_blocks=%d hashes_registered=%d pending_imports=%d",
            pending.request_id,
            len(pending.prompt),
            start,
            len(local_blocks) - start,
            n,
            len(self._pending_kv_imports),
        )

        self._handoff_import_owners[pending.request_id] = list(local_blocks)
        pending.local_blocks = []
        try:
            request_future = self.add_request(
                pending.request_id,
                pending.prompt,
                pending.sampling_params,
                precomputed_block_hashes=pending.hashes[:end] if end > 0 else None,
            )
        except Exception:
            self._release_handoff_import_owner(pending.request_id)
            raise
        if pending.request_id not in self.waiting_request_ids:
            self._release_handoff_import_owner(pending.request_id)

        def _relay_result(src: asyncio.Future) -> None:
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

    def _admission_flags(self) -> list:
        """Per-pending (done, exception) pairs, with the done flags agreed
        across the model-parallel group.

        Each rank polls its own transfer handles; the done flags are
        AND-reduced over the MP group so every rank admits the same imports on
        the same step. Pending order is identical across ranks (the submits
        arrive via the TP broadcast in order). A poll failure is recorded and
        re-raised by the caller's quarantine path; it flags as done because
        the failure is terminal on this rank either way."""
        local = []
        for p in self._pending_kv_imports:
            try:
                done = all(handle.poll() for handle in self._pending_transfer_handles(p))
                local.append((done, None))
            except Exception as exc:  # quarantined by the caller
                local.append((True, exc))
        mp_group = getattr(self.pg_collection, "mp", None)
        if (
            mp_group is not None
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size(mp_group) > 1
        ):
            flags = torch.tensor(
                [1 if d else 0 for d, _ in local],
                dtype=torch.int32,
                device=self.context.memory_buffer.device,
            )
            torch.distributed.all_reduce(flags, op=torch.distributed.ReduceOp.MIN, group=mp_group)
            local = [(bool(f), exc) for f, (_, exc) in zip(flags.tolist(), local)]
        return local

    def _poll_pending_kv_imports(self) -> int:
        if not self._pending_kv_imports:
            return 0
        admission = deque(self._admission_flags())
        ready = 0
        remaining = deque()
        while self._pending_kv_imports:
            pending = self._pending_kv_imports.popleft()
            done, poll_exc = admission.popleft()
            try:
                if poll_exc is not None:
                    raise poll_exc
                if done:
                    self._finalize_kv_handoff_import(pending)
                    ready += 1
                else:
                    remaining.append(pending)
            except Exception as exc:
                safe_to_release = self._wait_for_transfer_handles(
                    *self._pending_transfer_handles(pending)
                )
                if safe_to_release:
                    self._release_pending_kv_import(pending)
                else:
                    remaining.append(pending)
                    logging.error(
                        "Quarantining request %d cache storage after transfer timeout",
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
