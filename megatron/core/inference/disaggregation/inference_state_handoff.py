# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Engine-side lifecycle for disaggregated prefill/decode KV-cache handoff."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict

import torch

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
    top_level_blocks: list[int]  # Stage-0 blocks carried in the top-level wire field.
    kv_meta: Any


def _pack_int_records(records: list[tuple[int, list[int]]]) -> list[int]:
    """Encode request records for a tensor-based pipeline-stage gather.

    Each record is encoded as ``[request_id, value_count, *values]``, with the
    number of records at the front of the payload. For example,
    ``[(7, [10, 11]), (8, [12])]`` becomes ``[2, 7, 2, 10, 11, 8, 1, 12]``.
    The flat integer payload lets :func:`_all_gather_int_records` exchange
    request block IDs without Python object serialization.
    """

    payload = [len(records)]
    for request_id, values in records:
        payload.extend((int(request_id), len(values), *map(int, values)))
    return payload


def _unpack_int_records(payload: list[int]) -> list[tuple[int, list[int]]]:
    """Decode request records gathered from another pipeline stage.

    The first value is the record count, followed by
    ``[request_id, value_count, *values]`` for each record. For example,
    ``[1, 7, 2, 10, 11]`` becomes ``[(7, [10, 11])]``. Bounds and trailing-data
    checks reject malformed peer metadata before it is used for a handoff.
    """

    if not payload:
        raise RuntimeError("handoff metadata payload is empty")
    num_records = int(payload[0])
    offset = 1
    records = []
    for _ in range(num_records):
        if offset + 2 > len(payload):
            raise RuntimeError("handoff metadata payload has a truncated record header")
        request_id, count = map(int, payload[offset : offset + 2])
        offset += 2
        if count < 0 or offset + count > len(payload):
            raise RuntimeError("handoff metadata payload has an invalid record length")
        records.append((request_id, [int(value) for value in payload[offset : offset + count]]))
        offset += count
    if offset != len(payload):
        raise RuntimeError("handoff metadata payload contains trailing values")
    return records


def _all_gather_int_records(
    records: list[tuple[int, list[int]]], group: Any, device: torch.device
) -> list[list[tuple[int, list[int]]]]:
    """All-gather a batch of integer records without Python object serialization."""

    world_size = get_pg_size(group)
    if world_size == 1 or not torch.distributed.is_initialized():
        return [records]

    payload = _pack_int_records(records)
    header = torch.tensor((len(records), len(payload)), dtype=torch.int64, device=device)
    gathered_header = torch.empty(world_size * 2, dtype=torch.int64, device=device)
    torch.distributed.all_gather_into_tensor(gathered_header, header, group=group)
    headers = gathered_header.view(world_size, 2).cpu().tolist()

    record_counts = {int(values[0]) for values in headers}
    if len(record_counts) != 1:
        raise RuntimeError(f"Handoff request counts differ across ranks: {sorted(record_counts)}")
    payload_sizes = [int(values[1]) for values in headers]
    max_payload_size = max(payload_sizes)
    local_payload = torch.zeros(max_payload_size, dtype=torch.int64, device=device)
    local_payload[: len(payload)] = torch.tensor(payload, dtype=torch.int64, device=device)
    gathered_payload = torch.empty(world_size * max_payload_size, dtype=torch.int64, device=device)
    torch.distributed.all_gather_into_tensor(gathered_payload, local_payload, group=group)
    return [
        _unpack_int_records(rank_payload[:payload_size])
        for rank_payload, payload_size in zip(
            gathered_payload.view(world_size, max_payload_size).cpu().tolist(), payload_sizes
        )
    ]


class InferenceStateHandoffMixin:
    """Optional KV-cache handoff behavior composed into the dynamic engine."""

    def _initialize_disaggregation_state(self) -> None:
        """Initialize state without importing or constructing a transfer backend."""

        self._pinned_handoff_blocks: Dict[int, list] = {}
        self._kv_transfer_agent = None
        self._kv_peer_metas = None
        self._pp_kv_peer_metas = None
        self._deferred_kv_handoffs = deque()
        self._pending_kv_imports = deque()
        self._handoff_import_owners: Dict[int, list[int]] = {}
        self._pending_kv_pushes: list = []

    @property
    def pending_kv_import_count(self) -> int:
        """Number of decode requests waiting for capacity or transfer completion."""

        return len(self._deferred_kv_handoffs) + len(self._pending_kv_imports)

    @property
    def pending_kv_push_count(self) -> int:
        """Number of prefill sends waiting for transport completion."""

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

        This method must be called collectively by every model-parallel rank in
        the engine because each rank participates in PP and TP metadata gathers.

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

    def _prepare_handoff_metadata_batch(
        self, requests_and_blocks: list[tuple["DynamicInferenceRequest", list[int]]]
    ) -> dict[int, _PreparedHandoffMetadata]:
        """Assemble metadata for all handoffs completed by one engine step."""

        handoffs = [
            (request, list(block_ids))
            for request, block_ids in requests_and_blocks
            if getattr(request.sampling_params, "do_kv_handoff", False)
        ]
        if not handoffs:
            return {}
        if self._kv_peer_metas is None:
            raise RuntimeError("KV handoff requested before transfer setup")

        local_records = []
        local_blocks_by_request = {}
        for request, block_ids in handoffs:
            # Prefix-cache entries must be immutable. The last prompt block is
            # still writable when the prompt is not block-aligned, so transfer
            # only complete blocks and recompute the bounded tail on decode.
            num_complete = len(request.prompt_tokens) // self.context.block_size_tokens
            complete_blocks = block_ids[:num_complete]
            dropped_blocks = block_ids[num_complete:]
            if dropped_blocks:
                self._release_pinned_handoff_blocks(dropped_blocks)
            local_blocks_by_request[request.request_id] = complete_blocks
            local_records.append((request.request_id, complete_blocks))

        pp_size = get_pg_size(self.pg_collection.pp)
        try:
            if pp_size > 1:
                gathered_records = _all_gather_int_records(
                    local_records, self.pg_collection.pp, self.context.memory_buffer.device
                )
            else:
                gathered_records = [local_records]
            local_request_ids = [request_id for request_id, _ in local_records]
            for rank_records in gathered_records:
                rank_request_ids = [request_id for request_id, _ in rank_records]
                if rank_request_ids != local_request_ids:
                    raise RuntimeError(
                        "Pipeline ranks completed different handoff requests "
                        f"(local={local_request_ids}, peer={rank_request_ids})"
                    )
            for record_index in range(len(local_records)):
                block_counts = {
                    len(rank_records[record_index][1]) for rank_records in gathered_records
                }
                if len(block_counts) != 1:
                    raise RuntimeError(
                        "Pipeline ranks produced different handoff block counts "
                        f"for request {local_request_ids[record_index]}: {sorted(block_counts)}"
                    )
            static_pp_metas = self._pp_kv_peer_metas or [self._kv_peer_metas]
            if len(static_pp_metas) != pp_size:
                raise RuntimeError(
                    f"Expected static metadata for {pp_size} pipeline stages, "
                    f"got {len(static_pp_metas)}"
                )
        except Exception:
            for block_ids in local_blocks_by_request.values():
                self._release_pinned_handoff_blocks(block_ids)
            raise

        prepared = {}
        for record_index, (request_id, local_blocks) in enumerate(local_records):
            stage_blocks = [rank_records[record_index][1] for rank_records in gathered_records]
            if pp_size > 1:
                kv_meta: Any = {
                    "pp_metas": [
                        {"tp_metas": stage_meta, "block_ids": blocks}
                        for stage_meta, blocks in zip(static_pp_metas, stage_blocks)
                    ]
                }
            else:
                kv_meta = self._kv_peer_metas
            prepared[request_id] = _PreparedHandoffMetadata(
                local_blocks=local_blocks, top_level_blocks=stage_blocks[0], kv_meta=kv_meta
            )
        return prepared

    def _capture_handoff_meta(
        self,
        request: "DynamicInferenceRequest",
        block_ids: list,
        prepared: _PreparedHandoffMetadata | None = None,
    ) -> None:
        """Attach prepared transfer metadata and retain the request's blocks."""

        rid = request.request_id
        if prepared is None:
            prepared = self._prepare_handoff_metadata_batch([(request, block_ids)])[rid]
        block_ids = prepared.local_blocks
        kv_meta = prepared.kv_meta
        top_level_block_ids = prepared.top_level_blocks

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
            "block_ids": top_level_block_ids,
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
        """Start or capacity-queue a KV pull and return its completion future."""
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
        """Start one capacity-safe import, or return false without mutation."""

        allocator = self.context.kv_block_allocator
        cached_blocks = []
        if not getattr(self._kv_transfer_agent, "is_push", False):
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
        owned_blocks_tensor = torch.tensor(local_blocks, dtype=torch.int32, device="cpu")

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

        if not self._all_ranks_started_handoff(start_error is None):
            # A peer may fail after this rank has posted a two-sided receive.
            # Waiting here could block forever because its matching send may
            # never be posted. Keep any potentially active destination blocks
            # out of the allocator until the engine is restarted.
            safe_to_release = handle is None and getattr(
                start_error, "transfer_destinations_safe", True
            )
            if safe_to_release:
                allocator.release_memory_blocks(owned_blocks_tensor)
            else:
                logging.error(
                    "Quarantining KV blocks after a failed handoff submission: %s", local_blocks
                )
            error = start_error or RuntimeError(
                "KV handoff submission failed on a model-parallel peer"
            )
            if not handoff.future.done():
                handoff.future.set_exception(error)
            raise error

        pending = PendingKvImport(
            request_id=handoff.request_id,
            prompt=handoff.prompt,
            sampling_params=handoff.sampling_params,
            local_blocks=local_blocks,
            hashes=handoff.hashes,
            hashes_to_register=max(
                0, min(handoff.num_blocks, len(handoff.hashes)) - len(cached_blocks)
            ),
            hash_registration_start=len(cached_blocks),
            handle=handle,
            future=handoff.future,
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
        """Find the contiguous handoff prefix already cached on decode."""

        allocator = self.context.kv_block_allocator
        cached_blocks = []
        for block_hash in hashes[:num_blocks]:
            block_id = allocator.kv_hash_to_block_id.get(block_hash)
            if block_id is None:
                break
            cached_blocks.append(int(block_id))
        return cached_blocks

    def _handoff_capacity_available(self, num_blocks: int, cached_blocks: list[int]) -> bool:
        """Agree on KV capacity before any model-parallel rank mutates its allocator."""

        allocator = self.context.kv_block_allocator
        potential_matched_count = 0
        if cached_blocks:
            block_tensor = torch.tensor(cached_blocks, dtype=torch.int32, device="cpu")
            potential_matched_count = int((allocator.block_ref_counts[block_tensor] == 0).sum())
        local_available = allocator.is_memory_available(
            num_blocks, potential_matched_count=potential_matched_count
        )

        mp_group = getattr(self.pg_collection, "mp", None)
        world_size = (
            torch.distributed.get_world_size(mp_group)
            if (mp_group is not None and torch.distributed.is_initialized())
            else 1
        )
        if world_size == 1:
            return local_available

        agreement = torch.tensor(
            [num_blocks, -num_blocks, int(local_available)],
            dtype=torch.int32,
            device=self.context.memory_buffer.device,
        )
        torch.distributed.all_reduce(agreement, op=torch.distributed.ReduceOp.MIN, group=mp_group)
        min_blocks, neg_max_blocks, all_available = agreement.tolist()
        if min_blocks != -neg_max_blocks:
            raise RuntimeError(
                "Model-parallel ranks computed different KV handoff block counts "
                f"(min={min_blocks}, max={-neg_max_blocks})"
            )
        return bool(all_available)

    def _all_ranks_started_handoff(self, local_started: bool) -> bool:
        """Agree that every model-parallel rank submitted its local transfer."""

        mp_group = getattr(self.pg_collection, "mp", None)
        world_size = (
            torch.distributed.get_world_size(mp_group)
            if (mp_group is not None and torch.distributed.is_initialized())
            else 1
        )
        if world_size == 1:
            return local_started

        started = torch.tensor(
            int(local_started), dtype=torch.int32, device=self.context.memory_buffer.device
        )
        torch.distributed.all_reduce(started, op=torch.distributed.ReduceOp.MIN, group=mp_group)
        return bool(started.item())

    def _drain_deferred_kv_handoffs(self) -> int:
        """Start queued handoffs in FIFO order while the queue head fits."""

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
        """Per-pending (done, failed, exception) tuples agreed across MP ranks.

        Pending order is identical across ranks because submissions arrive via
        the TP broadcast in order. One SUM reduction establishes that all ranks
        completed or that any rank failed, keeping admission and failure control
        flow identical across the model-parallel group.
        """
        local = []
        for p in self._pending_kv_imports:
            try:
                done = all(handle.poll() for handle in self._pending_transfer_handles(p))
                local.append((done, False, None))
            except Exception as exc:  # quarantined by the caller
                local.append((False, True, exc))
        mp_group = getattr(self.pg_collection, "mp", None)
        world_size = (
            torch.distributed.get_world_size(mp_group)
            if (mp_group is not None and torch.distributed.is_initialized())
            else 1
        )
        if world_size > 1:
            # A failure contributes -world_size, so the reduced value stays
            # negative even when every other rank reports completion.
            flags = torch.tensor(
                [-world_size if failed else int(done) for done, failed, _ in local],
                dtype=torch.int32,
                device=self.context.memory_buffer.device,
            )
            torch.distributed.all_reduce(flags, op=torch.distributed.ReduceOp.SUM, group=mp_group)
            local = [
                (value == world_size, value < 0, exc)
                for value, (_, _, exc) in zip(flags.tolist(), local)
            ]
        return local

    def _poll_pending_kv_imports(self) -> int:
        self._drain_deferred_kv_handoffs()
        if not self._pending_kv_imports:
            return 0
        admission = deque(self._admission_flags())
        ready = 0
        remaining = deque()
        while self._pending_kv_imports:
            pending = self._pending_kv_imports.popleft()
            done, failed, poll_exc = admission.popleft()
            try:
                if failed:
                    raise poll_exc or RuntimeError(
                        "KV handoff transfer failed on a model-parallel peer"
                    )
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
