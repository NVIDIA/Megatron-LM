# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Engine-side lifecycle for disaggregated prefill/decode state handoff."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

try:
    import msgpack

    HAVE_MSGPACK = True
except ImportError:
    msgpack = None
    HAVE_MSGPACK = False

from megatron.core.inference.contexts.mamba_slot_allocator import MambaSlotCapacityError
from megatron.core.inference.disaggregation.pending_handoff_imports import (
    DeferredKvHandoff,
    PendingKvImport,
    PendingMambaImport,
)
from megatron.core.inference.disaggregation.transfer_backends.base import (
    construct_kv_transfer_backend_class,
)
from megatron.core.inference.disaggregation.utils import (
    drop_transfer_prefix_blocks,
    transfer_block_count,
)
from megatron.core.inference.headers import Headers
from megatron.core.utils import get_pg_rank, get_pg_size

if TYPE_CHECKING:
    from megatron.core.inference.inference_request import DynamicInferenceRequest
    from megatron.core.inference.sampling_params import SamplingParams

_MAMBA_STATE_KINDS = ("conv", "ssm")


def _common_mamba_positions(entries: list) -> list:
    """Return positions cached by every participating Mamba rank."""

    if not entries:
        return []
    peer_positions = [{int(position) for position in entry["positions"]} for entry in entries[1:]]
    return [
        int(position)
        for position in entries[0]["positions"]
        if all(int(position) in positions for positions in peer_positions)
    ]


# One recurrent checkpoint summarizes all preceding tokens, so a handoff needs
# only the farthest executable position. Earlier positions could seed reuse for
# future requests that diverge sooner, but would add transfer traffic and
# durable slot pressure for the current request.
def _executable_mamba_position(positions: list, prompt_length: int, block_size_tokens: int) -> list:
    """Return the farthest checkpoint that leaves executable prompt tokens."""

    max_skip_tokens = max(0, prompt_length - 2)
    executable = [
        int(position)
        for position in positions
        if (int(position) + 1) * block_size_tokens <= max_skip_tokens
    ]
    return [max(executable)] if executable else []


def _select_mamba_positions(meta: dict, source_positions: list, selected_positions: list) -> dict:
    """Filter one rank's transfer metadata to selected checkpoint positions."""

    block_ids = meta.get("block_ids")
    if block_ids is None or len(block_ids) != len(source_positions):
        raise RuntimeError("Mamba handoff metadata must contain one block ID per cached position")
    position_to_index = {int(position): index for index, position in enumerate(source_positions)}
    filtered = dict(meta)
    filtered["block_ids"] = [
        block_ids[position_to_index[int(position)]] for position in selected_positions
    ]
    return filtered


def _select_mamba_state_meta(meta: Any, source_positions: list, selected_positions: list) -> Any:
    """Filter a TP rank metadata dictionary or its gathered list."""

    if isinstance(meta, list):
        return [
            _select_mamba_positions(rank_meta, source_positions, selected_positions)
            for rank_meta in meta
        ]
    return _select_mamba_positions(meta, source_positions, selected_positions)


class InferenceStateHandoffMixin:
    """Optional KV/Mamba handoff behavior composed into the dynamic engine."""

    def _initialize_disaggregation_state(self) -> None:
        """Initialize state without importing or constructing a transfer backend."""

        self._pinned_handoff_blocks: Dict[int, list] = {}
        self._pinned_handoff_mamba_slots: Dict[int, list] = {}
        self._kv_transfer_agent = None
        self._kv_peer_metas = None
        self._mamba_transfer_agents = {}
        self._mamba_peer_metas = {}
        self._deferred_kv_handoffs = deque()
        self._pending_kv_imports = deque()
        self._handoff_import_owners: Dict[int, list[int]] = {}
        self._pending_kv_pushes: list = []

    @property
    def pending_kv_import_count(self) -> int:
        """Number of decode requests waiting for capacity or transfer completion."""

        return len(self._deferred_kv_handoffs) + len(self._pending_kv_imports)

    def _reset_pending_kv_imports(self) -> None:
        """Drain and release pending imports before an engine reset."""

        if not hasattr(self, "_deferred_kv_handoffs"):
            self._deferred_kv_handoffs = deque()
        while self._deferred_kv_handoffs:
            deferred = self._deferred_kv_handoffs.popleft()
            if not deferred.future.done():
                deferred.future.cancel()

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
        if unsafe:
            raise RuntimeError(
                "Cannot reset while KV handoff transfers may still write to cache storage"
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

        # Compute this PP rank's global attention- and Mamba-layer ranges.
        pp_size = get_pg_size(self.pg_collection.pp)
        pp_rank = get_pg_rank(self.pg_collection.pp)
        local_num_layers = self.context.num_attention_layers
        local_num_mamba_layers = getattr(self.context, "num_mamba_layers", 0)

        if pp_size > 1 and torch.distributed.is_initialized():
            layer_counts: list = [None] * pp_size
            torch.distributed.all_gather_object(
                layer_counts,
                (local_num_layers, local_num_mamba_layers),
                group=self.pg_collection.pp,
            )
            layer_start = sum(counts[0] for counts in layer_counts[:pp_rank])
            num_layers_global = sum(counts[0] for counts in layer_counts)
            mamba_layer_start = sum(counts[1] for counts in layer_counts[:pp_rank])
        else:
            layer_start = 0
            num_layers_global = local_num_layers
            mamba_layer_start = 0
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

        # Mamba uses the same transport segments as KV, with conv channels or
        # SSM heads as the fragment axis.
        self._mamba_transfer_agents = {}
        self._mamba_peer_metas = {}
        if getattr(self.context, "is_hybrid_model", False):
            from megatron.core.inference.disaggregation.ssm_reshard import (
                SSMShardLayout,
                SSMStateDims,
            )

            msa = getattr(self.context, "mamba_slot_allocator", None)
            if msa is None:
                raise RuntimeError(
                    "Hybrid model KV handoff requires the Mamba state cache. "
                    "Pass --inference-dynamic-batching-prefix-caching and "
                    "--inference-dynamic-batching-prefix-caching-mamba-gb <GB> "
                    "so the decode engine can restore transferred Mamba state."
                )
            conv_shape = msa.conv_states.shape
            ssm_shape = msa.ssm_states.shape
            nheads_local = int(ssm_shape[-3])
            nheads_global = nheads_local * tp_size
            configured_nheads = model_config.mamba_num_heads
            if configured_nheads is not None and configured_nheads != nheads_global:
                raise ValueError(
                    f"Mamba state shape implies {nheads_global} global heads, "
                    f"but model config specifies {configured_nheads}"
                )
            ssm_dims = SSMStateDims(
                nheads=nheads_global,
                headdim=int(ssm_shape[-2]),
                d_state=int(ssm_shape[-1]),
                ngroups=int(model_config.mamba_num_groups),
                d_conv=int(conv_shape[-1]),
            )
            ssm_layout = SSMShardLayout(
                global_rank=rank,
                tp_size=tp_size,
                tp_rank=tp_rank,
                layer_start=mamba_layer_start,
                num_layers=local_num_mamba_layers,
                dims=ssm_dims,
            )
            if int(conv_shape[-2]) != ssm_layout.conv_dim_local:
                raise ValueError(
                    "Mamba conv state shape does not match the model TP layout: "
                    f"{conv_shape[-2]} vs {ssm_layout.conv_dim_local}"
                )
            state_specs = {
                "conv": (msa.conv_states, msa.conv_states.shape[-2], msa.conv_states.shape[-1]),
                "ssm": (
                    msa.ssm_states,
                    msa.ssm_states.shape[-3],
                    msa.ssm_states.shape[-2] * msa.ssm_states.shape[-1],
                ),
            }
            backend_factory = getattr(self._kv_transfer_agent, "new_registered_buffer", backend_cls)
            for state_kind, (memory_buffer, width, state_dim) in state_specs.items():
                ssm_state_kind = "recurrent" if state_kind == "ssm" else state_kind
                self._mamba_transfer_agents[state_kind] = backend_factory(
                    agent_name=f"{role}-mamba-{state_kind}-rank{rank}",
                    memory_buffer=memory_buffer,
                    expected_num_blocks=msa.max_slots,
                    heads_per_partition=width,
                    head_dim=state_dim,
                    tokens_per_block=1,
                    ssm_layout=ssm_layout,
                    ssm_state_kind=ssm_state_kind,
                )
            self._mamba_peer_metas = {
                state_kind: agent.export_meta()
                for state_kind, agent in self._mamba_transfer_agents.items()
            }

        # Shared-agent metadata covers every registered state buffer, so export
        # it only after the optional Mamba buffers have been registered.
        self._kv_peer_metas = self._kv_transfer_agent.export_meta()
        if torch.distributed.is_initialized() and tp_size > 1:
            gathered: list = [None] * tp_size
            torch.distributed.all_gather_object(
                gathered, self._kv_peer_metas, group=self.pg_collection.tp
            )
            self._kv_peer_metas = gathered

    def push_handoff_kv(self, request_id: int, decode_metas: list) -> None:
        """Push a pinned hand-off's KV (and Mamba snapshots) to the decode
        instance described by `decode_metas` (two-sided transports only).

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
        if self._mamba_transfer_agents:
            # Reuse the exact slots advertised in the handoff metadata. The
            # allocator can acquire more cached Mamba states between metadata
            # capture and SEND_KV; recomputing here would make the sender post
            # more NCCL operations than the decode peer posted receives for.
            slots = self._pinned_handoff_mamba_slots.get(request_id, [])
            if slots:
                for state_kind, agent in self._mamba_transfer_agents.items():
                    peer = {
                        "tp_metas": [
                            e["mamba"][state_kind]
                            for e in decode_metas
                            if isinstance(e, dict) and e.get("mamba")
                        ]
                    }
                    handles.append(agent.begin_push_blocks(peer, slots))
        self._pending_kv_pushes.append((request_id, handles))
        logging.info(
            "DISAGG_PREFILL_PUSH request_id=%d blocks=%d mamba=%d",
            request_id,
            len(block_ids),
            len(handles) - 1,
        )

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

        local_mamba = None
        local_mamba_slots = []
        local_position_to_slot = {}
        if self._mamba_transfer_agents:
            msa = self.context.mamba_slot_allocator
            positions = []
            slots = []
            for pos, block in enumerate(block_ids):
                slot = msa.get_slot(int(block))
                if slot >= 0:
                    positions.append(pos)
                    slots.append(int(slot))
            local_mamba_slots = slots
            local_position_to_slot = dict(zip(positions, slots))
            local_mamba = {
                "positions": positions,
                **{
                    state_kind: {**meta, "block_ids": slots}
                    for state_kind, meta in self._mamba_peer_metas.items()
                },
            }

        tp_size = get_pg_size(self.pg_collection.tp)
        if tp_size > 1 and torch.distributed.is_initialized():
            gathered_mamba: list = [None] * tp_size
            torch.distributed.all_gather_object(
                gathered_mamba, local_mamba, group=self.pg_collection.tp
            )
            present_mamba = [entry for entry in gathered_mamba if entry is not None]
            if present_mamba:
                if len(present_mamba) != tp_size:
                    raise RuntimeError("Mamba handoff agents are not configured on every TP rank")
                positions = _common_mamba_positions(present_mamba)
                local_mamba = {
                    "positions": positions,
                    **{
                        state_kind: [
                            _select_mamba_positions(
                                entry[state_kind], entry["positions"], positions
                            )
                            for entry in present_mamba
                        ]
                        for state_kind in _MAMBA_STATE_KINDS
                    },
                }
            else:
                local_mamba = None

        pp_size = get_pg_size(self.pg_collection.pp)
        if pp_size > 1 and torch.distributed.is_initialized():
            local_entry = {
                "kv_meta": local_kv,
                "block_ids": list(block_ids),
                "mamba_meta": local_mamba,
            }
            gathered: list = [None] * pp_size
            torch.distributed.all_gather_object(gathered, local_entry, group=self.pg_collection.pp)
            kv_meta: Any = {
                "pp_metas": [
                    {"tp_metas": e["kv_meta"], "block_ids": e["block_ids"]} for e in gathered
                ]
            }
            top_block_ids: Any = gathered[0]["block_ids"]
            mamba_stages = [
                entry["mamba_meta"] for entry in gathered if entry["mamba_meta"] is not None
            ]
            if mamba_stages:
                positions = _executable_mamba_position(
                    _common_mamba_positions(mamba_stages),
                    len(request.prompt_tokens),
                    self.context.block_size_tokens,
                )
                mamba_meta = {
                    "positions": positions,
                    **{
                        state_kind: {
                            "pp_metas": [
                                {
                                    "tp_metas": _select_mamba_state_meta(
                                        stage[state_kind], stage["positions"], positions
                                    )
                                }
                                for stage in mamba_stages
                            ]
                        }
                        for state_kind in _MAMBA_STATE_KINDS
                    },
                }
            else:
                mamba_meta = None
        else:
            kv_meta = local_kv
            top_block_ids = block_ids
            mamba_meta = local_mamba
            if mamba_meta is not None:
                source_positions = mamba_meta["positions"]
                positions = _executable_mamba_position(
                    source_positions, len(request.prompt_tokens), self.context.block_size_tokens
                )
                mamba_meta = {
                    "positions": positions,
                    **{
                        state_kind: _select_mamba_state_meta(
                            mamba_meta[state_kind], source_positions, positions
                        )
                        for state_kind in _MAMBA_STATE_KINDS
                    },
                }

        if mamba_meta is not None:
            local_mamba_slots = [
                local_position_to_slot[position] for position in mamba_meta["positions"]
            ]
        self._pinned_handoff_mamba_slots[rid] = local_mamba_slots

        if isinstance(kv_meta, list):
            kv_meta = {"tp_metas": kv_meta}
        else:
            # TP=1 caches one static metadata dictionary for the engine. Keep
            # request-specific Mamba metadata out of that shared object so a
            # later handoff cannot overwrite an earlier request's positions.
            kv_meta = dict(kv_meta)
        if mamba_meta is not None:
            kv_meta["mamba"] = mamba_meta

        request.disaggregated_params = {
            "request_id": rid,
            "block_ids": top_block_ids,
            "kv_meta": kv_meta,
        }
        logging.info(
            "DISAGG_PREFILL_HANDOFF request_id=%d pinned_blocks=%d mamba_blocks=%d",
            rid,
            len(block_ids),
            len(mamba_meta["positions"]) if mamba_meta is not None else 0,
        )

    def release_handoff_blocks(self, request_id: int) -> None:
        """Release blocks pinned by a previous do_kv_handoff completion."""
        block_ids = self._pinned_handoff_blocks.pop(request_id, None)
        self._pinned_handoff_mamba_slots.pop(request_id, None)
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
        """Start a capacity-safe state pull, or defer it locally in FIFO order."""
        from megatron.core.inference.inference_request import compute_block_hashes_batched

        allocator = self.context.kv_block_allocator
        if not allocator.enable_prefix_caching:
            raise RuntimeError(
                "add_request_with_kv_handoff requires --enable-prefix-caching on the "
                "decode engine; the prefill-skip path uses the prefix-cache match logic."
            )

        mamba_meta = kv_meta.get("mamba") if isinstance(kv_meta, dict) else None
        local_has_mamba = bool(self._mamba_transfer_agents)
        if local_has_mamba and mamba_meta is None:
            raise RuntimeError(
                "Decode has Mamba state transfer agents but the handoff contains no "
                "Mamba metadata; prefill and decode must use the same hybrid model"
            )
        if self._kv_transfer_agent is None:
            raise RuntimeError("KV handoff received without a transfer backend")

        prompt_tensor = torch.tensor(prompt, dtype=torch.int64)
        hashes = compute_block_hashes_batched(
            prompt_tensor, self.context.block_size_tokens, include_partial=True
        )
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

        # Preserve receive order under backpressure. This is required by NCCL's
        # two-sided transport and also prevents a stream of small handoffs from
        # starving an older, larger one.
        if self._deferred_kv_handoffs:
            self._deferred_kv_handoffs.append(handoff)
            logging.debug(
                "DISAGG_DECODE_CAPACITY_QUEUE request_id=%d queued=%d",
                request_id,
                len(self._deferred_kv_handoffs),
            )
            return future

        started, capacity_error = self._try_start_kv_handoff_import(handoff)
        if not started:
            self._deferred_kv_handoffs.append(handoff)
            logging.debug(
                "DISAGG_DECODE_CAPACITY_QUEUE request_id=%d required=%d available=%d queued=%d",
                request_id,
                capacity_error.required,
                capacity_error.available,
                len(self._deferred_kv_handoffs),
            )
        return future

    def _try_start_kv_handoff_import(
        self, handoff: DeferredKvHandoff
    ) -> tuple[bool, Optional[MambaSlotCapacityError]]:
        """Reserve cache state before starting one handoff transfer."""

        allocator = self.context.kv_block_allocator
        num_blocks = handoff.num_blocks
        cached_blocks = []
        if not getattr(self._kv_transfer_agent, "is_push", False):
            cached_blocks = self._retain_cached_handoff_prefix(handoff.hashes, num_blocks)

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
        mamba_import = None
        capacity_error = None
        try:
            transfer_meta, transfer_src_blocks = drop_transfer_prefix_blocks(
                handoff.kv_meta, handoff.src_block_ids, len(cached_blocks)
            )
            mamba_meta = handoff.kv_meta.get("mamba") if isinstance(handoff.kv_meta, dict) else None
            if mamba_meta and self._mamba_transfer_agents:
                try:
                    mamba_import = self._reserve_mamba_handoff_import(
                        mamba_meta, local_blocks, handoff.hashes
                    )
                except MambaSlotCapacityError as exc:
                    capacity_error = exc

            capacity_error = self._agree_mamba_handoff_capacity(mamba_meta, capacity_error)
            if capacity_error is not None:
                if mamba_import is not None:
                    msa = self.context.mamba_slot_allocator
                    for block_id in mamba_import.target_blocks:
                        msa.invalidate_block(block_id)
                allocator.release_memory_blocks(owned_blocks_tensor)
                return False, capacity_error

            handle = self._kv_transfer_agent.begin_pull_blocks(
                transfer_meta, transfer_src_blocks, imported_blocks
            )
            if mamba_import is not None:
                self._start_mamba_handoff_import(handoff.request_id, mamba_meta, mamba_import)
        except Exception as exc:
            safe_to_release = getattr(exc, "transfer_destinations_safe", True)
            handles = [handle]
            if mamba_import is not None:
                handles.extend(mamba_import.handles.values())
            safe_to_release &= self._wait_for_transfer_handles(*handles)
            if safe_to_release:
                allocator.release_memory_blocks(owned_blocks_tensor)
                if mamba_import is not None:
                    msa = self.context.mamba_slot_allocator
                    for block_id in mamba_import.target_blocks:
                        msa.invalidate_block(block_id)
            else:
                logging.error(
                    "Quarantining cache storage after a timed-out handoff submission: "
                    "KV blocks=%s, Mamba slots=%s",
                    local_blocks,
                    mamba_import.local_slots if mamba_import is not None else [],
                )
            raise

        pending = PendingKvImport(
            request_id=handoff.request_id,
            prompt=handoff.prompt,
            sampling_params=handoff.sampling_params,
            local_blocks=local_blocks,
            hashes=handoff.hashes,
            hashes_to_register=max(0, min(num_blocks, len(handoff.hashes)) - len(cached_blocks)),
            hash_registration_start=len(cached_blocks),
            handle=handle,
            future=handoff.future,
            mamba=mamba_import,
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
        return True, None

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

    def _agree_mamba_handoff_capacity(
        self, mamba_meta: Optional[dict], local_error: Optional[MambaSlotCapacityError]
    ) -> Optional[MambaSlotCapacityError]:
        """Agree on decode capacity before any model-parallel rank starts transport."""

        positions = mamba_meta.get("positions", []) if isinstance(mamba_meta, dict) else []
        mp_group = getattr(self.pg_collection, "mp", None)
        if (
            not positions
            or mp_group is None
            or not torch.distributed.is_initialized()
            or torch.distributed.get_world_size(mp_group) == 1
        ):
            return local_error

        agreement = torch.tensor(
            [
                0 if local_error is not None else 1,
                local_error.available if local_error is not None else torch.iinfo(torch.int64).max,
                -local_error.required if local_error is not None else 0,
            ],
            dtype=torch.int64,
            device=self.context.memory_buffer.device,
        )
        torch.distributed.all_reduce(agreement, op=torch.distributed.ReduceOp.MIN, group=mp_group)
        all_succeeded, available, neg_required = agreement.tolist()
        if all_succeeded:
            return None
        return MambaSlotCapacityError(required=-neg_required, available=available)

    def _drain_deferred_kv_handoffs(self) -> int:
        """Start capacity-queued handoffs in FIFO order while they fit."""

        started_count = 0
        while self._deferred_kv_handoffs:
            handoff = self._deferred_kv_handoffs[0]
            started, _ = self._try_start_kv_handoff_import(handoff)
            if not started:
                break
            self._deferred_kv_handoffs.popleft()
            started_count += 1
            logging.debug(
                "DISAGG_DECODE_CAPACITY_ADMIT request_id=%d queued=%d",
                handoff.request_id,
                len(self._deferred_kv_handoffs),
            )
        return started_count

    @staticmethod
    def _pending_transfer_handles(pending: PendingKvImport) -> list:
        handles = [pending.handle]
        if pending.mamba is not None:
            handles.extend(pending.mamba.handles.values())
        return [handle for handle in handles if handle is not None]

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

        if pending.mamba is not None:
            self._complete_mamba_handoff_import(pending.request_id, pending.mamba, pending.hashes)

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

        # Coordinator-native mode: tell the coordinator the read drained so it
        # releases the prefill's pinned blocks and a flow-control slot. In the
        # Dynamo mode the client triggers the release instead.
        if getattr(self, "_disagg_config", None) is not None and self.is_mp_coordinator:
            assert HAVE_MSGPACK, "the coordinator-native disagg mode requires msgpack"
            self.socket_for_receiving_requests.send(
                msgpack.packb([Headers.KV_READ_DONE.value, pending.request_id], use_bin_type=True)
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
        if pending.mamba is not None:
            msa = self.context.mamba_slot_allocator
            if msa is not None:
                for block_id in pending.mamba.target_blocks:
                    msa.invalidate_block(int(block_id))

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
                [1 if d else 0 for d, _ in local], dtype=torch.int32, device="cuda"
            )
            torch.distributed.all_reduce(flags, op=torch.distributed.ReduceOp.MIN, group=mp_group)
            local = [(bool(f), exc) for f, (_, exc) in zip(flags.tolist(), local)]
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

    def _reserve_mamba_handoff_import(
        self, mamba_meta: dict, local_blocks: list, hashes: list
    ) -> Optional[PendingMambaImport]:
        """Atomically reserve destination slots without starting a transfer."""

        positions = [int(pos) for pos in mamba_meta.get("positions", [])]
        if not positions:
            return None
        if not self._mamba_transfer_agents:
            raise RuntimeError(
                "Received Mamba handoff state but this decode engine has no "
                "Mamba transfer agents. Ensure it runs a hybrid model with "
                "--inference-dynamic-batching-prefix-caching-mamba-gb set and "
                "KV transfer enabled."
            )
        msa = self.context.mamba_slot_allocator
        if msa is None:
            raise RuntimeError(
                "Mamba handoff requires the decode engine's Mamba state cache; "
                "pass --inference-dynamic-batching-prefix-caching-mamba-gb."
            )

        if any(pos < 0 or pos >= len(local_blocks) or pos >= len(hashes) for pos in positions):
            raise ValueError(
                f"Mamba handoff positions are outside the imported KV blocks: {positions}"
            )
        target_blocks = [int(local_blocks[p]) for p in positions]
        local_slots = msa.allocate_slots_batch(target_blocks)
        return PendingMambaImport(
            handles={}, local_slots=local_slots, target_blocks=target_blocks, positions=positions
        )

    def _start_mamba_handoff_import(
        self, request_id: int, mamba_meta: dict, pending: PendingMambaImport
    ) -> None:
        """Post transfers into slots already reserved for one handoff."""

        handles = {}
        pending.handles = handles
        for state_kind, agent in self._mamba_transfer_agents.items():
            handles[state_kind] = agent.begin_pull_blocks(
                mamba_meta[state_kind], [], pending.local_slots
            )
        logging.debug(
            "DISAGG_DECODE_MAMBA_IMPORT_SUBMIT request_id=%d mamba_blocks=%d",
            request_id,
            len(pending.target_blocks),
        )

    def _complete_mamba_handoff_import(
        self, request_id: int, pending: PendingMambaImport, hashes: list
    ) -> None:
        msa = self.context.mamba_slot_allocator
        if msa is None:
            raise RuntimeError("Mamba handoff completed but the decode cache is unavailable.")
        msa.register_block_hashes_batch(
            pending.target_blocks, [hashes[p] for p in pending.positions]
        )
        logging.debug(
            "DISAGG_DECODE_MAMBA_IMPORT request_id=%d mamba_blocks=%d",
            request_id,
            len(pending.target_blocks),
        )
