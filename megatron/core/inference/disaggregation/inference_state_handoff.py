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

from megatron.core.inference.disaggregation.pending_handoff_imports import (
    PendingKvImport,
    PendingMambaImport,
)
from megatron.core.inference.disaggregation.transfer_backends.base import (
    construct_kv_transfer_backend_class,
)
from megatron.core.inference.disaggregation.utils import transfer_block_count
from megatron.core.inference.headers import Headers
from megatron.core.utils import get_pg_rank, get_pg_size

if TYPE_CHECKING:
    from megatron.core.inference.inference_request import DynamicInferenceRequest
    from megatron.core.inference.sampling_params import SamplingParams

_MAMBA_STATE_KINDS = ("conv", "ssm")


class InferenceStateHandoffMixin:
    """Optional KV/Mamba handoff behavior composed into the dynamic engine."""

    def _initialize_disaggregation_state(self) -> None:
        """Initialize state without importing or constructing a transfer backend."""

        self._pinned_handoff_blocks: Dict[int, list] = {}
        self._kv_transfer_agent = None
        self._kv_peer_metas = None
        self._mamba_transfer_agents = {}
        self._mamba_peer_metas = {}
        self._pending_kv_imports = deque()
        self._pending_kv_pushes: list = []
        self._instance_transfer_meta = None

    @property
    def pending_kv_import_count(self) -> int:
        """Number of decode requests waiting for state transfer completion."""

        return len(self._pending_kv_imports)

    def _reset_pending_kv_imports(self) -> None:
        """Drain and release pending imports before an engine reset."""

        if not hasattr(self, "_pending_kv_imports"):
            self._pending_kv_imports = deque()
            return
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

    def setup_kv_transfer(self, role: str, backend: str = "nixl") -> None:
        """Bring up the KV transfer agents for this engine.

        Args:
            role: "prefill" or "decode"; used to name the local transfer agent.
            backend: transfer backend name, resolved through the explicit
                registry ("nixl"; "nccl" selects the two-sided push family).
        """
        backend_cls = construct_kv_transfer_backend_class(backend)

        # Pinned hand-off blocks are held as prefix-cache references. Without
        # prefix caching, a context reset (e.g. the EP idle dummy forward)
        # returns pinned blocks to the free pool while a peer may still read
        # them, and the decode side cannot admit imports at all.
        allocator = self.context.kv_block_allocator
        assert allocator.enable_prefix_caching, (
            "KV handoff requires prefix caching on both prefill and decode "
            "engines (--inference-dynamic-batching-prefix-caching)."
        )
        allocator.enable_handoff_pinning = True

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

        # Cache static peer metadata for every request.
        self._kv_peer_metas = self._kv_transfer_agent.export_meta()
        if torch.distributed.is_initialized() and tp_size > 1:
            gathered: list = [None] * tp_size
            torch.distributed.all_gather_object(
                gathered, self._kv_peer_metas, group=self.pg_collection.tp
            )
            self._kv_peer_metas = gathered

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
            for state_kind, (memory_buffer, width, state_dim) in state_specs.items():
                ssm_state_kind = "recurrent" if state_kind == "ssm" else state_kind
                self._mamba_transfer_agents[state_kind] = backend_cls(
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

        # Gather this instance's per-rank transfer metadata (KV plus Mamba
        # kinds), model-parallel-wide. The MP coordinator registers it with
        # the shared coordinator; push transports ship it to the prefill in
        # SEND_KV so both sides enumerate the same reshard plan.
        entry = self._kv_transfer_agent.export_meta()
        if self._mamba_transfer_agents:
            entry["mamba"] = dict(self._mamba_peer_metas)
        mp_group = self.pg_collection.mp
        if torch.distributed.is_initialized() and get_pg_size(mp_group) > 1:
            gathered: list = [None] * get_pg_size(mp_group)
            torch.distributed.all_gather_object(gathered, entry, group=mp_group)
        else:
            gathered = [entry]
        self._instance_transfer_meta = gathered

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
            msa = self.context.mamba_slot_allocator
            slots = [
                int(msa.get_slot(int(block)))
                for block in block_ids
                if msa.get_slot(int(block)) >= 0
            ]
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
        if self._mamba_transfer_agents:
            msa = self.context.mamba_slot_allocator
            positions = []
            slots = []
            for pos, block in enumerate(block_ids):
                slot = msa.get_slot(int(block))
                if slot >= 0:
                    positions.append(pos)
                    slots.append(int(slot))
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
                positions = present_mamba[0]["positions"]
                if any(entry["positions"] != positions for entry in present_mamba):
                    raise RuntimeError("Mamba cached block positions differ across source TP ranks")
                local_mamba = {
                    "positions": positions,
                    **{
                        state_kind: [entry[state_kind] for entry in present_mamba]
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
                positions = mamba_stages[0]["positions"]
                if any(stage["positions"] != positions for stage in mamba_stages):
                    raise RuntimeError("Mamba cached block positions differ across source PP ranks")
                mamba_meta = {
                    "positions": positions,
                    **{
                        state_kind: {
                            "pp_metas": [{"tp_metas": stage[state_kind]} for stage in mamba_stages]
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

        if isinstance(kv_meta, list):
            kv_meta = {"tp_metas": kv_meta}
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
        """Submit an async NIXL pull, then admit the request when KV is local."""
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
        local_blocks_tensor = allocator.allocate_memory_blocks(num_blocks)
        if local_blocks_tensor is None:
            raise RuntimeError(f"add_request_with_kv_handoff: OOM allocating {num_blocks} blocks")
        local_blocks = [int(b) for b in local_blocks_tensor.tolist()]

        handle = None
        try:
            handle = self._kv_transfer_agent.begin_pull_blocks(kv_meta, src_block_ids, local_blocks)
            mamba_import = (
                self._begin_mamba_handoff_import(request_id, mamba_meta, local_blocks, hashes)
                if mamba_meta and local_has_mamba
                else None
            )
        except Exception as exc:
            safe_to_release = getattr(exc, "transfer_destinations_safe", True)
            if handle is not None:
                safe_to_release &= self._wait_for_transfer_handles(handle)
            if safe_to_release:
                allocator.release_memory_blocks(local_blocks_tensor)
            else:
                logging.error(
                    "Quarantining KV blocks after a timed-out handoff submission: %s", local_blocks
                )
            raise

        future = self._loop.create_future()
        pending = PendingKvImport(
            request_id=request_id,
            prompt=prompt,
            sampling_params=sampling_params,
            local_blocks=local_blocks,
            hashes=hashes,
            hashes_to_register=min(num_blocks, len(hashes)),
            handle=handle,
            future=future,
            mamba=mamba_import,
        )
        self._pending_kv_imports.append(pending)
        logging.info(
            "DISAGG_DECODE_PULL_SUBMIT request_id=%d prompt_tokens=%d blocks=%d pending_imports=%d",
            request_id,
            len(prompt),
            num_blocks,
            len(self._pending_kv_imports),
        )
        self._loop.call_soon_threadsafe(self._loop.create_task, self._notify_cond_for_new_request())
        return future

    @staticmethod
    def _pending_transfer_handles(pending: PendingKvImport) -> list:
        handles = [pending.handle]
        if pending.mamba is not None:
            handles.extend(pending.mamba.handles.values())
        return [handle for handle in handles if handle is not None]

    def _finalize_kv_handoff_import(self, pending: PendingKvImport) -> None:
        allocator = self.context.kv_block_allocator
        n = pending.hashes_to_register
        local_blocks = pending.local_blocks

        if n > 0:
            parent_hashes = [pending.hashes[index - 1] if index > 0 else 0 for index in range(n)]
            allocator.register_kv_block_hashes(
                local_blocks[:n], pending.hashes[:n], parent_hashes=parent_hashes
            )

        local_blocks_idx = torch.tensor(local_blocks, dtype=torch.int64)
        allocator.block_ref_counts[local_blocks_idx] -= 1

        if pending.mamba is not None:
            self._complete_mamba_handoff_import(pending.request_id, pending.mamba, pending.hashes)

        logging.info(
            "DISAGG_DECODE_IMPORT request_id=%d prompt_tokens=%d "
            "imported_blocks=%d hashes_registered=%d pending_imports=%d",
            pending.request_id,
            len(pending.prompt),
            len(local_blocks),
            n,
            len(self._pending_kv_imports),
        )
        request_future = self.add_request(
            pending.request_id,
            pending.prompt,
            pending.sampling_params,
            precomputed_block_hashes=pending.hashes[:n] if n > 0 else None,
        )

        # Coordinator-native mode: tell the coordinator the read drained so it
        # releases the prefill's pinned blocks and a flow-control slot. In the
        # Dynamo mode the client triggers the release instead.
        if getattr(self, "_disagg_config", None) is not None and self.is_mp_coordinator:
            assert HAVE_MSGPACK, "the coordinator-native disagg mode requires msgpack"
            self.socket_for_receiving_requests.send(
                msgpack.packb([Headers.KV_READ_DONE.value, pending.request_id], use_bin_type=True)
            )

        def _relay_result(src: asyncio.Future) -> None:
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
        if pending.local_blocks:
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

    def _begin_mamba_handoff_import(
        self, request_id: int, mamba_meta: dict, local_blocks: list, hashes: list
    ) -> Optional[PendingMambaImport]:
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
        handles = {}
        try:
            for state_kind, agent in self._mamba_transfer_agents.items():
                handles[state_kind] = agent.begin_pull_blocks(
                    mamba_meta[state_kind], [], local_slots
                )
        except Exception as exc:
            safe_to_release = getattr(exc, "transfer_destinations_safe", True)
            safe_to_release &= self._wait_for_transfer_handles(*handles.values())
            if safe_to_release:
                for block_id in target_blocks:
                    msa.invalidate_block(block_id)
            else:
                logging.error(
                    "Quarantining Mamba slots after a timed-out handoff submission: %s", local_slots
                )
            raise
        logging.info(
            "DISAGG_DECODE_MAMBA_IMPORT_SUBMIT request_id=%d mamba_blocks=%d",
            request_id,
            len(target_blocks),
        )
        return PendingMambaImport(handles=handles, target_blocks=target_blocks, positions=positions)

    def _complete_mamba_handoff_import(
        self, request_id: int, pending: PendingMambaImport, hashes: list
    ) -> None:
        msa = self.context.mamba_slot_allocator
        if msa is None:
            raise RuntimeError("Mamba handoff completed but the decode cache is unavailable.")
        msa.register_block_hashes_batch(
            pending.target_blocks, [hashes[p] for p in pending.positions]
        )
        logging.info(
            "DISAGG_DECODE_MAMBA_IMPORT request_id=%d mamba_blocks=%d",
            request_id,
            len(pending.target_blocks),
        )
