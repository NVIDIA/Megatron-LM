# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Engine-side lifecycle for disaggregated prefill/decode state handoff."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict

import torch

from megatron.core.inference.disaggregation.decode_admission import (
    additional_decode_blocks,
    admit_prefilled_decode,
    can_admit_prefilled_decode,
)
from megatron.core.inference.disaggregation.handoff_completion_tracker import (
    HandoffCompletionTracker,
)
from megatron.core.inference.disaggregation.handoff_wire_protocol import (
    strip_registered_nixl_agent_metadata,
)
from megatron.core.inference.disaggregation.pending_handoff_imports import (
    DeferredKvHandoff,
    PendingKvImport,
    PendingSSMImport,
)
from megatron.core.inference.disaggregation.ssm_reshard import SSMShardLayout, SSMStateDims
from megatron.core.inference.disaggregation.transfer_backends.base import (
    construct_kv_transfer_backend_class,
)
from megatron.core.inference.disaggregation.utils import (
    drop_transfer_prefix_blocks,
    transfer_block_count,
)
from megatron.core.inference.inference_request import (
    DynamicInferenceEvent,
    DynamicInferenceEventType,
    Status,
    compute_block_hashes_batched,
)
from megatron.core.utils import get_pg_rank, get_pg_size

if TYPE_CHECKING:
    from megatron.core.inference.inference_request import DynamicInferenceRequest
    from megatron.core.inference.sampling_params import SamplingParams

_SSM_STATE_KINDS = ("conv", "recurrent")


@dataclass(frozen=True)
class _PreparedHandoffMetadata:
    """Per-request metadata assembled once for a completed prefill batch."""

    local_blocks: list[int]  # Full prompt block table, including a partial tail.
    local_ssm_slot: int | None
    kv_meta: Any
    ssm_meta: Any
    resume_tokens: list[int]


class InferenceStateHandoffMixin:
    """Optional KV/SSM handoff behavior composed into the dynamic engine."""

    def _initialize_disaggregation_state(self) -> None:
        """Initialize state without importing or constructing a transfer backend."""

        self._disagg_config = None
        self._pinned_handoff_blocks: Dict[int, list] = {}  # Request ID -> pinned KV block IDs.
        self._pinned_handoff_ssm_slots: Dict[int, int] = {}  # Request ID -> detached live slot.
        self._kv_transfer_agent = None
        self._kv_peer_metas = None  # KV descriptors for this PP stage's TP ranks.
        self._pp_kv_peer_metas = None  # Each PP stage's set of TP KV descriptors.
        self._ssm_transfer_agents = {}
        self._pp_ssm_peer_metas = None  # Each PP stage's TP SSM descriptors.
        self._deferred_kv_handoffs = deque()
        self._pending_kv_imports = deque()
        self._quarantined_kv_imports: list[PendingKvImport] = []
        self._handoff_completion_tracker: HandoffCompletionTracker | None = None
        self._handoff_completion_notifications: dict[int, bool] = {}  # Request ID -> failed.
        self._pending_kv_pushes: list = []
        self._kv_transfer_role: str | None = None

    def _notify_kv_read_done(self, request_id: int) -> None:
        """Hook for control planes that release source storage after decode admission."""

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
    def has_admittable_kv_import(self) -> bool:
        """Whether a completed import can be handled at the next admission point."""

        for pending in self._pending_kv_imports:
            failed = self._handoff_completion_notifications.get(pending.request_id)
            if failed is None:
                continue
            if failed or can_admit_prefilled_decode(self.context, len(pending.resume_tokens)):
                return True
        return False

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
        quarantined = []
        for pending in self._quarantined_kv_imports:
            safe_to_release = pending.destinations_safe and self._wait_for_transfer_handles(
                *self._pending_transfer_handles(pending)
            )
            if safe_to_release:
                self._release_pending_kv_import(pending)
            else:
                quarantined.append(pending)
        self._quarantined_kv_imports = quarantined
        if unsafe_pushes or unsafe or quarantined:
            raise RuntimeError(
                "Cannot reset while KV handoff transfers may still access cache storage"
            )
        if self._pinned_handoff_blocks or self._pinned_handoff_ssm_slots:
            raise RuntimeError(
                "Cannot reset while handoff state remains pinned; wait for RELEASE_KV"
            )
        self._handoff_completion_notifications.clear()

    def schedule_waiting_requests(self) -> None:
        """Reject prompt scheduling on a dedicated disaggregated decode engine.

        Side: decode engine; pull and push transport paths.
        """

        if self._kv_transfer_role == "decode":
            # Async scheduling reaches this point after consuming the previous
            # forward's logits. The legacy path does not retain logits across steps.
            self._admit_pending_kv_imports()
            if self.waiting_request_ids:
                raise RuntimeError(
                    "A disaggregated decode engine cannot schedule prompt prefill requests"
                )
        super().schedule_waiting_requests()

    def setup_kv_transfer(self, role: str, backend: str = "nixl") -> None:
        """Bring up the KV transfer agents for this engine.

        This method must be called collectively by every model-parallel rank in
        the engine because each rank participates in PP and TP metadata gathers.

        Args:
            role: "prefill" or "decode"; used to name the local transfer agent.
            backend: transfer backend name, resolved through the explicit
                registry ("nixl"; "nccl" selects the two-sided push family).
        """
        if role not in ("prefill", "decode"):
            raise ValueError(f"KV transfer role must be 'prefill' or 'decode', got {role!r}")
        if self.context.is_hybrid_model:
            if role == "decode" and self.context.mamba_slot_allocator is not None:
                raise RuntimeError(
                    "A hybrid decode handoff writes directly into live SSM state; "
                    "do not configure prefix_caching_mamba_gb on the decode engine"
                )
        self._kv_transfer_role = role
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

        # Compute this PP rank's global attention- and SSM-layer ranges.
        pp_size = get_pg_size(self.pg_collection.pp)
        pp_rank = get_pg_rank(self.pg_collection.pp)
        local_num_layers = self.context.num_attention_layers
        local_num_ssm_layers = self.context.num_mamba_layers

        if pp_size > 1 and torch.distributed.is_initialized():
            layer_counts: list = [None] * pp_size
            torch.distributed.all_gather_object(
                layer_counts, (local_num_layers, local_num_ssm_layers), group=self.pg_collection.pp
            )
            layer_start = sum(counts[0] for counts in layer_counts[:pp_rank])
            num_layers_global = sum(counts[0] for counts in layer_counts)
            ssm_layer_start = sum(counts[1] for counts in layer_counts[:pp_rank])
        else:
            layer_start = 0
            num_layers_global = local_num_layers
            ssm_layer_start = 0
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

        # Both roles transfer directly between live recurrent-state slots. Prefill
        # detaches its source slot until decode acknowledges transfer completion.
        if self.context.is_hybrid_model:
            conv_states = self.context.mamba_conv_states
            recurrent_states = self.context.mamba_ssm_states
            state_slot_count = self.context.max_requests

            conv_shape = conv_states.shape
            ssm_shape = recurrent_states.shape
            nheads_local = int(ssm_shape[-3])
            nheads_global = nheads_local * tp_size
            configured_nheads = model_config.mamba_num_heads
            if configured_nheads is not None and configured_nheads != nheads_global:
                raise ValueError(
                    f"SSM state shape implies {nheads_global} global heads, "
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
                layer_start=ssm_layer_start,
                num_layers=local_num_ssm_layers,
                dims=ssm_dims,
            )
            if int(conv_shape[-2]) != ssm_layout.conv_dim_local:
                raise ValueError(
                    "SSM conv state shape does not match the model TP layout: "
                    f"{conv_shape[-2]} vs {ssm_layout.conv_dim_local}"
                )
            state_specs = {
                "conv": (conv_states, conv_states.shape[-2], conv_states.shape[-1]),
                "recurrent": (
                    recurrent_states,
                    recurrent_states.shape[-3],
                    recurrent_states.shape[-2] * recurrent_states.shape[-1],
                ),
            }
            for state_kind, (memory_buffer, width, state_dim) in state_specs.items():
                self._ssm_transfer_agents[state_kind] = (
                    self._kv_transfer_agent.new_registered_buffer(
                        agent_name=f"{role}-ssm-{state_kind}-rank{rank}",
                        memory_buffer=memory_buffer,
                        expected_num_blocks=state_slot_count,
                        heads_per_partition=width,
                        head_dim=state_dim,
                        tokens_per_block=1,
                        ssm_layout=ssm_layout,
                        ssm_state_kind=state_kind,
                    )
                )
            ssm_peer_metas = {
                state_kind: agent.export_meta()
                for state_kind, agent in self._ssm_transfer_agents.items()
            }
        else:
            ssm_peer_metas = {}

        # Transport descriptors do not change between requests. Collect each
        # stage's TP descriptors once and reuse them for every handoff.
        local_peer_metas = {"kv": self._kv_transfer_agent.export_meta(), "ssm": ssm_peer_metas}
        if torch.distributed.is_initialized() and tp_size > 1:
            gathered_tp_metas: list = [None] * tp_size
            torch.distributed.all_gather_object(
                gathered_tp_metas, local_peer_metas, group=self.pg_collection.tp
            )
            ssm_state_kinds = [set(entry["ssm"]) for entry in gathered_tp_metas]
            if any(kinds != ssm_state_kinds[0] for kinds in ssm_state_kinds[1:]):
                raise RuntimeError("SSM handoff agents are not configured on every TP rank")
            self._kv_peer_metas = [entry["kv"] for entry in gathered_tp_metas]
            tp_ssm_peer_metas = {
                state_kind: [entry["ssm"][state_kind] for entry in gathered_tp_metas]
                for state_kind in ssm_state_kinds[0]
            }
        else:
            self._kv_peer_metas = local_peer_metas["kv"]
            tp_ssm_peer_metas = dict(local_peer_metas["ssm"])

        local_stage_metas = {"kv": self._kv_peer_metas, "ssm": tp_ssm_peer_metas}
        gathered_stage_metas = [local_stage_metas]
        if torch.distributed.is_initialized() and pp_size > 1:
            gathered_stage_metas = [None] * pp_size
            torch.distributed.all_gather_object(
                gathered_stage_metas, local_stage_metas, group=self.pg_collection.pp
            )
        self._pp_kv_peer_metas = [entry["kv"] for entry in gathered_stage_metas]
        self._pp_ssm_peer_metas = [entry["ssm"] for entry in gathered_stage_metas]

    def push_handoff_kv(self, request_id: int, decode_metas: list) -> None:
        """Push a pinned hand-off's KV and live SSM state to the decode
        instance described by `decode_metas` (two-sided transports only).

        The decode posted its matching receives when SUBMIT_REQUEST_WITH_KV
        arrived; the sends are reaped asynchronously and the pins stay until
        the coordinator's RELEASE_KV.

        Side: prefill engine; push transport path only.
        """
        block_ids = self._pinned_handoff_blocks.get(request_id)
        if not block_ids:
            if self._ssm_transfer_agents and request_id in self._pinned_handoff_ssm_slots:
                raise RuntimeError(
                    f"Handoff request {request_id} has detached SSM state but no pinned KV blocks"
                )
            logging.warning(
                "SEND_KV for request %d with no pinned hand-off blocks; skipping", request_id
            )
            return

        ssm_pushes = []
        if self._ssm_transfer_agents:
            slot = self._pinned_handoff_ssm_slots.get(request_id)
            if slot is None:
                raise RuntimeError(f"No detached SSM state for handoff request {request_id}")
            for state_kind in _SSM_STATE_KINDS:
                try:
                    peer_metas = [entry["ssm"][state_kind] for entry in decode_metas]
                except (KeyError, TypeError) as error:
                    raise RuntimeError(
                        f"Decode metadata is missing {state_kind} SSM transfer state"
                    ) from error
                ssm_pushes.append((self._ssm_transfer_agents[state_kind], peer_metas, [slot]))

        kv_peer = {"tp_metas": list(decode_metas)}
        handles = [self._kv_transfer_agent.begin_push_blocks(kv_peer, block_ids)]
        # The source live slot remains detached until RELEASE_KV, so request
        # cleanup cannot reuse it while these sends are active.
        for agent, peer_metas, slots in ssm_pushes:
            handles.append(agent.begin_push_blocks({"tp_metas": peer_metas}, slots))
        self._pending_kv_pushes.append((request_id, handles))
        logging.info(
            "DISAGG_PREFILL_PUSH request_id=%d blocks=%d ssm=%d",
            request_id,
            len(block_ids),
            len(handles) - 1,
        )

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
        self,
        requests_and_state: list[tuple["DynamicInferenceRequest", list[int], int | None]],
        decode_tokens_by_request: Dict[int, list[int]],
    ) -> dict[int, _PreparedHandoffMetadata]:
        """Assemble metadata for all handoffs completed by one engine step.

        Side: prefill engine; pull and push transport paths.
        """

        handoffs = []
        for request, block_ids, ssm_slot in requests_and_state:
            if not request.sampling_params.do_kv_handoff:
                continue
            prompt_block_count = (
                len(request.prompt_tokens) + self.context.block_size_tokens - 1
            ) // self.context.block_size_tokens
            handoffs.append((request, list(block_ids[:prompt_block_count]), ssm_slot))
        if not handoffs:
            return {}

        try:
            if self._kv_peer_metas is None:
                raise RuntimeError("KV handoff requested before transfer setup")

            pp_size = get_pg_size(self.pg_collection.pp)
            static_pp_metas = self._pp_kv_peer_metas or [self._kv_peer_metas]
            static_pp_ssm_metas = self._pp_ssm_peer_metas or [{}]
            if len(static_pp_metas) != pp_size:
                raise RuntimeError(
                    f"Expected static metadata for {pp_size} pipeline stages, "
                    f"got {len(static_pp_metas)}"
                )
            if len(static_pp_ssm_metas) != pp_size:
                raise RuntimeError(
                    f"Expected static SSM metadata for {pp_size} pipeline stages, "
                    f"got {len(static_pp_ssm_metas)}"
                )

            prepared = {}
            for request, candidate_blocks, ssm_slot in handoffs:
                resume_tokens = list(decode_tokens_by_request.get(request.request_id, []))
                expected_resume_tokens = self.context.num_speculative_tokens + 1
                if len(resume_tokens) != expected_resume_tokens:
                    raise RuntimeError(
                        "Cannot create a decode-only handoff without one sampled token plus "
                        f"the configured MTP proposals: expected {expected_resume_tokens}, "
                        f"got {len(resume_tokens)}"
                    )
                if not candidate_blocks:
                    raise RuntimeError(
                        "Cannot create a decode-only handoff without the prompt KV state"
                    )
                if self._ssm_transfer_agents and ssm_slot is None:
                    raise RuntimeError(
                        "Cannot create a hybrid decode-only handoff without the exact final "
                        "SSM state"
                    )
                block_ids = candidate_blocks
                if pp_size > 1:
                    # Model-parallel ranks receive the same request stream and
                    # use synchronized KV and recurrent-state cache capacities.
                    # Their deterministic allocators therefore assign the same
                    # physical IDs, while transport descriptors remain stage- and
                    # rank-specific metadata collected once during setup.
                    kv_meta: Any = {
                        "pp_metas": [
                            {"tp_metas": stage_meta, "block_ids": block_ids}
                            for stage_meta in static_pp_metas
                        ]
                    }
                else:
                    kv_meta = self._kv_peer_metas

                ssm_meta = None
                if self._ssm_transfer_agents:
                    stage_state_metas = [
                        self._ssm_stage_transfer_meta(static_meta, ssm_slot)
                        for static_meta in static_pp_ssm_metas
                    ]
                    ssm_meta = {}
                    for state_kind in _SSM_STATE_KINDS:
                        if pp_size > 1:
                            ssm_meta[state_kind] = {
                                "pp_metas": [
                                    {"tp_metas": stage_meta[state_kind]}
                                    for stage_meta in stage_state_metas
                                ]
                            }
                        else:
                            ssm_meta[state_kind] = stage_state_metas[0][state_kind]

                prepared[request.request_id] = _PreparedHandoffMetadata(
                    local_blocks=block_ids,
                    local_ssm_slot=ssm_slot,
                    kv_meta=kv_meta,
                    ssm_meta=ssm_meta,
                    resume_tokens=resume_tokens,
                )
            return prepared
        except Exception:
            # The controller temporarily transfers ownership for every finished
            # request on a prefill engine, including regular requests. Return all
            # of it if metadata preparation aborts before the per-request loop.
            for _, block_ids, ssm_slot in requests_and_state:
                self._release_pinned_handoff_blocks(block_ids)
                self._release_pinned_handoff_ssm_slot(ssm_slot)
            raise

    @staticmethod
    def _ssm_stage_transfer_meta(static_metas: dict, selected_slot: int) -> dict:
        """Attach the final recurrent-state slot to one stage's descriptors."""

        state_metas = {}
        selected_slots = [selected_slot]
        for state_kind in _SSM_STATE_KINDS:
            state_static_metas = static_metas.get(state_kind)
            if state_static_metas is None:
                raise RuntimeError(f"Missing static {state_kind} SSM handoff metadata")
            if isinstance(state_static_metas, list):
                state_metas[state_kind] = [
                    {**meta, "block_ids": selected_slots} for meta in state_static_metas
                ]
            else:
                state_metas[state_kind] = {**state_static_metas, "block_ids": selected_slots}
        return state_metas

    def _capture_handoff_meta(
        self, request: "DynamicInferenceRequest", prepared: _PreparedHandoffMetadata | None
    ) -> None:
        """Attach prepared transfer metadata and retain the request's blocks.

        Side: prefill engine; pull and push transport paths.
        """

        if prepared is None:
            raise RuntimeError(
                f"No handoff metadata was prepared for completed request {request.request_id}"
            )

        rid = request.request_id
        block_ids = prepared.local_blocks
        kv_meta = prepared.kv_meta
        ssm_meta = prepared.ssm_meta

        self._pinned_handoff_blocks[rid] = list(block_ids)
        if prepared.local_ssm_slot is not None:
            self._pinned_handoff_ssm_slots[rid] = prepared.local_ssm_slot

        if isinstance(kv_meta, list):
            kv_meta = {"tp_metas": kv_meta}
        else:
            # TP=1 caches one static metadata dictionary for the engine. Keep
            # request-specific SSM metadata out of that shared object.
            kv_meta = dict(kv_meta)
        kv_meta["resume_tokens"] = prepared.resume_tokens
        if ssm_meta is not None:
            kv_meta["ssm"] = ssm_meta
        if (
            self._disagg_config is not None
            and self._disagg_config["kv_transport_backend"] == "nixl"
        ):
            # Native engines register static NIXL agent blobs once. Keep each
            # request payload limited to addresses, layouts, and block mappings.
            kv_meta = strip_registered_nixl_agent_metadata(kv_meta)

        request.disaggregated_params = {
            "request_id": rid,
            "block_ids": block_ids,
            "kv_meta": kv_meta,
        }
        logging.info(
            "DISAGG_PREFILL_HANDOFF request_id=%d pinned_blocks=%d ssm_blocks=%d",
            rid,
            len(block_ids),
            int(ssm_meta is not None),
        )

    def release_handoff_blocks(self, request_id: int) -> None:
        """Release blocks pinned by a previous do_kv_handoff completion.

        Side: prefill engine; pull and push transport paths.
        """
        block_ids = self._pinned_handoff_blocks.pop(request_id, None)
        ssm_slot = self._pinned_handoff_ssm_slots.pop(request_id, None)
        if not block_ids and ssm_slot is None:
            return
        released = self._release_pinned_handoff_blocks(block_ids or [])
        self._release_pinned_handoff_ssm_slot(ssm_slot)
        logging.info(
            "DISAGG_PREFILL_RELEASE request_id=%d released_blocks=%d ssm=%d",
            request_id,
            released,
            int(ssm_slot is not None),
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

    def _release_pinned_handoff_ssm_slot(self, ssm_slot: int | None) -> None:
        """Release a prefill live-state slot after its handoff ownership ends."""

        if ssm_slot is not None:
            self.context.mamba_metadata.free_slot(ssm_slot)

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

        ssm_meta = kv_meta.get("ssm") if isinstance(kv_meta, dict) else None
        local_has_ssm = bool(self._ssm_transfer_agents)
        if self.context.is_hybrid_model and not local_has_ssm:
            raise RuntimeError("Hybrid decode received a handoff before SSM transfer setup")
        if local_has_ssm and ssm_meta is None:
            raise RuntimeError(
                "Decode has SSM state transfer agents but the handoff contains no "
                "SSM metadata; prefill and decode must use the same hybrid model"
            )
        if local_has_ssm:
            if not isinstance(ssm_meta, dict):
                raise RuntimeError("SSM handoff metadata must be a mapping")
            missing_state_kinds = [kind for kind in _SSM_STATE_KINDS if kind not in ssm_meta]
            if missing_state_kinds:
                raise RuntimeError(
                    f"SSM handoff metadata is missing state kinds {missing_state_kinds}"
                )
        if not self.context.is_hybrid_model and ssm_meta is not None:
            raise RuntimeError("Transformer decode received SSM metadata from a hybrid prefill")
        if self._kv_transfer_agent is None:
            raise RuntimeError("KV handoff received without a transfer backend")

        resume_tokens = (
            [int(token) for token in kv_meta.get("resume_tokens", [])]
            if isinstance(kv_meta, dict)
            else []
        )
        expected_resume_tokens = self.context.num_speculative_tokens + 1
        if len(resume_tokens) != expected_resume_tokens:
            raise RuntimeError(
                "Decode-only handoff requires one sampled token plus the configured "
                f"MTP proposals: expected {expected_resume_tokens}, got {len(resume_tokens)}"
            )
        if sampling_params.return_log_probs or sampling_params.top_n_logprobs > 0:
            raise NotImplementedError(
                "Decode-only handoff does not yet transfer prompt or first-token log probabilities"
            )

        prompt_tensor = torch.tensor(prompt, dtype=torch.int64)
        hashes = compute_block_hashes_batched(prompt_tensor, self.context.block_size_tokens)
        num_blocks = transfer_block_count(kv_meta, src_block_ids)
        expected_blocks = (
            len(prompt) + self.context.block_size_tokens - 1
        ) // self.context.block_size_tokens
        if num_blocks != expected_blocks:
            raise RuntimeError(
                "Decode-only handoff requires every prompt KV block, including the partial tail: "
                f"expected {expected_blocks}, got {num_blocks}"
            )
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

        started = self._try_start_kv_handoff_import(handoff)
        if not started:
            self._deferred_kv_handoffs.append(handoff)
            logging.debug(
                "DISAGG_DECODE_CAPACITY_QUEUE request_id=%d queued=%d",
                request_id,
                len(self._deferred_kv_handoffs),
            )
        return future

    def _try_start_kv_handoff_import(self, handoff: DeferredKvHandoff) -> bool:
        """Reserve cache state and start one import, or return without mutation.

        Side: decode engine; pull and push transport paths.
        """

        allocator = self.context.kv_block_allocator
        cached_blocks = []
        if not self._kv_transfer_agent.is_push:
            cached_blocks = self._find_cached_handoff_prefix(handoff.hashes, handoff.num_blocks)

        num_blocks_to_import = handoff.num_blocks - len(cached_blocks)
        resume_tokens = (
            [int(token) for token in handoff.kv_meta.get("resume_tokens", [])]
            if isinstance(handoff.kv_meta, dict)
            else []
        )
        ssm_meta = handoff.kv_meta.get("ssm") if isinstance(handoff.kv_meta, dict) else None
        continuation_block_count = (
            additional_decode_blocks(
                len(handoff.prompt), len(resume_tokens), self.context.block_size_tokens
            )
            if resume_tokens
            else 0
        )
        if not self._handoff_capacity_available(
            num_blocks_to_import + continuation_block_count, cached_blocks
        ):
            return False
        if ssm_meta and self.context.mamba_metadata.mamba_state_free_slot_count < 1:
            return False

        allocator.retain_memory_blocks(cached_blocks)
        allocated_blocks_tensor = allocator.allocate_memory_blocks(
            num_blocks_to_import + continuation_block_count
        )
        if allocated_blocks_tensor is None:
            if cached_blocks:
                allocator.release_memory_blocks(
                    torch.tensor(cached_blocks, dtype=torch.int32, device="cpu")
                )
            raise RuntimeError("KV allocator capacity changed during handoff admission")
        allocated_blocks = [int(block) for block in allocated_blocks_tensor.tolist()]
        imported_blocks = allocated_blocks[:num_blocks_to_import]
        continuation_blocks = allocated_blocks[num_blocks_to_import:]
        local_blocks = cached_blocks + imported_blocks
        owned_blocks_tensor = torch.tensor(
            local_blocks + continuation_blocks, dtype=torch.int32, device="cpu"
        )

        handle = None
        start_error = None
        ssm_import = None
        if ssm_meta:
            try:
                ssm_import = self._reserve_ssm_handoff_import()
            except Exception:
                allocator.release_memory_blocks(owned_blocks_tensor)
                raise

        try:
            transfer_meta, transfer_src_blocks = drop_transfer_prefix_blocks(
                handoff.kv_meta, handoff.src_block_ids, len(cached_blocks)
            )
            handle = self._kv_transfer_agent.begin_pull_blocks(
                transfer_meta, transfer_src_blocks, imported_blocks
            )
            if ssm_import is not None:
                self._start_ssm_handoff_import(handoff.request_id, ssm_meta, ssm_import)
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
            ssm=ssm_import,
            resume_tokens=resume_tokens,
            continuation_blocks=continuation_blocks,
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

        started_count = 0
        while self._deferred_kv_handoffs:
            handoff = self._deferred_kv_handoffs[0]
            started = self._try_start_kv_handoff_import(handoff)
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
        """Return this decode import's active KV and SSM transfer handles.

        Side: decode engine; pull and push transport paths.
        """

        handles = [pending.handle]
        if pending.ssm is not None:
            handles.extend(pending.ssm.handles)
        return [handle for handle in handles if handle is not None]

    def _validate_decode_ready_handoff(self, pending: PendingKvImport) -> None:
        """Validate that transferred state can start decode without prompt execution."""

        expected_tokens = self.context.num_speculative_tokens + 1
        if len(pending.resume_tokens) != expected_tokens:
            raise RuntimeError(
                "Decode-only handoff is missing its sampled token or MTP proposals: "
                f"expected {expected_tokens}, got {len(pending.resume_tokens)}"
            )
        if self.context.is_hybrid_model:
            if pending.ssm is None:
                raise RuntimeError("Hybrid decode-only handoff is missing transferred SSM state")

    def _finalize_kv_handoff_import(self, pending: PendingKvImport) -> None:
        """Register transferred blocks and admit the decode request.

        Side: decode engine; pull and push transport paths.
        """

        allocator = self.context.kv_block_allocator
        local_blocks = pending.local_blocks
        cached_prefix_block_count = pending.cached_prefix_block_count
        registration_end = min(len(local_blocks), len(pending.hashes))
        num_hashes_to_register = registration_end - cached_prefix_block_count

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

        request_future = self.add_request(
            pending.request_id,
            pending.prompt,
            pending.sampling_params,
            precomputed_block_hashes=(
                pending.hashes[:registration_end] if registration_end > 0 else None
            ),
        )

        request = self.get_request(pending.request_id)
        if request.status == Status.FAILED:
            # add_request() reports validation failures through its own future.
            self._release_pending_kv_import(pending)
        else:
            queued_request_id = self.waiting_request_ids[0]
            if queued_request_id != pending.request_id:
                raise RuntimeError(
                    "Decode-only admission expected the imported request at the waiting queue "
                    f"head: expected {pending.request_id}, got {queued_request_id}"
                )
            self.waiting_request_ids.popleft()
            self._validate_decode_ready_handoff(pending)
            first_token = pending.resume_tokens[0]
            if request.sampling_params.num_tokens_to_generate > 0:
                request.generated_tokens.append(first_token)
                if self.track_generated_token_events:
                    first_token_event = request.add_event_generated_token(first_token)
                else:
                    first_token_event = DynamicInferenceEvent(
                        type=DynamicInferenceEventType.GENERATED_TOKEN,
                        payload={"token_id": first_token},
                    )
                request.ttft = first_token_event.timestamp - request.event_add_engine.timestamp

                stop_word_hit = False
                if request.stop_word_ids:
                    stop_word_hit, _ = self._check_stop_words_for_request_post_append(request)
                if first_token == request.sampling_params.termination_id or stop_word_hit:
                    request.sampling_params.num_tokens_to_generate = len(request.generated_tokens)

            request.num_cached_tokens = len(pending.prompt)
            if len(request.generated_tokens) >= request.sampling_params.num_tokens_to_generate:
                self._release_pending_kv_import(pending)
                self._complete_handoff_request_without_forward(pending.request_id)
            else:
                admit_prefilled_decode(
                    self.context,
                    request,
                    local_blocks,
                    pending.continuation_blocks,
                    pending.resume_tokens,
                    ssm_state_idx=(pending.ssm.live_slot if pending.ssm is not None else None),
                )
                pending.ssm = None
                pending.local_blocks = []
                pending.continuation_blocks = []
                if self.use_coordinator and self.is_mp_coordinator:
                    self._try_send_streaming_partials()
            self._notify_kv_read_done(pending.request_id)

        def _relay_result(src: asyncio.Future) -> None:
            """Forward decode completion to the handoff future."""

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

    def _complete_handoff_request_without_forward(self, request_id: int) -> None:
        """Complete an imported request whose transferred token already satisfies it."""

        request_entry = self.requests.pop(request_id)
        request = request_entry.record[-1]
        request.generated_length = len(request.generated_tokens)
        request.status = Status.COMPLETED
        request.add_event_finish()
        self.finished_request_count += 1

        if self.use_coordinator and self.is_mp_coordinator:
            self._send_request_records_to_coordinator([request_entry.record])
            self._partial_emit_lengths.pop(request_id, None)
        elif not self.use_coordinator:
            if request.prompt is None:
                request.prompt = self.controller.detokenize(
                    self.controller.tokenizer, request.prompt_tokens.tolist(), remove_EOD=False
                )
            request.generated_text = self.controller.detokenize(
                self.controller.tokenizer,
                request.generated_tokens,
                remove_EOD=not request.sampling_params.detokenize_stop_sequence,
            )

        request_entry.future.set_result(request_entry.record)

    def _release_pending_kv_import(self, pending: PendingKvImport) -> None:
        """Release storage owned by an unadmitted decode import.

        Side: decode engine; pull and push transport paths.
        """

        owned_blocks = pending.local_blocks + pending.continuation_blocks
        if owned_blocks:
            block_tensor = torch.tensor(owned_blocks, dtype=torch.int32, device="cpu")
            self.context.kv_block_allocator.release_memory_blocks(block_tensor)
        pending.local_blocks = []
        pending.continuation_blocks = []
        if pending.ssm is not None:
            self.context.mamba_metadata.free_slot(pending.ssm.live_slot)
            pending.ssm = None

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
        """Progress imports without mutating the active decode batch.

        Side: decode engine; pull and push transport paths.
        """

        self._drain_deferred_kv_handoffs()
        if not self._pending_kv_imports:
            return 0
        self._report_completed_kv_imports()
        return int(self.has_admittable_kv_import)

    def _admit_pending_kv_imports(self) -> int:
        """Handle completed imports at an engine scheduling admission point.

        Side: decode engine; pull and push transport paths.
        """

        ready = 0
        remaining = deque()
        while self._pending_kv_imports:
            pending = self._pending_kv_imports.popleft()
            failed = self._handoff_completion_notifications.get(pending.request_id)
            if failed is None:
                remaining.append(pending)
                continue
            try:
                if failed:
                    raise pending.local_error or RuntimeError(
                        "KV handoff transfer failed on a model-parallel peer"
                    )
                self._validate_decode_ready_handoff(pending)
                if not can_admit_prefilled_decode(self.context, len(pending.resume_tokens)):
                    remaining.append(pending)
                    continue
                self._handoff_completion_notifications.pop(pending.request_id)
                self._finalize_kv_handoff_import(pending)
                ready += 1
            except Exception as exc:
                self._handoff_completion_notifications.pop(pending.request_id, None)
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
                    self._quarantined_kv_imports.append(pending)
                    logging.error(
                        "Quarantining request %d cache storage after an incomplete handoff",
                        pending.request_id,
                    )
                if not pending.future.done():
                    pending.future.set_exception(exc)
                logging.exception("DISAGG_DECODE_PULL_FAILED request_id=%d", pending.request_id)
                if failed:
                    continue
                remaining.extend(self._pending_kv_imports)
                self._pending_kv_imports = remaining
                raise
        self._pending_kv_imports = remaining
        if ready:
            self._loop.call_soon_threadsafe(
                self._loop.create_task, self._notify_cond_for_new_request()
            )
        return ready

    def _reserve_ssm_handoff_import(self) -> PendingSSMImport:
        """Reserve the live request slot that receives exact SSM state."""

        if not self._ssm_transfer_agents:
            raise RuntimeError(
                "Received SSM handoff state but this decode engine has no "
                "SSM transfer agents. Ensure KV transfer was initialized on a "
                "hybrid decode engine."
            )
        live_slot = self.context.mamba_metadata.allocate_slot()
        if live_slot is None:
            raise RuntimeError("Live SSM slot capacity changed during handoff admission")
        return PendingSSMImport(handles=[], live_slot=int(live_slot))

    def _start_ssm_handoff_import(
        self, request_id: int, ssm_meta: dict, pending: PendingSSMImport
    ) -> None:
        """Post transfers into slots already reserved for one handoff."""

        pending.handles.clear()
        for state_kind in _SSM_STATE_KINDS:
            agent = self._ssm_transfer_agents[state_kind]
            pending.handles.append(
                agent.begin_pull_blocks(ssm_meta[state_kind], [], [pending.live_slot])
            )
        logging.debug("DISAGG_DECODE_SSM_IMPORT_SUBMIT request_id=%d ssm_blocks=%d", request_id, 1)
