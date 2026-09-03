# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Coordinator-side orchestration for native prefill/decode disaggregation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import msgpack

from megatron.core.inference.config import PrefixCachingCoordinatorPolicy
from megatron.core.inference.disaggregation.coordinator_scheduler import (
    DECODE,
    PREFILL,
    DisaggCoordinatorScheduler,
)
from megatron.core.inference.disaggregation.handoff_wire_protocol import (
    make_submit_request_with_kv_message,
    restore_registered_nixl_agent_metadata,
)
from megatron.core.inference.headers import Headers


@dataclass
class _RequestState:
    prompt: Any
    sampling_params: dict
    block_hashes: list[int]


def _instance_transfer_signature(instance_meta: Any) -> tuple:
    """Return model-wide state geometry shared by compatible instances."""

    if not isinstance(instance_meta, list) or not instance_meta:
        raise ValueError("state transfer metadata must be a non-empty list")
    if not all(isinstance(entry, dict) for entry in instance_meta):
        raise ValueError("every model-parallel transfer descriptor must be a dictionary")

    kv_fields = (
        "tokens_per_block",
        "element_size",
        "head_dim",
        "num_layers_global",
        "num_kv_heads_global",
    )
    kv_signatures = {tuple(entry.get(field) for field in kv_fields) for entry in instance_meta}
    if len(kv_signatures) != 1 or None in next(iter(kv_signatures)):
        raise ValueError("model-parallel ranks advertise inconsistent KV transfer geometry")

    ssm_entries = [entry.get("ssm") or {} for entry in instance_meta]
    if not all(isinstance(entry, dict) for entry in ssm_entries):
        raise ValueError("SSM transfer metadata must be a dictionary")
    state_kinds = {state_kind for entry in ssm_entries for state_kind in entry}
    ssm_signatures = []
    for state_kind in sorted(state_kinds):
        signatures = set()
        layer_ranges = set()
        for entry in ssm_entries:
            state_meta = entry.get(state_kind)
            layout = state_meta.get("ssm_layout") if isinstance(state_meta, dict) else None
            dims = layout.get("dims") if isinstance(layout, dict) else None
            if not isinstance(dims, dict):
                raise ValueError(f"model-parallel metadata is missing {state_kind} SSM geometry")
            signatures.add((state_meta.get("element_size"), tuple(sorted(dims.items()))))
            layer_ranges.add((int(layout["layer_start"]), int(layout["num_layers"])))
        if len(signatures) != 1 or next(iter(signatures))[0] is None:
            raise ValueError(
                f"model-parallel ranks advertise inconsistent {state_kind} SSM geometry"
            )
        layer_count = 0
        for layer_start, local_count in sorted(layer_ranges):
            if layer_start != layer_count:
                raise ValueError(f"model-parallel {state_kind} SSM layers are not contiguous")
            layer_count = max(layer_count, layer_start + local_count)
        ssm_signatures.append((state_kind, next(iter(signatures)), layer_count))
    return (next(iter(kv_signatures)), tuple(ssm_signatures))


class DisaggCoordinatorRuntime:
    """Own the state and two-hop routing used only by a disaggregated coordinator."""

    def __init__(self, coordinator: Any) -> None:
        self.coordinator = coordinator
        self.scheduler = DisaggCoordinatorScheduler()
        self.hop1_request_ids: set[int] = set()
        self.requests: dict[int, _RequestState] = {}
        self.engine_role: dict[Any, str] = {}
        self.engine_transport: dict[Any, str] = {}
        self.engine_metadata: dict[Any, Any] = {}
        self.terminating_request_ids: set[int] = set()
        self._transfer_signature: tuple | None = None

    def register_engine(self, identity, role: str, transport: str, instance_meta: Any) -> None:
        """Register one prefill or decode instance and its transfer metadata."""

        previous_role = self.engine_role.get(identity)
        previous_transport = self.engine_transport.get(identity)
        if previous_role is not None:
            logging.warning("Coordinator: replacing reconnected engine %r", identity)
            self.coordinator._remove_engine(identity)
            if previous_role != role or previous_transport != transport:
                raise ValueError(
                    f"engine {identity!r} cannot change disaggregated role or transport"
                )
        if self.engine_transport and transport not in self.engine_transport.values():
            raise ValueError("prefill and decode engines must use the same state transport")
        signature = _instance_transfer_signature(instance_meta)
        if self._transfer_signature is not None and signature != self._transfer_signature:
            raise ValueError("prefill and decode engines have incompatible transfer geometry")

        capacity = self.scheduler.register_engine(identity, role, instance_meta)
        if self._transfer_signature is None:
            self._transfer_signature = signature
        coordinator = self.coordinator
        if identity not in coordinator.identities_of_data_parallel_ranks:
            coordinator.identities_of_data_parallel_ranks.append(identity)
            coordinator._register_rank_identity(identity)
        self.engine_role[identity] = role
        self.engine_transport[identity] = transport
        self.engine_metadata[identity] = instance_meta
        logging.info(
            "Coordinator: registered %s engine %s with live SSM capacity=%s",
            role,
            identity,
            capacity,
        )

    def remove_engine(self, identity) -> None:
        """Remove an engine and discard work that can no longer complete."""

        for request_id in self.scheduler.pop_queued_for_engine(identity):
            self.drop_request(
                request_id, f"queued on removed engine {identity!r}", source_safe=True
            )
        affected = set(self.scheduler.requests_involving(identity))
        affected.update(self.scheduler.reservations_for_engine(identity))
        for request_id in affected:
            self.drop_request(
                request_id,
                f"engine {identity!r} removed",
                source_safe=(
                    self.scheduler.reserved_engine(PREFILL, request_id) is None
                    or self.scheduler.assigned_engine(DECODE, request_id) != identity
                ),
            )
        self.engine_role.pop(identity, None)
        self.engine_transport.pop(identity, None)
        self.engine_metadata.pop(identity, None)
        self.scheduler.remove_engine(identity)
        if not self.engine_role:
            self._transfer_signature = None

    def _request_hashes(self, prompt: Any) -> list[int]:
        hashes = self.coordinator.compute_request_hashes(prompt)
        if (
            self.coordinator.prefix_caching_coordinator_policy
            == PrefixCachingCoordinatorPolicy.FIRST_PREFIX_BLOCK
        ):
            return hashes[:1]
        return hashes

    def _make_routing_score(self, block_hashes: list[int], role: str):
        """Build a role-local routing score, computing prefix affinity once.

        Decode affinity avoids allocating and transferring prefix blocks that
        are already cached on the selected decode engine.
        """

        policy = self.coordinator.prefix_caching_coordinator_policy
        if block_hashes and policy != PrefixCachingCoordinatorPolicy.LOAD_BALANCED:
            matches, recencies = self.coordinator._match_vector(block_hashes)
        else:
            matches = recencies = None

        def score(identity) -> tuple:
            free_capacity = self.scheduler.available_fraction(identity, role)
            recency = 0.0
            if matches is None:
                combined = free_capacity
            else:
                rank_index = self.coordinator.identity_to_rank_index[identity]
                match = float(matches[rank_index])
                recency = float(recencies[rank_index])
                alpha = self.coordinator.prefix_caching_routing_alpha
                combined = alpha * match + (1.0 - alpha) * free_capacity
            load = (
                self.scheduler.prefill_load(identity)
                if role == PREFILL
                else self.scheduler.decode_load(identity)
            )
            return (-combined, -recency, *load)

        return score

    def _record_hash_assignment(self, identity, block_hashes: list[int]) -> None:
        if block_hashes:
            self.coordinator._update_rank_hashes(identity, block_hashes)

    def route_submit(self, request_id: int, prompt: Any, sampling_params: dict) -> None:
        """Reserve capacity and send a client request to a prefill instance."""

        block_hashes = self._request_hashes(prompt)
        self.requests[request_id] = _RequestState(prompt, sampling_params, block_hashes)
        if sampling_params.get("return_log_probs") and not sampling_params.get(
            "skip_prompt_log_probs", True
        ):
            self.drop_request(
                request_id,
                "prompt log probabilities are not supported by native disaggregation",
                source_safe=True,
            )
            return
        try:
            prefill_id = self.scheduler.select_engine(
                PREFILL, request_id, self._make_routing_score(block_hashes, PREFILL)
            )
        except RuntimeError as error:
            self.drop_request(request_id, f"cannot route to prefill: {error}", source_safe=True)
            return

        capacity = self.scheduler.capacity(prefill_id)
        slot_cost = self.scheduler.prefill_slot_cost(prefill_id)
        if not self.scheduler.can_ever_fit(prefill_id, slot_cost):
            self.drop_request(
                request_id,
                f"request requires {slot_cost} live SSM slots, but prefill "
                f"engine {prefill_id!r} advertises only {capacity}",
                source_safe=True,
            )
            return
        if self.scheduler.has_queued_prefill(prefill_id) or not self.scheduler.try_reserve_prefill(
            prefill_id, request_id, slot_cost
        ):
            self.scheduler.enqueue_prefill(prefill_id, request_id, slot_cost)
            return
        self._submit_prefill(prefill_id, request_id, prompt, sampling_params)

    def _submit_prefill(
        self, prefill_id, request_id: int, prompt: Any, sampling_params: dict
    ) -> None:
        """Submit a request whose prefill capacity has already been reserved."""

        self.hop1_request_ids.add(request_id)
        prefill_params = dict(sampling_params)
        prefill_params["do_kv_handoff"] = True
        prefill_params["num_tokens_to_generate"] = 0
        prefill_params["skip_prompt_log_probs"] = True
        if "num_tokens_total" in prefill_params:
            prefill_params["num_tokens_total"] = None
        if self._send(prefill_id, Headers.SUBMIT_REQUEST, request_id, prompt, prefill_params):
            self._record_hash_assignment(prefill_id, self.requests[request_id].block_hashes)

    def _drain_prefill_queue(self, prefill_id) -> None:
        while True:
            request = self.scheduler.pop_next_prefill(prefill_id)
            if request is None:
                return
            state = self.requests[request.request_id]
            assert state.prompt is not None, "completed request remained in the prefill queue"
            self._submit_prefill(
                prefill_id, request.request_id, state.prompt, state.sampling_params
            )

    def handle_prefill_done(self, request_id: int, finished_request: dict) -> None:
        """Route a completed prefill handoff to a decode instance."""

        self.hop1_request_ids.discard(request_id)
        if request_id in self.terminating_request_ids:
            self._finish_abort(request_id)
            return
        request_state = self.requests.get(request_id)
        handoff = finished_request.get("disaggregated_params")
        if request_state is None or not isinstance(handoff, dict):
            self.drop_request(
                request_id, "prefill reply carried no handoff metadata", source_safe=True
            )
            return
        kv_meta = handoff.get("kv_meta")
        block_ids = handoff.get("block_ids")
        if (
            not isinstance(kv_meta, dict)
            or not isinstance(block_ids, list)
            or any(type(block_id) is not int for block_id in block_ids)
        ):
            self.drop_request(
                request_id, "prefill reply carried invalid handoff metadata", source_safe=True
            )
            return
        try:
            decode_id = self.scheduler.select_engine(
                DECODE, request_id, self._make_routing_score(request_state.block_hashes, DECODE)
            )
        except RuntimeError as error:
            self.drop_request(request_id, f"cannot route to decode: {error}", source_safe=True)
            return

        prefill_id = self.scheduler.reserved_engine(PREFILL, request_id)
        if prefill_id is not None and self.engine_transport.get(prefill_id) == "nixl":
            try:
                kv_meta = restore_registered_nixl_agent_metadata(
                    kv_meta, self.engine_metadata[prefill_id]
                )
            except ValueError as error:
                self.drop_request(
                    request_id, f"invalid NIXL handoff metadata: {error}", source_safe=True
                )
                return

        payload = msgpack.packb(
            make_submit_request_with_kv_message(
                Headers.SUBMIT_REQUEST_WITH_KV.value,
                request_id,
                request_state.prompt,
                request_state.sampling_params,
                kv_meta,
                block_ids,
            ),
            use_bin_type=True,
        )
        # The serialized handoff owns these values until decode receives it.
        request_state.prompt = None
        request_state.sampling_params = {}
        slot_cost = self.scheduler.slot_cost_from_handoff(handoff)
        capacity = self.scheduler.capacity(decode_id)
        if not self.scheduler.can_ever_fit(decode_id, slot_cost):
            self.drop_request(
                request_id,
                f"SSM handoff requires {slot_cost} live slots, but decode "
                f"engine {decode_id!r} advertises only {capacity}",
                source_safe=True,
            )
            return
        if self.scheduler.has_queued(decode_id) or not self.scheduler.try_reserve(
            decode_id, request_id, slot_cost
        ):
            self.scheduler.enqueue(decode_id, request_id, payload, slot_cost)
            return
        self._send_decode_handoff(decode_id, request_id, payload)

    def _send_decode_handoff(self, decode_id, request_id: int, payload: bytes) -> bool:
        """Send a handoff; an unreachable engine is removed with its assigned work."""

        sent = self.coordinator._send_to_engine(decode_id, payload, remove_unreachable=False)
        if sent:
            request = self.requests[request_id]
            self._record_hash_assignment(decode_id, request.block_hashes)
            request.block_hashes = []
        else:
            # The handoff was not delivered, so its prefill source is safe.
            self._release_prefill(request_id)
            self.coordinator._remove_engine(decode_id)
        return sent

    def handle_kv_transfer_ready(
        self, sender_identity, request_id: int, cached_prefix_blocks: int
    ) -> None:
        """Start a two-sided send after the assigned decode commits destinations."""

        if self.scheduler.assigned_engine(DECODE, request_id) != sender_identity:
            logging.warning(
                "Coordinator: ignoring KV_TRANSFER_READY for request %d from %r",
                request_id,
                sender_identity,
            )
            return
        prefill_id = self.scheduler.reserved_engine(PREFILL, request_id)
        if prefill_id is None or self.engine_transport.get(prefill_id) != "nccl":
            self.drop_request(
                request_id, "unexpected KV transfer readiness notification", source_safe=False
            )
            return
        self._send(
            prefill_id,
            Headers.SEND_KV,
            request_id,
            self.engine_metadata[sender_identity],
            cached_prefix_blocks,
        )

    def _drain_decode_queue(self, decode_id) -> None:
        while True:
            handoff = self.scheduler.pop_next_admissible(decode_id)
            if handoff is None:
                return
            if not self._send_decode_handoff(decode_id, handoff.request_id, handoff.payload):
                return

    def _release_prefill(self, request_id: int) -> None:
        """Release prefill pins after decode has imported the handoff."""

        prefill_id = self.scheduler.release_prefill(request_id)
        if prefill_id is None:
            return
        self.scheduler.forget_assignment(PREFILL, request_id)
        sent = self._send(prefill_id, Headers.RELEASE_KV, request_id)
        if sent:
            self._drain_prefill_queue(prefill_id)

    def handle_kv_read_done(self, sender_identity, request_id: int) -> None:
        """Release source state after the assigned decode reports a completed read."""

        if self.scheduler.reserved_engine(PREFILL, request_id) is None:
            return
        assigned_decode = self.scheduler.assigned_engine(DECODE, request_id)
        if assigned_decode != sender_identity:
            logging.warning(
                "Coordinator: ignoring KV_READ_DONE for request %d from %r; assigned decode is %r",
                request_id,
                sender_identity,
                assigned_decode,
            )
            return
        self._release_prefill(request_id)
        if request_id not in self.requests:
            self.scheduler.forget_request(request_id)

    def handle_decode_done(self, request_id: int) -> None:
        """Release all coordinator state after the decode reply is returned."""

        decode_id = self.scheduler.release_decode(request_id)
        if decode_id is not None:
            self._drain_decode_queue(decode_id)
        self._release_prefill(request_id)
        self.scheduler.forget_request(request_id)
        self.requests.pop(request_id, None)

    def drop_request(self, request_id: int, reason: str, *, source_safe: bool) -> None:
        """Fail a client request and discard its coordinator state."""

        logging.error("Coordinator: dropping disagg request %s: %s", request_id, reason)
        coordinator = self.coordinator
        client_identity = coordinator.request_id_to_client_id.get(request_id)
        client_request_id = coordinator.request_id_to_client_request_id.get(request_id)
        if client_identity is not None and client_request_id is not None:
            coordinator.router_socket.send_multipart(
                [
                    client_identity,
                    msgpack.packb(
                        [Headers.REQUEST_ERROR.value, client_request_id, reason, source_safe],
                        use_bin_type=True,
                    ),
                ]
            )
        self.scheduler.remove_queued(request_id)
        decode_id = self.scheduler.release_decode(request_id)
        if decode_id is not None:
            self._drain_decode_queue(decode_id)
        self.requests.pop(request_id, None)
        self.hop1_request_ids.discard(request_id)
        if source_safe:
            self.terminating_request_ids.discard(request_id)
            self._release_prefill(request_id)
            self.scheduler.forget_request(request_id)
            coordinator.request_id_to_client_id.pop(request_id, None)
            client_request_id = coordinator.request_id_to_client_request_id.pop(request_id, None)
            if client_identity is not None and client_request_id is not None:
                coordinator.client_request_to_request_id.pop(
                    (client_identity, client_request_id), None
                )
        else:
            self.terminating_request_ids.add(request_id)

    def handle_engine_failure(
        self, request_id: int, reason: str, *, source_safe: bool = False
    ) -> None:
        """Fail a decode handoff without releasing a source still in use."""

        self.drop_request(request_id, reason, source_safe=source_safe)

    def abort_request(self, request_id: int) -> None:
        """Cancel queued work or forward cancellation to its active engine."""

        self.terminating_request_ids.add(request_id)
        if self.scheduler.remove_queued(request_id):
            self._finish_abort(request_id)
            return
        decode_id = self.scheduler.assigned_engine(DECODE, request_id)
        if decode_id is not None:
            self._send(decode_id, Headers.ABORT_REQUEST, request_id)
            return
        prefill_id = self.scheduler.reserved_engine(PREFILL, request_id)
        if prefill_id is not None:
            self._send(prefill_id, Headers.ABORT_REQUEST, request_id)
            return
        self._finish_abort(request_id)

    def handle_engine_aborted(self, request_id: int, *, source_safe: bool) -> None:
        """Finish a cancellation after the engine reports transport safety."""

        if source_safe and request_id in self.terminating_request_ids:
            self._finish_abort(request_id)

    def _finish_abort(self, request_id: int) -> None:
        coordinator = self.coordinator
        client_identity = coordinator.request_id_to_client_id.get(request_id)
        client_request_id = coordinator.request_id_to_client_request_id.get(request_id)
        self._release_prefill(request_id)
        decode_id = self.scheduler.release_decode(request_id)
        if decode_id is not None:
            self._drain_decode_queue(decode_id)
        self.scheduler.forget_request(request_id)
        self.requests.pop(request_id, None)
        self.hop1_request_ids.discard(request_id)
        self.terminating_request_ids.discard(request_id)
        coordinator.request_id_to_client_id.pop(request_id, None)
        coordinator.request_id_to_client_request_id.pop(request_id, None)
        if client_identity is not None and client_request_id is not None:
            coordinator.client_request_to_request_id.pop((client_identity, client_request_id), None)
            coordinator.router_socket.send_multipart(
                [
                    client_identity,
                    msgpack.packb(
                        [Headers.REQUEST_ABORTED.value, client_request_id, True], use_bin_type=True
                    ),
                ]
            )

    def _send(self, identity, header, *parts) -> bool:
        payload = msgpack.packb([header.value, *parts], use_bin_type=True)
        return self.coordinator._send_to_engine(identity, payload)
