# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Coordinator-side orchestration for native prefill/decode disaggregation."""

from __future__ import annotations

import logging
from typing import Any

import msgpack

from megatron.core.inference.disaggregation.coordinator_flow_control import DisaggStateFlowControl
from megatron.core.inference.disaggregation.coordinator_routing import make_disagg_router
from megatron.core.inference.disaggregation.handoff_wire_protocol import (
    make_submit_request_with_kv_message,
    restore_registered_nixl_agent_metadata,
)
from megatron.core.inference.headers import Headers


class DisaggCoordinatorRuntime:
    """Own the state and two-hop routing used only by a disaggregated coordinator."""

    def __init__(self, coordinator: Any, router_name: str, max_outstanding: int) -> None:
        self.coordinator = coordinator
        self.router = make_disagg_router(router_name)
        self.flow = DisaggStateFlowControl()
        self.max_outstanding = max_outstanding
        self.hop1_request_ids: set[int] = set()
        self.request_metadata: dict[int, tuple[Any, dict]] = {}
        self.prefill_by_request: dict[int, Any] = {}
        self.engine_transport: dict[Any, str] = {}
        self.engine_metadata: dict[Any, Any] = {}

    def register_engine(self, identity, role: str, transport: str, instance_meta: Any) -> None:
        """Register one prefill or decode instance and its transfer metadata."""

        capacity = self.flow.register_engine(identity, role, instance_meta)
        coordinator = self.coordinator
        if identity not in coordinator.identities_of_data_parallel_ranks:
            coordinator.identities_of_data_parallel_ranks.append(identity)
            coordinator._register_rank_identity(identity)
        self.router.register(identity, role)
        self.engine_transport[identity] = transport
        self.engine_metadata[identity] = instance_meta
        logging.info(
            "Coordinator: registered %s engine %s with SSM cache capacity=%s",
            role,
            identity,
            capacity,
        )

    def remove_engine(self, identity) -> None:
        """Remove an engine and discard work that can no longer complete."""

        for request_id in self.flow.pop_queued_for_engine(identity):
            self.drop_request(request_id, f"queued on removed engine {identity!r}")
        affected = set(self.router.requests_involving(identity))
        affected.update(self.flow.reservations_for_engine(identity))
        for request_id in affected:
            self.drop_request(request_id, f"engine {identity!r} removed")
        self.router.remove(identity)
        self.engine_transport.pop(identity, None)
        self.engine_metadata.pop(identity, None)
        self.flow.remove_engine(identity)

    def route_submit(self, request_id: int, prompt: Any, sampling_params: dict) -> None:
        """Reserve capacity and send a client request to a prefill instance."""

        self.request_metadata[request_id] = (prompt, sampling_params)
        try:
            prefill_id = self.router.route_submit(request_id)
        except RuntimeError as error:
            self.drop_request(request_id, f"cannot route to prefill: {error}")
            return

        capacity = self.flow.capacity(prefill_id)
        slot_cost = self.flow.prefill_slot_cost(prefill_id)
        if not self.flow.can_ever_fit(prefill_id, slot_cost):
            self.drop_request(
                request_id,
                f"request requires {slot_cost} durable SSM slots, but prefill "
                f"engine {prefill_id!r} advertises only {capacity}",
            )
            return
        if self.flow.has_queued_prefill(prefill_id) or not self.flow.try_reserve_prefill(
            prefill_id, request_id, slot_cost, self.max_outstanding
        ):
            self.flow.enqueue_prefill(prefill_id, request_id, prompt, sampling_params, slot_cost)
            return
        self._submit_prefill(prefill_id, request_id, prompt, sampling_params)

    def _submit_prefill(
        self, prefill_id, request_id: int, prompt: Any, sampling_params: dict
    ) -> None:
        """Submit a request whose prefill capacity has already been reserved."""

        self.prefill_by_request[request_id] = prefill_id
        self.hop1_request_ids.add(request_id)
        prefill_params = dict(sampling_params)
        prefill_params["do_kv_handoff"] = True
        prefill_params["num_tokens_to_generate"] = 0
        prefill_params["skip_prompt_log_probs"] = True
        if "num_tokens_total" in prefill_params:
            prefill_params["num_tokens_total"] = None
        self._send(prefill_id, Headers.SUBMIT_REQUEST, request_id, prompt, prefill_params)

    def _drain_prefill_queue(self, prefill_id) -> None:
        while True:
            request = self.flow.pop_next_prefill(prefill_id, self.max_outstanding)
            if request is None:
                return
            self._submit_prefill(
                prefill_id, request.request_id, request.prompt, request.sampling_params
            )

    def handle_prefill_done(self, request_id: int, finished_request: dict) -> None:
        """Route a completed prefill handoff to a decode instance."""

        self.hop1_request_ids.discard(request_id)
        request_meta = self.request_metadata.get(request_id)
        handoff = finished_request.get("disaggregated_params")
        if request_meta is None or not handoff:
            self.drop_request(request_id, "prefill reply carried no handoff metadata")
            return
        prompt, sampling_params = request_meta
        try:
            _, decode_id = self.router.route_prefill_done(request_id)
        except RuntimeError as error:
            self.drop_request(request_id, f"cannot route to decode: {error}")
            return

        kv_meta = handoff["kv_meta"]
        prefill_id = self.prefill_by_request.get(request_id)
        if prefill_id is not None and self.engine_transport.get(prefill_id) == "nixl":
            try:
                kv_meta = restore_registered_nixl_agent_metadata(
                    kv_meta, self.engine_metadata[prefill_id]
                )
            except ValueError as error:
                self.drop_request(request_id, f"invalid NIXL handoff metadata: {error}")
                return

        payload = msgpack.packb(
            make_submit_request_with_kv_message(
                Headers.SUBMIT_REQUEST_WITH_KV.value,
                request_id,
                prompt,
                sampling_params,
                kv_meta,
                handoff["block_ids"],
            ),
            use_bin_type=True,
        )
        slot_cost = self.flow.slot_cost_from_handoff(handoff)
        capacity = self.flow.capacity(decode_id)
        if not self.flow.can_ever_fit(decode_id, slot_cost):
            self.drop_request(
                request_id,
                f"SSM handoff requires {slot_cost} durable slots, but decode "
                f"engine {decode_id!r} advertises only {capacity}",
            )
            return
        if self.flow.has_queued(decode_id) or not self.flow.try_reserve(
            decode_id, request_id, slot_cost
        ):
            self.flow.enqueue(decode_id, request_id, payload, slot_cost)
            return
        self._send_decode_handoff(decode_id, request_id, payload)

    def _send_decode_handoff(self, decode_id, request_id: int, payload: bytes) -> bool:
        if not self.coordinator._send_to_engine(decode_id, payload):
            return False
        prefill_id = self.prefill_by_request.get(request_id)
        if prefill_id is not None and self.engine_transport.get(prefill_id) == "nccl":
            self._send(prefill_id, Headers.SEND_KV, request_id, self.engine_metadata[decode_id])
        return True

    def _drain_decode_queue(self, decode_id) -> None:
        while True:
            handoff = self.flow.pop_next_admissible(decode_id)
            if handoff is None:
                return
            if not self._send_decode_handoff(decode_id, handoff.request_id, handoff.payload):
                return

    def handle_kv_read_done(self, request_id: int) -> None:
        """Release prefill pins after decode has imported the handoff."""

        prefill_id = self.prefill_by_request.pop(request_id, None)
        if prefill_id is None:
            return
        self._send(prefill_id, Headers.RELEASE_KV, request_id)
        released_prefill = self.flow.release_prefill(request_id)
        if released_prefill is not None:
            self._drain_prefill_queue(released_prefill)

    def handle_decode_done(self, request_id: int) -> None:
        """Release all coordinator state after the decode reply is returned."""

        decode_id = self.flow.release_decode(request_id)
        if decode_id is not None:
            self._drain_decode_queue(decode_id)
        self.router.forget(request_id)
        self.request_metadata.pop(request_id, None)
        self.handle_kv_read_done(request_id)

    def drop_request(self, request_id: int, reason: str) -> None:
        """Discard coordinator state for a request that cannot complete."""

        logging.error("Coordinator: dropping disagg request %s: %s", request_id, reason)
        self.flow.remove_queued(request_id)
        decode_id = self.flow.release_decode(request_id)
        if decode_id is not None:
            self._drain_decode_queue(decode_id)
        self.router.forget(request_id)
        self.request_metadata.pop(request_id, None)
        self.hop1_request_ids.discard(request_id)
        self.handle_kv_read_done(request_id)

        coordinator = self.coordinator
        coordinator.request_id_to_client_id.pop(request_id, None)
        client_request_id = coordinator.request_id_to_client_request_id.pop(request_id, None)
        if client_request_id is not None:
            for key, mapped_request_id in list(coordinator.client_request_to_request_id.items()):
                if mapped_request_id == request_id:
                    coordinator.client_request_to_request_id.pop(key)

    def _send(self, identity, header, *parts) -> bool:
        payload = msgpack.packb([header.value, *parts], use_bin_type=True)
        return self.coordinator._send_to_engine(identity, payload)
