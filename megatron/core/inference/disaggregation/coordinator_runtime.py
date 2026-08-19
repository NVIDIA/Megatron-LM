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

    def __init__(self, coordinator: Any, router_name: str) -> None:
        self.coordinator = coordinator
        self.router = make_disagg_router(router_name)
        self.flow = DisaggStateFlowControl()
        self.hop1_request_ids: set[int] = set()
        self.request_metadata: dict[int, tuple[Any, dict]] = {}
        self.prefill_by_request: dict[int, Any] = {}
        self.engine_role: dict[Any, str] = {}
        self.engine_transport: dict[Any, str] = {}
        self.engine_metadata: dict[Any, Any] = {}
        self.cancelled_request_ids: set[int] = set()

    def register_engine(self, identity, role: str, transport: str, instance_meta: Any) -> None:
        """Register one prefill or decode instance and its transfer metadata."""

        previous_role = self.engine_role.get(identity)
        previous_transport = self.engine_transport.get(identity)
        if previous_role is not None and (previous_role != role or previous_transport != transport):
            raise ValueError(f"engine {identity!r} cannot change disaggregated role or transport")
        if self.engine_transport and transport not in set(self.engine_transport.values()):
            raise ValueError("prefill and decode engines must use the same state transport")

        capacity = self.flow.register_engine(identity, role, instance_meta)
        coordinator = self.coordinator
        if identity not in coordinator.identities_of_data_parallel_ranks:
            coordinator.identities_of_data_parallel_ranks.append(identity)
            coordinator._register_rank_identity(identity)
        self.router.register(identity, role)
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

        for request_id in self.flow.pop_queued_for_engine(identity):
            self.drop_request(
                request_id, f"queued on removed engine {identity!r}", source_safe=True
            )
        affected = set(self.router.requests_involving(identity))
        affected.update(self.flow.reservations_for_engine(identity))
        for request_id in affected:
            self.drop_request(
                request_id,
                f"engine {identity!r} removed",
                source_safe=self.router.decode_for_request(request_id) != identity,
            )
        self.router.remove(identity)
        self.engine_role.pop(identity, None)
        self.engine_transport.pop(identity, None)
        self.engine_metadata.pop(identity, None)
        self.flow.remove_engine(identity)

    def route_submit(self, request_id: int, prompt: Any, sampling_params: dict) -> None:
        """Reserve capacity and send a client request to a prefill instance."""

        self.request_metadata[request_id] = (prompt, sampling_params)
        try:
            prefill_id = self.router.route_submit(request_id, self.flow.prefill_load)
        except RuntimeError as error:
            self.drop_request(request_id, f"cannot route to prefill: {error}", source_safe=True)
            return

        capacity = self.flow.capacity(prefill_id)
        slot_cost = self.flow.prefill_slot_cost(prefill_id)
        if not self.flow.can_ever_fit(prefill_id, slot_cost):
            self.drop_request(
                request_id,
                f"request requires {slot_cost} live SSM slots, but prefill "
                f"engine {prefill_id!r} advertises only {capacity}",
                source_safe=True,
            )
            return
        if self.flow.has_queued_prefill(prefill_id) or not self.flow.try_reserve_prefill(
            prefill_id, request_id, slot_cost
        ):
            self.flow.enqueue_prefill(prefill_id, request_id, slot_cost)
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
            request = self.flow.pop_next_prefill(prefill_id)
            if request is None:
                return
            prompt, sampling_params = self.request_metadata[request.request_id]
            self._submit_prefill(prefill_id, request.request_id, prompt, sampling_params)

    def handle_prefill_done(self, request_id: int, finished_request: dict) -> None:
        """Route a completed prefill handoff to a decode instance."""

        self.hop1_request_ids.discard(request_id)
        if request_id in self.cancelled_request_ids:
            self._finish_abort(request_id, source_safe=True)
            return
        request_meta = self.request_metadata.get(request_id)
        handoff = finished_request.get("disaggregated_params")
        if request_meta is None or not handoff:
            self.drop_request(
                request_id, "prefill reply carried no handoff metadata", source_safe=True
            )
            return
        prompt, sampling_params = request_meta
        try:
            _, decode_id = self.router.route_prefill_done(request_id, self.flow.decode_load)
        except RuntimeError as error:
            self.drop_request(request_id, f"cannot route to decode: {error}", source_safe=True)
            return

        kv_meta = handoff["kv_meta"]
        prefill_id = self.prefill_by_request.get(request_id)
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
                f"SSM handoff requires {slot_cost} live slots, but decode "
                f"engine {decode_id!r} advertises only {capacity}",
                source_safe=True,
            )
            return
        if self.flow.has_queued(decode_id) or not self.flow.try_reserve(
            decode_id, request_id, slot_cost
        ):
            self.flow.enqueue(decode_id, request_id, payload, slot_cost)
            return
        self._send_decode_handoff(decode_id, request_id, payload)

    def _send_decode_handoff(self, decode_id, request_id: int, payload: bytes) -> bool:
        return self.coordinator._send_to_engine(decode_id, payload)

    def handle_kv_transfer_ready(
        self, sender_identity, request_id: int, cached_prefix_blocks: int
    ) -> None:
        """Start a two-sided send after the assigned decode commits destinations."""

        if self.router.decode_for_request(request_id) != sender_identity:
            logging.warning(
                "Coordinator: ignoring KV_TRANSFER_READY for request %d from %r",
                request_id,
                sender_identity,
            )
            return
        prefill_id = self.prefill_by_request.get(request_id)
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
            handoff = self.flow.pop_next_admissible(decode_id)
            if handoff is None:
                return
            if not self._send_decode_handoff(decode_id, handoff.request_id, handoff.payload):
                return

    def _release_prefill(self, request_id: int) -> None:
        """Release prefill pins after decode has imported the handoff."""

        prefill_id = self.prefill_by_request.pop(request_id, None)
        if prefill_id is None:
            return
        self._send(prefill_id, Headers.RELEASE_KV, request_id)
        released_prefill = self.flow.release_prefill(request_id)
        if released_prefill is not None:
            self._drain_prefill_queue(released_prefill)

    def handle_kv_read_done(self, sender_identity, request_id: int) -> None:
        """Release source state after the assigned decode reports a completed read."""

        if request_id not in self.prefill_by_request:
            return
        assigned_decode = self.router.decode_for_request(request_id)
        if assigned_decode != sender_identity:
            logging.warning(
                "Coordinator: ignoring KV_READ_DONE for request %d from %r; assigned decode is %r",
                request_id,
                sender_identity,
                assigned_decode,
            )
            return
        self._release_prefill(request_id)
        if request_id not in self.request_metadata:
            self.router.forget(request_id)

    def handle_decode_done(self, request_id: int) -> None:
        """Release all coordinator state after the decode reply is returned."""

        decode_id = self.flow.release_decode(request_id)
        if decode_id is not None:
            self._drain_decode_queue(decode_id)
        self._release_prefill(request_id)
        self.router.forget(request_id)
        self.request_metadata.pop(request_id, None)

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
        self.flow.remove_queued(request_id)
        decode_id = self.flow.release_decode(request_id)
        if decode_id is not None:
            self._drain_decode_queue(decode_id)
        self.request_metadata.pop(request_id, None)
        self.hop1_request_ids.discard(request_id)
        self.cancelled_request_ids.discard(request_id)
        if source_safe:
            self._release_prefill(request_id)
            self.router.forget(request_id)

        coordinator.request_id_to_client_id.pop(request_id, None)
        client_request_id = coordinator.request_id_to_client_request_id.pop(request_id, None)
        if client_identity is not None and client_request_id is not None:
            coordinator.client_request_to_request_id.pop((client_identity, client_request_id), None)

    def handle_engine_failure(
        self, request_id: int, reason: str, *, source_safe: bool = False
    ) -> None:
        """Fail a decode handoff without releasing a source still in use."""

        self.drop_request(request_id, reason, source_safe=source_safe)

    def abort_request(self, request_id: int) -> None:
        """Cancel queued work or forward cancellation to its active engine."""

        self.cancelled_request_ids.add(request_id)
        if self.flow.remove_queued(request_id):
            self._finish_abort(request_id, source_safe=True)
            return
        decode_id = self.router.decode_for_request(request_id)
        if decode_id is not None:
            self._send(decode_id, Headers.ABORT_REQUEST, request_id)
            return
        prefill_id = self.prefill_by_request.get(request_id)
        if prefill_id is not None:
            self._send(prefill_id, Headers.ABORT_REQUEST, request_id)
            return
        self._finish_abort(request_id, source_safe=True)

    def handle_engine_aborted(self, request_id: int, *, source_safe: bool) -> None:
        """Finish a cancellation after the engine reports transport safety."""

        if request_id in self.cancelled_request_ids:
            self._finish_abort(request_id, source_safe=source_safe)

    def _finish_abort(self, request_id: int, *, source_safe: bool) -> None:
        coordinator = self.coordinator
        client_identity = coordinator.request_id_to_client_id.get(request_id)
        client_request_id = coordinator.request_id_to_client_request_id.get(request_id)
        if source_safe:
            self._release_prefill(request_id)
        decode_id = self.flow.release_decode(request_id)
        if decode_id is not None:
            self._drain_decode_queue(decode_id)
        if source_safe:
            self.router.forget(request_id)
        self.request_metadata.pop(request_id, None)
        self.hop1_request_ids.discard(request_id)
        self.cancelled_request_ids.discard(request_id)
        coordinator.request_id_to_client_id.pop(request_id, None)
        coordinator.request_id_to_client_request_id.pop(request_id, None)
        if client_identity is not None and client_request_id is not None:
            coordinator.client_request_to_request_id.pop((client_identity, client_request_id), None)
            coordinator.router_socket.send_multipart(
                [
                    client_identity,
                    msgpack.packb(
                        [Headers.REQUEST_ABORTED.value, client_request_id, source_safe],
                        use_bin_type=True,
                    ),
                ]
            )

    def _send(self, identity, header, *parts) -> bool:
        payload = msgpack.packb([header.value, *parts], use_bin_type=True)
        return self.coordinator._send_to_engine(identity, payload)
