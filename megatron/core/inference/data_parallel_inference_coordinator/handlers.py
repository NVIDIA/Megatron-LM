# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Message handlers for the data parallel inference coordinator.

Each handler is a free function decorated with @message_handler, which records
it in the module-level HANDLERS registry keyed by message header. The
coordinator builds its dispatch table from this registry, so a new message type
is supported simply by adding a decorated function here; the coordinator's event
loop never changes.

Handlers have the signature ``(coordinator, sender_identity, payload) -> bool | None``
where ``payload`` is the already-deserialized message. Returning a truthy value
signals the coordinator's event loop to stop.
"""

import logging

import torch

from megatron.core.inference.config import PrefixCachingCoordinatorPolicy
from megatron.core.inference.disaggregation.handoff_wire_protocol import (
    make_submit_request_with_kv_message,
)
from megatron.core.inference.headers import Headers

from .state import CONTROL_TRANSITIONS, CoordinatorState

try:
    import msgpack
except ImportError:
    msgpack = None


# Maps a message header value to the function that handles it. Populated by the
# @message_handler decorator at import time.
HANDLERS = {}


def message_handler(*headers):
    """Register a function as the handler for one or more message headers.

    A new message type is supported by writing a handler function and decorating
    it with the header(s) it serves; it is added to HANDLERS, which the
    coordinator turns into its dispatch table. The event loop never needs to
    change when a header is added.
    """

    def decorator(fn):
        for header in headers:
            assert header not in HANDLERS, f"duplicate handler for {header}"
            HANDLERS[header] = fn
        return fn

    return decorator


@message_handler(Headers.REGISTER_ROLE)
def handle_register_role(coordinator, sender_identity, payload):
    """Register a coordinator-native prefill or decode engine."""

    try:
        if coordinator.disagg is None:
            raise ValueError("REGISTER_ROLE requires a disaggregated coordinator")
        _, role, transport, instance_meta = payload
        coordinator.disagg.register_engine(sender_identity, role, transport, instance_meta)
    except (KeyError, TypeError, ValueError) as error:
        logging.warning(
            "Coordinator: rejecting role registration from %r: %s", sender_identity, error
        )
        coordinator.router_socket.send_multipart(
            [
                sender_identity,
                msgpack.packb([Headers.REQUEST_ERROR.value, str(error)], use_bin_type=True),
            ]
        )
        return
    coordinator.router_socket.send_multipart(
        [sender_identity, msgpack.packb([Headers.REGISTER_ROLE_ACK.value], use_bin_type=True)]
    )


@message_handler(Headers.CONNECT)
def handle_connect(coordinator, sender_identity, payload):
    """Handshake with a new client, replying with a CONNECT_ACK."""
    if sender_identity in coordinator.known_clients:
        logging.info(f"Client {sender_identity} sent a duplicate connect request. Ignoring ..")
        return

    coordinator.known_clients.add(sender_identity)
    coordinator.router_socket.send_multipart(
        [sender_identity, msgpack.packb([Headers.CONNECT_ACK.value], use_bin_type=True)]
    )


@message_handler(Headers.SUBMIT_REQUEST)
def handle_submit_request(coordinator, sender_identity, payload):
    """Route a client request to a data parallel rank.

    Returns True (stopping the loop) if no engines are reachable.
    """
    # ToDo [Siddharth]: We might want to tokenize the prompt on the
    # assigned data parallel rank for this process instead
    # of the coordinator.

    # Message from a known client
    if sender_identity not in coordinator.known_clients:
        logging.info(f"Received message from unknown client {sender_identity}. Ignoring.")
        return
    # this is a message from a client.
    # route it to a data parallel rank
    # Payload is [SUBMIT_REQUEST, client_request_id, prompt, sampling_params,
    # multi_modal_data].
    fields = payload[1:]
    if len(fields) == 3:
        client_request_id, prompt, sampling_params = fields
        multi_modal_data = None
    else:
        client_request_id, prompt, sampling_params, multi_modal_data = fields[:4]

    if coordinator.disagg is not None:
        assert not multi_modal_data, "native disaggregation does not support multimodal requests"

    # map client request_id to server request_id
    # necessary because multiple clients might have the same request_id.
    request_id = coordinator.next_request_id
    coordinator.next_request_id += 1
    coordinator.request_id_to_client_id[request_id] = sender_identity
    coordinator.request_id_to_client_request_id[request_id] = client_request_id
    coordinator.client_request_to_request_id[(sender_identity, client_request_id)] = request_id

    # Serialize prompt.
    if isinstance(prompt, (str, list)):
        pass
    elif isinstance(prompt, torch.Tensor):
        prompt = prompt.tolist()
    else:
        raise Exception("specialize for <%s> prompt." % type(prompt).__name__)

    if coordinator.disagg is not None:
        coordinator.disagg.route_submit(request_id, prompt, sampling_params)
        return

    engine_payload = msgpack.packb(
        [Headers.SUBMIT_REQUEST.value, request_id, prompt, sampling_params, multi_modal_data],
        use_bin_type=True,
    )

    # Skip prefix-aware routing for image-bearing requests. Prefix *caching*
    # itself is disabled for these requests in _build_vlm_request, so cross-image
    # cache reuse can't happen; clearing hashes here just prevents affinity
    # routing that would concentrate multimodal requests onto whichever rank
    # happened to serve a text-identical prompt first.
    if multi_modal_data:
        request_hashes = []
    else:
        request_hashes = coordinator.compute_request_hashes(prompt)
        if (
            coordinator.prefix_caching_coordinator_policy
            == PrefixCachingCoordinatorPolicy.FIRST_PREFIX_BLOCK
        ):
            request_hashes = request_hashes[:1]

    # Account for the fact that some engines may have died.
    for _ in range(len(coordinator.identities_of_data_parallel_ranks)):
        next_identity = coordinator.get_best_data_parallel_rank(request_hashes)
        if coordinator._send_to_engine(next_identity, engine_payload):
            break
    else:
        # If all engines have died, we are in an abnormal state, and must exit cleanly.
        logging.error("Coordinator: no reachable engines for request %d", request_id)
        del coordinator.request_id_to_client_id[request_id]
        del coordinator.request_id_to_client_request_id[request_id]
        del coordinator.client_request_to_request_id[(sender_identity, client_request_id)]
        return True

    coordinator.request_id_to_rank[request_id] = next_identity
    coordinator._pending_counts[coordinator.identity_to_rank_index[next_identity]] += 1
    if request_hashes:
        coordinator._update_rank_hashes(next_identity, request_hashes)
    if coordinator.schedule_records is not None:
        coordinator.schedule_records.append(
            {
                "request_id": request_id,
                "rank_index": coordinator.identity_to_rank_index[next_identity],
                "num_hashes": len(request_hashes),
            }
        )


@message_handler(
    Headers.PAUSE,
    Headers.UNPAUSE,
    Headers.SUSPEND,
    Headers.RESUME,
    Headers.SET_GENERATION_EPOCH,
    Headers.STOP,
)
def handle_control_signal(coordinator, sender_identity, payload):
    """Validate a control signal against the transition table and broadcast it."""
    if sender_identity not in coordinator.known_clients:
        logging.warning("Coordinator: ignoring signal from unknown client.")
        return

    header = Headers(payload[0])
    transition = CONTROL_TRANSITIONS[header]
    if coordinator.state not in transition.allowed_from:
        # Silently ignore redundant signals; warn on genuinely invalid ones.
        if coordinator.state not in transition.idempotent_in:
            logging.warning("Coordinator: ignoring %s in state %s", header.name, coordinator.state)
        return
    if transition.new_state is not None:
        coordinator.state = transition.new_state

    # Broadcast the control signal. Forward the full deserialized payload so
    # that data-bearing signals (e.g. SET_GENERATION_EPOCH) retain their args.
    coordinator._broadcast_to_engines(payload)

    # STOP affects engines; reset coordinator to RUNNING to allow future engines.
    if header == Headers.STOP:
        coordinator.state = CoordinatorState.RUNNING


@message_handler(Headers.START_CUDA_PROFILER, Headers.STOP_CUDA_PROFILER)
def handle_cuda_profiler_signal(coordinator, sender_identity, payload):
    """Broadcast a CUDA profiler control signal to every connected DP engine.

    Profiler control is not a coordinator state transition, so there are no
    CoordinatorState checks — the signal is simply forwarded to all engines.
    """
    if sender_identity not in coordinator.known_clients:
        logging.warning("Coordinator: ignoring profiler signal from unknown client.")
        return
    coordinator._broadcast_to_engines(payload)


@message_handler(Headers.ENGINE_REPLY)
def handle_engine_reply(coordinator, sender_identity, payload):
    """Route completed requests from an engine back to their originating clients."""
    # This is the output of a single engine step on some data parallel rank.
    if sender_identity not in coordinator.identities_of_data_parallel_ranks:
        # A removed engine's final replies may still be queued up.
        # Only exit with an assert if the sender was never connected to the coordinator.
        assert (
            sender_identity in coordinator.removed_engine_identities
        ), f"ENGINE_REPLY from never-connected sender {sender_identity!r}"
        logging.warning("Coordinator: ENGINE_REPLY from removed engine %r", sender_identity)
    finished_requests = payload[1]

    for finished_request in finished_requests:
        fid = finished_request["request_id"]
        if coordinator.disagg is not None and fid in coordinator.disagg.hop1_request_ids:
            coordinator.disagg.handle_prefill_done(fid, finished_request)
            continue

        if fid not in coordinator.request_id_to_client_id:
            logging.warning(
                "Coordinator: ignoring duplicate or late ENGINE_REPLY for request %d from %r",
                fid,
                sender_identity,
            )
            continue
        coordinator.detokenize(finished_request)
        client_identity = coordinator.request_id_to_client_id[fid]
        client_request_id = coordinator.request_id_to_client_request_id[fid]
        del coordinator.request_id_to_client_id[fid]
        del coordinator.request_id_to_client_request_id[fid]
        del coordinator.client_request_to_request_id[(client_identity, client_request_id)]
        assigned_rank = coordinator.request_id_to_rank.pop(fid, None)
        if assigned_rank is not None:
            idx = coordinator.identity_to_rank_index.get(assigned_rank)
            if idx is not None:
                assert coordinator._pending_counts[idx] >= 1
                coordinator._pending_counts[idx] -= 1
        if coordinator.disagg is not None:
            coordinator.disagg.handle_decode_done(fid)

        coordinator.router_socket.send_multipart(
            [
                client_identity,
                msgpack.packb(
                    [Headers.ENGINE_REPLY.value, client_request_id, finished_request],
                    use_bin_type=True,
                ),
            ]
        )


@message_handler(Headers.KV_READ_DONE)
def handle_kv_read_done(coordinator, sender_identity, payload):
    """Release prefill-owned cache storage after decode imports it."""

    if coordinator.disagg is None:
        logging.warning("Coordinator: ignoring KV_READ_DONE without disaggregation enabled")
        return
    coordinator.disagg.handle_kv_read_done(sender_identity, int(payload[1]))


@message_handler(Headers.KV_TRANSFER_READY)
def handle_kv_transfer_ready(coordinator, sender_identity, payload):
    """Start NCCL sends after decode commits the matching destinations."""

    if coordinator.disagg is None:
        logging.warning("Coordinator: ignoring KV_TRANSFER_READY without disaggregation enabled")
        return
    coordinator.disagg.handle_kv_transfer_ready(sender_identity, int(payload[1]), int(payload[2]))


@message_handler(Headers.REQUEST_ERROR)
def handle_request_error(coordinator, sender_identity, payload):
    """Forward a terminal engine-side request failure to its client."""

    request_id, reason, source_safe = int(payload[1]), str(payload[2]), bool(payload[3])
    if coordinator.disagg is not None:
        coordinator.disagg.handle_engine_failure(request_id, reason, source_safe=source_safe)
        return

    client_identity = coordinator.request_id_to_client_id.get(request_id)
    client_request_id = coordinator.request_id_to_client_request_id.get(request_id)
    if client_identity is None or client_request_id is None:
        return
    coordinator.router_socket.send_multipart(
        [
            client_identity,
            msgpack.packb(
                [Headers.REQUEST_ERROR.value, client_request_id, reason, source_safe],
                use_bin_type=True,
            ),
        ]
    )
    if source_safe:
        coordinator.request_id_to_client_id.pop(request_id, None)
        coordinator.request_id_to_client_request_id.pop(request_id, None)
        coordinator.client_request_to_request_id.pop((client_identity, client_request_id), None)
        assigned_rank = coordinator.request_id_to_rank.pop(request_id, None)
        if assigned_rank is not None:
            index = coordinator.identity_to_rank_index.get(assigned_rank)
            if index is not None and coordinator._pending_counts[index] > 0:
                coordinator._pending_counts[index] -= 1


@message_handler(Headers.REQUEST_ABORTED)
def handle_request_aborted(coordinator, sender_identity, payload):
    """Forward engine cancellation completion to the requesting client."""

    request_id, source_safe = int(payload[1]), bool(payload[2])
    if coordinator.disagg is not None:
        coordinator.disagg.handle_engine_aborted(request_id, source_safe=source_safe)
        return
    if not source_safe:
        return
    client_identity = coordinator.request_id_to_client_id.pop(request_id, None)
    client_request_id = coordinator.request_id_to_client_request_id.pop(request_id, None)
    assigned_rank = coordinator.request_id_to_rank.pop(request_id, None)
    if assigned_rank is not None:
        index = coordinator.identity_to_rank_index.get(assigned_rank)
        if index is not None and coordinator._pending_counts[index] > 0:
            coordinator._pending_counts[index] -= 1
    if client_identity is None or client_request_id is None:
        return
    coordinator.client_request_to_request_id.pop((client_identity, client_request_id), None)
    coordinator.router_socket.send_multipart(
        [
            client_identity,
            msgpack.packb(
                [Headers.REQUEST_ABORTED.value, client_request_id, source_safe], use_bin_type=True
            ),
        ]
    )


@message_handler(Headers.SUBMIT_REQUEST_WITH_KV)
def handle_submit_request_with_kv(coordinator, sender_identity, payload):
    """Route a client-supplied handoff to a decode engine."""

    if sender_identity not in coordinator.known_clients:
        logging.info(
            "Received SUBMIT_REQUEST_WITH_KV from unknown client %s; ignoring.", sender_identity
        )
        return
    if len(payload) != 6:
        logging.error(
            "Coordinator: malformed SUBMIT_REQUEST_WITH_KV payload with %d fields", len(payload) - 1
        )
        return
    client_request_id, prompt, sampling_params, kv_meta, src_block_ids = payload[1:]

    if isinstance(prompt, torch.Tensor):
        prompt = prompt.tolist()
    elif not isinstance(prompt, (str, list)):
        logging.error("Coordinator: unsupported handoff prompt type %s", type(prompt).__name__)
        return
    if (
        not isinstance(sampling_params, dict)
        or not isinstance(kv_meta, dict)
        or not isinstance(src_block_ids, list)
        or any(type(block_id) is not int for block_id in src_block_ids)
    ):
        logging.error("Coordinator: malformed handoff transfer metadata")
        return

    request_id = coordinator.next_request_id
    coordinator.next_request_id += 1
    coordinator.request_id_to_client_id[request_id] = sender_identity
    coordinator.request_id_to_client_request_id[request_id] = client_request_id
    coordinator.client_request_to_request_id[(sender_identity, client_request_id)] = request_id
    engine_payload = msgpack.packb(
        make_submit_request_with_kv_message(
            Headers.SUBMIT_REQUEST_WITH_KV.value,
            request_id,
            prompt,
            sampling_params,
            kv_meta,
            src_block_ids,
        ),
        use_bin_type=True,
    )
    request_hashes = coordinator.compute_request_hashes(prompt)
    if (
        coordinator.prefix_caching_coordinator_policy
        == PrefixCachingCoordinatorPolicy.FIRST_PREFIX_BLOCK
    ):
        request_hashes = request_hashes[:1]

    for _ in range(len(coordinator.identities_of_data_parallel_ranks)):
        next_identity = coordinator.get_best_data_parallel_rank(request_hashes)
        if coordinator._send_to_engine(next_identity, engine_payload):
            break
    else:
        logging.error("Coordinator: no reachable engines for handoff request %d", request_id)
        del coordinator.request_id_to_client_id[request_id]
        del coordinator.request_id_to_client_request_id[request_id]
        coordinator.client_request_to_request_id.pop((sender_identity, client_request_id), None)
        return True
    coordinator.request_id_to_rank[request_id] = next_identity
    coordinator._pending_counts[coordinator.identity_to_rank_index[next_identity]] += 1
    if request_hashes:
        coordinator._update_rank_hashes(next_identity, request_hashes)
    if coordinator.schedule_records is not None:
        coordinator.schedule_records.append(
            {
                "request_id": request_id,
                "rank_index": coordinator.identity_to_rank_index[next_identity],
                "num_hashes": len(request_hashes),
            }
        )


@message_handler(Headers.RELEASE_KV)
def handle_release_kv(coordinator, sender_identity, payload):
    """Broadcast a client handoff release to every engine."""

    if sender_identity not in coordinator.known_clients:
        logging.warning("Coordinator: ignoring RELEASE_KV from unknown client.")
        return
    coordinator._broadcast_to_engines([Headers.RELEASE_KV.value, int(payload[1])])


@message_handler(Headers.ENGINE_REPLY_PARTIAL)
def handle_engine_reply_partial(coordinator, sender_identity, payload):
    """Route incremental engine replies without releasing request routing state."""
    if sender_identity not in coordinator.identities_of_data_parallel_ranks:
        assert (
            sender_identity in coordinator.removed_engine_identities
        ), f"ENGINE_REPLY_PARTIAL from never-connected sender {sender_identity!r}"
        logging.warning("Coordinator: ENGINE_REPLY_PARTIAL from removed engine %r", sender_identity)
        return
    for partial in payload[1]:
        request_id = partial["request_id"]
        client_identity = coordinator.request_id_to_client_id.get(request_id)
        client_request_id = coordinator.request_id_to_client_request_id.get(request_id)
        if client_identity is None or client_request_id is None:
            logging.warning(
                "Coordinator: ignoring late ENGINE_REPLY_PARTIAL for request %d", request_id
            )
            continue
        # Partial tokens are detokenized incrementally by the client-facing streaming layer.
        coordinator.router_socket.send_multipart(
            [
                client_identity,
                msgpack.packb(
                    [Headers.ENGINE_REPLY_PARTIAL.value, client_request_id, partial],
                    use_bin_type=True,
                ),
            ]
        )


@message_handler(Headers.ABORT_REQUEST)
def handle_abort_request(coordinator, sender_identity, payload):
    """Forward a client cancellation to the engine serving that request."""
    if sender_identity not in coordinator.known_clients:
        logging.warning("Coordinator: ignoring abort from unknown client.")
        return
    client_request_id = int(payload[1])
    request_id = coordinator.client_request_to_request_id.get((sender_identity, client_request_id))
    if request_id is None:
        return
    if coordinator.disagg is not None:
        coordinator.disagg.abort_request(request_id)
        return
    assigned_rank = coordinator.request_id_to_rank.get(request_id)
    if assigned_rank is not None:
        coordinator._send_to_engine(
            assigned_rank,
            msgpack.packb([Headers.ABORT_REQUEST.value, request_id], use_bin_type=True),
        )


@message_handler(Headers.SHUTDOWN)
def handle_shutdown(coordinator, sender_identity, payload):
    """Stop the coordinator event loop on request from a known client."""
    if sender_identity not in coordinator.known_clients:
        logging.warning("Coordinator: ignoring signal from unknown client.")
        return
    return True


@message_handler(Headers.DISCONNECT)
def handle_disconnect(coordinator, sender_identity, payload):
    """Remove a disconnecting engine from the routing pool."""
    if sender_identity in coordinator.identities_of_data_parallel_ranks:
        coordinator._remove_engine(sender_identity)
