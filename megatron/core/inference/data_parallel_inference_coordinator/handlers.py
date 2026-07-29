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
    client_request_id, prompt, sampling_params = payload[1:]
    # map client request_id to server request_id
    # necessary because multiple clients might have the same request_id.
    request_id = coordinator.next_request_id
    coordinator.next_request_id += 1
    coordinator.request_id_to_client_id[request_id] = sender_identity
    coordinator.request_id_to_client_request_id[request_id] = client_request_id

    # Serialize prompt.
    if isinstance(prompt, (str, list)):
        pass
    elif isinstance(prompt, torch.Tensor):
        prompt = prompt.tolist()
    else:
        raise Exception("specialize for <%s> prompt." % type(prompt).__name__)

    engine_payload = msgpack.packb(
        [Headers.SUBMIT_REQUEST.value, request_id, prompt, sampling_params], use_bin_type=True
    )

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
        coordinator.detokenize(finished_request)
        fid = finished_request["request_id"]
        client_identity = coordinator.request_id_to_client_id[fid]
        client_request_identity = coordinator.request_id_to_client_request_id[fid]
        del coordinator.request_id_to_client_id[fid]
        del coordinator.request_id_to_client_request_id[fid]
        assigned_rank = coordinator.request_id_to_rank.pop(fid, None)
        if assigned_rank is not None:
            idx = coordinator.identity_to_rank_index.get(assigned_rank)
            if idx is not None:
                assert coordinator._pending_counts[idx] >= 1
                coordinator._pending_counts[idx] -= 1

        coordinator.router_socket.send_multipart(
            [
                client_identity,
                msgpack.packb(
                    [Headers.ENGINE_REPLY.value, client_request_identity, finished_request],
                    use_bin_type=True,
                ),
            ]
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
