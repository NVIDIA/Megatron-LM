# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Message handlers for the data parallel inference coordinator.

Each handler is a free function decorated with @message_handler, which records
it in the module-level HANDLERS registry keyed by message header. The
coordinator builds its dispatch table from this registry, so a new message type
is supported simply by adding a decorated function here; the coordinator's event
loop never changes.

Handlers have the signature
``(coordinator, sender_identity, metadata, bodies) -> bool | None``.

``metadata`` is the decoded contents of the message's first frame: a header
followed by whatever the coordinator needs in order to route the message. It is
small by construction and does not grow with prompt length.

``bodies`` is the list of remaining raw frames, still packed. These carry the
prompt (inbound) or the finished request (outbound) -- the bulk of the bytes.
The coordinator forwards them as opaque frames and only decodes one when it has
to mutate it, which keeps its per-request cost flat in prompt length.

Returning a truthy value signals the event loop to stop.
"""

import logging

from megatron.core.inference.config import (
    MediaCacheCoordinatorPolicy,
    PrefixCachingCoordinatorPolicy,
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


@message_handler(Headers.CONNECT)
def handle_connect(coordinator, sender_identity, metadata, bodies):
    """Handshake with a new client, replying with a CONNECT_ACK.

    Sent by ``InferenceClient.start``.

    ``metadata``: ``[header]``.
    ``bodies``: empty.
    """
    if sender_identity in coordinator.known_clients:
        logging.info(f"Client {sender_identity} sent a duplicate connect request. Ignoring ..")
        return

    coordinator.known_clients.add(sender_identity)
    coordinator.router_socket.send_multipart(
        [sender_identity, msgpack.packb([Headers.CONNECT_ACK.value], use_bin_type=True)]
    )


@message_handler(Headers.SUBMIT_REQUEST)
def handle_submit_request(coordinator, sender_identity, metadata, bodies):
    """Route a client request to a data parallel rank.

    Sent by ``InferenceClient.add_request`` / ``add_request_streaming``.

    ``metadata``: ``[header, client_request_id, sampling_params, multi_modal_data]``,
        where ``sampling_params`` is the serialized dict and ``multi_modal_data``
        carries the media identity the routing policy keys on. Both are
        constant-size, which is why they are decoded here on every request.
    ``bodies``: ``[prompt]`` -- one frame holding the packed prompt (a string or
        a token id list). Forwarded to the engine untouched, and decoded here
        only when the routing policy needs block hashes.

    Returns True (stopping the loop) if no engines are reachable.
    """
    # Message from a known client
    if sender_identity not in coordinator.known_clients:
        logging.info(f"Received message from unknown client {sender_identity}. Ignoring.")
        return

    _, client_request_id, sampling_params, multi_modal_data = metadata
    prompt_frame = bodies[0]

    # map client request_id to server request_id
    # necessary because multiple clients might have the same request_id.
    request_id = coordinator.next_request_id
    coordinator.next_request_id += 1
    coordinator.request_id_to_client_id[request_id] = sender_identity
    coordinator.request_id_to_client_request_id[request_id] = client_request_id
    coordinator.client_request_to_request_id[(sender_identity, client_request_id)] = request_id

    # Rebuilding the metadata frame is cheap: it holds no prompt tokens.
    engine_metadata = msgpack.packb(
        [Headers.SUBMIT_REQUEST.value, request_id, sampling_params, multi_modal_data],
        use_bin_type=True,
    )

    # Media identity is read straight from the metadata frame; it salts the
    # routing hashes and is passed to rank selection for media affinity.
    media_cache_key = (
        multi_modal_data.get("media_cache_key") if isinstance(multi_modal_data, dict) else None
    )

    # Only prefix-affinity routing consults the block hashes. When nothing will
    # look at them, skip hashing *and* the prompt decode it needs -- avoiding
    # that decode is the reason the prompt travels in its own frame.
    if (
        coordinator.enable_prefix_caching
        and coordinator.block_size_tokens is not None
        and coordinator.prefix_caching_coordinator_policy
        != PrefixCachingCoordinatorPolicy.LOAD_BALANCED
    ):
        request_hashes = coordinator.compute_request_hashes(
            msgpack.unpackb(prompt_frame, raw=False), cache_salt=media_cache_key
        )
    else:
        request_hashes = []

    if (
        coordinator.prefix_caching_coordinator_policy
        == PrefixCachingCoordinatorPolicy.FIRST_PREFIX_BLOCK
    ):
        request_hashes = request_hashes[:1]

    # Account for the fact that some engines may have died.
    for _ in range(len(coordinator.identities_of_data_parallel_ranks)):
        next_identity = coordinator.get_best_data_parallel_rank(
            request_hashes, media_cache_key=media_cache_key
        )
        if coordinator._send_to_engine(next_identity, [engine_metadata, prompt_frame]):
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
    if (
        isinstance(media_cache_key, str)
        and coordinator.vision_embedding_cache_enabled
        and coordinator.media_cache_coordinator_policy == MediaCacheCoordinatorPolicy.AFFINITY
    ):
        coordinator._update_media_affinity(media_cache_key, next_identity)
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


@message_handler(Headers.SUBMIT_REQUEST_WITH_KV)
def handle_submit_request_with_kv(coordinator, sender_identity, metadata, bodies):
    """Route a client-supplied KV handoff to a decode engine.

    Sent by ``InferenceClient.add_request_with_kv_handoff`` /
    ``add_request_with_kv_handoff_streaming``.

    ``metadata``: ``[header, client_request_id, sampling_params, kv_meta]``,
        where ``kv_meta`` is the peer's NIXL agent/layout export. It is bounded
        by TP size and num_speculative_tokens, not by prompt length, so it stays
        in the metadata frame.
    ``bodies``: ``[prompt, src_block_ids]``. ``src_block_ids`` names one remote
        block per block_size_tokens of prompt, so it grows with the prompt and
        travels as its own frame. Both bodies are forwarded to the engine
        untouched, so the coordinator decodes nothing sequence-dependent. In
        disaggregated serving every decode request arrives here, so this is as
        hot as a plain submission.
    """

    if sender_identity not in coordinator.known_clients:
        logging.info(
            "Received SUBMIT_REQUEST_WITH_KV from unknown client %s; ignoring.", sender_identity
        )
        return
    if len(metadata) != 4 or len(bodies) != 2:
        logging.error(
            "Coordinator: malformed SUBMIT_REQUEST_WITH_KV with %d metadata fields, %d bodies",
            len(metadata) - 1,
            len(bodies),
        )
        return

    _, client_request_id, sampling_params, kv_meta = metadata
    request_id = coordinator.next_request_id
    coordinator.next_request_id += 1
    coordinator.request_id_to_client_id[request_id] = sender_identity
    coordinator.request_id_to_client_request_id[request_id] = client_request_id
    coordinator.client_request_to_request_id[(sender_identity, client_request_id)] = request_id

    # Rebuilding the metadata frame is cheap: it holds no prompt tokens.
    engine_metadata = msgpack.packb(
        [Headers.SUBMIT_REQUEST_WITH_KV.value, request_id, sampling_params, kv_meta],
        use_bin_type=True,
    )

    for _ in range(len(coordinator.identities_of_data_parallel_ranks)):
        next_identity = coordinator.get_least_loaded_data_parallel_rank()
        if coordinator._send_to_engine(next_identity, [engine_metadata, *bodies]):
            break
    else:
        logging.error("Coordinator: no reachable engines for handoff request %d", request_id)
        del coordinator.request_id_to_client_id[request_id]
        del coordinator.request_id_to_client_request_id[request_id]
        del coordinator.client_request_to_request_id[(sender_identity, client_request_id)]
        return True

    coordinator.request_id_to_rank[request_id] = next_identity
    coordinator._pending_counts[coordinator.identity_to_rank_index[next_identity]] += 1


@message_handler(Headers.RELEASE_KV)
def handle_release_kv(coordinator, sender_identity, metadata, bodies):
    """Broadcast release of prefill blocks retained for a completed handoff.

    Sent by ``InferenceClient.release_handoff``. Broadcast to every engine;
    engines not holding that request id treat it as a no-op.

    ``metadata``: ``[header, client_request_id]``.
    ``bodies``: empty.
    """

    if sender_identity not in coordinator.known_clients:
        logging.warning("Coordinator: ignoring RELEASE_KV from unknown client.")
        return
    coordinator._broadcast_to_engines([Headers.RELEASE_KV.value, int(metadata[1])])


@message_handler(
    Headers.PAUSE,
    Headers.UNPAUSE,
    Headers.SUSPEND,
    Headers.RESUME,
    Headers.SET_GENERATION_EPOCH,
    Headers.STOP,
)
def handle_control_signal(coordinator, sender_identity, metadata, bodies):
    """Validate a control signal against the transition table and broadcast it.

    Serves PAUSE, UNPAUSE, SUSPEND, RESUME, SET_GENERATION_EPOCH and STOP, all
    sent by ``InferenceClient._send_signal_to_engines``.

    ``metadata``: ``[header, *args]``. Every signal but one carries no args;
        SET_GENERATION_EPOCH carries ``[header, generation_epoch]``. The whole
        list is rebroadcast verbatim so data-bearing signals keep their args.
    ``bodies``: empty.
    """
    if sender_identity not in coordinator.known_clients:
        logging.warning("Coordinator: ignoring signal from unknown client.")
        return

    header = Headers(metadata[0])
    transition = CONTROL_TRANSITIONS[header]
    if coordinator.state not in transition.allowed_from:
        # Silently ignore redundant signals; warn on genuinely invalid ones.
        if coordinator.state not in transition.idempotent_in:
            logging.warning("Coordinator: ignoring %s in state %s", header.name, coordinator.state)
        return
    if transition.new_state is not None:
        coordinator.state = transition.new_state

    # Broadcast the control signal. Forward the full metadata so that
    # data-bearing signals (e.g. SET_GENERATION_EPOCH) retain their args.
    coordinator._broadcast_to_engines(metadata)

    # STOP affects engines; reset coordinator to RUNNING to allow future engines.
    if header == Headers.STOP:
        coordinator.state = CoordinatorState.RUNNING


@message_handler(Headers.START_CUDA_PROFILER, Headers.STOP_CUDA_PROFILER)
def handle_cuda_profiler_signal(coordinator, sender_identity, metadata, bodies):
    """Broadcast a CUDA profiler control signal to every connected DP engine.

    Serves START_CUDA_PROFILER and STOP_CUDA_PROFILER, sent by
    ``InferenceClient._send_signal_to_engines``.

    Profiler control is not a coordinator state transition, so there are no
    CoordinatorState checks — the signal is simply forwarded to all engines.

    ``metadata``: ``[header]``, rebroadcast verbatim.
    ``bodies``: empty.
    """
    if sender_identity not in coordinator.known_clients:
        logging.warning("Coordinator: ignoring profiler signal from unknown client.")
        return
    coordinator._broadcast_to_engines(metadata)


@message_handler(Headers.ENGINE_REPLY)
def handle_engine_reply(coordinator, sender_identity, metadata, bodies):
    """Route completed requests from an engine back to their originating clients.

    Sent by the engine via ``_engine_reply_frames``.

    ``metadata``: ``[header, [[request_id, needs_detokenize], ...]]`` -- one
        entry per finished request, in the same order as ``bodies``.
    ``bodies``: one frame per entry, each a packed finished request. A finished
        request echoes the prompt back, so these are the largest frames the
        coordinator handles. A body is decoded only when ``needs_detokenize`` is
        set; otherwise it reaches the client as the frame the engine produced.
    """
    # This is the output of a single engine step on some data parallel rank.
    if sender_identity not in coordinator.identities_of_data_parallel_ranks:
        # A removed engine's final replies may still be queued up.
        # Only exit with an assert if the sender was never connected to the coordinator.
        assert (
            sender_identity in coordinator.removed_engine_identities
        ), f"ENGINE_REPLY from never-connected sender {sender_identity!r}"
        logging.warning("Coordinator: ENGINE_REPLY from removed engine %r", sender_identity)

    for (fid, needs_detokenize), body in zip(metadata[1], bodies):
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

        if needs_detokenize:
            # Detokenizing writes generated_text into the reply, so this one has
            # to be decoded and re-encoded. Clients that detokenize for
            # themselves (the OpenAI frontend does) never take this path.
            finished_request = msgpack.unpackb(body, raw=False)
            coordinator.detokenize(finished_request)
            body = msgpack.packb(finished_request, use_bin_type=True)

        reply_metadata = msgpack.packb(
            [Headers.ENGINE_REPLY.value, client_request_id], use_bin_type=True
        )
        coordinator.router_socket.send_multipart([client_identity, reply_metadata, body])


@message_handler(Headers.ENGINE_REPLY_PARTIAL)
def handle_engine_reply_partial(coordinator, sender_identity, metadata, bodies):
    """Route incremental engine replies without releasing request routing state.

    Sent by the engine for streaming requests at each streaming interval.

    ``metadata``: ``[header, [request_id, ...]]`` -- one id per partial, in the
        same order as ``bodies``. No detokenize flag: partials are always
        detokenized incrementally by the client-facing streaming layer.
    ``bodies``: one frame per id, each a packed partial
        (``{"request_id": int, "new_tokens": [...]}``), always forwarded
        untouched.
    """
    if sender_identity not in coordinator.identities_of_data_parallel_ranks:
        assert (
            sender_identity in coordinator.removed_engine_identities
        ), f"ENGINE_REPLY_PARTIAL from never-connected sender {sender_identity!r}"
        logging.warning("Coordinator: ENGINE_REPLY_PARTIAL from removed engine %r", sender_identity)
        return
    for request_id, body in zip(metadata[1], bodies):
        client_identity = coordinator.request_id_to_client_id[request_id]
        client_request_id = coordinator.request_id_to_client_request_id[request_id]
        # Partial tokens are detokenized incrementally by the client-facing
        # streaming layer, so the body is always forwarded untouched.
        coordinator.router_socket.send_multipart(
            [
                client_identity,
                msgpack.packb(
                    [Headers.ENGINE_REPLY_PARTIAL.value, client_request_id], use_bin_type=True
                ),
                body,
            ]
        )


@message_handler(Headers.ABORT_REQUEST)
def handle_abort_request(coordinator, sender_identity, metadata, bodies):
    """Forward a client cancellation to the engine serving that request.

    Sent by ``InferenceClient.abort_request``. Unknown or already-completed
    request ids are dropped silently.

    ``metadata``: ``[header, client_request_id]``.
    ``bodies``: empty.
    """
    if sender_identity not in coordinator.known_clients:
        logging.warning("Coordinator: ignoring abort from unknown client.")
        return
    client_request_id = int(metadata[1])
    request_id = coordinator.client_request_to_request_id.get((sender_identity, client_request_id))
    if request_id is None:
        return
    assigned_rank = coordinator.request_id_to_rank.get(request_id)
    if assigned_rank is not None:
        coordinator._send_to_engine(
            assigned_rank,
            [msgpack.packb([Headers.ABORT_REQUEST.value, request_id], use_bin_type=True)],
        )


@message_handler(Headers.SHUTDOWN)
def handle_shutdown(coordinator, sender_identity, metadata, bodies):
    """Stop the coordinator event loop on request from a known client.

    Sent by ``InferenceClient.shutdown_engines``.

    ``metadata``: ``[header]``.
    ``bodies``: empty.
    """
    if sender_identity not in coordinator.known_clients:
        logging.warning("Coordinator: ignoring signal from unknown client.")
        return
    return True


@message_handler(Headers.DISCONNECT)
def handle_disconnect(coordinator, sender_identity, metadata, bodies):
    """Remove a disconnecting engine from the routing pool.

    Sent by an engine as it exits -- the only handler here whose sender is an
    engine rather than a client.

    ``metadata``: ``[header]``.
    ``bodies``: empty.
    """
    if sender_identity in coordinator.identities_of_data_parallel_ranks:
        coordinator._remove_engine(sender_identity)
