# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
from unittest.mock import MagicMock, patch

import msgpack
import pytest
import zmq

from megatron.core.inference.headers import Headers
from megatron.core.inference.inference_client import InferenceClient
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints.common import (
    abort_requests,
)

pytestmark = pytest.mark.asyncio


def _make_client(deserialize: bool = False):
    """Build an InferenceClient with a mocked zmq Context/Socket.

    The real ctor binds a DEALER socket against a TCP endpoint; we only care
    about the bytes it sends/receives, so we replace zmq.Context wholesale.
    """
    fake_socket = MagicMock(name="zmq_socket")
    fake_context = MagicMock(name="zmq_context")
    fake_context.socket.return_value = fake_socket
    with patch("megatron.core.inference.inference_client.zmq.Context", return_value=fake_context):
        client = InferenceClient("tcp://127.0.0.1:5555", deserialize=deserialize)
    return client, fake_context, fake_socket


async def test_inference_client_lifecycle():
    """End-to-end lifecycle of InferenceClient with mocked zmq sockets:
    construct → start (CONNECT handshake) → add_request (SUBMIT_REQUEST) →
    listener delivers ENGINE_REPLY → control signal (pause + set epoch) →
    stop (cancels listener, cancels pending futures, closes socket).

    Per reviewer guidance, the per-step assertions are intentionally bundled
    into one test because the contract is the ordering, not the steps in
    isolation."""
    client, fake_context, fake_socket = _make_client()

    # Construction: DEALER socket connected, HWMs at 0, counters initialized.
    fake_socket.connect.assert_called_once_with("tcp://127.0.0.1:5555")
    opts = {call.args[0]: call.args[1] for call in fake_socket.setsockopt.call_args_list}
    assert opts[zmq.SNDHWM] == 0 and opts[zmq.RCVHWM] == 0
    assert client.next_request_id == 0
    assert client.completion_futures == {}

    # start(): handshake sends CONNECT, expects CONNECT_ACK, spawns listener task.
    # We stage two recv() replies: the CONNECT_ACK during handshake, and an
    # ENGINE_REPLY for the request we'll add below. Subsequent recvs raise
    # zmq.Again so the listener loop yields back to the event loop.
    recv_queue = [
        msgpack.packb([Headers.CONNECT_ACK.value], use_bin_type=True),
        msgpack.packb([Headers.ENGINE_REPLY.value, 0, {"foo": "bar"}], use_bin_type=True),
    ]

    def fake_recv(*args, **kwargs):
        if recv_queue:
            return recv_queue.pop(0)
        raise zmq.Again()

    fake_socket.recv.side_effect = fake_recv

    client.start()
    assert isinstance(client.listener_task, asyncio.Task)
    sent_connect = fake_socket.send.call_args.args[0]
    assert msgpack.unpackb(sent_connect, raw=False)[0] == Headers.CONNECT.value

    # add_request: SUBMIT_REQUEST payload (header, id, prompt, sampling-dict), counter increments.
    fut = client.add_request("hello", SamplingParams(temperature=0.5))
    assert isinstance(fut, asyncio.Future)
    assert client.next_request_id == 1
    assert 0 in client.request_submission_times
    submit_payload = msgpack.unpackb(fake_socket.send.call_args.args[0], raw=False)
    assert submit_payload[0] == Headers.SUBMIT_REQUEST.value
    assert submit_payload[1] == 0
    assert submit_payload[2] == "hello"
    assert submit_payload[3]["temperature"] == 0.5

    # Listener delivers the reply: future resolves with payload + injected latency.
    # Submission-time entry is popped on completion.
    result = await asyncio.wait_for(fut, timeout=2.0)
    assert result["foo"] == "bar"
    assert "latency" in result
    assert 0 not in client.request_submission_times
    assert 0 not in client.completion_futures

    # Control helpers send the matching Headers byte (PAUSE used as a representative;
    # the dispatch table is one ctype-style mapping shared across all helpers).
    fake_socket.send.reset_mock()
    client.pause_engines()
    assert msgpack.unpackb(fake_socket.send.call_args.args[0], raw=False)[0] == Headers.PAUSE.value
    client.set_generation_epoch(42)
    epoch_payload = msgpack.unpackb(fake_socket.send.call_args.args[0], raw=False)
    assert epoch_payload[0] == Headers.SET_GENERATION_EPOCH.value
    assert epoch_payload[1] == 42

    # Submit a second request so we have a pending future for stop() to cancel.
    pending = client.add_request("p2", SamplingParams())

    # stop(): cancels listener, cancels pending futures, closes socket + terminates ctx.
    client.stop()
    await asyncio.sleep(0)  # allow cancellation to propagate
    assert client.listener_task.cancelled() or client.listener_task.done()
    assert pending.cancelled()
    assert client.completion_futures == {}
    fake_socket.close.assert_called_once_with(linger=0)
    fake_context.term.assert_called_once_with()


async def test_inference_client_connect_handshake_rejects_unexpected_reply():
    """If the coordinator replies with anything other than CONNECT_ACK during
    the handshake, the client raises AssertionError synchronously — this is a
    fatal protocol mismatch, not a recoverable error. Separated from the
    lifecycle test because it short-circuits before any state is established."""
    client, _, fake_socket = _make_client()
    fake_socket.recv.return_value = msgpack.packb([Headers.STOP.value], use_bin_type=True)
    with pytest.raises(AssertionError):
        client._connect_with_inference_coordinator()


async def test_add_request_with_kv_handoff_returns_future():
    """Non-streaming handoffs use the normal final-reply future path."""
    client, _, fake_socket = _make_client()
    params = SamplingParams(temperature=0.5)

    future = client.add_request_with_kv_handoff([1, 2, 3], params, {"agent": "prefill"}, [10, 11])

    assert isinstance(future, asyncio.Future)
    assert params.streaming is False
    assert client.completion_futures == {0: future}
    assert client.streams == {}
    payload = msgpack.unpackb(fake_socket.send.call_args.args[0], raw=False)
    assert payload[0] == Headers.SUBMIT_REQUEST_WITH_KV.value
    assert payload[1] == 0
    assert payload[2] == [1, 2, 3]
    assert payload[3]["streaming"] is False
    assert payload[4] == {"agent": "prefill"}
    assert payload[5] == [10, 11]
    future.cancel()


async def test_add_request_with_id_returns_the_id_abort_needs():
    """The id handed back is the one that reaches the coordinator as ABORT_REQUEST.

    A non-streaming HTTP response writes nothing to the socket while it
    generates, so a client that disconnects mid-generation is never discovered
    as a broken pipe. The handler has to abort explicitly, and abort_request
    takes an id -- with only the future in hand there is nothing to name, and
    cancelling the future alone leaves the engine generating.
    """
    client, _, fake_socket = _make_client()

    request_id, future = client.add_request_with_id("hello", SamplingParams())

    assert isinstance(future, asyncio.Future)
    assert client.completion_futures == {request_id: future}
    submitted = msgpack.unpackb(fake_socket.send.call_args.args[0], raw=False)
    assert submitted[0] == Headers.SUBMIT_REQUEST.value
    assert submitted[1] == request_id

    client.abort_request(request_id)

    aborted = msgpack.unpackb(fake_socket.send.call_args.args[0], raw=False)
    assert aborted == [Headers.ABORT_REQUEST.value, request_id]
    # The abort has to drop local state too, or the handler leaks a future that
    # nothing will ever resolve.
    assert request_id not in client.completion_futures
    assert request_id in client.aborted_request_ids


async def test_add_request_delegates_to_add_request_with_id():
    """add_request keeps returning only a future, drawing from the same id sequence."""
    client, _, _ = _make_client()

    first = client.add_request("a", SamplingParams())
    second_id, second = client.add_request_with_id("b", SamplingParams())

    assert isinstance(first, asyncio.Future)
    assert second_id == 1, "add_request must consume an id from the same counter"
    assert client.completion_futures == {0: first, 1: second}

    first.cancel()
    second.cancel()


async def test_abort_requests_helper_is_best_effort():
    """One failing abort must not stop the others, and must not raise.

    abort_requests runs on paths that are already unwinding -- a handler
    cancelled by client disconnect, or one returning a 500 -- where letting a
    second exception escape would replace the real one.
    """
    client = MagicMock()
    client.abort_request.side_effect = [None, RuntimeError("coordinator gone"), None]

    abort_requests(client, [7, 8, 9], "client disconnected")

    assert [call.args[0] for call in client.abort_request.call_args_list] == [7, 8, 9]

    client.reset_mock(side_effect=True)
    abort_requests(client, [], "client disconnected")
    client.abort_request.assert_not_called()
