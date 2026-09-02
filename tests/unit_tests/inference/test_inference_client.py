# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
from unittest.mock import MagicMock, patch

import msgpack
import pytest
import zmq

from megatron.core.inference.headers import Headers
from megatron.core.inference.inference_client import InferenceClient, InferenceRequestError
from megatron.core.inference.sampling_params import SamplingParams

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


async def test_abort_preserves_fire_and_forget_api_and_wait_uses_running_loop():
    client, _, _ = _make_client()

    assert client.abort_request(7) is None
    future = client.abort_request_and_wait(8)

    assert future.get_loop() is asyncio.get_running_loop()
    future.cancel()


async def test_terminal_error_and_abort_acknowledgement():
    client, _, fake_socket = _make_client()
    recv_queue = [msgpack.packb([Headers.CONNECT_ACK.value], use_bin_type=True)]

    def fake_recv(*args, **kwargs):
        if recv_queue:
            return recv_queue.pop(0)
        raise zmq.Again()

    fake_socket.recv.side_effect = fake_recv
    client.start()

    failed = client.add_request("failed", SamplingParams())
    recv_queue.append(msgpack.packb([Headers.REQUEST_ERROR.value, 0, "transfer failed", True]))
    with pytest.raises(InferenceRequestError, match="transfer failed") as error:
        await asyncio.wait_for(failed, timeout=2.0)
    assert error.value.source_safe

    unsafe = client.add_request("unsafe", SamplingParams())
    recv_queue.append(msgpack.packb([Headers.REQUEST_ERROR.value, 1, "read failed", False]))
    with pytest.raises(InferenceRequestError, match="read failed") as error:
        await asyncio.wait_for(unsafe, timeout=2.0)
    assert not error.value.source_safe
    abort_ack = client.abort_request_and_wait(1)
    assert not abort_ack.done()
    abort_payload = msgpack.unpackb(fake_socket.send.call_args.args[0], raw=False)
    assert abort_payload == [Headers.ABORT_REQUEST.value, 1]
    recv_queue.append(msgpack.packb([Headers.REQUEST_ABORTED.value, 1, True]))
    assert await asyncio.wait_for(abort_ack, timeout=2.0)
    await asyncio.sleep(0)
    assert 1 not in client.abort_futures
    assert 1 not in client.aborted_request_ids

    unsafe = client.add_request("unsafe", SamplingParams())
    recv_queue.append(msgpack.packb([Headers.REQUEST_ERROR.value, 2, "read failed", False]))
    with pytest.raises(InferenceRequestError):
        await asyncio.wait_for(unsafe, timeout=2.0)
    assert 2 not in client.abort_futures
    pending_ack = client.abort_request_and_wait(2)
    abort_payload = msgpack.unpackb(fake_socket.send.call_args.args[0], raw=False)
    assert abort_payload == [Headers.ABORT_REQUEST.value, 2]
    recv_queue.append(msgpack.packb([Headers.REQUEST_ABORTED.value, 2, True]))
    assert await asyncio.wait_for(asyncio.shield(pending_ack), timeout=2.0)
    await asyncio.sleep(0)
    assert 2 not in client.abort_futures
    assert 2 not in client.aborted_request_ids

    recv_queue.append(msgpack.packb([Headers.ENGINE_REPLY.value, 1, {}], use_bin_type=True))
    await asyncio.sleep(0.01)
    assert not client.listener_task.done()

    client.stop()


async def test_fire_and_forget_abort_acknowledgement_clears_late_reply_guard():
    client, _, fake_socket = _make_client()
    recv_queue = [msgpack.packb([Headers.CONNECT_ACK.value], use_bin_type=True)]

    def fake_recv(*args, **kwargs):
        if recv_queue:
            return recv_queue.pop(0)
        raise zmq.Again()

    fake_socket.recv.side_effect = fake_recv
    client.start()

    request_id, _ = client.add_request_with_id("aborted", SamplingParams())
    client.abort_request(request_id)
    assert request_id in client.aborted_request_ids
    assert request_id not in client.abort_futures

    recv_queue.append(
        msgpack.packb([Headers.REQUEST_ABORTED.value, request_id, True], use_bin_type=True)
    )
    for _ in range(100):
        if request_id not in client.aborted_request_ids:
            break
        await asyncio.sleep(0.005)

    assert request_id not in client.aborted_request_ids
    client.stop()


async def test_add_request_with_id_returns_the_id_abort_needs():
    """The id handed back is the one that reaches the coordinator as ABORT_REQUEST.

    A non-streaming HTTP response writes nothing to the socket while it
    generates, so a client that disconnects mid-generation is never discovered
    as a broken pipe. The handler has to abort explicitly, and abort_request
    takes an id -- with only the future in hand there is nothing to name, and
    cancelling the future alone leaves the engine generating. add_request keeps
    returning the future alone, delegating here for the id sequence.
    """
    client, _, fake_socket = _make_client()

    delegated = client.add_request("a", SamplingParams())
    request_id, future = client.add_request_with_id("hello", SamplingParams())

    assert isinstance(future, asyncio.Future)
    assert request_id == 1, "add_request must consume an id from the same counter"
    assert client.completion_futures == {0: delegated, request_id: future}
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

    delegated.cancel()


def _consume_reply_for_a_completed_request(client):
    """Submit, then stand in for _recv_task delivering the final reply."""
    request_id, future = client.add_request_with_id("hello", SamplingParams())
    # _recv_task pops the future from completion_futures immediately before
    # resolving it.
    client.completion_futures.pop(request_id)
    future.set_result({"tokens": [1]})
    return request_id


def _consume_reply_for_a_finished_stream(client):
    """Same, for the streaming path, whose AsyncStream aborts on close."""
    stream = client.add_request_streaming("hello", SamplingParams())
    # _recv_task pops the stream before delivering the final frame.
    client.streams.pop(stream.request_id)
    return stream.request_id


@pytest.mark.parametrize(
    "finish_request",
    [_consume_reply_for_a_completed_request, _consume_reply_for_a_finished_stream],
    ids=["completed_future", "finished_stream"],
)
async def test_abort_request_ignores_a_request_with_no_local_state(finish_request):
    """A completed request must not be recorded in aborted_request_ids.

    That set is pruned in exactly one place -- _recv_task, when an ENGINE_REPLY
    arrives for the id. Once the reply has been consumed no further reply will
    ever arrive, so recording the id there leaks an entry for the lifetime of
    the process: the same unbounded growth this abort path exists to prevent,
    moved from the engine's batch into the client. The new callers hit this
    routinely -- gather propagates on the first failure while siblings that
    already succeeded are aborted after completion.
    """
    client, _, fake_socket = _make_client()

    request_id = finish_request(client)
    fake_socket.send.reset_mock()

    client.abort_request(request_id)

    assert request_id not in client.aborted_request_ids
    # The coordinator has already dropped its mapping for a finished request,
    # so the ABORT_REQUEST send would be wasted too.
    fake_socket.send.assert_not_called()
