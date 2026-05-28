# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for InferenceClient.add_request_streaming.

The streaming contract (see InferenceClient.add_request_streaming):

- ``SamplingParams.streaming`` is forced True before submission.
- The submit payload uses the existing ``SUBMIT_REQUEST`` header; the engine
  reads ``streaming`` off the sampling params to decide whether to emit
  partials.
- ENGINE_REPLY_PARTIAL frames arrive as ``[header, request_id, partial_dict]``
  and surface as ``{"partial": partial_dict}`` items on the iterator.
- A terminal ENGINE_REPLY surfaces as a single ``{"final": reply}`` item, after
  which the iterator stops.
"""

import asyncio
from unittest.mock import MagicMock, patch

import msgpack
import pytest
import zmq

from megatron.core.inference.headers import Headers
from megatron.core.inference.inference_client import InferenceClient
from megatron.core.inference.sampling_params import SamplingParams

pytestmark = pytest.mark.asyncio


def _make_client():
    fake_socket = MagicMock(name="zmq_socket")
    fake_context = MagicMock(name="zmq_context")
    fake_context.socket.return_value = fake_socket
    with patch("megatron.core.inference.inference_client.zmq.Context", return_value=fake_context):
        client = InferenceClient("tcp://127.0.0.1:5555", deserialize=False)
    return client, fake_socket


async def test_add_request_streaming_emits_partials_then_final():
    """Two ENGINE_REPLY_PARTIAL frames followed by an ENGINE_REPLY terminate the iterator."""
    client, fake_socket = _make_client()

    recv_queue = [
        msgpack.packb([Headers.CONNECT_ACK.value], use_bin_type=True),
        msgpack.packb(
            [Headers.ENGINE_REPLY_PARTIAL.value, 0, {"request_id": 0, "new_tokens": [1, 2]}],
            use_bin_type=True,
        ),
        msgpack.packb(
            [Headers.ENGINE_REPLY_PARTIAL.value, 0, {"request_id": 0, "new_tokens": [3]}],
            use_bin_type=True,
        ),
        msgpack.packb(
            [Headers.ENGINE_REPLY.value, 0, {"request_id": 0, "generated_tokens": [1, 2, 3]}],
            use_bin_type=True,
        ),
    ]

    def fake_recv(*args, **kwargs):
        if recv_queue:
            return recv_queue.pop(0)
        raise zmq.Again()

    fake_socket.recv.side_effect = fake_recv
    client.start()

    params = SamplingParams(temperature=0.7)
    assert params.streaming is False  # default

    iterator = client.add_request_streaming("hi", params)

    # Submission side-effects.
    assert params.streaming is True
    assert 0 in client.stream_queues
    submit_payload = msgpack.unpackb(fake_socket.send.call_args.args[0], raw=False)
    assert submit_payload[0] == Headers.SUBMIT_REQUEST.value
    assert submit_payload[3]["streaming"] is True

    # Drain the iterator.
    items = []
    async for item in iterator:
        items.append(item)

    assert len(items) == 3
    assert items[0] == {"partial": {"request_id": 0, "new_tokens": [1, 2]}}
    assert items[1] == {"partial": {"request_id": 0, "new_tokens": [3]}}
    assert "final" in items[2]
    assert items[2]["final"]["generated_tokens"] == [1, 2, 3]

    # Routing state for the request is fully cleaned up.
    assert 0 not in client.stream_queues
    assert 0 not in client.request_submission_times

    client.stop()


async def test_streaming_partial_for_unknown_request_is_dropped():
    """ENGINE_REPLY_PARTIAL frames whose request_id has no stream queue are silently ignored."""
    client, fake_socket = _make_client()

    recv_queue = [
        msgpack.packb([Headers.CONNECT_ACK.value], use_bin_type=True),
        msgpack.packb(
            [Headers.ENGINE_REPLY_PARTIAL.value, 42, {"request_id": 42, "new_tokens": [9]}],
            use_bin_type=True,
        ),
    ]

    def fake_recv(*args, **kwargs):
        if recv_queue:
            return recv_queue.pop(0)
        raise zmq.Again()

    fake_socket.recv.side_effect = fake_recv
    client.start()

    # Give the listener a chance to consume the stray partial.
    await asyncio.sleep(0.02)
    assert client.stream_queues == {}

    client.stop()


async def test_client_stop_terminates_open_streams():
    """stop() pushes the sentinel into open stream queues so awaiters can exit."""
    client, fake_socket = _make_client()

    recv_queue = [msgpack.packb([Headers.CONNECT_ACK.value], use_bin_type=True)]

    def fake_recv(*args, **kwargs):
        if recv_queue:
            return recv_queue.pop(0)
        raise zmq.Again()

    fake_socket.recv.side_effect = fake_recv
    client.start()

    iterator = client.add_request_streaming("hi", SamplingParams())
    client.stop()

    items = [item async for item in iterator]
    assert items == []
