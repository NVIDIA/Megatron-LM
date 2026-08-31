# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for ``InferenceClient.add_request_streaming``."""

import asyncio
from unittest.mock import MagicMock, patch

import msgpack
import pytest
import zmq

from megatron.core.inference.async_stream import AsyncStream
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

    # Partials and finals both arrive as [metadata, body] frames.
    recv_queue = [
        [msgpack.packb([Headers.CONNECT_ACK.value], use_bin_type=True)],
        [
            msgpack.packb([Headers.ENGINE_REPLY_PARTIAL.value, 0], use_bin_type=True),
            msgpack.packb(
                {"request_id": 0, "new_tokens": [1, 2], "new_log_probs": [-0.1, -0.2]},
                use_bin_type=True,
            ),
        ],
        [
            msgpack.packb([Headers.ENGINE_REPLY_PARTIAL.value, 0], use_bin_type=True),
            msgpack.packb(
                {"request_id": 0, "new_tokens": [3], "new_log_probs": [-0.3]}, use_bin_type=True
            ),
        ],
        [
            msgpack.packb([Headers.ENGINE_REPLY.value, 0], use_bin_type=True),
            msgpack.packb({"request_id": 0, "generated_tokens": [1, 2, 3]}, use_bin_type=True),
        ],
    ]

    def fake_recv(*args, **kwargs):
        if recv_queue:
            return recv_queue.pop(0)
        raise zmq.Again()

    fake_socket.recv_multipart.side_effect = fake_recv
    client.start()

    params = SamplingParams(temperature=0.7, return_log_probs=True)
    assert params.streaming is False  # default

    iterator = client.add_request_streaming("hi", params)

    assert isinstance(iterator, AsyncStream)
    assert params.streaming is True
    assert 0 in client.streams
    submit_meta, _ = fake_socket.send_multipart.call_args.args[0]
    submit_payload = msgpack.unpackb(submit_meta, raw=False)
    assert submit_payload[0] == Headers.SUBMIT_REQUEST.value
    assert submit_payload[2]["streaming"] is True

    items = []
    async for item in iterator:
        items.append(item)

    assert len(items) == 3
    assert items[0] == {
        "partial": {"request_id": 0, "new_tokens": [1, 2], "new_log_probs": [-0.1, -0.2]}
    }
    assert items[1] == {"partial": {"request_id": 0, "new_tokens": [3], "new_log_probs": [-0.3]}}
    assert "final" in items[2]
    assert items[2]["final"]["generated_tokens"] == [1, 2, 3]

    assert 0 not in client.streams
    assert 0 not in client.request_submission_times

    client.stop()


async def test_streaming_partial_for_unknown_request_is_dropped():
    """ENGINE_REPLY_PARTIAL frames whose request_id has no stream are silently ignored."""
    client, fake_socket = _make_client()

    recv_queue = [
        [msgpack.packb([Headers.CONNECT_ACK.value], use_bin_type=True)],
        [
            msgpack.packb([Headers.ENGINE_REPLY_PARTIAL.value, 42], use_bin_type=True),
            msgpack.packb({"request_id": 42, "new_tokens": [9]}, use_bin_type=True),
        ],
    ]

    def fake_recv(*args, **kwargs):
        if recv_queue:
            return recv_queue.pop(0)
        raise zmq.Again()

    fake_socket.recv_multipart.side_effect = fake_recv
    client.start()

    await asyncio.sleep(0.02)
    assert client.streams == {}

    client.stop()


async def test_client_stop_terminates_open_streams():
    """stop() finishes open streams so awaiters can exit."""
    client, fake_socket = _make_client()

    recv_queue = [[msgpack.packb([Headers.CONNECT_ACK.value], use_bin_type=True)]]

    def fake_recv(*args, **kwargs):
        if recv_queue:
            return recv_queue.pop(0)
        raise zmq.Again()

    fake_socket.recv_multipart.side_effect = fake_recv
    client.start()

    iterator = client.add_request_streaming("hi", SamplingParams())
    client.stop()

    items = [item async for item in iterator]
    assert items == []


async def test_stream_close_sends_abort_and_terminates_iterator():
    """Closing a live stream sends ABORT_REQUEST and wakes its consumer."""
    client, fake_socket = _make_client()
    iterator = client.add_request_streaming("hi", SamplingParams())

    await iterator.aclose()

    payload = msgpack.unpackb(fake_socket.send.call_args.args[0], raw=False)
    assert payload == [Headers.ABORT_REQUEST.value, iterator.request_id]
    assert iterator.request_id not in client.streams
    assert iterator.request_id not in client.request_submission_times
    assert [item async for item in iterator] == []


async def test_add_request_with_kv_handoff_streaming_returns_stream():
    """Streaming handoffs opt into partial replies and register an AsyncStream."""
    client, fake_socket = _make_client()
    params = SamplingParams(temperature=0.5)

    stream = client.add_request_with_kv_handoff_streaming(
        [1, 2, 3], params, {"agent": "prefill"}, [10, 11]
    )

    assert isinstance(stream, AsyncStream)
    assert params.streaming is True
    assert client.streams == {0: stream}
    assert client.completion_futures == {}
    # Framed as [metadata, prompt, src_block_ids]: src_block_ids names one block
    # per block_size_tokens of prompt, so it grows with the prompt and travels as
    # its own body rather than in the metadata frame the coordinator decodes.
    meta_frame, prompt_frame, blocks_frame = fake_socket.send_multipart.call_args.args[0]
    metadata = msgpack.unpackb(meta_frame, raw=False)
    assert metadata[0] == Headers.SUBMIT_REQUEST_WITH_KV.value
    assert metadata[1] == 0
    assert metadata[2]["streaming"] is True
    assert metadata[3] == {"agent": "prefill"}
    assert msgpack.unpackb(prompt_frame, raw=False) == [1, 2, 3]
    assert msgpack.unpackb(blocks_frame, raw=False) == [10, 11]

    await stream.aclose()
