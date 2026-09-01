# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
from unittest.mock import MagicMock, patch

import msgpack
import pytest
import zmq

from megatron.core.inference.headers import Headers
from megatron.core.inference.inference_client import InferenceClient
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
    # We stage two recv_multipart() replies: the CONNECT_ACK during handshake,
    # and an ENGINE_REPLY for the request we'll add below. Messages arrive as
    # [metadata, body] frames; the body is only present on replies that carry
    # one. Subsequent recvs raise zmq.Again so the listener yields back to the
    # event loop.
    recv_queue = [
        [msgpack.packb([Headers.CONNECT_ACK.value], use_bin_type=True)],
        [
            msgpack.packb([Headers.ENGINE_REPLY.value, 0], use_bin_type=True),
            msgpack.packb({"foo": "bar"}, use_bin_type=True),
        ],
    ]

    def fake_recv(*args, **kwargs):
        if recv_queue:
            return recv_queue.pop(0)
        raise zmq.Again()

    fake_socket.recv_multipart.side_effect = fake_recv

    client.start()
    assert isinstance(client.listener_task, asyncio.Task)
    sent_connect = fake_socket.send.call_args.args[0]
    assert msgpack.unpackb(sent_connect, raw=False)[0] == Headers.CONNECT.value

    # add_request frames the submission as [metadata, prompt, block_hashes, media]
    # so the coordinator can route it without decoding the prompt or the media.
    fut = client.add_request("hello", SamplingParams(temperature=0.5))
    assert isinstance(fut, asyncio.Future)
    assert client.next_request_id == 1
    assert 0 in client.request_submission_times
    submit_meta, submit_prompt, submit_hashes, submit_media = (
        fake_socket.send_multipart.call_args.args[0]
    )
    submit_payload = msgpack.unpackb(submit_meta, raw=False)
    assert submit_payload[0] == Headers.SUBMIT_REQUEST.value
    assert submit_payload[1] == 0
    assert submit_payload[2]["temperature"] == 0.5
    assert msgpack.unpackb(submit_prompt, raw=False) == "hello"
    # This client was told no block size, so it reports None -- "I did not hash" --
    # and the coordinator hashes on its behalf.
    assert msgpack.unpackb(submit_hashes, raw=False) is None
    # Text-only, so the media frame is present but empty: the frame count is fixed
    # so the coordinator can reject a malformed submission on arity alone.
    assert msgpack.unpackb(submit_media, raw=False) is None

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
    fake_socket.recv_multipart.return_value = [
        msgpack.packb([Headers.STOP.value], use_bin_type=True)
    ]
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
    # Framed as [metadata, prompt, src_block_ids]: src_block_ids names one block
    # per block_size_tokens of prompt, so it grows with the prompt and travels as
    # its own body rather than in the metadata frame the coordinator decodes.
    meta_frame, prompt_frame, blocks_frame = fake_socket.send_multipart.call_args.args[0]
    metadata = msgpack.unpackb(meta_frame, raw=False)
    assert metadata[0] == Headers.SUBMIT_REQUEST_WITH_KV.value
    assert metadata[1] == 0
    assert metadata[2]["streaming"] is False
    assert metadata[3] == {"agent": "prefill"}
    assert msgpack.unpackb(prompt_frame, raw=False) == [1, 2, 3]
    assert msgpack.unpackb(blocks_frame, raw=False) == [10, 11]
    future.cancel()


def _configured_client(policy=None, block_size=4):
    from megatron.core.inference.config import PrefixCachingCoordinatorPolicy

    fake_socket = MagicMock(name="zmq_socket")
    fake_context = MagicMock(name="zmq_context")
    fake_context.socket.return_value = fake_socket
    with patch("megatron.core.inference.inference_client.zmq.Context", return_value=fake_context):
        client = InferenceClient(
            "tcp://127.0.0.1:5555",
            block_size_tokens=block_size,
            prefix_caching_coordinator_policy=policy
            or PrefixCachingCoordinatorPolicy.LONGEST_PREFIX,
        )
    return client, fake_socket


def _hash_frame(fake_socket):
    return msgpack.unpackb(fake_socket.send_multipart.call_args.args[0][2], raw=False)


async def test_unconfigured_client_says_it_did_not_hash():
    """The regression that matters: None, never [].

    Callers that build an InferenceClient directly -- MegatronAsyncLLM,
    megatron.rl, the coordinator example -- configure neither the block size nor
    the policy. Reporting an empty list for them reads as "hashed, nothing
    matched", so the coordinator skips its own hashing and prefix-affinity
    routing silently degrades to load balancing with nothing raising.
    """
    fake_socket = MagicMock(name="zmq_socket")
    fake_context = MagicMock(name="zmq_context")
    fake_context.socket.return_value = fake_socket
    with patch("megatron.core.inference.inference_client.zmq.Context", return_value=fake_context):
        client = InferenceClient("tcp://127.0.0.1:5555")
    client.add_request(list(range(8)), SamplingParams()).cancel()
    assert _hash_frame(fake_socket) is None


async def test_configured_client_hashes_a_text_prompt():
    import torch

    from megatron.core.inference.inference_request import compute_block_hashes_batched

    client, fake_socket = _configured_client()
    tokens = list(range(8))
    client.add_request(tokens, SamplingParams()).cancel()
    assert _hash_frame(fake_socket) == compute_block_hashes_batched(
        torch.tensor(tokens, dtype=torch.int64), block_size=4
    )


async def test_multimodal_hashes_are_salted_with_the_media_key():
    """Same tokens, different media, disjoint hashes.

    The client is the only place holding both the tokens and the media key, so
    it is the only place that can salt them without deriving the key twice --
    deriving it digests the media, hundreds of MB for video.
    """
    import torch

    from megatron.core.inference.inference_request import (
        compute_block_hashes_batched,
        compute_media_cache_key,
    )

    client, fake_socket = _configured_client()
    tokens = list(range(8))

    client.add_request(tokens, SamplingParams()).cancel()
    text_hashes = _hash_frame(fake_socket)

    client.add_request(tokens, SamplingParams(), multi_modal_data={"image": b"jpeg"}).cancel()
    media_hashes = _hash_frame(fake_socket)

    assert set(text_hashes).isdisjoint(media_hashes)
    assert media_hashes == compute_block_hashes_batched(
        torch.tensor(tokens, dtype=torch.int64),
        block_size=4,
        cache_salt=compute_media_cache_key("image", [b"jpeg"]),
    )


async def test_media_bytes_travel_in_their_own_frame():
    """Media rides in a body frame; only its bounded descriptor is metadata.

    The coordinator decodes and repacks the metadata frame for every request, so
    anything unbounded in it is paid for on a loop shared by every rank. Raw
    video reaches hundreds of megabytes, which is three orders of magnitude more
    than the prompt decode the earlier split removed.
    """
    from megatron.core.inference.inference_request import compute_media_cache_key

    client, fake_socket = _configured_client()
    image = b"jpeg-payload"

    client.add_request(list(range(8)), SamplingParams(), multi_modal_data={"image": image}).cancel()
    meta_frame, _prompt, _hashes, media_frame = fake_socket.send_multipart.call_args.args[0]

    media_meta = msgpack.unpackb(meta_frame, raw=False)[3]
    assert media_meta == {
        "media_cache_key": compute_media_cache_key("image", [image]),
        "modality": "image",
    }
    # The bytes are in the body frame, and nowhere in the metadata frame.
    assert msgpack.unpackb(media_frame, raw=False) == [image]
    assert image not in meta_frame


async def test_media_frames_round_trip_back_to_serialized_form():
    """What the engine reassembles from the two frames is what was serialized."""
    from megatron.core.inference.inference_request import (
        merge_multimodal_data,
        serialize_multimodal_data,
    )

    client, fake_socket = _configured_client()
    multi_modal_data = {"image": [b"a", b"bb"], "media_tokens_preexpanded": True}

    client.add_request(list(range(8)), SamplingParams(), multi_modal_data=multi_modal_data).cancel()
    meta_frame, _prompt, _hashes, media_frame = fake_socket.send_multipart.call_args.args[0]

    reassembled = merge_multimodal_data(
        msgpack.unpackb(meta_frame, raw=False)[3], msgpack.unpackb(media_frame, raw=False)
    )
    assert reassembled == serialize_multimodal_data(multi_modal_data)


async def test_load_balanced_policy_skips_hashing():
    """Nobody reads them under LOAD_BALANCED, so computing them is pure overhead."""
    from megatron.core.inference.config import PrefixCachingCoordinatorPolicy

    client, fake_socket = _configured_client(PrefixCachingCoordinatorPolicy.LOAD_BALANCED)
    client.add_request(list(range(8)), SamplingParams()).cancel()
    assert _hash_frame(fake_socket) == []


async def test_string_prompt_is_left_to_the_coordinator():
    """Hashing needs token ids, and the client has no tokenizer."""
    client, fake_socket = _configured_client()
    client.add_request("some prompt text", SamplingParams()).cancel()
    assert _hash_frame(fake_socket) is None
