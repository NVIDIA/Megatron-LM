# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio

import msgpack
import pytest
import zmq
import zmq.asyncio

from megatron.core.inference.headers import Headers
from megatron.inference.integrations.dynamo.handoff_channel import HandoffChannel

_ACK = (Headers.RELEASE_KV_ACK.value, 7, "source")
_RELEASE = [Headers.RELEASE_KV.value, 7, "source"]


@pytest.fixture
def router():
    context = zmq.asyncio.Context()
    socket = context.socket(zmq.ROUTER)
    port = socket.bind_to_random_port("tcp://127.0.0.1")
    yield socket, context, f"tcp://127.0.0.1:{port}"
    socket.close(linger=0)
    context.term()


@pytest.mark.asyncio
async def test_late_reply_resolves_abandoned_request_not_its_retry(router):
    socket, context, address = router
    channel = HandoffChannel(context, address)
    try:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(channel.request(_RELEASE, _ACK), 0.05)
        retry = asyncio.create_task(channel.request(_RELEASE, _ACK))
        identity = (await socket.recv_multipart())[0]
        await socket.recv_multipart()
        await socket.send_multipart([identity, msgpack.packb([*_ACK, "late"])])
        await asyncio.sleep(0.05)
        assert not retry.done()
        await socket.send_multipart([identity, msgpack.packb([*_ACK, "retry"])])
        assert await asyncio.wait_for(retry, 1) == [*_ACK, "retry"]
    finally:
        await channel.aclose()


@pytest.mark.asyncio
async def test_retired_channel_finishes_waiting_requests_before_closing(router):
    socket, context, address = router
    channel = HandoffChannel(context, address)
    try:
        waiting = asyncio.create_task(channel.request(_RELEASE, _ACK))
        identity = (await socket.recv_multipart())[0]
        channel.retire()
        with pytest.raises(ConnectionError):
            await channel.request(_RELEASE, _ACK)
        assert not channel.closed
        await socket.send_multipart([identity, msgpack.packb(list(_ACK))])
        assert await asyncio.wait_for(waiting, 1) == list(_ACK)
        assert channel.closed
    finally:
        await channel.aclose()
