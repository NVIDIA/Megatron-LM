# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for AsyncZmqEndpoint -- no torch.distributed required."""

import asyncio
import multiprocessing

import pytest

try:
    import zmq
    import zmq.asyncio

    HAVE_ZMQ = True
except ImportError:
    HAVE_ZMQ = False

try:
    import msgpack

    HAVE_MSGPACK = True
except ImportError:
    HAVE_MSGPACK = False

pytestmark = [
    pytest.mark.skipif(not HAVE_ZMQ, reason="pyzmq required"),
    pytest.mark.skipif(not HAVE_MSGPACK, reason="msgpack required"),
    pytest.mark.asyncio,
]

from megatron.core.inference.async_zmq_communicator import AsyncZmqEndpoint
from megatron.core.inference.headers import Headers

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
async def zmq_ctx():
    """Fresh ZMQ async context, terminated after test."""
    ctx = zmq.asyncio.Context()
    yield ctx
    ctx.term()


@pytest.fixture
async def push_pull():
    """PUSH/PULL AsyncZmqEndpoint pair on a random TCP port."""
    sender = AsyncZmqEndpoint("PUSH", bind=True)
    receiver = AsyncZmqEndpoint("PULL", connect=sender.address)
    sender.start()
    receiver.start()
    yield sender, receiver
    await sender.shutdown()
    await receiver.shutdown()


@pytest.fixture
async def router_dealer():
    """ROUTER/DEALER AsyncZmqEndpoint pair on a random TCP port."""
    router = AsyncZmqEndpoint("ROUTER", bind=True)
    dealer = AsyncZmqEndpoint("DEALER", identity="test-dealer", connect=router.address)
    router.start()
    dealer.start()
    # Brief sleep to let ZMQ handshake complete.
    await asyncio.sleep(0.05)
    yield router, dealer
    await router.shutdown()
    await dealer.shutdown()


# ── TestAsyncZmqEndpoint ────────────────────────────────────────────────────


class TestAsyncZmqEndpoint:
    """Tests for the AsyncZmqEndpoint send/recv and lifecycle."""

    # -- Roundtrip tests --

    async def test_roundtrip_push_pull(self, push_pull):
        """PUSH/PULL roundtrip with serialized data."""
        sender, receiver = push_pull

        data = [1, "hello", {"key": True}]
        sender._isend(Headers.SUBMIT_REQUEST, data)

        identity, header, received = await asyncio.wait_for(receiver._irecv(), timeout=2.0)

        assert identity is None
        assert header == Headers.SUBMIT_REQUEST
        assert received == data

    async def test_roundtrip_no_data(self, push_pull):
        """Header-only message (no payload)."""
        sender, receiver = push_pull

        sender._isend(Headers.PAUSE)

        identity, header, received = await asyncio.wait_for(receiver._irecv(), timeout=2.0)

        assert identity is None
        assert header == Headers.PAUSE
        assert received is None

    async def test_roundtrip_serialize_false(self, push_pull):
        """Raw bytes with serialize=False / deserialize=False."""
        sender, receiver = push_pull

        raw_data = b"\x01\x02\x03\xff"
        sender._isend(Headers.ENGINE_REPLY, raw_data, serialize=False)

        identity, header, received = await asyncio.wait_for(
            receiver._irecv(deserialize=False), timeout=2.0
        )

        assert identity is None
        assert header == Headers.ENGINE_REPLY
        assert received == raw_data

    async def test_roundtrip_router_dealer(self, router_dealer):
        """ROUTER/DEALER roundtrip with identity routing."""
        router, dealer = router_dealer

        # ROUTER -> DEALER: send with explicit identity.
        router._isend(Headers.ACK, [42], identity=b"test-dealer")

        identity, header, data = await asyncio.wait_for(dealer._irecv(), timeout=2.0)
        assert identity is None  # DEALER strips identity frame
        assert header == Headers.ACK
        assert data == [42]

        # DEALER -> ROUTER: send without identity; ROUTER adds it.
        dealer._isend(Headers.ENGINE_CONNECT)

        identity, header, data = await asyncio.wait_for(router._irecv(), timeout=2.0)
        assert identity == b"test-dealer"
        assert header == Headers.ENGINE_CONNECT
        assert data is None

    # -- Send queue behavior tests --

    async def test_send_drains_queue(self, push_pull):
        """Send queue drains and delivers multiple sends in order."""
        sender, receiver = push_pull

        n = 5
        for i in range(n):
            sender._isend(Headers.SUBMIT_REQUEST, [i])

        received = []
        for _ in range(n):
            _, header, data = await asyncio.wait_for(receiver._irecv(), timeout=2.0)
            assert header == Headers.SUBMIT_REQUEST
            received.append(data[0])

        assert received == list(range(n))

    async def test_shutdown_empty(self):
        """shutdown() on an endpoint with an empty send queue exits cleanly."""
        ep = AsyncZmqEndpoint("PUSH", bind=True)
        ep.start()
        await asyncio.sleep(0.01)
        await asyncio.wait_for(ep.shutdown(), timeout=2.0)

    async def test_ehostunreach_handled(self):
        """EHOSTUNREACH on ROUTER is logged as a warning, not raised."""
        router = AsyncZmqEndpoint("ROUTER", bind=True)
        router.start()

        # Send to a non-existent identity -- ROUTER_MANDATORY makes this fail
        # with EHOSTUNREACH.  The send task catches it and logs a warning.
        router._isend(Headers.ACK, identity=b"nonexistent")

        await asyncio.wait_for(router.shutdown(), timeout=2.0)

    async def test_shutdown_preserves_pending(self):
        """A send issued before shutdown() is delivered to the receiver.

        shutdown() drains the send queue before closing sockets, so the
        message is fully sent before the sender shuts down.
        """
        sender = AsyncZmqEndpoint("PUSH", bind=True)
        receiver = AsyncZmqEndpoint("PULL", connect=sender.address)
        sender.start()
        receiver.start()

        sender._isend(Headers.DISCONNECT)
        await asyncio.wait_for(sender.shutdown(), timeout=2.0)

        # The message should have been sent before shutdown completed.
        _, header, _ = await asyncio.wait_for(receiver._irecv(), timeout=2.0)
        assert header == Headers.DISCONNECT

        await receiver.shutdown()


# ── TestRecvDoesNotBlockOtherCoroutines ──────────────────────────────────────


class TestRecvDoesNotBlockOtherCoroutines:
    """Regression test: blocking recv must not prevent other coroutines from running."""

    async def test_recv_does_not_block_lock(self, zmq_ctx):
        """A coroutine blocked on recv does not prevent another from acquiring a lock.

        If recv were called inside a held lock (an old bug), this test would
        deadlock because the recv coroutine holds the lock while blocked on
        recv, and no other coroutine can acquire it.
        """
        lock = asyncio.Lock()

        # Create a PULL socket that will never receive anything.
        pull = zmq_ctx.socket(zmq.PULL)
        pull.bind_to_random_port("tcp://127.0.0.1")

        recv_started = asyncio.Event()

        async def mock_coord_recv_task():
            """Simulates _mp_coord_recv_task with recv OUTSIDE the lock."""
            recv_started.set()
            # This recv blocks forever (no sender), but does NOT hold the lock.
            await pull.recv_multipart()

        got_lock = asyncio.Event()

        async def mock_schedule_requests():
            """Simulates schedule_requests acquiring the lock."""
            await recv_started.wait()
            # Brief sleep to ensure recv task is blocked on recv.
            await asyncio.sleep(0.05)
            async with lock:
                got_lock.set()

        t1 = asyncio.create_task(mock_coord_recv_task())
        t2 = asyncio.create_task(mock_schedule_requests())

        # If this times out, the lock pattern is broken.
        await asyncio.wait_for(got_lock.wait(), timeout=2.0)

        t1.cancel()
        t2.cancel()
        try:
            await t1
        except asyncio.CancelledError:
            pass
        try:
            await t2
        except asyncio.CancelledError:
            pass

        pull.close(linger=0)


# ── TestStartupBuffering ─────────────────────────────────────────────────────


class _DummyTokenizer:
    """Minimal tokenizer for coordinator tests."""

    vocab_size = 10
    bos = None
    eod = 0
    pad = 0

    def tokenize(self, prompt):
        return [0]

    def detokenize(self, tokens, skip_special_tokens=False):
        return ""


class TestStartupBuffering:
    """Test that client sends are buffered until all engines connect."""

    async def test_client_ack_buffered_until_engine_connects(self):
        """Client ACK is not sent until the required engines have connected.

        1. Spawn coordinator with data_parallel_size=1.
        2. Connect client DEALER, send CLIENT_CONNECT.
        3. Verify no ACK arrives (coordinator buffers it).
        4. Connect engine DEALER, send ENGINE_CONNECT.
        5. Verify client now receives ACK.
        """
        from megatron.core.inference.data_parallel_inference_coordinator import (
            DataParallelInferenceCoordinator,
        )

        spawn_ctx = multiprocessing.get_context("spawn")
        pipe_parent, pipe_child = spawn_ctx.Pipe()
        ready_event = spawn_ctx.Event()

        proc = spawn_ctx.Process(
            target=DataParallelInferenceCoordinator.entrypoint,
            args=(pipe_child, ready_event, 1, _DummyTokenizer()),
        )
        proc.start()

        try:
            # Wait for coordinator to bind and send address.
            assert pipe_parent.poll(timeout=10.0), "Coordinator didn't start"
            addr = pipe_parent.recv()
            pipe_parent.close()

            ctx = zmq.asyncio.Context()

            # Connect a client.
            client = ctx.socket(zmq.DEALER)
            client.setsockopt(zmq.IDENTITY, b"test-client")
            client.connect(addr)
            await asyncio.sleep(0.05)

            # Send CLIENT_CONNECT.
            client.send_multipart([Headers.CLIENT_CONNECT.value.to_bytes()])

            # The coordinator should NOT send ACK yet (no engines connected).
            await asyncio.sleep(0.2)
            with pytest.raises(zmq.Again):
                client.recv_multipart(flags=zmq.NOBLOCK)

            # Now connect an engine.
            engine = ctx.socket(zmq.DEALER)
            engine.setsockopt(zmq.IDENTITY, b"test-engine")
            engine.connect(addr)
            await asyncio.sleep(0.05)
            engine.send_multipart([Headers.ENGINE_CONNECT.value.to_bytes()])

            # Client should now receive the buffered ACK.
            frames = await asyncio.wait_for(client.recv_multipart(), timeout=5.0)
            header = Headers(int.from_bytes(frames[0]))
            assert header == Headers.ACK

            # Clean up: send SHUTDOWN.
            client.send_multipart([Headers.SHUTDOWN.value.to_bytes()])
            await asyncio.sleep(0.1)

            engine.close(linger=0)
            client.close(linger=0)
            ctx.term()
        finally:
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2.0)
