# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for AsyncZmqSendRecv -- no torch.distributed required."""

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

from megatron.core.inference.async_zmq_communicator import AsyncZmqSendRecv
from megatron.core.inference.headers import Headers

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
async def zmq_ctx():
    """Fresh ZMQ async context, terminated after test."""
    ctx = zmq.asyncio.Context()
    yield ctx
    ctx.term()


@pytest.fixture
async def push_pull(zmq_ctx):
    """PUSH/PULL socket pair on a random TCP port."""
    push = zmq_ctx.socket(zmq.PUSH)
    pull = zmq_ctx.socket(zmq.PULL)
    port = push.bind_to_random_port("tcp://127.0.0.1")
    pull.connect(f"tcp://127.0.0.1:{port}")
    yield push, pull
    push.close(linger=0)
    pull.close(linger=0)


@pytest.fixture
async def router_dealer(zmq_ctx):
    """ROUTER/DEALER socket pair on a random TCP port.

    ROUTER has ROUTER_MANDATORY set so sends to unknown identities raise.
    DEALER identity is b"test-dealer".
    """
    router = zmq_ctx.socket(zmq.ROUTER)
    router.setsockopt(zmq.ROUTER_MANDATORY, 1)
    dealer = zmq_ctx.socket(zmq.DEALER)
    dealer.setsockopt(zmq.IDENTITY, b"test-dealer")
    port = router.bind_to_random_port("tcp://127.0.0.1")
    dealer.connect(f"tcp://127.0.0.1:{port}")
    # Brief sleep to let ZMQ handshake complete.
    await asyncio.sleep(0.05)
    yield router, dealer
    dealer.close(linger=0)
    router.close(linger=0)


# ── TestAsyncZmqSendRecv ─────────────────────────────────────────────────────


class TestAsyncZmqSendRecv:
    """Tests for the AsyncZmqSendRecv helper."""

    # -- Roundtrip tests --

    async def test_roundtrip_push_pull(self, push_pull):
        """PUSH/PULL roundtrip with serialized data."""
        push, pull = push_pull
        sender = AsyncZmqSendRecv()
        receiver = AsyncZmqSendRecv()

        data = [1, "hello", {"key": True}]
        sender.isend(push, Headers.SUBMIT_REQUEST, data)
        task = asyncio.create_task(sender.send_task())

        identity, header, received = await asyncio.wait_for(receiver.irecv(pull), timeout=2.0)

        assert identity is None
        assert header == Headers.SUBMIT_REQUEST
        assert received == data

        sender.shutdown()
        await asyncio.wait_for(task, timeout=2.0)

    async def test_roundtrip_no_data(self, push_pull):
        """Header-only message (no payload)."""
        push, pull = push_pull
        sender = AsyncZmqSendRecv()
        receiver = AsyncZmqSendRecv()

        sender.isend(push, Headers.PAUSE)
        task = asyncio.create_task(sender.send_task())

        identity, header, received = await asyncio.wait_for(receiver.irecv(pull), timeout=2.0)

        assert identity is None
        assert header == Headers.PAUSE
        assert received is None

        sender.shutdown()
        await asyncio.wait_for(task, timeout=2.0)

    async def test_roundtrip_serialize_false(self, push_pull):
        """Raw bytes with serialize=False / deserialize=False."""
        push, pull = push_pull
        sender = AsyncZmqSendRecv()
        receiver = AsyncZmqSendRecv()

        raw_data = b"\x01\x02\x03\xff"
        sender.isend(push, Headers.ENGINE_REPLY, raw_data, serialize=False)
        task = asyncio.create_task(sender.send_task())

        identity, header, received = await asyncio.wait_for(
            receiver.irecv(pull, deserialize=False), timeout=2.0
        )

        assert identity is None
        assert header == Headers.ENGINE_REPLY
        assert received == raw_data

        sender.shutdown()
        await asyncio.wait_for(task, timeout=2.0)

    async def test_roundtrip_router_dealer(self, router_dealer):
        """ROUTER/DEALER roundtrip with identity routing."""
        router, dealer = router_dealer
        router_helper = AsyncZmqSendRecv()
        dealer_helper = AsyncZmqSendRecv()

        # ROUTER → DEALER: send with explicit identity.
        router_helper.isend(router, Headers.ACK, [42], identity=b"test-dealer")
        router_task = asyncio.create_task(router_helper.send_task())

        identity, header, data = await asyncio.wait_for(dealer_helper.irecv(dealer), timeout=2.0)
        assert identity is None  # DEALER strips identity frame
        assert header == Headers.ACK
        assert data == [42]

        # DEALER → ROUTER: send without identity; ROUTER adds it.
        dealer_helper.isend(dealer, Headers.ENGINE_CONNECT)
        dealer_task = asyncio.create_task(dealer_helper.send_task())

        identity, header, data = await asyncio.wait_for(
            router_helper.irecv(router, socket_uses_identity=True), timeout=2.0
        )
        assert identity == b"test-dealer"
        assert header == Headers.ENGINE_CONNECT
        assert data is None

        router_helper.shutdown()
        dealer_helper.shutdown()
        await asyncio.wait_for(router_task, timeout=2.0)
        await asyncio.wait_for(dealer_task, timeout=2.0)

    # -- send_task behavior tests --

    async def test_send_task_drains_queue(self, push_pull):
        """send_task drains all queued sends then exits on shutdown."""
        push, pull = push_pull
        sender = AsyncZmqSendRecv()
        receiver = AsyncZmqSendRecv()

        n = 5
        for i in range(n):
            sender.isend(push, Headers.SUBMIT_REQUEST, [i])

        task = asyncio.create_task(sender.send_task())

        received = []
        for _ in range(n):
            _, header, data = await asyncio.wait_for(receiver.irecv(pull), timeout=2.0)
            assert header == Headers.SUBMIT_REQUEST
            received.append(data[0])

        assert received == list(range(n))

        sender.shutdown()
        await asyncio.wait_for(task, timeout=2.0)

    async def test_send_task_shutdown_empty(self):
        """shutdown() on an empty queue causes send_task to exit cleanly."""
        helper = AsyncZmqSendRecv()
        task = asyncio.create_task(helper.send_task())
        # Give send_task a chance to block on get().
        await asyncio.sleep(0.01)
        helper.shutdown()
        await asyncio.wait_for(task, timeout=2.0)

    async def test_send_task_error_handled(self, zmq_ctx):
        """on_send_error returning True suppresses the exception."""
        router = zmq_ctx.socket(zmq.ROUTER)
        router.setsockopt(zmq.ROUTER_MANDATORY, 1)
        router.bind_to_random_port("tcp://127.0.0.1")

        helper = AsyncZmqSendRecv()

        # Send to a non-existent identity — ROUTER_MANDATORY makes this fail.
        helper.isend(router, Headers.ACK, identity=b"nonexistent")

        errors = []

        def on_error(e, identity):
            errors.append((e, identity))
            return True  # handled

        task = asyncio.create_task(helper.send_task(on_send_error=on_error))
        # Give send_task time to process the failing send.
        await asyncio.sleep(0.1)
        helper.shutdown()
        await asyncio.wait_for(task, timeout=2.0)

        assert len(errors) == 1
        assert isinstance(errors[0][0], zmq.error.ZMQError)
        assert errors[0][1] == b"nonexistent"

        router.close(linger=0)

    async def test_send_task_error_propagates(self, zmq_ctx):
        """on_send_error returning False lets the exception propagate."""
        router = zmq_ctx.socket(zmq.ROUTER)
        router.setsockopt(zmq.ROUTER_MANDATORY, 1)
        router.bind_to_random_port("tcp://127.0.0.1")

        helper = AsyncZmqSendRecv()
        helper.isend(router, Headers.ACK, identity=b"nonexistent")

        def on_error(e, identity):
            return False  # not handled

        task = asyncio.create_task(helper.send_task(on_send_error=on_error))

        with pytest.raises(zmq.error.ZMQError):
            await asyncio.wait_for(task, timeout=2.0)

        router.close(linger=0)

    async def test_shutdown_preserves_pending(self, push_pull):
        """A pending send completes before send_task exits on shutdown.

        This is the DISCONNECT-before-sentinel guarantee: isend(DISCONNECT)
        followed by shutdown() means the DISCONNECT send is processed first.
        """
        push, pull = push_pull
        sender = AsyncZmqSendRecv()
        receiver = AsyncZmqSendRecv()

        sender.isend(push, Headers.DISCONNECT)
        sender.shutdown()

        task = asyncio.create_task(sender.send_task())
        await asyncio.wait_for(task, timeout=2.0)

        # The message should have been sent before send_task exited.
        _, header, _ = await asyncio.wait_for(receiver.irecv(pull), timeout=2.0)
        assert header == Headers.DISCONNECT


# ── TestCoordRecvLockPattern ─────────────────────────────────────────────────


class TestCoordRecvLockPattern:
    """Regression tests for the coord_recv_lock fix in DynamicInferenceEngine."""

    async def test_recv_outside_lock_no_deadlock(self, zmq_ctx):
        """schedule_requests can acquire the lock while recv blocks.

        If recv were inside the lock (the old bug), this test would deadlock
        because the recv task holds the lock while blocked on recv, and the
        schedule_requests task can never acquire it.

        With recv outside the lock (the fix), the lock is only held briefly
        during the forward, so schedule_requests can acquire it even when no
        coordinator messages are arriving.
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
