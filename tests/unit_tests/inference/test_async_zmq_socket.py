# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for AsyncZmqEndpoint and coordinator -- no torch.distributed required."""

import asyncio
import multiprocessing
from collections import deque

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

try:
    import torch  # noqa: F401

    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

pytestmark = [
    pytest.mark.skipif(not HAVE_ZMQ, reason="pyzmq required"),
    pytest.mark.skipif(not HAVE_MSGPACK, reason="msgpack required"),
    pytest.mark.asyncio,
]

from megatron.core.inference.async_zmq_communicator import AsyncZmqEndpoint
from megatron.core.inference.data_parallel_inference_coordinator import (
    DataParallelInferenceCoordinator,
)
from megatron.core.inference.headers import Headers

if HAVE_TORCH:
    from megatron.core.inference.engines.dynamic_engine import EngineState
    from megatron.core.inference.engines.engine_coordinator_client import (
        EngineCoordinatorClient,
    )


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


def _spawn_coordinator(dp_size=1):
    """Spawn a DataParallelInferenceCoordinator process, return (proc, addr, ready_event)."""
    spawn_ctx = multiprocessing.get_context("spawn")
    pipe_parent, pipe_child = spawn_ctx.Pipe()
    ready_event = spawn_ctx.Event()

    proc = spawn_ctx.Process(
        target=DataParallelInferenceCoordinator.entrypoint,
        args=(pipe_child, ready_event, dp_size, _DummyTokenizer()),
    )
    proc.start()
    assert pipe_parent.poll(timeout=10.0), "Coordinator didn't send address"
    addr = pipe_parent.recv()
    pipe_parent.close()
    return proc, addr, ready_event


async def test_async_zmq_endpoint():
    """Exercise AsyncZmqEndpoint send/recv, queue draining, error handling, and lifecycle."""

    # ── PUSH/PULL roundtrips ──
    sender = AsyncZmqEndpoint("PUSH", bind=True)
    receiver = AsyncZmqEndpoint("PULL", connect=sender.address)
    sender.start()
    receiver.start()

    # Serialized data
    data = [1, "hello", {"key": True}]
    sender._isend(Headers.SUBMIT_REQUEST, data)
    identity, header, received = await asyncio.wait_for(receiver._irecv(), timeout=2.0)
    assert identity is None and header == Headers.SUBMIT_REQUEST and received == data

    # Header-only (no payload)
    sender._isend(Headers.PAUSE)
    _, header, received = await asyncio.wait_for(receiver._irecv(), timeout=2.0)
    assert header == Headers.PAUSE and received is None

    # Raw bytes (serialize=False / deserialize=False)
    raw = b"\x01\x02\x03\xff"
    sender._isend(Headers.ENGINE_REPLY, raw, serialize=False)
    _, header, received = await asyncio.wait_for(receiver._irecv(deserialize=False), timeout=2.0)
    assert header == Headers.ENGINE_REPLY and received == raw

    # Queue draining: 5 sends arrive in order
    for i in range(5):
        sender._isend(Headers.SUBMIT_REQUEST, [i])
    for i in range(5):
        _, _, d = await asyncio.wait_for(receiver._irecv(), timeout=2.0)
        assert d == [i]

    await sender.shutdown()
    await receiver.shutdown()

    # ── ROUTER/DEALER roundtrips ──
    router = AsyncZmqEndpoint("ROUTER", bind=True)
    dealer = AsyncZmqEndpoint("DEALER", identity="test-dealer", connect=router.address)
    router.start()
    dealer.start()
    await asyncio.sleep(0.05)

    router._isend(Headers.ACK, [42], identity=b"test-dealer")
    _, header, d = await asyncio.wait_for(dealer._irecv(), timeout=2.0)
    assert header == Headers.ACK and d == [42]

    dealer._isend(Headers.ENGINE_CONNECT)
    identity, header, _ = await asyncio.wait_for(router._irecv(), timeout=2.0)
    assert identity == b"test-dealer" and header == Headers.ENGINE_CONNECT

    await router.shutdown()
    await dealer.shutdown()

    # ── EHOSTUNREACH handled gracefully ──
    router = AsyncZmqEndpoint("ROUTER", bind=True)
    router.start()
    router._isend(Headers.ACK, identity=b"nonexistent")
    await asyncio.wait_for(router.shutdown(), timeout=2.0)

    # ── Shutdown drains pending sends, then exits cleanly even when empty ──
    s = AsyncZmqEndpoint("PUSH", bind=True)
    r = AsyncZmqEndpoint("PULL", connect=s.address)
    s.start()
    r.start()
    s._isend(Headers.DISCONNECT)
    await asyncio.wait_for(s.shutdown(), timeout=2.0)
    _, header, _ = await asyncio.wait_for(r._irecv(), timeout=2.0)
    assert header == Headers.DISCONNECT
    await r.shutdown()

    ep = AsyncZmqEndpoint("PUSH", bind=True)
    ep.start()
    await asyncio.sleep(0.01)
    await asyncio.wait_for(ep.shutdown(), timeout=2.0)


async def test_coordinator_lifecycle():
    """End-to-end coordinator test: startup buffering, routing, engine disconnect, clean shutdown.

    Spawns a coordinator process (dp_size=1) and exercises the full lifecycle through
    raw ZMQ sockets:
    1. Client connects BEFORE any engine -> ACK is buffered (startup buffering).
    2. Engine connects -> quorum reached -> client receives buffered ACK.
    3. Request routes to the only engine.
    4. Second engine registers dynamically; first engine disconnects.
    5. Subsequent requests all route to the survivor (_remove_engine pruned scoring).
    6. SHUTDOWN exits the coordinator cleanly within 2 s (no SIGTERM needed).
    """
    proc, addr, ready_event = _spawn_coordinator(dp_size=1)

    try:
        ctx = zmq.asyncio.Context()

        # ── Startup buffering: client ACK withheld until engine connects ──
        client = ctx.socket(zmq.DEALER)
        client.setsockopt(zmq.IDENTITY, b"test-client")
        client.connect(addr)
        await asyncio.sleep(0.05)
        client.send_multipart([Headers.CLIENT_CONNECT.value.to_bytes()])

        await asyncio.sleep(0.2)
        with pytest.raises(zmq.Again):
            client.recv_multipart(flags=zmq.NOBLOCK)

        # Engine-a connects -> quorum -> client gets ACK.
        engine_a = ctx.socket(zmq.DEALER)
        engine_a.setsockopt(zmq.IDENTITY, b"engine-a")
        engine_a.connect(addr)
        await asyncio.sleep(0.05)
        engine_a.send_multipart([Headers.ENGINE_CONNECT.value.to_bytes()])

        frames = await asyncio.wait_for(client.recv_multipart(), timeout=5.0)
        assert Headers(int.from_bytes(frames[0])) == Headers.ACK
        assert ready_event.wait(timeout=5.0)

        # ── Request routing to single engine ──
        req = msgpack.packb([0, "prompt-0", {}], use_bin_type=True)
        client.send_multipart([Headers.SUBMIT_REQUEST.value.to_bytes(), req])
        frames = await asyncio.wait_for(engine_a.recv_multipart(), timeout=5.0)
        assert Headers(int.from_bytes(frames[0])) == Headers.SUBMIT_REQUEST

        # ── Dynamic registration + disconnect: route to survivor ──
        engine_b = ctx.socket(zmq.DEALER)
        engine_b.setsockopt(zmq.IDENTITY, b"engine-b")
        engine_b.connect(addr)
        await asyncio.sleep(0.05)
        engine_b.send_multipart([Headers.ENGINE_CONNECT.value.to_bytes()])
        await asyncio.sleep(0.1)

        engine_a.send_multipart([Headers.DISCONNECT.value.to_bytes()])
        await asyncio.sleep(0.1)

        for i in range(3):
            req = msgpack.packb([i + 1, f"prompt-{i + 1}", {}], use_bin_type=True)
            client.send_multipart([Headers.SUBMIT_REQUEST.value.to_bytes(), req])

        for _ in range(3):
            frames = await asyncio.wait_for(engine_b.recv_multipart(), timeout=5.0)
            assert Headers(int.from_bytes(frames[0])) == Headers.SUBMIT_REQUEST

        with pytest.raises(zmq.Again):
            engine_a.recv_multipart(flags=zmq.NOBLOCK)

        # ── Clean shutdown ──
        client.send_multipart([Headers.SHUTDOWN.value.to_bytes()])
        proc.join(timeout=2.0)
        assert not proc.is_alive(), (
            "Coordinator did not exit within 2 s of SHUTDOWN"
        )

        engine_a.close(linger=0)
        engine_b.close(linger=0)
        client.close(linger=0)
        ctx.term()
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=2.0)


@pytest.mark.skipif(not HAVE_TORCH, reason="torch required for EngineCoordinatorClient")
@pytest.mark.parametrize(
    "is_leader,requests,epoch,signals,expect_state,expect_n_added,expect_epoch,expect_putback",
    [
        # Only requests
        (True, [(1, "a", None), (2, "b", None)], None, [], "RUNNING", 2, 0, 0),
        # Only epoch
        (True, [], 42, [], "RUNNING", 0, 42, 0),
        # PAUSE signal
        (True, [], None, ["PAUSE"], "PAUSING", 0, 0, 0),
        # All together
        (True, [(1, "a", None)], 7, ["PAUSE"], "PAUSING", 1, 7, 0),
        # Leader puts back extra signals
        (True, [], None, ["PAUSE", "STOP"], "PAUSING", 0, 0, 1),
        # Follower discards extra signals
        (False, [], None, ["PAUSE", "STOP"], "PAUSING", 0, 0, 0),
        # Empty (no-op)
        (True, [], None, [], "RUNNING", 0, 0, 0),
    ],
    ids=[
        "requests-only",
        "epoch-only",
        "pause-signal",
        "all-together",
        "leader-puts-back-extras",
        "follower-discards-extras",
        "empty-noop",
    ],
)
async def test_deferred_application(
    is_leader, requests, epoch, signals,
    expect_state, expect_n_added, expect_epoch, expect_putback,
):
    """Verify apply_deferred commits all deferred state to the engine correctly.

    Exercises the pipelined defer/apply mechanism by populating _deferred_requests,
    _deferred_epoch, and _deferred_signals on a mock EngineCoordinatorClient, then
    calling apply_deferred and checking that the engine state matches expectations.
    """
    # Minimal client mock — bypass __init__, set only the fields apply_deferred reads.
    client = object.__new__(EngineCoordinatorClient)
    client.is_mp_coordinator = is_leader
    client.pending_messages = deque()
    client._deferred_requests = list(requests)
    client._deferred_epoch = epoch
    client._deferred_signals = [Headers[s] for s in signals]

    # Minimal engine mock.
    class Engine:
        def __init__(self):
            self.state = EngineState.RUNNING
            self._state_events = {
                s: asyncio.Event()
                for s in (
                    EngineState.RUNNING, EngineState.PAUSED, EngineState.SUSPENDED,
                    EngineState.RESUMED, EngineState.STOPPED,
                )
            }
            self._state_events[EngineState.RUNNING].set()
            self.requests = {}
            self.waiting_request_ids = deque()
            self._generation_epoch = 0
            self._added = []

        def add_request(self, rid, prompt, sp):
            self._added.append((rid, prompt, sp))

        def suspend(self):
            pass

        def resume(self):
            pass

    engine = Engine()
    client.apply_deferred(engine)

    assert engine.state == EngineState[expect_state]
    assert len(engine._added) == expect_n_added
    if expect_n_added:
        assert engine._added == list(requests)
    assert engine._generation_epoch == expect_epoch
    assert len(client.pending_messages) == expect_putback
    # All deferred state must be cleared after apply.
    assert client._deferred_requests == []
    assert client._deferred_epoch is None
    assert client._deferred_signals == []
