# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import threading
from collections import Counter
from unittest.mock import patch

import msgpack
import pytest
import torch
import zmq
from zmq.utils.monitor import recv_monitor_message

from megatron.core.inference.headers import Headers
from megatron.core.inference.inference_client import InferenceClient
from megatron.core.inference.inference_request import DynamicInferenceRequest, Status
from megatron.core.inference.sampling_params import SamplingParams
from tests.unit_tests.inference.coordinator_pairwise_utils import CoordinatorThread, until


class _ObservedSocket:
    """Count actual terminal frames without replacing transport or client parsing."""

    def __init__(self, socket):
        self.socket, self.replies = socket, Counter()

    def __getattr__(self, name):
        return getattr(self.socket, name)

    def recv_multipart(self, *args, **kwargs):
        frames = self.socket.recv_multipart(*args, **kwargs)
        metadata = msgpack.unpackb(frames[0], raw=False)
        if metadata[0] == Headers.ENGINE_REPLY.value:
            self.replies[metadata[1]] += 1
        return frames


class _CoordinatorRuntime(CoordinatorThread):
    def __init__(self):
        self.monitor_endpoint = "inproc://coordinator-monitor"
        super().__init__(
            0, None, setup=lambda c: c.router_socket.monitor(self.monitor_endpoint, zmq.EVENT_ALL)
        )
        assert self.ready.wait(timeout=5.0)
        self.monitor = self.coordinator.context.socket(zmq.PAIR)
        self.monitor.connect(self.monitor_endpoint)

    @property
    def processed(self):
        return Counter(event["header"] for event in self.events)

    def assert_healthy(self):
        if self.errors:
            raise AssertionError("coordinator thread failed") from self.errors[0]
        assert self.thread.is_alive(), "coordinator loop exited unexpectedly"

    def shutdown(self):
        if self.thread.is_alive():
            context = zmq.Context()
            socket = context.socket(zmq.DEALER)
            socket.connect(self.address)
            socket.send(msgpack.packb([Headers.CONNECT.value], use_bin_type=True))
            if socket.poll(5000) & zmq.POLLIN:
                socket.recv_multipart()
                socket.send(msgpack.packb([Headers.SHUTDOWN.value], use_bin_type=True))
            socket.close(linger=100)
            context.term()
        self.monitor.close(linger=0)
        self.thread.join(timeout=5.0)
        assert not self.thread.is_alive(), "coordinator thread did not stop"
        if self.errors:
            raise AssertionError("coordinator thread failed") from self.errors[0]

    async def wait_for_disconnect(self):
        def disconnected():
            while self.monitor.poll(0):
                if recv_monitor_message(self.monitor)["event"] == zmq.EVENT_DISCONNECTED:
                    return True

        await _eventually(disconnected, "coordinator did not observe disconnect")

    async def wait_for_engine(self, engine):
        ranks = self.coordinator.identities_of_data_parallel_ranks
        await _eventually(lambda: engine.identity in ranks, "engine did not register")


class _EnginePeer:
    def __init__(self, address, identity):
        self.identity, self.context = identity, zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.IDENTITY, identity)
        self.socket.connect(address)
        self.socket.send(b"")

    def receive_request(self):
        assert self.socket.poll(5000) & zmq.POLLIN, "engine did not receive request"
        metadata = msgpack.unpackb(self.socket.recv_multipart()[0], raw=False)
        assert Headers(metadata[0]) in (Headers.SUBMIT_REQUEST, Headers.SUBMIT_REQUEST_WITH_KV)
        return metadata[1]

    def disconnect(self):
        self.socket.send(msgpack.packb([Headers.DISCONNECT.value], use_bin_type=True))

    def send_reply(self, request_id, *, partial=False):
        header = Headers.ENGINE_REPLY_PARTIAL if partial else Headers.ENGINE_REPLY
        request_ids = [request_id] if partial else [[request_id, False]]
        body = (
            {"request_id": request_id, "new_tokens": [1]}
            if partial
            else DynamicInferenceRequest(
                request_id=request_id,
                prompt_tokens=torch.empty(0, dtype=torch.int64),
                sampling_params=SamplingParams(detokenize_generations=False),
                status=Status.COMPLETED,
            ).serialize()
        )
        metadata = msgpack.packb([header.value, request_ids], use_bin_type=True)
        self.socket.send_multipart([metadata, msgpack.packb(body, use_bin_type=True)])

    def close(self):
        self.socket.close(linger=0)
        self.context.term()


@pytest.fixture
def coordinator_runtime():
    runtime = _CoordinatorRuntime()
    yield runtime
    runtime.shutdown()


def _start_client(runtime, *, deserialize=False):
    client = InferenceClient(runtime.address, deserialize=deserialize)
    client.socket = _ObservedSocket(client.socket)
    client.start(connect_timeout_seconds=5.0)
    return client


async def _eventually(predicate, message):
    try:
        await until(predicate, timeout=5.0)
    except TimeoutError:
        raise AssertionError(message) from None


def _assert_reply(reply, status=Status.FAILED):
    if isinstance(reply, DynamicInferenceRequest):
        reply = reply.serialize()
    assert reply["status"] == status.name
    if status == Status.FAILED:
        assert [event["type"] for event in reply["events"]].count("FAIL") == 1


def test_send_to_client_reraises_other_errors(coordinator_runtime):
    coordinator = coordinator_runtime.coordinator
    error = zmq.ZMQError(zmq.EINVAL)
    with patch.object(coordinator.router_socket, "send_multipart", side_effect=error):
        with pytest.raises(zmq.ZMQError) as caught:
            coordinator._send_to_client([b"client", b"reply"])
    assert caught.value is error


@pytest.mark.asyncio
@pytest.mark.parametrize("delivery", ["partial", "final"])
async def test_closed_client_delivery_does_not_disrupt_live_client(coordinator_runtime, delivery):
    runtime = coordinator_runtime
    engine = _EnginePeer(runtime.address, b"live-engine")
    closed_client, live_client = [_start_client(runtime) for _ in range(2)]
    closed = False
    try:
        await runtime.wait_for_engine(engine)
        closed_stream = closed_client.add_request_streaming(
            [1, 2], SamplingParams(num_tokens_to_generate=2)
        )
        closed_request_id = engine.receive_request()
        live_future = live_client.add_request([3, 4], SamplingParams(num_tokens_to_generate=1))
        resolved = []
        live_future.add_done_callback(resolved.append)
        live_request_id = engine.receive_request()
        assert closed_request_id != live_request_id
        closed_client.stop()
        closed = True
        await runtime.wait_for_disconnect()
        engine.send_reply(closed_request_id, partial=delivery == "partial")
        # Same-peer FIFO makes this live reply a barrier after the failed delivery.
        engine.send_reply(live_request_id)
        reply = await asyncio.wait_for(live_future, timeout=5.0)
        assert reply["status"] == Status.COMPLETED.name
        runtime.assert_healthy()
        assert resolved == [live_future]
        assert live_client.socket.replies == {0: 1}
        if delivery == "partial":
            # A later final still owns cleanup even though its client is gone.
            engine.send_reply(closed_request_id)
            await _eventually(
                lambda: not runtime.coordinator.request_id_to_client_id,
                "closed client request was not cleaned up",
            )
        runtime.assert_requests_retired()
        assert runtime.coordinator._pending_counts.tolist() == [0]
        assert closed_stream.finished
    finally:
        if not closed:
            closed_client.stop()
        live_client.stop()
        engine.close()


@pytest.mark.asyncio
async def test_disconnect_before_connect_ack(coordinator_runtime, monkeypatch):
    runtime = coordinator_runtime
    received, release = [], threading.Event()
    original = runtime.coordinator._handlers[Headers.CONNECT]

    def delayed(c, sender, metadata, bodies):
        received.append(sender)
        assert release.wait(5.0)
        return original(c, sender, metadata, bodies)

    monkeypatch.setitem(runtime.coordinator._handlers, Headers.CONNECT, delayed)
    client = InferenceClient(runtime.address)
    try:
        client.socket.send(msgpack.packb([Headers.CONNECT.value], use_bin_type=True))
        await _eventually(lambda: received, "CONNECT was not received")
        client.stop()
        await runtime.wait_for_disconnect()
        release.set()
        await _eventually(lambda: runtime.errors or runtime.processed[Headers.CONNECT], "stalled")
        runtime.assert_healthy()
        assert received[0] not in runtime.coordinator.known_clients
        client = _start_client(runtime)
    finally:
        release.set()
        runtime.coordinator._handlers[Headers.CONNECT] = original
        client.stop()
