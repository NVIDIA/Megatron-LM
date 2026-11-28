# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import msgpack
import zmq

from megatron.inference.integrations.dynamo.telemetry import (
    EngineEventReceiver,
    EngineEventReporter,
)


def test_kv_events_bypass_request_coordinator():
    received = []
    ready = threading.Event()

    def observe(kind, payload):
        received.append((kind, payload))
        if len(received) == 2:
            ready.set()

    receiver = EngineEventReceiver(observe, "127.0.0.1")
    address = receiver.start()
    helper = MagicMock()
    engine = SimpleNamespace(rank=0, context=SimpleNamespace(dynamo_helper=helper))
    reporter = EngineEventReporter(engine, address)
    reporter.start()
    try:
        reporter.observe("ready", {"version": 3})
        listener = helper.add_kv_event_listener.call_args.args[0]
        listener("stored", {"block_hashes": [101]})
        assert ready.wait(timeout=2.0)
        assert received == [("ready", {"version": 3}), ("stored", {"block_hashes": [101]})]
    finally:
        receiver.stop()
        reporter.stop()


def _push_socket(address):
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(address)
    return context, socket


def test_receiver_logs_and_skips_a_malformed_event(caplog):
    received = []
    ready = threading.Event()

    def observe(kind, payload):
        if kind == "boom":
            raise RuntimeError("handler failure")
        received.append((kind, payload))
        ready.set()

    receiver = EngineEventReceiver(observe, "127.0.0.1")
    address = receiver.start()
    context, socket = _push_socket(address)
    try:
        with caplog.at_level(logging.WARNING):
            # A frame that is not msgpack, one that is not a [kind, payload] pair, and one
            # whose handler raises are each logged and skipped.
            socket.send(b"\xc1")
            socket.send(msgpack.packb(["stored", {}, "unexpected"], use_bin_type=True))
            socket.send(msgpack.packb(["boom", {}], use_bin_type=True))
            socket.send(msgpack.packb(["progress", {}], use_bin_type=True))
            assert ready.wait(timeout=2.0)
        assert received == [("progress", {})]
        messages = [record.getMessage() for record in caplog.records]
        assert sum("malformed engine event" in message for message in messages) == 2
        assert any("Failed to handle engine event 'boom'" in message for message in messages)
    finally:
        receiver.stop()
        socket.close()
        context.term()


def test_reporter_logs_and_drops_an_unserializable_event(caplog):
    received = []
    ready = threading.Event()

    def observe(kind, payload):
        received.append((kind, payload))
        ready.set()

    receiver = EngineEventReceiver(observe, "127.0.0.1")
    address = receiver.start()
    engine = SimpleNamespace(rank=0, context=SimpleNamespace(dynamo_helper=MagicMock()))
    reporter = EngineEventReporter(engine, address)
    reporter.start()
    try:
        with caplog.at_level(logging.WARNING):
            reporter.observe("stored", {"block_hashes": [object()]})
            reporter.observe("progress", {})
            assert ready.wait(timeout=2.0)
        assert received == [("progress", {})]
        assert any(
            "Dropping engine event 'stored'" in record.getMessage() for record in caplog.records
        )
    finally:
        receiver.stop()
        reporter.stop()


def test_reporter_logs_a_transport_failure_after_startup(caplog):
    engine = SimpleNamespace(rank=0, context=SimpleNamespace(dynamo_helper=MagicMock()))
    reporter = EngineEventReporter(engine, "tcp://127.0.0.1:1")
    reporter.start()
    thread = reporter._thread
    with (
        caplog.at_level(logging.ERROR),
        patch.object(zmq.Socket, "send", side_effect=zmq.ZMQError(zmq.ETERM)),
    ):
        reporter.observe("progress", {})
        thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert any("Engine event reporter failed" in r.getMessage() for r in caplog.records)
    assert reporter._ready.empty()
    reporter.stop()
