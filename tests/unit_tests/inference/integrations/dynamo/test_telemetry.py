# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

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
