import threading

from megatron.inference.integrations.dynamo.telemetry import (
    EngineEventReceiver,
    EngineEventReporter,
)


class _Engine:
    rank = 0

    def add_kv_event_listener(self, listener):
        self.listener = listener


def test_kv_events_bypass_request_coordinator():
    received = []
    ready = threading.Event()

    def observe(kind, payload):
        received.append((kind, payload))
        if len(received) == 2:
            ready.set()

    receiver = EngineEventReceiver(observe, "127.0.0.1")
    address = receiver.start()
    engine = _Engine()
    reporter = EngineEventReporter(engine, address)
    reporter.start()
    try:
        reporter.observe("ready", {"version": 3})
        engine.listener("stored", {"block_hashes": [101]})
        assert ready.wait(timeout=2.0)
        assert received == [("ready", {"version": 3}), ("stored", {"block_hashes": [101]})]
    finally:
        receiver.stop()
        reporter.stop()
