# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from megatron.core.pipeline_parallel.utils import ScheduleNode


class _FakeStorage:
    def __init__(self, trace):
        self.trace = trace

    def resize_(self, size):
        self.trace.append(("resize", size))


class _FakeTensor:
    def __init__(self, trace):
        self.trace = trace
        self.storage = _FakeStorage(trace)

    def record_stream(self, stream):
        self.trace.append(("record_stream", stream))

    def untyped_storage(self):
        return self.storage


class _FakeEvent:
    def __init__(self, trace):
        self.trace = trace

    def wait(self, stream):
        self.trace.append(("wait", stream))


def _make_node(trace, *, handback_stream=None):
    return ScheduleNode(
        forward_func=lambda: None,
        stream="consumer",
        event=_FakeEvent(trace),
        free_input=True,
        free_input_handback_stream=handback_stream,
    )


def test_free_forward_inputs_record_stream_fallback():
    trace = []
    node = _make_node(trace)

    node._free_forward_inputs((_FakeTensor(trace), None))

    assert trace == [("record_stream", "consumer"), ("resize", 0)]


def test_handback_argument_preserves_positional_name_compatibility():
    node = ScheduleNode(lambda: None, "consumer", _FakeEvent([]), None, True, "custom-name")

    assert node.name == "custom-name"
    assert node.free_input_handback_stream is None


def test_free_forward_inputs_hands_back_before_release():
    trace = []
    creator_calls = []

    def get_creator_stream():
        creator_calls.append(True)
        return "creator"

    node = _make_node(trace, handback_stream=get_creator_stream)

    node._free_forward_inputs((_FakeTensor(trace), None))
    node._free_forward_inputs((_FakeTensor(trace),))

    assert creator_calls == [True]
    assert trace == [("wait", "creator"), ("resize", 0), ("wait", "creator"), ("resize", 0)]
