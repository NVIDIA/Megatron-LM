# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_handback_reuses_creator_stream_storage_without_corrupting_consumer():
    """Exercise the real allocator while the consumer stream trails the producer."""

    def run(use_handback):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        creator = torch.cuda.Stream()
        consumer = torch.cuda.Stream()
        event = torch.cuda.Event()
        shape = (4 * 1024 * 1024,)
        outputs = []
        expected = []
        pointers = []

        def consume(value):
            # Force enough in-flight depth for record_stream to retain several
            # blocks while keeping the test independent of GEMM libraries.
            torch.cuda._sleep(10_000_000)
            return value.sum(dtype=torch.float32)

        node = ScheduleNode(
            consume,
            consumer,
            event,
            free_input=True,
            free_input_handback_stream=creator if use_handback else None,
        )

        for index in range(16):
            value = float(index + 1)
            with torch.cuda.stream(creator):
                tensor = torch.full(shape, value, dtype=torch.bfloat16, device="cuda")
                pointers.append(tensor.data_ptr())
                event.record(creator)
            outputs.append(node.forward(tensor))
            expected.append(value * shape[0])

        torch.cuda.synchronize()
        actual = torch.stack(outputs).cpu()
        torch.testing.assert_close(
            actual, torch.tensor(expected, dtype=torch.float32), rtol=0, atol=0
        )
        return actual, len(set(pointers)), torch.cuda.max_memory_reserved()

    control_output, control_pointers, control_reserved = run(False)
    handback_output, handback_pointers, handback_reserved = run(True)

    torch.testing.assert_close(handback_output, control_output, rtol=0, atol=0)
    assert handback_pointers < control_pointers
    assert handback_reserved < control_reserved
