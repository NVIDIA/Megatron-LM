# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Helpers for parsing ``torch.profiler`` events in the mfsdp_v2 tests."""

from torch.autograd import DeviceType
from torch.autograd.profiler_util import FunctionEvent


def events_overlap(first: FunctionEvent, second: FunctionEvent) -> bool:
    return (
        first.time_range.start < second.time_range.end
        and second.time_range.start < first.time_range.end
    )


def collect_linked_device_events(
    events: list[FunctionEvent], cpu_event_name_substring: str
) -> list[list[FunctionEvent]]:
    """Collect device events linked to each matching CPU op instance.

    Device events are attributed by their launching CPU op rather than searched by their
    own name: device-side names vary across GPU architectures and kernel libraries -- for
    example a matmul kernel is named ``nvjet_``/``cutlass_``/``cublas_``... while its CPU
    op is simply ``aten::mm``, and under zero-CTA the all-gather is not even a distinct
    kernel, just a generic copy-engine ``Memcpy``.

    Returns a list, in event order, where each entry contains the device events for one
    matching op instance.
    """
    # A correlation id is shared by a device event and the leaf runtime op that issued it,
    # not the enclosing matched op, so walk cpu_parent up from each correlated leaf. Id 0
    # is the "no device correlation" sentinel and is skipped.
    op_by_correlation: dict[int, FunctionEvent] = {}
    for event in events:
        if event.device_type != DeviceType.CPU or not event.linked_correlation_id:
            continue
        node = event
        while node is not None:
            if cpu_event_name_substring in node.name:
                op_by_correlation[event.linked_correlation_id] = node
                break
            node = node.cpu_parent

    device_events_by_op: dict[FunctionEvent, list[FunctionEvent]] = {}
    for event in events:
        if event.device_type != DeviceType.CUDA:
            continue
        op = op_by_correlation.get(event.linked_correlation_id)
        if op is not None:
            device_events_by_op.setdefault(op, []).append(event)
    return list(device_events_by_op.values())
