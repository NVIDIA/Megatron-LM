# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Helpers for parsing ``torch.profiler`` events in the mfsdp_v2 tests."""

from typing import Optional

from torch.autograd import DeviceType
from torch.autograd.profiler_util import FunctionEvent
from torch.profiler import profile as TorchProfiler


class EventGroup(list[FunctionEvent]):
    """CUDA events linked to one CPU profiler event."""

    def __repr__(self) -> str:
        return repr([event.name for event in self])


def events_overlap(first: FunctionEvent, second: FunctionEvent) -> bool:
    return (
        first.time_range.start < second.time_range.end
        and second.time_range.start < first.time_range.end
    )


def _linked_correlation_ids(
    prof: TorchProfiler, events: list[FunctionEvent]
) -> dict[int, Optional[int]]:
    """Return each event's Kineto linked correlation id, keyed by ``id(event)``.

    Newer torch releases expose ``FunctionEvent.linked_correlation_id`` directly. Older
    releases keep it only on the raw Kineto events, so recover it from the Kineto event each
    ``FunctionEvent`` was built from: same correlation id, device type, and trace-relative
    time range, computed exactly as ``torch.autograd.profiler`` computes it. An event with no
    unambiguous source maps to ``None``.
    """
    if all(hasattr(event, "linked_correlation_id") for event in events):
        return {id(event): event.linked_correlation_id for event in events}

    kineto_results = prof.profiler.kineto_results
    trace_start_ns = kineto_results.trace_start_ns()
    linked_by_key: dict[tuple, set[int]] = {}
    for kineto_event in kineto_results.events():
        key = (
            kineto_event.correlation_id(),
            kineto_event.device_type(),
            (kineto_event.start_ns() - trace_start_ns) / 1000,
            (kineto_event.end_ns() - trace_start_ns) / 1000,
        )
        linked_by_key.setdefault(key, set()).add(kineto_event.linked_correlation_id())

    linked: dict[int, Optional[int]] = {}
    for event in events:
        key = (event.id, event.device_type, event.time_range.start, event.time_range.end)
        candidates = linked_by_key.get(key)
        linked[id(event)] = next(iter(candidates)) if candidates and len(candidates) == 1 else None
    return linked


def event_groups_overlap(first: EventGroup, second: EventGroup) -> bool:
    for first_event in first:
        for second_event in second:
            if events_overlap(first_event, second_event):
                return True
    return False


def collect_linked_event_groups(
    prof: TorchProfiler, cpu_event_name_substring: str
) -> list[EventGroup]:
    """Collect CUDA events grouped by linked CPU event.

    Device events are attributed by their launching CPU op rather than searched by their
    own name: device-side names vary across GPU architectures and kernel libraries -- for
    example a matmul kernel is named ``nvjet_``/``cutlass_``/``cublas_``... while its
    CPU op is simply ``aten::mm``.

    Each group contains the CUDA events linked to one matching CPU event, including kernels
    and memcpys.
    """
    # A correlation id is shared by a device event and the leaf runtime op that issued it,
    # not the enclosing matched op, so walk cpu_parent up from each correlated leaf. Id 0
    # is the "no device correlation" sentinel and is skipped.
    events = prof.events()
    linked_correlation_ids = _linked_correlation_ids(prof, events)
    groups_by_correlation: dict[int, EventGroup] = {}
    for event in events:
        correlation_id = linked_correlation_ids[id(event)]
        if event.device_type != DeviceType.CPU or not correlation_id:
            continue
        node = event
        while node is not None:
            if cpu_event_name_substring in node.name:
                groups_by_correlation[correlation_id] = EventGroup()
                break
            node = node.cpu_parent

    for event in events:
        if event.device_type != DeviceType.CUDA:
            continue
        correlation_id = linked_correlation_ids[id(event)]
        assert (
            correlation_id is not None
        ), f"Could not recover the linked correlation id of device event {event.name!r}"
        group = groups_by_correlation.get(correlation_id)
        if group is None:
            continue
        group.append(event)

    return list(groups_by_correlation.values())
