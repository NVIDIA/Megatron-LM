# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Helpers for parsing ``torch.profiler`` events in the mfsdp_v2 tests."""

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
    # A CPU operation can issue several correlated runtime calls (including event
    # recording calls with no device event). Group by the enclosing operation,
    # not by each leaf correlation, or those calls inflate the collective count.
    events = prof.events()
    groups_by_operation: dict[int, EventGroup] = {}
    groups_by_correlation: dict[int, EventGroup] = {}
    for event in events:
        if event.device_type != DeviceType.CPU:
            continue
        node = event
        operation = None
        while node is not None:
            if cpu_event_name_substring in node.name:
                operation = node
            node = node.cpu_parent
        if operation is None:
            continue
        group = groups_by_operation.setdefault(operation.id, EventGroup())
        # Id 0 is the "no device correlation" sentinel. Keep the operation's
        # empty group: zero-CTA collectives can intentionally launch no kernels.
        if event.linked_correlation_id:
            groups_by_correlation[event.linked_correlation_id] = group

    for event in events:
        if event.device_type != DeviceType.CUDA:
            continue
        group = groups_by_correlation.get(event.linked_correlation_id)
        if group is None:
            continue
        group.append(event)

    return list(groups_by_operation.values())
