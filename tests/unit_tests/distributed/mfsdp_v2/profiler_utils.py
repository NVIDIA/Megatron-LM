# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Helpers for parsing ``torch.profiler`` events in the mfsdp_v2 tests."""

from torch.autograd import DeviceType
from torch.autograd.profiler_util import FunctionEvent


def events_overlap(first: FunctionEvent, second: FunctionEvent) -> bool:
    return (
        first.time_range.start < second.time_range.end
        and second.time_range.start < first.time_range.end
    )


def _linked_correlation_id(event: FunctionEvent) -> int | None:
    return getattr(event, "linked_correlation_id", None)


def _is_device_work(event: FunctionEvent) -> bool:
    return event.device_type == DeviceType.CUDA and not event.name.startswith("nccl:")


def collect_linked_device_events(
    events: list[FunctionEvent], cpu_event_name_substring: str
) -> list[FunctionEvent]:
    """Collect device events linked to matching CPU op instances.

    Device events are attributed by their launching CPU op rather than searched by their
    own name: device-side names vary across GPU architectures and kernel libraries -- for
    example a matmul kernel is named ``nvjet_``/``cutlass_``/``cublas_``... while its CPU
    op is simply ``aten::mm``, and under zero-CTA the all-gather is not even a distinct
    kernel, just a generic copy-engine ``Memcpy``.

    Returns a flat list of device events in matching CPU-op order.
    """
    return [
        event
        for group in collect_linked_device_event_groups(events, cpu_event_name_substring)
        for event in group
    ]


def collect_linked_device_event_groups(
    events: list[FunctionEvent], cpu_event_name_substring: str
) -> list[list[FunctionEvent]]:
    """Collect device events grouped by matching CPU op instance.

    One logical CPU op can emit multiple device events; zero-CTA all-gather, for example,
    decomposes into several copy-engine memcpys.
    """
    # A correlation id is shared by a device event and the leaf runtime op that issued it,
    # not the enclosing matched op, so walk cpu_parent up from each correlated leaf. Id 0
    # is the "no device correlation" sentinel and is skipped.
    correlation_to_group: dict[int, int] = {}
    group_index_by_cpu_event: dict[int, int] = {}
    groups: list[list[FunctionEvent]] = []
    for event in events:
        correlation_id = _linked_correlation_id(event)
        if event.device_type != DeviceType.CPU or not correlation_id:
            continue
        node = event
        while node is not None:
            if cpu_event_name_substring in node.name:
                cpu_event_id = id(node)
                group_index = group_index_by_cpu_event.get(cpu_event_id)
                if group_index is None:
                    group_index = len(groups)
                    group_index_by_cpu_event[cpu_event_id] = group_index
                    groups.append([])
                correlation_to_group[correlation_id] = group_index
                break
            node = node.cpu_parent

    for event in events:
        correlation_id = _linked_correlation_id(event)
        group_index = correlation_to_group.get(correlation_id)
        if _is_device_work(event) and group_index is not None:
            groups[group_index].append(event)

    return [group for group in groups if group]
