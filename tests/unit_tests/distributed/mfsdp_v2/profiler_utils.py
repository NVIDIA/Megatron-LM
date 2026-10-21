# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Helpers for parsing ``torch.profiler`` events in the mfsdp_v2 tests."""

from typing import Optional

from torch.autograd import DeviceType
from torch.autograd.profiler_util import FunctionEvent
from torch.profiler import profile as TorchProfiler

# CUPTI names its non-kernel device activities (copy-engine memcpys and memsets) with these
# prefixes. Only consulted when the installed torch predates ``FunctionEvent.activity_type``.
_NON_KERNEL_DEVICE_ACTIVITY_PREFIXES = ("Memcpy", "Memset")


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


def _is_device_kernel(event: FunctionEvent) -> bool:
    if event.device_type != DeviceType.CUDA:
        return False
    activity_type = getattr(event, "activity_type", None)
    if activity_type is not None:
        return activity_type == "kernel"
    return not event.is_user_annotation and not event.name.startswith(
        _NON_KERNEL_DEVICE_ACTIVITY_PREFIXES
    )


def collect_linked_kernels(
    prof: TorchProfiler, cpu_event_name_substring: str
) -> list[FunctionEvent]:
    """Collect device kernel events linked to matching CPU op instances.

    Device events are attributed by their launching CPU op rather than searched by their
    own name: device-side names vary across GPU architectures and kernel libraries -- for
    example a matmul kernel is named ``nvjet_``/``cutlass_``/``cublas_``... while its
    CPU op is simply ``aten::mm``.

    Zero-CTA all-gather copy-engine memcpys are not kernels and are intentionally not
    returned.
    """
    # A correlation id is shared by a device event and the leaf runtime op that issued it,
    # not the enclosing matched op, so walk cpu_parent up from each correlated leaf. Id 0
    # is the "no device correlation" sentinel and is skipped.
    events = prof.events()
    linked_correlation_ids = _linked_correlation_ids(prof, events)
    matching_correlations: set[int] = set()
    for event in events:
        correlation_id = linked_correlation_ids[id(event)]
        if event.device_type != DeviceType.CPU or not correlation_id:
            continue
        node = event
        while node is not None:
            if cpu_event_name_substring in node.name:
                matching_correlations.add(correlation_id)
                break
            node = node.cpu_parent

    linked_kernels: list[FunctionEvent] = []
    for event in events:
        if not _is_device_kernel(event):
            continue
        correlation_id = linked_correlation_ids[id(event)]
        assert (
            correlation_id is not None
        ), f"Could not recover the linked correlation id of device kernel {event.name!r}"
        if correlation_id not in matching_correlations:
            continue
        linked_kernels.append(event)

    return linked_kernels
