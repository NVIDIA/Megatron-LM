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


def _is_device_kernel(event: FunctionEvent) -> bool:
    return (
        event.device_type == DeviceType.CUDA
        and event.activity_type == "kernel"
        and not event.name.startswith("nccl:")
    )


def collect_linked_device_events(
    events: list[FunctionEvent], cpu_event_name_substring: str
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
    matching_correlations: set[int] = set()
    for event in events:
        correlation_id = _linked_correlation_id(event)
        if event.device_type != DeviceType.CPU or not correlation_id:
            continue
        node = event
        while node is not None:
            if cpu_event_name_substring in node.name:
                matching_correlations.add(correlation_id)
                break
            node = node.cpu_parent

    return [
        event
        for event in events
        if _is_device_kernel(event) and _linked_correlation_id(event) in matching_correlations
    ]
