# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Helpers for parsing ``torch.profiler`` events in the mfsdp_v2 tests."""

from torch.autograd import DeviceType
from torch.autograd.profiler_util import FunctionEvent
from torch.profiler import profile as TorchProfiler


def events_overlap(first: FunctionEvent, second: FunctionEvent) -> bool:
    return (
        first.time_range.start < second.time_range.end
        and second.time_range.start < first.time_range.end
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
    matching_correlations: set[int] = set()
    for event in events:
        if event.device_type != DeviceType.CPU or not event.linked_correlation_id:
            continue
        node = event
        while node is not None:
            if cpu_event_name_substring in node.name:
                matching_correlations.add(event.linked_correlation_id)
                break
            node = node.cpu_parent

    linked_kernels: list[FunctionEvent] = []
    for event in events:
        if event.device_type != DeviceType.CUDA:
            continue
        if event.activity_type != "kernel":
            continue
        if event.linked_correlation_id not in matching_correlations:
            continue
        linked_kernels.append(event)

    return linked_kernels
