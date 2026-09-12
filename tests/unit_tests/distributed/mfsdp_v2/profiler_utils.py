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


def _event_correlation_id(event: FunctionEvent) -> int:
    """Return the profiler correlation key across old and new PyTorch schemas."""
    correlation_id = getattr(event, "linked_correlation_id", None)
    return event.id if correlation_id is None else correlation_id


def _is_kernel_event(event: FunctionEvent) -> bool:
    """Distinguish kernels from CUDA runtime annotations across profiler schemas."""
    activity_type = getattr(event, "activity_type", None)
    if activity_type is not None:
        return activity_type == "kernel"
    return not getattr(event, "is_user_annotation", False) and not event.name.startswith(
        ("Memcpy ", "Memset ")
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
        correlation_id = _event_correlation_id(event)
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
        if event.device_type != DeviceType.CUDA or not _is_kernel_event(event):
            continue
        if _event_correlation_id(event) not in matching_correlations:
            continue
        linked_kernels.append(event)

    return linked_kernels
