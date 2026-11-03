# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU regression tests for collective profiler event attribution."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from torch.autograd import DeviceType

MODULE_PATH = Path(__file__).parents[2] / "tests/unit_tests/distributed/mfsdp_v2/profiler_utils.py"
spec = importlib.util.spec_from_file_location("mfsdp_profiler_utils", MODULE_PATH)
profiler_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profiler_utils)


def _event(event_id, name, *, parent=None, correlation=0, device=DeviceType.CPU):
    return SimpleNamespace(
        id=event_id,
        name=name,
        cpu_parent=parent,
        linked_correlation_id=correlation,
        device_type=device,
    )


def _groups(events):
    return profiler_utils.collect_linked_event_groups(
        SimpleNamespace(events=lambda: events), "allgather"
    )


def test_multiple_runtime_calls_belong_to_one_collective():
    operation = _event(1, "c10d::allgather")
    record = _event(2, "cudaEventRecord", parent=operation, correlation=11)
    launch = _event(3, "cudaLaunchKernel", parent=operation, correlation=12)
    copy = _event(4, "cudaMemcpyAsync", parent=operation, correlation=13)
    kernel = _event(5, "ncclKernel", correlation=12, device=DeviceType.CUDA)
    memcpy = _event(6, "Memcpy DtoD", correlation=13, device=DeviceType.CUDA)
    groups = _groups([operation, record, launch, copy, kernel, memcpy])
    assert groups == [[kernel, memcpy]]


def test_nested_matching_backend_events_do_not_double_count():
    operation = _event(1, "c10d::allgather")
    backend = _event(2, "nccl:allgather", parent=operation, correlation=11)
    launch = _event(3, "cudaLaunchKernel", parent=backend, correlation=12)
    kernel = _event(4, "ncclKernel", correlation=12, device=DeviceType.CUDA)
    assert _groups([operation, backend, launch, kernel]) == [[kernel]]


def test_separate_and_zero_kernel_collectives_are_preserved():
    empty = _event(1, "c10d::allgather")
    record = _event(2, "cudaEventRecord", parent=empty, correlation=11)
    operation = _event(3, "c10d::allgather")
    launch = _event(4, "cudaLaunchKernel", parent=operation, correlation=12)
    unrelated = _event(5, "unrelated", correlation=0, device=DeviceType.CUDA)
    kernel = _event(6, "ncclKernel", correlation=12, device=DeviceType.CUDA)
    assert _groups([empty, record, operation, launch, unrelated, kernel]) == [[], [kernel]]
