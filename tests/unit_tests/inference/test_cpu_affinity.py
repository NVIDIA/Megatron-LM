# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from megatron.core.inference import cpu_affinity as affinity


@pytest.mark.parametrize(
    "main,helper,fails",
    [({2}, {3}, False), ({2, 4}, {3}, True), ({2}, {3, 4}, True), ({2}, set(), True)],
)
def test_thread_masks(monkeypatch, main, helper, fails):
    monkeypatch.setattr(affinity.sys, "platform", "linux")
    monkeypatch.setattr(affinity.threading, "get_native_id", lambda: 101)
    monkeypatch.setattr(affinity, "_gpu_local_cpus", lambda: ("GPU-test", {2, 3}))
    monkeypatch.setattr(
        affinity.Path, "iterdir", lambda _: [SimpleNamespace(name=str(t)) for t in (102, 103)]
    )
    monkeypatch.setattr(
        affinity.os,
        "sched_getaffinity",
        Mock(side_effect=[main, ProcessLookupError(), helper]),
        raising=False,
    )
    setter = Mock()
    monkeypatch.setattr(affinity.os, "sched_setaffinity", setter, raising=False)
    with pytest.raises(RuntimeError, match="GPU=GPU-test") if fails else nullcontext():
        affinity.validate_inference_cpu_affinity(all_threads=True)
    setter.assert_not_called()


def test_unknown_topology(monkeypatch):
    monkeypatch.setattr(affinity.sys, "platform", "linux")
    monkeypatch.setattr(affinity.os, "sched_getaffinity", Mock(), raising=False)
    monkeypatch.setattr(affinity, "_gpu_local_cpus", Mock(side_effect=RuntimeError("unavailable")))
    with pytest.raises(RuntimeError, match="Cannot determine GPU-local"):
        affinity.validate_inference_cpu_affinity()


def test_gpu_uuid_and_sparse_cpu_mask(monkeypatch):
    cuda = Mock()
    cuda.get_device_properties.return_value.uuid = "GPU-remapped"
    nvml = Mock()
    nvml.nvmlDeviceGetCpuAffinity.return_value = [0, 0, 8]
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    monkeypatch.setattr(affinity.Path, "read_text", lambda _: "0-3,128-131")
    monkeypatch.setattr(affinity.ctypes, "sizeof", lambda _: 8)
    assert affinity._gpu_local_cpus() == ("GPU-remapped", {131})
    nvml.nvmlDeviceGetHandleByUUID.assert_called_once_with("GPU-remapped")
    nvml.nvmlDeviceGetCpuAffinity.assert_called_once_with(
        nvml.nvmlDeviceGetHandleByUUID.return_value, 3
    )
    nvml.nvmlShutdown.assert_called_once()
