# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import sys
from unittest.mock import Mock

import pytest

from megatron.core.inference import cpu_affinity as affinity


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("58f3a1b2-0000-1111-2222-333344445555", "GPU-58f3a1b2-0000-1111-2222-333344445555"),
        ("GPU-58f3a1b2-0000-1111-2222-333344445555", "GPU-58f3a1b2-0000-1111-2222-333344445555"),
        ("MIG-58f3a1b2-0000-1111-2222-333344445555", "MIG-58f3a1b2-0000-1111-2222-333344445555"),
    ],
)
def test_nvml_uuid(raw, expected):
    assert affinity._nvml_uuid(raw) == expected


@pytest.mark.parametrize("possible,count", [("0", 1), ("0-3", 4), ("0-3,128-131\n", 132)])
def test_possible_cpu_count(possible, count):
    assert affinity._possible_cpu_count(possible) == count


@pytest.mark.parametrize(
    "cpus,expected",
    [
        (set(), ""),
        ({5}, "5"),
        (set(range(70)), "0-69"),
        ({0, 1, 2, 4, 70, 71, 139}, "0-2,4,70-71,139"),
    ],
)
def test_format_cpus(cpus, expected):
    assert affinity._format_cpus(cpus) == expected


def test_mask_to_cpus_sparse():
    assert affinity._mask_to_cpus([0b101, 0, 1 << 3], word_bits=64) == {0, 2, 131}


@pytest.mark.parametrize(
    "masks,expected",
    [
        ({1: {2}, 3: {3}}, []),
        ({1: {2, 4}, 3: {3}}, ["tid=1 allowed=[2,4]"]),
        ({1: {2}, 3: {3, 4}}, ["tid=3 allowed=[3-4]"]),
        ({1: {2}, 3: set()}, ["tid=3 allowed=[]"]),
    ],
)
def test_find_violations(masks, expected):
    def get_affinity(tid):
        if tid not in masks:
            raise ProcessLookupError(tid)
        return masks[tid]

    # tid 2 exits between enumeration and inspection and is skipped.
    assert affinity._find_violations([1, 2, 3], 1, {2, 3}, get_affinity) == expected


def test_find_violations_caller_missing_raises():
    with pytest.raises(ProcessLookupError):
        affinity._find_violations([1], 1, {0}, Mock(side_effect=ProcessLookupError()))


def test_gpu_local_cpus_queries_nvml_by_uuid(monkeypatch):
    nvml = Mock()
    nvml.nvmlDeviceGetCpuAffinity.return_value = [0, 1 << 3]
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    monkeypatch.setattr(affinity.Path, "read_text", lambda _: "0-127\n")
    word_bits = affinity.ctypes.sizeof(affinity.ctypes.c_ulong) * 8

    cpus = affinity._gpu_local_cpus("GPU-abc")

    assert cpus == {word_bits + 3}
    nvml.nvmlDeviceGetHandleByUUID.assert_called_once_with("GPU-abc")
    nvml.nvmlDeviceGetCpuAffinity.assert_called_once_with(
        nvml.nvmlDeviceGetHandleByUUID.return_value, 128 // word_bits
    )
    nvml.nvmlShutdown.assert_called_once()


@pytest.mark.skipif(not hasattr(os, "sched_getaffinity"), reason="Linux only")
def test_unknown_topology_raises(monkeypatch):
    monkeypatch.setattr(affinity, "_gpu_local_cpus", Mock(side_effect=RuntimeError("no NVML")))
    monkeypatch.setattr(affinity.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(affinity.torch.cuda, "get_device_properties", lambda _: Mock(uuid="abc"))
    with pytest.raises(RuntimeError, match="Cannot determine GPU-local"):
        affinity.validate_inference_cpu_affinity()
