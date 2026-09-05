# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys
import uuid
from types import ModuleType, SimpleNamespace

from megatron.core.pipeline_parallel import utils as pipeline_utils


def test_set_ideal_affinity_tolerates_restricted_nvml(monkeypatch):
    """NVML affinity is an optional optimization in restricted containers."""
    cuda = ModuleType("cuda")
    cuda.__path__ = []
    bindings = ModuleType("cuda.bindings")
    bindings.__path__ = []
    driver = ModuleType("cuda.bindings.driver")
    runtime = ModuleType("cuda.bindings.runtime")

    cuda.bindings = bindings
    bindings.driver = driver
    bindings.runtime = runtime

    cuda_success = object()
    driver_success = object()
    runtime.cudaError_t = SimpleNamespace(cudaSuccess=cuda_success)
    runtime.cudaGetDevice = lambda: (cuda_success, 0)
    driver.CUresult = SimpleNamespace(CUDA_SUCCESS=driver_success)
    driver.cuDeviceGetUuid = lambda _device_id: (
        driver_success,
        SimpleNamespace(bytes=uuid.UUID(int=0).bytes),
    )

    pynvml = ModuleType("pynvml")

    class NVMLError(Exception):
        pass

    pynvml.NVMLError = NVMLError
    pynvml.nvmlInit = lambda: None
    pynvml.nvmlDeviceGetHandleByUUID = lambda _uuid: object()

    def fail_to_set_affinity(_handle):
        raise NVMLError("Unknown Error")

    pynvml.nvmlDeviceSetCpuAffinity = fail_to_set_affinity

    monkeypatch.setitem(sys.modules, "cuda", cuda)
    monkeypatch.setitem(sys.modules, "cuda.bindings", bindings)
    monkeypatch.setitem(sys.modules, "cuda.bindings.driver", driver)
    monkeypatch.setitem(sys.modules, "cuda.bindings.runtime", runtime)
    monkeypatch.setitem(sys.modules, "pynvml", pynvml)

    messages = []
    monkeypatch.setattr(
        pipeline_utils,
        "log_single_rank",
        lambda _logger, _level, message: messages.append(message),
    )

    pipeline_utils.set_ideal_affinity_for_current_gpu()

    assert messages == [
        "Could not set CPU affinity through NVML; continuing without it: Unknown Error"
    ]
