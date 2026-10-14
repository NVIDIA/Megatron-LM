# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Regression coverage for optimizer capture concurrent with data-loader pinning."""

import ctypes
import threading
from concurrent.futures import ThreadPoolExecutor

import torch

from megatron.core.optimizer.optimizer_cuda_graph import OptimizerCudaGraphWrapper
from tests.unit_tests.test_utilities import Utils


def test_optimizer_capture_allows_background_host_allocation(monkeypatch):
    """A fresh pinned allocation in another thread must not invalidate optimizer capture."""
    Utils.initialize_distributed()
    device = torch.cuda.current_device()
    for name, value in (("curr_iteration", 0), ("cuda_graph", None), ("result", None)):
        monkeypatch.setattr(OptimizerCudaGraphWrapper, name, value)

    # Call the runtime directly so PyTorch's pinned-memory cache cannot hide the race.
    # CUDA 12 and 13 use their major version as the runtime library's SONAME.
    cudart = ctypes.CDLL(f"libcudart.so.{torch.version.cuda.split('.')[0]}")
    cudart.cudaHostAlloc.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
        ctypes.c_uint,
    ]
    cudart.cudaHostAlloc.restype = ctypes.c_int
    cudart.cudaFreeHost.argtypes = [ctypes.c_void_p]
    cudart.cudaFreeHost.restype = ctypes.c_int
    host_ptr = ctypes.c_void_p()
    ready = threading.Event()
    allocate = threading.Event()
    value = torch.zeros(1, device="cuda")

    def allocate_pinned_memory():
        # Initialize the worker's CUDA device before capture begins.
        torch.cuda.set_device(device)
        ready.set()
        if not allocate.wait(timeout=30):
            raise TimeoutError("Optimizer capture did not request the pinned allocation")
        return cudart.cudaHostAlloc(ctypes.byref(host_ptr), 4096, 0)

    with ThreadPoolExecutor(max_workers=1) as executor:
        allocation = executor.submit(allocate_pinned_memory)

        def optimizer_step():
            if torch.cuda.is_current_stream_capturing():
                # Keep capture open until the independent allocation has completed.
                allocate.set()
                assert allocation.result(timeout=30) == 0
            value.add_(1)
            return value

        optimizer_step_graph = OptimizerCudaGraphWrapper(optimizer_step, cuda_graph_warmup_steps=1)
        try:
            assert ready.wait(timeout=30)
            optimizer_step_graph()  # Warmup.
            optimizer_step_graph()  # Capture with a concurrent cudaHostAlloc, then replay.
            optimizer_step_graph()  # Replay again without running the Python callback.
            torch.cuda.synchronize()
            assert host_ptr.value is not None
            torch.testing.assert_close(value, torch.full_like(value, 3), rtol=0, atol=0)
        finally:
            allocate.set()
            # Join before freeing the allocation, including if capture or an assertion failed.
            executor.shutdown(wait=True)
            del optimizer_step_graph
            if host_ptr.value is not None:
                assert cudart.cudaFreeHost(host_ptr) == 0
