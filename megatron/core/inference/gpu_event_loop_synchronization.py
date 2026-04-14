# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import ctypes
from typing import Optional

import torch

_libcudart: Optional[ctypes.CDLL] = None
_CUDA_HOST_FN_T = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


def _get_cudart() -> ctypes.CDLL:
    """Lazily load and configure the CUDA runtime library."""
    global _libcudart
    if _libcudart is None:
        cuda_major = torch.version.cuda.split('.')[0]
        _libcudart = ctypes.CDLL(f"libcudart.so.{cuda_major}")
        _libcudart.cudaLaunchHostFunc.restype = ctypes.c_int
        _libcudart.cudaLaunchHostFunc.argtypes = [
            ctypes.c_void_p,  # cudaStream_t
            _CUDA_HOST_FN_T,  # cudaHostFn_t
            ctypes.c_void_p,  # void* userData
        ]
    return _libcudart


class GPUFuture:
    """Awaitable that resolves when all preceding work on a CUDA stream completes.

    Instead of blocking the CPU with ``torch.cuda.synchronize()`` or
    ``event.synchronize()``, this uses ``cudaLaunchHostFunc`` to enqueue a
    host-side callback on the CUDA stream.  The callback resolves an asyncio
    ``Future`` via ``call_soon_threadsafe``, allowing other asyncio tasks to
    run while the GPU is busy.

    Usage:
        gpu_done = GPUFuture(loop)
        launch_gpu_work(...)
        gpu_done.record()
        launch_more_gpu_work(...)
        await gpu_done          # other tasks run while GPU is busy
    """

    # Prevent garbage-collection of live ctypes callbacks.
    # Keyed by id() because ctypes function pointers are not hashable.
    _prevent_gc: dict = {}

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._future: asyncio.Future = loop.create_future()

    def record(self, stream: Optional[torch.cuda.Stream] = None) -> None:
        """Enqueue a host callback that resolves this future.

        Args:
            stream: CUDA stream to attach to. Defaults to the current stream.
        """
        if stream is None:
            stream = torch.cuda.current_stream()

        # This closure prevents the ctypes wrapper from being collected
        # while the callback is in the stream.
        prevent_gc_ref: Optional[object] = None

        def _host_fn(_user_data: ctypes.c_void_p) -> None:
            # Runs on CUDA's internal callback thread.
            # MUST NOT call any CUDA API.
            try:
                self._loop.call_soon_threadsafe(self._future.set_result, None)
            except RuntimeError:
                # Event loop closed; nothing to do.
                pass
            GPUFuture._prevent_gc.pop(id(prevent_gc_ref), None)

        c_fn = _CUDA_HOST_FN_T(_host_fn)
        prevent_gc_ref = c_fn
        GPUFuture._prevent_gc[id(c_fn)] = c_fn

        err = _get_cudart().cudaLaunchHostFunc(
            ctypes.c_void_p(stream.cuda_stream), c_fn, ctypes.c_void_p(0)
        )
        if err != 0:
            # Allow the callback to be garbage-collected on failure.
            GPUFuture._prevent_gc.pop(id(c_fn), None)
            raise RuntimeError(f"cudaLaunchHostFunc failed with CUDA error {err}")

    def __await__(self):
        return self._future.__await__()
