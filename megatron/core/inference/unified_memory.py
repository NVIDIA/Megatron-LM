# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import os
import signal
import warnings
from contextlib import contextmanager
from enum import Enum, auto
from pathlib import Path

from torch.cuda.memory import CUDAPluggableAllocator
from torch.utils.cpp_extension import CUDA_HOME, load_inline

from megatron.core.utils import is_torch_min_version

try:
    if is_torch_min_version("2.8.0"):
        from torch.cuda.memory import MemPool
    else:
        from torch.cuda import MemPool
    _has_mem_pool = True
except ImportError:
    _has_mem_pool = False


class CompilationState(Enum):
    """Enum to distinguish between unified memory (UVM) compilation states."""

    UNATTEMPTED = auto()  # Compilation has not been attempted.
    FAILURE = auto()  # Compilation attempted, but failed.
    SUCCESS = auto()  # Compilation attempted, and succeeded.


class UnifiedMemoryUnsupportedError(Exception):
    """Unified memory is not supported on this system."""


class UnifiedMemoryCompileTimeoutError(UnifiedMemoryUnsupportedError):
    """Unified memory compilation timed out."""


# Compilation vars.
_compilation_state = CompilationState.UNATTEMPTED
_alloc = None  # must remain global until process exit.
_mod = None  # must remain global until process exit.


@contextmanager
def _compile_timeout(timeout_s: int):
    """Context manager to timeout compilation.

    Args:
        timeout_s (int): Timeout in seconds.
    """

    def _handler(signum, frame):
        raise UnifiedMemoryCompileTimeoutError(
            "Unified memory compilation has been forcefully timed out. "
            "This is almost certainly due to stale lock files associated with your Unix user. "
            "The official PyTorch advice is to resolve this issue with the following command:\n"
            "`rm -rf /tmp/torch_extensions/`\n"
            "Alternately, the TORCH_EXTENSIONS_DIR env var may be set to a different path. "
            "Please clean up your stale cache and try again."
        )

    curr_handler = signal.signal(signal.SIGALRM, _handler)
    try:
        signal.alarm(timeout_s)
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, curr_handler)


def compile_allocator():
    """Attempt to compile UVM allocator."""

    global _compilation_state, _alloc, _mod

    if _compilation_state != CompilationState.UNATTEMPTED:
        return

    _mempool_c_src = r"""
    #include <cuda_runtime_api.h>
    #include <cstddef>

    #define EXPORT extern "C"

    EXPORT void* managed_malloc(size_t size, int device, void* stream) {
      (void)stream;
      int prev_device = -1;
      cudaGetDevice(&prev_device);
      if (device != prev_device && device >= 0) cudaSetDevice(device);

      // cudaMallocManaged allows for more memory to be allocated than the device memory size.
      // The cudaMemAttachGlobal flag makes the memory accessible from both host and device.
      void* ptr = nullptr;
      cudaError_t err = cudaMallocManaged(&ptr, (size_t)size, cudaMemAttachGlobal);
      if (err != cudaSuccess) return nullptr;

      if (device >= 0) {
        // cudaMemAdviseSetPreferredLocation sets the preferred location for the memory.
        // This is a hint that tries to prevent data from being migrated away from the device.

        #if CUDART_VERSION >= 13000
          // For CUDA >= 13, the cudaMemAdvise device arg is type cudaMemLocation
          // instead of an int, so we setup the location and conditionally use it
          // in calls to cudaMemAdvise.
          cudaMemLocation location;
          location.type = cudaMemLocationTypeDevice;
          location.id = device;

          cudaMemAdvise(ptr, (size_t)size, cudaMemAdviseSetPreferredLocation, location);

          // cudaMemAdviseSetAccessedBy ensures the memory always lives in the device's page table.
          // Even if the memory has to be migrated away from the device, it still does not page fault.
          // The CUDA docs claim that cudaMemAdviseSetPreferredLocation completely overrides this flag,
          // but there is no harm in adding this flag as well for future-proofing.
          cudaMemAdvise(ptr, (size_t)size, cudaMemAdviseSetAccessedBy, location);
        #else
          cudaMemAdvise(ptr, (size_t)size, cudaMemAdviseSetPreferredLocation, device);
          // cudaMemAdviseSetAccessedBy ensures the memory always lives in the device's page table.
          // Even if the memory has to be migrated away from the device, it still does not page fault.
          // The CUDA docs claim that cudaMemAdviseSetPreferredLocation completely overrides this flag,
          // but there is no harm in adding this flag as well for future-proofing.
          cudaMemAdvise(ptr, (size_t)size, cudaMemAdviseSetAccessedBy, device);
        #endif
      }
      if (device != prev_device && prev_device >= 0) cudaSetDevice(prev_device);
      return ptr;
    }

    EXPORT void managed_free(void* ptr, size_t size, int device, void* stream) {
      // Memory allocated with cudaMallocManaged should be released with cudaFree.
      (void)size; (void)device; (void)stream;
      if (ptr) cudaFree(ptr);
    }
    """

    # Define a timeout of 30s for how long the build is allowed to run.
    timeout_s = 30

    # Build the .so upon import; this avoids issues.
    if _has_mem_pool:
        _extra_ldflags = ["-lcudart"]
        if CUDA_HOME:
            _cuda_lib = os.path.join(CUDA_HOME, "lib64")
            if os.path.isdir(_cuda_lib):
                _extra_ldflags = [f"-L{_cuda_lib}", "-lcudart"]
        try:
            with _compile_timeout(timeout_s):
                _mod = load_inline(
                    name="managed_alloc_runtime",
                    cpp_sources=[_mempool_c_src],
                    functions=[],
                    with_cuda=True,
                    extra_ldflags=_extra_ldflags,
                    verbose=True,
                )
                _so_path = Path(_mod.__file__).as_posix()
                _cpa = CUDAPluggableAllocator(_so_path, "managed_malloc", "managed_free")
                _alloc = _cpa.allocator()
                _compilation_state = CompilationState.SUCCESS
        except (RuntimeError, ImportError, OSError, UnifiedMemoryCompileTimeoutError) as e:
            warnings.warn(f"Failed to create unified memory mempool: '{e}'.")
            _compilation_state = CompilationState.FAILURE

        # Synchronize failure state across ranks. (For currently unknown reasons,
        # one rank can show as FAILURE while the remaining ranks show as SUCCESS.)
        import torch

        local_state = torch.tensor(
            [_compilation_state.value], dtype=torch.uint8, device=torch.cuda.current_device()
        )
        world_states = [
            torch.empty(1, dtype=torch.uint8, device=torch.cuda.current_device())
            for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(world_states, local_state)
        world_states = set(s.item() for s in world_states)
        if CompilationState.FAILURE.value in world_states:
            _compilation_state = CompilationState.FAILURE


def create_unified_mempool() -> "MemPool":
    """Create a unified memory mempool using CUDA managed memory.

    Returns:
        (MemPool) Unified memory mempool.
    """

    # Attempt to compile allocator.
    compile_allocator()

    # Return mempool.
    if _compilation_state != CompilationState.SUCCESS:
        raise UnifiedMemoryUnsupportedError()
    else:
        return MemPool(allocator=_alloc)
