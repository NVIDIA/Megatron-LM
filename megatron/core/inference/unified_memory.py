# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import ctypes
import os
import signal
import threading
import warnings
from contextlib import contextmanager
from enum import Enum, auto
from pathlib import Path

import torch
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
_so_path = None  # path to compiled extension .so (must remain global until exit).
_ctypes_lib = None  # ctypes handle to compiled extension
_ctypes_lock = threading.Lock()
_compilation_error: str | None = None  # store last failure reason for better error messages


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

    global _compilation_state, _alloc, _mod, _so_path, _ctypes_lib, _compilation_error

    if _compilation_state != CompilationState.UNATTEMPTED:
        return

    if not _has_mem_pool:
        _compilation_state = CompilationState.FAILURE
        _compilation_error = (
            "PyTorch does not expose CUDA MemPool on this build/version. "
            "UVM mempool requires torch.cuda.MemPool or torch.cuda.memory.MemPool."
        )
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

    // Prefetch managed memory to a device (or to CPU with cudaCpuDeviceId == -1).
    EXPORT int managed_prefetch(void* ptr, size_t size, int device, void* stream) {
      cudaStream_t s = (cudaStream_t)stream;
      cudaError_t err;
      #if CUDART_VERSION >= 13000
        cudaMemLocation location;
        if (device == (int)-1) {
          location.type = cudaMemLocationTypeHost;
          location.id = 0;
        } else {
          location.type = cudaMemLocationTypeDevice;
          location.id = device;
        }
        err = cudaMemPrefetchAsync(ptr, (size_t)size, location, 0, s);
      #else
        err = cudaMemPrefetchAsync(ptr, (size_t)size, device, s);
      #endif
      return (int)err;
    }

    // Update preferred location advice for managed memory (GPU device id, or CPU with cudaCpuDeviceId == -1).
    EXPORT int managed_advise_preferred_location(void* ptr, size_t size, int device) {
      cudaError_t err;
      #if CUDART_VERSION >= 13000
        cudaMemLocation location;
        if (device == (int)-1) {
          location.type = cudaMemLocationTypeHost;
          location.id = 0;
        } else {
          location.type = cudaMemLocationTypeDevice;
          location.id = device;
        }
        err = cudaMemAdvise(ptr, (size_t)size, cudaMemAdviseSetPreferredLocation, location);
      #else
        err = cudaMemAdvise(ptr, (size_t)size, cudaMemAdviseSetPreferredLocation, device);
      #endif
      return (int)err;
    }

    // Ensure a device is in the page table for this managed region.
    EXPORT int managed_advise_accessed_by(void* ptr, size_t size, int device) {
      cudaError_t err;
      #if CUDART_VERSION >= 13000
        cudaMemLocation location;
        location.type = cudaMemLocationTypeDevice;
        location.id = device;
        err = cudaMemAdvise(ptr, (size_t)size, cudaMemAdviseSetAccessedBy, location);
      #else
        err = cudaMemAdvise(ptr, (size_t)size, cudaMemAdviseSetAccessedBy, device);
      #endif
      return (int)err;
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
                _compilation_error = None
        except (RuntimeError, ImportError, OSError, UnifiedMemoryCompileTimeoutError) as e:
            _compilation_error = str(e)
            warnings.warn(f"Failed to create unified memory mempool: '{e}'.")
            _compilation_state = CompilationState.FAILURE
            _so_path = None
            _ctypes_lib = None

        # Synchronize failure state across ranks. (For currently unknown reasons,
        # one rank can show as FAILURE while the remaining ranks show as SUCCESS.)
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
        details = _compilation_error
        if details is None:
            details = "Unknown reason (allocator compilation did not succeed)."
        raise UnifiedMemoryUnsupportedError(
            "Unified virtual memory (UVM) mempool is unsupported or failed to initialize: "
            + details
        )
    else:
        return MemPool(allocator=_alloc)


def _get_ctypes_lib() -> "ctypes.CDLL":
    """Return a ctypes handle to the compiled UVM extension (.so)."""
    global _ctypes_lib
    compile_allocator()
    if _compilation_state != CompilationState.SUCCESS or _so_path is None:
        raise UnifiedMemoryUnsupportedError()
    if _ctypes_lib is not None:
        return _ctypes_lib
    with _ctypes_lock:
        if _ctypes_lib is None:
            _ctypes_lib = ctypes.CDLL(_so_path)
            # Configure argtypes/restype for exported helpers.
            _ctypes_lib.managed_prefetch.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
                ctypes.c_void_p,
            ]
            _ctypes_lib.managed_prefetch.restype = ctypes.c_int
            _ctypes_lib.managed_advise_preferred_location.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
            ]
            _ctypes_lib.managed_advise_preferred_location.restype = ctypes.c_int
            _ctypes_lib.managed_advise_accessed_by.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
            ]
            _ctypes_lib.managed_advise_accessed_by.restype = ctypes.c_int
    return _ctypes_lib


def prefetch_managed_tensor(tensor, *, device: int, stream=None) -> None:
    """Prefetch a CUDA tensor allocated from the UVM mempool to a specific device.

    This uses `cudaMemPrefetchAsync` to physically migrate the pages backing the tensor.
    The virtual address (pointer) remains unchanged, making this safe for use with
    recorded CUDA graphs.

    Args:
        tensor (torch.Tensor): CUDA tensor allocated from the UVM mempool.
        device (int): Target device ID. Use -1 (cudaCpuDeviceId) to prefetch to CPU.
        stream (torch.cuda.Stream, optional): Stream to use for the asynchronous prefetch.
            Defaults to the current stream.
    """
    if tensor is None:
        return
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("prefetch_managed_tensor expects a torch.Tensor")
    if tensor.numel() == 0:
        return
    if not tensor.is_cuda:
        raise ValueError("prefetch_managed_tensor expects a CUDA tensor")

    lib = _get_ctypes_lib()
    nbytes = tensor.nbytes
    if stream is None:
        stream = torch.cuda.current_stream()
    # torch.cuda.Stream exposes a cuda_stream integer handle.
    stream_ptr = ctypes.c_void_p(int(stream.cuda_stream))
    err = lib.managed_prefetch(
        ctypes.c_void_p(int(tensor.data_ptr())), ctypes.c_size_t(nbytes), int(device), stream_ptr
    )
    if err != 0:
        raise RuntimeError(f"cudaMemPrefetchAsync failed with cudaError={err}")


def advise_managed_tensor_preferred_location(tensor, *, device: int) -> None:
    """Set the preferred physical location hint for a managed tensor.

    This uses `cudaMemAdviseSetPreferredLocation`. It tells the CUDA driver where the
    pages should ideally reside. Unlike prefetch, this is a hint and does not
    immediately trigger migration unless the driver decides it is necessary.

    Args:
        tensor (torch.Tensor): CUDA tensor allocated from the UVM mempool.
        device (int): Preferred device ID. Use -1 (cudaCpuDeviceId) for CPU.
    """
    if tensor is None:
        return
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("advise_managed_tensor_preferred_location expects a torch.Tensor")
    if tensor.numel() == 0:
        return
    if not tensor.is_cuda:
        raise ValueError("advise_managed_tensor_preferred_location expects a CUDA tensor")

    lib = _get_ctypes_lib()
    nbytes = tensor.nbytes
    err = lib.managed_advise_preferred_location(
        ctypes.c_void_p(int(tensor.data_ptr())), ctypes.c_size_t(nbytes), int(device)
    )
    if err != 0:
        raise RuntimeError(f"cudaMemAdviseSetAccessedBy failed with cudaError={err}")


def advise_managed_tensor_accessed_by(tensor, *, device: int) -> None:
    """Hint that a specific device will access the managed tensor.

    This uses `cudaMemAdviseSetAccessedBy`. It ensures that the mapping for this
    memory region is established in the page tables of the specified device,
    reducing page fault latency when the device first touches the data.

    Args:
        tensor (torch.Tensor): CUDA tensor allocated from the UVM mempool.
        device (int): Device ID that will access the tensor. Must be a GPU ID.
    """
    if tensor is None:
        return
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("advise_managed_tensor_accessed_by expects a torch.Tensor")
    if tensor.numel() == 0:
        return
    if not tensor.is_cuda:
        raise ValueError("advise_managed_tensor_accessed_by expects a CUDA tensor")

    lib = _get_ctypes_lib()
    nbytes = tensor.nbytes
    err = lib.managed_advise_accessed_by(
        ctypes.c_void_p(int(tensor.data_ptr())), ctypes.c_size_t(nbytes), int(device)
    )
    if err != 0:
        raise RuntimeError(f"cudaMemAdviseSetAccessedBy failed with cudaError={err}")


def prefetch_managed_module_parameters(
    module, *, device: int, include_buffers: bool = False
) -> int:
    """Prefetch all UVM-allocated parameters (and optionally buffers) of a module.

    Iterates through all parameters of the module and initiates an asynchronous
    migration to the target device. This is typically used to offload weights to
    CPU during training or prefetch them to GPU before inference.

    Args:
        module (torch.nn.Module): The module containing UVM parameters.
        device (int): Target device ID (-1 for CPU).
        include_buffers (bool, optional): Whether to also prefetch module buffers.
            Defaults to False.

    Returns:
        int: The total number of bytes for which prefetch was initiated.
    """
    if module is None:
        return 0

    # Avoid duplicate prefetch on shared tensors.
    seen_ptrs: set[int] = set()
    total_nbytes = 0
    stream = torch.cuda.current_stream()

    for name, p in module.named_parameters(recurse=True):
        if p is None:
            continue
        t = p.data
        if not isinstance(t, torch.Tensor) or not t.is_cuda or t.numel() == 0:
            continue
        ptr = int(t.data_ptr())
        if ptr in seen_ptrs:
            continue
        seen_ptrs.add(ptr)
        nbytes = t.nbytes
        err = prefetch_managed_tensor(t, device=device, stream=stream)
        if err:
            raise RuntimeError(
                f"cudaMemPrefetchAsync failed (cudaError={err}) for parameter '{name}': "
                f"shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}, "
                f"data_ptr=0x{t.data_ptr():x}, nbytes={nbytes}. "
                "This tensor is not UVM-allocated."
            )
        total_nbytes += nbytes

    if include_buffers:
        for name, b in module.named_buffers(recurse=True):
            if b is None:
                continue
            if not isinstance(b, torch.Tensor) or not b.is_cuda or b.numel() == 0:
                continue
            ptr = int(b.data_ptr())
            if ptr in seen_ptrs:
                continue
            seen_ptrs.add(ptr)
            nbytes = b.nbytes
            err = prefetch_managed_tensor(b, device=device, stream=stream)
            if err:
                raise RuntimeError(
                    f"cudaMemPrefetchAsync failed (cudaError={err}) for buffer '{name}': "
                    f"shape={tuple(b.shape)}, dtype={b.dtype}, device={b.device}, "
                    f"data_ptr=0x{b.data_ptr():x}, nbytes={nbytes}. "
                    "This tensor is not UVM-allocated."
                )
            total_nbytes += nbytes

    return total_nbytes


def advise_managed_module_parameters_preferred_location(
    module, *, device: int, include_buffers: bool = False
) -> None:
    """Set the preferred physical location hint for all UVM parameters in a module.

    Args:
        module (torch.nn.Module): The module containing UVM parameters.
        device (int): Preferred device ID (-1 for CPU).
        include_buffers (bool, optional): Whether to also advise on module buffers.
            Defaults to False.
    """
    if module is None:
        return

    seen_ptrs: set[int] = set()
    for name, p in module.named_parameters(recurse=True):
        if p is None:
            continue
        t = p.data
        if not isinstance(t, torch.Tensor) or not t.is_cuda or t.numel() == 0:
            continue
        ptr = int(t.data_ptr())
        if ptr in seen_ptrs:
            continue
        seen_ptrs.add(ptr)
        err = advise_managed_tensor_preferred_location(t, device=device)
        if err:
            raise RuntimeError(
                f"cudaMemAdviseSetPreferredLocation failed (cudaError={err}) for param '{name}': "
                f"shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}, "
                f"data_ptr=0x{t.data_ptr():x}, nbytes={t.nbytes}. "
                "This tensor is not UVM-allocated."
            )

    if include_buffers:
        for name, b in module.named_buffers(recurse=True):
            if b is None:
                continue
            if not isinstance(b, torch.Tensor) or not b.is_cuda or b.numel() == 0:
                continue
            ptr = int(b.data_ptr())
            if ptr in seen_ptrs:
                continue
            seen_ptrs.add(ptr)
            err = advise_managed_tensor_preferred_location(b, device=device)
            if err:
                raise RuntimeError(
                    f"cudaMemAdviseSetPreferredLocation failed (err={err}) for buf '{name}': "
                    f"shape={tuple(b.shape)}, dtype={b.dtype}, device={b.device}, "
                    f"data_ptr=0x{b.data_ptr():x}, nbytes={b.nbytes}. "
                    "This tensor is not UVM-allocated."
                )
