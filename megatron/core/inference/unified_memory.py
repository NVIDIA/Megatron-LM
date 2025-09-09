# build_and_use_managed_allocator.py
import os, pathlib
import torch
from torch.utils.cpp_extension import load_inline, CUDA_HOME
from torch.cuda.memory import CUDAPluggableAllocator, change_current_allocator

# 1) Inline C++ source for a managed-memory allocator
src = r"""
#include <cuda_runtime_api.h>
#include <cstddef>

#define EXPORT extern "C"

// malloc: allocate Unified (Managed) memory
EXPORT void* my_managed_malloc(ssize_t size, int device, void* stream) {
  (void)stream;
  int cur = -1;
  cudaGetDevice(&cur);
  if (device != cur && device >= 0) cudaSetDevice(device);

  void* ptr = nullptr;
  cudaError_t err = cudaMallocManaged(&ptr, (size_t)size, cudaMemAttachGlobal);
  if (err != cudaSuccess) return nullptr;

  // Optional hints to reduce first-touch page faults:
  if (device >= 0) {
    cudaMemAdvise(ptr, (size_t)size, cudaMemAdviseSetPreferredLocation, device);
    cudaMemAdvise(ptr, (size_t)size, cudaMemAdviseSetAccessedBy, device);
  }
  return ptr;
}

// free
EXPORT void my_managed_free(void* ptr, ssize_t size, int device, void* stream) {
  (void)size; (void)device; (void)stream;
  if (ptr) cudaFree(ptr);
}
"""

# 2) JIT-compile a tiny extension that exports those C symbols.
#    We link against cudart; on most systems, -lcudart is enough if CUDA is on LD_LIBRARY_PATH.
extra_ldflags = ["-lcudart"]
if CUDA_HOME:
    # help linker find libcudart if not on the default path
    cand = os.path.join(CUDA_HOME, "lib64")
    if os.path.isdir(cand):
        extra_ldflags = [f"-L{cand}", "-lcudart"]

mod = load_inline(
    name="managed_alloc_runtime",
    cpp_sources=[src],
    functions=[],          # no pybind functions; we just want the .so with our exported symbols
    with_cuda=True,        # ensures CUDA include paths & NVCC toolchain are available
    extra_ldflags=extra_ldflags,
    verbose=False,
)

# 3) Locate the built shared library and plug it into PyTorch
so_path = pathlib.Path(mod.__file__).as_posix()
alloc = CUDAPluggableAllocator(so_path, "my_managed_malloc", "my_managed_free").allocator()
unified_memory_mempool = torch.cuda.MemPool(allocator=alloc)