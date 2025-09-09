# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import os
import warnings
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
    has_mem_pool = True
except ImportError:
    has_mem_pool = False

src = r"""
#include <cuda_runtime_api.h>
#include <cstddef>

#define EXPORT extern "C"

EXPORT void* managed_malloc(size_t size, int device, void* stream) {
  (void)stream;
  int cur = -1;
  cudaGetDevice(&cur);
  if (device != cur && device >= 0) cudaSetDevice(device);

  void* ptr = nullptr;
  cudaError_t err = cudaMallocManaged(&ptr, (size_t)size, cudaMemAttachGlobal);
  if (err != cudaSuccess) return nullptr;

  if (device >= 0) {
    cudaMemAdvise(ptr, (size_t)size, cudaMemAdviseSetPreferredLocation, device);
    cudaMemAdvise(ptr, (size_t)size, cudaMemAdviseSetAccessedBy, device);
  }
  return ptr;
}

EXPORT void managed_free(void* ptr, size_t size, int device, void* stream) {
  (void)size; (void)device; (void)stream;
  if (ptr) cudaFree(ptr);
}
"""

unified_memory_mempool = None

if has_mem_pool:
    extra_ldflags = ["-lcudart"]
    if CUDA_HOME:
        cuda_lib = os.path.join(CUDA_HOME, "lib64")
        if os.path.isdir(cuda_lib):
            extra_ldflags = [f"-L{cuda_lib}", "-lcudart"]
    try:
        mod = load_inline(
            name="managed_alloc_runtime",
            cpp_sources=[src],
            functions=[],
            with_cuda=True,
            extra_ldflags=extra_ldflags,
            verbose=False,
        )
        so_path = Path(mod.__file__).as_posix()
        alloc = CUDAPluggableAllocator(so_path, "managed_malloc", "managed_free").allocator()
        unified_memory_mempool = MemPool(allocator=alloc)
    except (RuntimeError, ImportError):
        warnings.warn("Unified memory mempool is not available.")
