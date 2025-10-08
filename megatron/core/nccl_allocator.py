# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import logging
import os
from contextlib import nullcontext

import torch

# This import is needed for the cpp extension to work.
# pylint: disable=unused-import
from torch.utils import cpp_extension

from megatron.core.utils import is_torch_min_version

# MCORE NCCL Allocator copies and modifies the APEX NCCL allocator.
# The original APEX NCCL allocator is available at:
# https://github.com/NVIDIA/apex/blob/master/apex/contrib/nccl_allocator.py
# https://github.com/NVIDIA/apex/blob/master/apex/contrib/csrc/nccl_allocator/NCCLAllocator.cpp

_allocator = None


def _build_nccl_allocator():
    global _allocator
    # If the allocator is already built, return
    if _allocator is not None:
        return

    nccl_allocator_source = """
    #include <c10/cuda/CUDACachingAllocator.h>
    #include <c10/util/Exception.h>
    #include <torch/csrc/cuda/CUDAPluggableAllocator.h>
    #include <torch/extension.h>

    #include <nccl.h>
    #include <iostream>
    #include <cstdio>

    extern "C" {
        #define NCCL_CHECK(cmd) do { \
        ncclResult_t r = cmd; \
        if (r != ncclSuccess) { \
            printf("Failed, NCCL error %s:%d '%s':", \
                __FILE__,__LINE__,ncclGetErrorString(r)); \
            exit(EXIT_FAILURE); \
        } \
        } while(0)

        void* nccl_alloc_plug(size_t size, int device, void* stream) {
            void* ptr;
            NCCL_CHECK(ncclMemAlloc(&ptr, size));
            return ptr;
        }

        void nccl_free_plug(void* ptr, size_t size, int device, void* stream) {
            NCCL_CHECK(ncclMemFree(ptr));
        }

        std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator> nccl_allocator;

        void maybe_init() {
            if (!nccl_allocator) {
                nccl_allocator = std::make_shared<
                    torch::cuda::CUDAPluggableAllocator::CUDAPluggableAllocator>(
                    nccl_alloc_plug, nccl_free_plug);
            }
        }

        std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>
        get_nccl_allocator() {
        maybe_init();
        return nccl_allocator;
        }

        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("get_nccl_allocator", []() { return get_nccl_allocator(); });
        };
    }
    """
    module_dir = os.path.dirname(__file__)
    source_dir = os.path.join(module_dir, "build")
    nccl_allocator_libname = "nccl_allocator"
    os.makedirs(source_dir, exist_ok=True)

    nccl_allocator = torch.utils.cpp_extension.load_inline(
        name=nccl_allocator_libname,
        cpp_sources=nccl_allocator_source,
        with_cuda=True,
        extra_ldflags=["-lnccl"],
        verbose=True,
        is_python_module=True,
        build_directory=source_dir,
    )

    _allocator = nccl_allocator.get_nccl_allocator()


def get_func_args(func):
    """
    Get the argument names of a function.
    """
    import inspect

    sig = inspect.signature(func)
    return [arg.name for arg in sig.parameters.values()]


def create_nccl_mem_pool(symmetric=None):  # symmetric: bool | None = None -> torch.cuda.MemPool:
    """
    Create a memory pool using the NCCL allocator.
    """
    _build_nccl_allocator()
    if not is_torch_min_version("2.9.0a0") and symmetric is True:
        logging.info(
            f"Symmetric memory pool is not supported with torch version < 2.9.0a0"
            f"Current torch version: {torch.__version__}"
            "falling back to non-symmetric memory pool"
        )
        symmetric = False

    assert _allocator is not None, "NCCL allocator is not initialized"
    if not symmetric:
        _pool = torch.cuda.MemPool(_allocator)
    else:
        if 'symmetric' in get_func_args(torch.cuda.MemPool):
            _pool = torch.cuda.MemPool(_allocator, symmetric=symmetric)
        elif 'symm_mem' in get_func_args(torch.cuda.MemPool):
            # This path handles argument name divergence between
            # nvidia pytorch and the official pytorch.
            _pool = torch.cuda.MemPool(_allocator, symm_mem=symmetric)
        else:
            raise ValueError(
                "symmetric setting with torch.cuda.MemPool requires " "higher PyTorch version"
            )
    return _pool


def init() -> None:
    """
    Initialize the NCCL allocator.

    PyTorch tracks memory registration at the pool level, not per allocation.
    If a pool already contains allocations from a previous context, attempting
    to register it again will re-register all existing allocations and may
    trigger NCCL errors. To avoid this, the pool is explicitly deregistered
    on entry and re-registered on exit for each context use.
    """
    # Enables NCCL NVLS algorithm
    os.environ["NCCL_NVLS_ENABLE"] = "1"
    # Disables the use of the tensor register allocator hook
    os.environ["TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK"] = "0"
    _build_nccl_allocator()
    print(f"[MCORE][NCCL_ALLOCATOR] Initialized NCCL Allocator")


# Preserve the original APEX NCCL allocator interface for backward compatibility
class nccl_mem:
    """
    An NCCL memory allocator, which inherits APEX nccl_allocator implementation.
    """

    def __init__(self, pool, enabled=True, device=None, group=None):
        self.device = None
        self.group = None
        self.mem_context = None
        self.pool = pool

        if enabled:
            if device is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            elif isinstance(device, int):
                self.device = torch.device("cuda", device)
            elif isinstance(device, str):
                assert "cuda" in device, "only cuda devices are supported"
                self.device = torch.device(device)

            if group is None:
                self.group = torch.distributed.distributed_c10d._get_default_group()
            else:
                self.group = group

            self.mem_context = torch.cuda.use_mem_pool(self.pool)
        else:
            self.mem_context = nullcontext()

    def __enter__(self):
        self.mem_context.__enter__()
        if self.group is not None:
            backend = self.group._get_backend(self.device)
            try:
                # Deregister first to avoid duplicate registration of previously
                # registered memory.
                backend.deregister_mem_pool(self.pool)
            except RuntimeError:
                desc = getattr(self.group, "group_desc", None)
                print(
                    f"[MCORE][NCCL_ALLOCATOR] Failed to deregister mem pool from"
                    f"{repr(self.group)}({desc}) group!!"
                )

    def __exit__(self, *args):
        if self.group is not None:
            backend = self.group._get_backend(self.device)
            try:
                backend.register_mem_pool(self.pool)
            except RuntimeError:
                desc = getattr(self.group, "group_desc", None)
                print(
                    f"[MCORE][NCCL_ALLOCATOR] Failed to register mem pool to"
                    f"{repr(self.group)}({desc}) group!!"
                )

        self.mem_context.__exit__(*args)


class MultiGroupMemPoolAllocator:
    """
    A custom allocator class that registers a single memory pool with multiple communication groups.

    Use cases:
    - [FSDP+EP] In case of FSDP with EP, expert layer (expert-dp) and non-expert layer (dp) use
      different communicator groups. The same memory pool has to be registered to both the groups.
    - [Hybrid FSDP/DP] In case of Hybrid FSDP/DP, there are inter-dp group and intra-dp group.
      The same memory pool has to be registered to both the groups.
    - [Hybrid FSDP/DP + EP] In case of Hybrid FSDP/DP + EP, there are inter-dp, intra-dp, and
      expert-dp groups. The same memory pool has to be registered to all the groups.

    Example:
        ```
        import megatron.core.nccl_allocator as nccl_allocator
        nccl_allocator.init()
        pool = nccl_allocator.create_nccl_mem_pool()
        group_1 = torch.distributed.new_group(ranks=[0, 1, 2, 3, 4, 5, 6, 7], backend="nccl")
        group_2 = torch.distributed.new_group(ranks=[0, 2, 4, 6], backend="nccl")
        with MultiGroupMemPoolAllocator(pool, [group_1, group_2]):
            a = torch.zeros(1024, dtype=torch.float32, device="cuda")
            b = torch.zeros(1024, dtype=torch.float32, device="cuda")
        ```
    """

    def __init__(
        self, pool, groups
    ):  # pool: torch.cuda.MemPool, groups: List[torch.distributed.ProcessGroup]
        self.pool = pool
        self.groups = groups
        self.mem_context = torch.cuda.use_mem_pool(self.pool)

        assert isinstance(self.pool, torch.cuda.MemPool), "pool must be a torch.cuda.MemPool"
        assert isinstance(self.groups, list), "groups must be a list"
        assert all(
            isinstance(group, torch.distributed.ProcessGroup) for group in self.groups
        ), "groups must be a list of torch.distributed.ProcessGroup"

    def __enter__(self):
        self.mem_context.__enter__()
        for group in self.groups:
            backend = group._get_backend(torch.device("cuda", torch.cuda.current_device()))
            try:
                # Since the registration is done in mempool granularity, we need to deregister
                # the tensors in the mempool and re-register the mempool including the newly created
                # tensors after the context is exited.
                backend.deregister_mem_pool(self.pool)
            except RuntimeError:
                desc = getattr(group, "group_desc", None)
                print(
                    f"[MCORE][MultiGroupMemPoolAllocator] Failed to deregister mem pool from"
                    f"{repr(group)}({desc}) group!!"
                )

    def __exit__(self, *args):
        for group in self.groups:
            backend = group._get_backend(torch.device("cuda", torch.cuda.current_device()))
            try:
                backend.register_mem_pool(self.pool)
            except RuntimeError:
                desc = getattr(group, "group_desc", None)
                print(
                    f"[MCORE][MultiGroupMemPoolAllocator] Failed to register mem pool to"
                    f"{repr(group)}({desc}) group!!"
                )
        self.mem_context.__exit__(*args)
