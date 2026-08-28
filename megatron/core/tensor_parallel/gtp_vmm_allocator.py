# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""CUDA VMM allocator for the GTP symmetric-memory pools.

Implements NCCL's requirements on user-allocated communication buffers
(https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#memory-allocator)
minimally: VMM allocations at the recommended granularity, exportable handle
types (POSIX FD, plus FABRIC when supported — dropped on retry if cuMemCreate
rejects it), and GPUDirect-RDMA-capable physical memory when supported. Unlike
``ncclMemAlloc``, the memory is mapped only on the allocation's device — the
requirements ask for no more, and ncclMemAlloc's extra mappings on every
P2P-visible peer GPU slow CPU-side kernel launching for the whole training
step (measured at -6% end-to-end throughput at 256 GPUs, recovered by this
allocator).

NCCL window registration (``ncclCommWindowRegister`` via
``nccl_allocator.register_mem_pool``) accepts this memory and runs its
symmetric kernels on it.
"""

import logging
import os

import torch

# This import is needed for the cpp extension to work.
# pylint: disable=unused-import
from torch.utils import cpp_extension

import megatron.core.nccl_allocator as nccl_allocator
from megatron.core.nccl_allocator import get_func_args
from megatron.core.utils import log_single_rank

logger = logging.getLogger(__name__)

_allocator = None


def _build_vmm_allocator():
    global _allocator
    # If the allocator is already built, return
    if _allocator is not None:
        return

    vmm_allocator_source = """
    #include <c10/cuda/CUDACachingAllocator.h>
    #include <c10/cuda/CUDAGuard.h>
    #include <c10/util/Exception.h>
    #include <torch/csrc/cuda/CUDAPluggableAllocator.h>
    #include <torch/extension.h>

    #include <cuda.h>
    #include <cstdio>

    extern "C" {
        // Hooks are called through a C ABI: no exceptions, fail hard.
        #define CU_CHECK(cmd) do { \\
        CUresult r = cmd; \\
        if (r != CUDA_SUCCESS) { \\
            const char* name = nullptr; \\
            cuGetErrorName(r, &name); \\
            printf("Failed, CU error %s:%d '%s':", \\
                __FILE__,__LINE__, name ? name : "unknown"); \\
            exit(EXIT_FAILURE); \\
        } \\
        } while(0)

        // ncclMemAlloc's handle types: POSIX FD, plus FABRIC when supported.
        int requested_handle_types(CUdevice device, bool* fabric_added) {
            *fabric_added = false;
            int types = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        #if CUDA_VERSION >= 12030
            int supported = 0;
            CUresult result = cuDeviceGetAttribute(
                &supported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, device);
            if (result == CUDA_SUCCESS && supported) {
                types |= CU_MEM_HANDLE_TYPE_FABRIC;
                *fabric_added = true;
            }
        #endif
            return types;
        }

        void* gtp_vmm_alloc_plug(size_t size, int device, void* stream) {
            (void)stream;
            // Make the allocation's device current for the driver calls.
            c10::cuda::CUDAGuard device_guard(static_cast<c10::DeviceIndex>(device));
            CU_CHECK(cuInit(0));

            CUdevice cu_device;
            CU_CHECK(cuDeviceGet(&cu_device, device));

            bool fabric_added = false;
            int handle_types = requested_handle_types(cu_device, &fabric_added);

            CUmemAllocationProp prop = {};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = cu_device;
            prop.requestedHandleTypes = (CUmemAllocationHandleType)handle_types;

            int gdr_supported = 0;
            CU_CHECK(cuDeviceGetAttribute(
                &gdr_supported, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED,
                cu_device));
            if (gdr_supported) {
                prop.allocFlags.gpuDirectRDMACapable = 1;
            }

            size_t granularity = 0;
            CU_CHECK(cuMemGetAllocationGranularity(
                &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
            size_t mapped_size = ((size + granularity - 1) / granularity) * granularity;

            CUmemGenericAllocationHandle handle = {};
            CUresult create_result = cuMemCreate(&handle, mapped_size, &prop, 0);
        #if CUDA_VERSION >= 12030
            // ncclMemAlloc's fallback: only the auto-added FABRIC bit may drop.
            if (create_result != CUDA_SUCCESS && fabric_added &&
                (create_result == CUDA_ERROR_NOT_PERMITTED ||
                 create_result == CUDA_ERROR_NOT_SUPPORTED)) {
                handle_types &= ~CU_MEM_HANDLE_TYPE_FABRIC;
                prop.requestedHandleTypes = (CUmemAllocationHandleType)handle_types;
                create_result = cuMemCreate(&handle, mapped_size, &prop, 0);
            }
        #endif
            CU_CHECK(create_result);

            CUdeviceptr address = 0;
            CU_CHECK(cuMemAddressReserve(&address, mapped_size, granularity, 0, 0));
            CU_CHECK(cuMemMap(address, mapped_size, 0, handle, 0));

            // The delta vs ncclMemAlloc: grant access on this device ONLY, not on
            // every P2P-visible peer (persistent peer mappings slow kernel launch).
            CUmemAccessDesc access = {};
            access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access.location.id = device;
            access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            CU_CHECK(cuMemSetAccess(address, mapped_size, &access, 1));

            // The mapping keeps the allocation alive; free recovers the handle.
            CU_CHECK(cuMemRelease(handle));
            return (void*)address;
        }

        void gtp_vmm_free_plug(void* ptr, size_t size, int device, void* stream) {
            (void)size;
            (void)stream;
            c10::cuda::CUDAGuard device_guard(static_cast<c10::DeviceIndex>(device));
            CUdeviceptr address = (CUdeviceptr)ptr;

            // Recover handle and extent before unmapping invalidates the pointer.
            CUmemGenericAllocationHandle handle = {};
            CU_CHECK(cuMemRetainAllocationHandle(&handle, ptr));
            CUdeviceptr base = 0;
            size_t mapped_size = 0;
            CU_CHECK(cuMemGetAddressRange(&base, &mapped_size, address));
            CU_CHECK(cuMemUnmap(address, mapped_size));
            CU_CHECK(cuMemRelease(handle));
            CU_CHECK(cuMemAddressFree(address, mapped_size));
        }

        std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator> vmm_allocator;

        void maybe_init() {
            if (!vmm_allocator) {
                vmm_allocator = std::make_shared<
                    torch::cuda::CUDAPluggableAllocator::CUDAPluggableAllocator>(
                    gtp_vmm_alloc_plug, gtp_vmm_free_plug);
            }
        }

        std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>
        get_vmm_allocator() {
        maybe_init();
        return vmm_allocator;
        }

        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("get_vmm_allocator", []() { return get_vmm_allocator(); });
        };
    }
    """
    # Same shared build dir as the nccl allocator extension; torch's file lock
    # serializes concurrent builds across local ranks.
    module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build_dir = os.path.join(module_dir, "build")
    os.makedirs(build_dir, exist_ok=True)
    try:
        vmm_allocator = torch.utils.cpp_extension.load_inline(
            name="gtp_vmm_allocator",
            cpp_sources=vmm_allocator_source,
            with_cuda=True,
            extra_ldflags=["-lcuda"],
            verbose=True,
            is_python_module=True,
            build_directory=build_dir,
        )
    except Exception as e:
        raise RuntimeError(
            "[GTP] Failed to build the GTP VMM allocator extension; "
            "--gtp-remat-nccl-ub/--gtp-expert-remat-nccl-ub require nvcc and libcuda "
            "at runtime."
        ) from e

    _allocator = vmm_allocator.get_vmm_allocator()


def create_vmm_mem_pool() -> torch.cuda.MemPool:
    """
    Create a symmetric memory pool using the VMM allocator. GTP symmetric pools are
    always symmetric (register_gtp_symm_pool enforces the torch >= 2.9 floor).
    """
    _build_vmm_allocator()
    assert _allocator is not None, "VMM allocator is not initialized"
    if 'symmetric' in get_func_args(torch.cuda.MemPool):
        # PyTorch >= 2.9.0a0 and before PyTorch PR #161238 takes the symmetric knob at
        # MemPool construction; since #161238 it lives in the registration function.
        return torch.cuda.MemPool(_allocator, symmetric=True)
    if 'symm_mem' in get_func_args(torch.cuda.MemPool):
        # Argument name divergence between nvidia pytorch and the official pytorch.
        return torch.cuda.MemPool(_allocator, symm_mem=True)
    # The symmetric knob is in the registration function.
    return torch.cuda.MemPool(_allocator)


def init() -> None:
    """
    Initialize the VMM allocator, including the NCCL environment its pools are
    registered under (same settings as nccl_allocator.init()).
    """
    # Enables NCCL NVLS algorithm
    os.environ["NCCL_NVLS_ENABLE"] = "1"
    # Disables the use of the tensor register allocator hook
    os.environ["TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK"] = "0"
    _build_vmm_allocator()
    log_single_rank(logger, logging.INFO, "[MCORE][GTP] Initialized the VMM Allocator")


def register_mem_pool(pool: torch.cuda.MemPool, group) -> None:
    """
    Window-register a VMM pool's segments on ``group`` (always symmetric).
    Delegating to nccl_allocator is safe because its (de)registration walks the
    pool's segments and never touches the allocator that produced them.
    """
    nccl_allocator.register_mem_pool(pool, group, symmetric=True)


def deregister_mem_pool(pool: torch.cuda.MemPool, group) -> None:
    """
    Deregister a VMM pool's windows from ``group``. Delegates to nccl_allocator.
    """
    nccl_allocator.deregister_mem_pool(pool, group)
