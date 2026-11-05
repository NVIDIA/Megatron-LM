// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// See LICENSE for license information.

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <torch/csrc/cuda/CUDAPluggableAllocator.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cstdio>
#include <cstdlib>

extern "C" {
// Hooks are called through a C ABI: no exceptions, fail hard.
#define CU_CHECK(cmd)                                                                           \
    do {                                                                                        \
        CUresult r = cmd;                                                                       \
        if (r != CUDA_SUCCESS) {                                                                \
            const char* name = nullptr;                                                         \
            cuGetErrorName(r, &name);                                                           \
            printf("Failed, CU error %s:%d '%s':", __FILE__, __LINE__, name ? name : "unknown");\
            exit(EXIT_FAILURE);                                                                 \
        }                                                                                       \
    } while (0)

// ncclMemAlloc's handle types: POSIX FD, plus FABRIC when supported.
int requested_handle_types(CUdevice device, bool* fabric_added) {
    *fabric_added = false;
    int types = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#if CUDA_VERSION >= 12030
    int supported = 0;
    CUresult result =
        cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, device);
    if (result == CUDA_SUCCESS && supported) {
        types |= CU_MEM_HANDLE_TYPE_FABRIC;
        *fabric_added = true;
    }
#endif
    return types;
}

void* vmm_alloc_plug(size_t size, int device, void* stream) {
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
        &gdr_supported, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, cu_device));
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
    // FABRIC is best-effort: the device attribute can report support on nodes
    // where IMEX is not actually provisioned, and that surfaces under several
    // error codes -- retry on any failure. Safe because fabric_added means we
    // added the bit ourselves; a genuine error still fails the retry below.
    if (create_result != CUDA_SUCCESS && fabric_added) {
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

void vmm_free_plug(void* ptr, size_t size, int device, void* stream) {
    (void)size;
    (void)stream;
    c10::cuda::CUDAGuard device_guard(static_cast<c10::DeviceIndex>(device));
    CUdeviceptr address = (CUdeviceptr)ptr;

    // Recover handle and extent before unmapping invalidates the pointer.
    // Requirement: the driver must support cuMemRetainAllocationHandle and
    // cuMemGetAddressRange on cuMemMap'd VMM ranges (the docs only mention
    // cuMemAlloc, but ncclMemFree relies on this same sequence).
    CUmemGenericAllocationHandle handle = {};
    CU_CHECK(cuMemRetainAllocationHandle(&handle, ptr));
    size_t mapped_size = 0;
    CU_CHECK(cuMemGetAddressRange(NULL, &mapped_size, address));
    CU_CHECK(cuMemUnmap(address, mapped_size));
    CU_CHECK(cuMemRelease(handle));
    CU_CHECK(cuMemAddressFree(address, mapped_size));
}

std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator> vmm_allocator;

void maybe_init() {
    if (!vmm_allocator) {
        vmm_allocator =
            std::make_shared<torch::cuda::CUDAPluggableAllocator::CUDAPluggableAllocator>(
                vmm_alloc_plug, vmm_free_plug);
    }
}

std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator> get_vmm_allocator() {
    maybe_init();
    return vmm_allocator;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_vmm_allocator", []() { return get_vmm_allocator(); });
};
}
