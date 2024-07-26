// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

#include "ds_kernel_utils.h"
#include "memory_access_utils.h"
#include "quantization.h"
#include "quantization_utils.h"
#include "reduction_utils.h"
#include "dequantization_utils.h"
#include <torch/extension.h>

namespace cg = cooperative_groups;

template <typename scalar_t, int vec_size>
__device__ __inline__ size_t load_to_local(
    scalar_t* __restrict__ local_buffer,
    scalar_t** __restrict__ model_params,
    const size_t* param_sizes,
    const size_t num_params,
    const size_t total_length,
    const size_t idx,
    const size_t param_offset) {

    size_t left = param_offset;
    size_t right = num_params - 1;
    size_t param_idx = num_params;

    // binary search for param list offset
    while (left <= right) {
        size_t mid = (left + right) / 2;
        size_t mid_start_idx = mid==0 ? 0 : param_sizes[mid-1];
        size_t mid_end_idx = param_sizes[mid];
        if (mid_end_idx <= idx) {
            left = mid + 1;
        } else if (idx < mid_start_idx) {
            right = mid - 1;
        } else {
            param_idx = mid;
            break;
        }
    }

    if (param_idx >= num_params) {

#pragma unroll
        for (int j = 0; j < vec_size; j++) {
            local_buffer[j] = 0;
        }

        return num_params;
    }

#pragma unroll
    for (int j = 0; j < vec_size; ) {
        if (idx + j >= total_length) {
            local_buffer[j] = 0; // Handle out-of-bounds by setting to zero or another appropriate value
            j++;
            continue;
        }

        size_t start_idx = param_idx==0 ? 0 : param_sizes[param_idx-1];
        size_t end_idx = param_sizes[param_idx];
        assert(("load_to_local failed, idx + j < start_idx", start_idx <= idx + j));
        for (; param_idx < num_params; ) {

            if (idx + j < end_idx) {
                if (idx + vec_size - 1 < end_idx) {
                    // IF [idx+j, idx+vec_size) is contiguous, load once is enough
                    // model_params[param_idx][idx + j - start_idx: idx + vec_size] -> local_buffer[j: vec_size]
                    if (vec_size - j >= 8) {
                        mem_access::load_global<8*sizeof(scalar_t)>(
                            local_buffer + j,
                            model_params[param_idx]+idx + j - start_idx);
                        j += 8;
                        break;
                    } else if (vec_size - j >= 4) {
                        mem_access::load_global<4*sizeof(scalar_t)>(
                            local_buffer + j,
                            model_params[param_idx]+idx + j - start_idx);
                        j += 4;
                        break;
                    } else if (vec_size - j >= 2) {
                        mem_access::load_global<2*sizeof(scalar_t)>(
                            local_buffer + j,
                            model_params[param_idx]+idx + j - start_idx);
                        j += 2;
                        break;
                    } else if (vec_size - j >= 1) {
                        mem_access::load_global<1*sizeof(scalar_t)>(
                            local_buffer + j,
                            model_params[param_idx]+idx + j - start_idx);
                        j += 1;
                        break;
                    } else {
                        assert(("load_to_local failed, vec_size - j < 1", false));
                    }
                } else {
                    // IF [idx+j, idx+vec_size) is not contiguous, only load [idx+j, end_idx)
                    if (end_idx - idx - j >= 8) {
                        mem_access::load_global<8*sizeof(scalar_t)>(
                            local_buffer + j,
                            model_params[param_idx]+idx + j - start_idx);
                        j += 8;
                        break;
                    } else if (end_idx - idx - j >= 4) {
                        mem_access::load_global<4*sizeof(scalar_t)>(
                            local_buffer + j,
                            model_params[param_idx]+idx + j - start_idx);
                        j += 4;
                        break;
                    } else if (end_idx - idx - j >= 2) {
                        mem_access::load_global<2*sizeof(scalar_t)>(
                            local_buffer + j,
                            model_params[param_idx]+idx + j - start_idx);
                        j += 2;
                        break;
                    } else if (end_idx - idx - j >= 1) {
                        mem_access::load_global<1*sizeof(scalar_t)>(
                            local_buffer + j,
                            model_params[param_idx]+idx + j - start_idx);
                        j += 1;
                        break;
                    } else {
                        assert(("load_to_local failed, end_idx - idx - j < 1", false));
                    }
                }
            }
            start_idx = end_idx;
            ++param_idx;
            end_idx = param_sizes[param_idx];
        }
        assert(("load_to_local failed, for loop search parameters finished without finding", param_idx<num_params));
    }
    return param_idx;
}

template <typename T, int numBits, dequantize::Type qType, int unroll, int threads>
__global__ void dequantize_kernel(
    T* __restrict__ dequant_data,
    __nv_bfloat16** model_params, // model params are real params for forward computation
    const size_t dp_param_offset,
    const size_t* model_param_size, // model param size are used to store size for all params
    const size_t num_params,
    const size_t total_size,
    const int8_t* __restrict__ q_data,
    const float* __restrict__ global_params,
    int elems_per_group,
    int64_t total_elems)
{
    extern __shared__ size_t shared_mem_address[]; // Shared memory declaration

    // Load model_param_size into shared memory
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    for (int i = tid; i < num_params; i += blockDim.x * blockDim.y) {
        shared_mem_address[i] = model_param_size[i];
    }
    __syncthreads(); // Ensure all threads have loaded the data

    if constexpr (numBits == 4 || numBits == 8) {

        cg::thread_block tb = cg::this_thread_block();
        cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

        // Load constants
        // TODO(cmikeh2): Refactor into functions?
        constexpr int load_granularity = (dequantize::granularity / (sizeof(T))) / (numBits == 8 ? 1 : 2);
        constexpr int load_step_stride = load_granularity * threads;
        constexpr int load_block_stride = load_step_stride * unroll;

        // Store constants
        constexpr int T_per_chunk = dequantize::granularity / sizeof(T);
        constexpr int store_step_stride = T_per_chunk * threads;
        constexpr int store_block_stride = store_step_stride * unroll;

        // Load offsets
        const int64_t load_block_offset = tb.group_index().x * load_block_stride;
        // Note: we can use `load_granularity` since the dtype is `int8_t`.
        const int load_thread_offset = tb.thread_index().x * load_granularity;
        const int8_t* load_base = q_data + load_block_offset + load_thread_offset;

        // Store offsets
        const int64_t store_block_offset = tb.group_index().x * store_block_stride;
        const int store_thread_offset = tb.thread_index().x * T_per_chunk;
        const int64_t elem_id_base = store_block_offset + store_thread_offset;

        int8_t local_load_buffer[load_granularity * unroll];
        T local_dequant_buffer[T_per_chunk * unroll];

        /*
        Note: Splitting this loop in half gave about 3-5% performance increase for reasons that aren't
        totally clear to me, so this is a deliberately weird code structure.
        */
    #pragma unroll
        for (int i = 0; i < unroll; i++) {
            const int64_t elem_id_iter = elem_id_base + i * store_step_stride;

            if (elem_id_iter < total_elems) {
                mem_access::load_global<load_granularity>(local_load_buffer + i * load_granularity,
                                                        load_base + i * load_step_stride);
            }
        }

        size_t param_offset = 0;
    #pragma unroll
        for (int i = 0; i < unroll; i++) {
            const int64_t elem_id_iter = elem_id_base + i * store_step_stride;
            if (elem_id_iter < total_elems) {
                // TODO(cmikeh2): Can we amortize this division? Perform once on the first iteration and
                // use indexing math to do division free interpolation of the successive groups?
                const int group_index = elem_id_iter / elems_per_group;
                dequantize::Params<qType, numBits> q_params(global_params, group_index);

                dequantize::chunk<T, numBits, qType>(local_dequant_buffer + i * T_per_chunk,
                                        local_load_buffer + i * load_granularity,
                                        q_params);

                T temp_param_model[T_per_chunk];
                T* data_cast = local_dequant_buffer + i * T_per_chunk;
                param_offset = load_to_local<T, T_per_chunk>(
                    temp_param_model, 
                    model_params, 
                    shared_mem_address, 
                    num_params, 
                    total_size, 
                    elem_id_iter + dp_param_offset, 
                    param_offset);

                for (int k = 0; k < T_per_chunk; k++) {
                    data_cast[k] = __hadd(data_cast[k], temp_param_model[k]);
                }
                mem_access::store_global<dequantize::granularity>(dequant_data + elem_id_iter,
                                                    local_dequant_buffer + i * T_per_chunk);
            }
        }
    } else if constexpr (numBits == 3) {
        // TODO(cmikeh2): Need this implementation
        assert(false);
    } else {
        assert(false);
    }
}

#define LAUNCH_DEQUANT_ADD_KERNEL(num_bits, q_type)                                          \
    dequantize_kernel<T, num_bits, q_type, unroll, threads><<<grid, block, shared_mem_size, stream>>>( \
        dequant_data, model_params, dp_param_offset, model_param_size, num_params, total_size, q_data, global_params, elems_per_group, total_elems);

template <typename T>
void launch_dequantize_kernel(
    T* dequant_data,
    T** model_params, // model params are real params for forward computation
    const size_t dp_param_offset,
    const size_t* model_param_size, // model param size are used to store size for all params
    const size_t num_params,
    const size_t total_size,
    const int8_t* q_data,
    const float* global_params,
    quantize::Type q_type,
    int num_bits,
    int elems_per_group,
    int64_t total_elems,
    cudaStream_t stream)
{
    constexpr int unroll = 8;
    constexpr int threads = 512;
    constexpr int elems_per_block = unroll * threads * dequantize::granularity / (sizeof(T));

    const dim3 block(threads);
    const dim3 grid((total_elems + elems_per_block - 1) / elems_per_block);

    size_t shared_mem_size = (num_params) * sizeof(size_t);

    // TODO(cmikeh2): It may make sense to tune unroll, there is perf benefit for large
    // problem sizes with this large unroll value.
    if (num_bits == 8 && q_type == quantize::Type::Symmetric) {
        LAUNCH_DEQUANT_ADD_KERNEL(8, quantize::Type::Symmetric);
    } else if (num_bits == 8 && q_type == quantize::Type::Asymmetric) {
        LAUNCH_DEQUANT_ADD_KERNEL(8, quantize::Type::Asymmetric);
    } else if (num_bits == 4 && q_type == quantize::Type::Symmetric) {
        LAUNCH_DEQUANT_ADD_KERNEL(4, quantize::Type::Symmetric);
    } else if (num_bits == 4 && q_type == quantize::Type::Asymmetric) {
        LAUNCH_DEQUANT_ADD_KERNEL(4, quantize::Type::Asymmetric);
    }
}

template <typename T>
at::Tensor fused_dequantize_add_cuda(
    at::Tensor& quantized_data,
    at::Tensor& params,
    at::Tensor& output_buffer,
    std::vector<at::Tensor> param_list,
    size_t dp_param_offset,
    int groups,
    int num_bits,
    quantize::Type quant_type) {

    const long int expected_buffer_elements = quantized_data.numel() * 8 / num_bits;
    if (std::is_same<T, float>::value) {
        TORCH_CHECK(output_buffer.scalar_type() == torch::kFloat32, "Output buffer must have same type with template");
    } else if (std::is_same<T, __half>::value){
        TORCH_CHECK(output_buffer.scalar_type() == torch::kFloat16, "Output buffer must have same type with template");
    } else if (std::is_same<T, __nv_bfloat16>::value){
        TORCH_CHECK(output_buffer.scalar_type() == torch::kBFloat16, "Output buffer must have same type with template");
    }
    TORCH_CHECK(output_buffer.numel() == expected_buffer_elements, "Output buffer does not have the correct number of elements.");

    const int64_t total_elems = at::numel(output_buffer);
    const int elems_per_group = total_elems / groups;

  // Calculate total size of the output tensor
    size_t total_size = 0;
    size_t num_params = param_list.size();
    std::vector<size_t> model_param_size(num_params);
    for (size_t i = 0; i < num_params; ++i) {
        total_size += param_list[i].size(0);
        model_param_size[i] = total_size;
    }
    // printf("total size: %ld, param buffer size: %ld, dp_param_offset: %ld\n", total_size, param_buffer_size, dp_param_offset);
    total_size = min(total_size, output_buffer.numel()+dp_param_offset);

    // Copy params ptr
    std::vector<void*> model_params(num_params);
    for (size_t i = 0; i < num_params; ++i) {
        model_params[i] = param_list[i].data_ptr();
    }

    // Allocate device memory for input pointers
    T** d_model_params;
    cudaMalloc(&d_model_params, param_list.size() * sizeof(T*));
    cudaMemcpy(d_model_params, model_params.data(), param_list.size() * sizeof(T*), cudaMemcpyHostToDevice);

    // Allocate device memory for input sizes
    size_t* d_model_param_size;
    cudaMalloc(&d_model_param_size, (num_params) * sizeof(size_t));
    cudaMemcpy(d_model_param_size, model_param_size.data(), num_params * sizeof(size_t), cudaMemcpyHostToDevice);

    // d_model_params, dp_param_offset, d_model_param_size, num_params, total_size,
    launch_dequantize_kernel(
        (T*)output_buffer.data_ptr(),
        d_model_params,
        dp_param_offset,
        d_model_param_size,
        num_params,
        total_size,
        (const int8_t*)quantized_data.data_ptr(),
        (const float*)params.data_ptr(),
        quant_type,
        num_bits,
        elems_per_group,
        total_elems,
        at::cuda::getCurrentCUDAStream());

    return output_buffer;

}

#ifdef BF16_AVAILABLE
template at::Tensor fused_dequantize_add_cuda<__nv_bfloat16>(
    at::Tensor& quantized_data,
    at::Tensor& params,
    at::Tensor& output_buffer,
    std::vector<at::Tensor> param_list,
    size_t dp_param_offset,
    int groups,
    int num_bits,
    quantize::Type quant_type);
#endif