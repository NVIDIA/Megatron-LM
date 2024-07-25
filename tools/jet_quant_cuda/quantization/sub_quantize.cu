// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

#include "ds_kernel_utils.h"
#include "memory_access_utils.h"
#include "quantization.h"
#include "quantization_utils.h"
#include "reduction_utils.h"
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

    if (param_idx == num_params) {

#pragma unroll
        for (int j = 0; j < vec_size; ) {
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
                    }
                }
            }
            start_idx = end_idx;
            ++param_idx;
            end_idx = param_sizes[param_idx];
        }
    }
    return param_idx;
}

#ifdef BF16_AVAILABLE
template <
    int q_bits,
    quantize::Type quant_type,
    int UNROLL,
    int internal_unroll,
    int threads_per_group,
    int max_threads>
__global__ void cached_quantization(
    int8_t* __restrict__ output_data,
    float* __restrict__ params,
    const __nv_bfloat16* __restrict__ param_buffer, // Updated to bfloat16
    __nv_bfloat16** model_params, // model params are real params for forward computation
    const size_t dp_param_offset,
    const size_t* model_param_size, // model param size are used to store size for all params
    const size_t num_params,
    const size_t total_size,
    int groups,
    int elems_per_group)
{
    extern __shared__ size_t shared_mem_address[]; // Shared memory declaration

    // Load model_param_size into shared memory
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    for (int i = tid; i < num_params; i += blockDim.x * blockDim.y) {
        shared_mem_address[i] = model_param_size[i];
    }
    __syncthreads(); // Ensure all threads have loaded the data

    size_t* shared_model_param_size = shared_mem_address;

    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    // Indexing offsets
    const int64_t block_offset =
        (static_cast<int64_t>(tb.group_index().x) * (max_threads / threads_per_group) * elems_per_group) +
        (tb.thread_index().y * elems_per_group);
    const int elem_offset = tb.thread_index().x * quantize::bf_per_load;
    const int64_t base_offset = block_offset + elem_offset;
    const int stride = tb.size() * quantize::bf_per_load;

    const __nv_bfloat16* input_base = param_buffer + base_offset;

    __nv_bfloat162 local_buffer[UNROLL * internal_unroll * quantize::bf2_per_load]; // Updated buffer type
    // size_t param_offset = d_block_start_param_offset[tb.group_index().x];
    size_t param_offset = 0;

#pragma unroll
    for (int i = 0; i < UNROLL; i++) {
        __nv_bfloat162* iteration_buffer = local_buffer + i * internal_unroll * quantize::bf2_per_load; // Updated pointer type
#pragma unroll
        for (int j = 0; j < internal_unroll; j++) {
            const int iteration = i * internal_unroll + j;
            __nv_bfloat16* data_cast = reinterpret_cast<__nv_bfloat16*>(iteration_buffer + j * quantize::bf2_per_load);
            __nv_bfloat16 temp_param_model[quantize::bf_per_load];
            mem_access::load_global<quantize::granularity>(
                iteration_buffer + j * quantize::bf2_per_load,
                input_base + iteration * stride,
                elem_offset + iteration * stride < elems_per_group);
            param_offset = load_to_local<__nv_bfloat16, quantize::bf_per_load>(temp_param_model, model_params, shared_model_param_size, num_params, total_size, base_offset + iteration * stride + dp_param_offset, param_offset);
#pragma unroll
            for (int k = 0; k < quantize::bf_per_load; k++) {
                data_cast[k] = (elem_offset + iteration * stride + k < elems_per_group) ? __hsub(data_cast[k], temp_param_model[k]) : __float2bfloat16(0.0f);
            }
        }
    }

    quantize::
        local_array<quant_type, q_bits, UNROLL * internal_unroll, threads_per_group, max_threads>(
            local_buffer, params, output_data, elems_per_group, groups);
}
#endif

/********* Launcher methods ***********/
#define LAUNCH_CACHED_QUANT_CALL(q_bits, quant_type) \
    cached_quantization<q_bits,                      \
                        quant_type,                  \
                        unroll_factor,               \
                        internal_unroll_l,           \
                        threads_per_group,           \
                        max_threads>                 \
        <<<grid, block, shared_mem_size, stream>>>(output_data, d_params, d_param_buffer, d_model_params, dp_param_offset, d_model_param_size, num_params, total_size, groups, elems_per_group);

#define LAUNCH_CACHED_QUANT(                                                        \
    q_bits, quant_type, unroll_factor_in, internal_unroll_in, threads_per_group_in) \
    const int unroll_factor = unroll_factor_in;                                     \
    const int internal_unroll_l = internal_unroll_in;                               \
    const int threads_per_group = threads_per_group_in;                             \
    if (q_bits == 4) {                                                              \
        if (quant_type == quantize::Type::Asymmetric) {                             \
            LAUNCH_CACHED_QUANT_CALL(4, quantize::Type::Asymmetric)                 \
        } else {                                                                    \
            LAUNCH_CACHED_QUANT_CALL(4, quantize::Type::Symmetric)                  \
        }                                                                           \
    } else {                                                                        \
        if (quant_type == quantize::Type::Asymmetric) {                             \
            LAUNCH_CACHED_QUANT_CALL(8, quantize::Type::Asymmetric)                 \
        } else {                                                                    \
            LAUNCH_CACHED_QUANT_CALL(8, quantize::Type::Symmetric)                  \
        }                                                                           \
    }

#ifdef BF16_AVAILABLE
void launch_sub_quant(
    int8_t* output_data,
    float* d_params,
    const __nv_bfloat16* d_param_buffer,  // param buffer are contiguous buffer place for all gather params
    const size_t param_buffer_size,
    std::vector<at::Tensor> param_list,
    const size_t dp_param_offset,
    const int groups,
    const int elems_per_group,
    const int num_bits,
    const quantize::Type quant_type,
    cudaStream_t stream)
{
    constexpr int max_threads = 256;
    constexpr int internal_unroll = 2;
    const bool is_subblock_schedule = (elems_per_group <= 128) ? true : false;
    const int bf_per_step = is_subblock_schedule ? quantize::bf_per_load
                                                : quantize::bf_per_load * internal_unroll;

    const int one_step_threads = next_pow2((elems_per_group + bf_per_step - 1) / bf_per_step);
    const int threads_per_group = (one_step_threads < max_threads) ? one_step_threads : max_threads;
    const int groups_per_block = is_subblock_schedule ? (max_threads + threads_per_group - 1) / threads_per_group : 1;
    const int groups_launch = (groups_per_block + groups - 1) / groups_per_block;

    dim3 block(threads_per_group, groups_per_block);
    dim3 grid(groups_launch);
    const int elems_per_step = threads_per_group * bf_per_step;
    const int external_unroll = (elems_per_group + elems_per_step - 1) / elems_per_step;

    // Calculate total size of the output tensor
    size_t total_size = 0;
    size_t num_params = param_list.size();
    std::vector<size_t> model_param_size(num_params);
    for (size_t i = 0; i < num_params; ++i) {
        total_size += param_list[i].size(0);
        model_param_size[i] = total_size;
    }
    // printf("total size: %ld, param buffer size: %ld, dp_param_offset: %ld\n", total_size, param_buffer_size, dp_param_offset);
    total_size = min(total_size, param_buffer_size+dp_param_offset);

    // Copy params ptr
    std::vector<void*> model_params(num_params);
    for (size_t i = 0; i < num_params; ++i) {
        model_params[i] = param_list[i].data_ptr();
    }

    // Allocate device memory for input pointers
    __nv_bfloat16** d_model_params;
    cudaMalloc(&d_model_params, param_list.size() * sizeof(__nv_bfloat16*));
    cudaMemcpy(d_model_params, model_params.data(), param_list.size() * sizeof(__nv_bfloat16*), cudaMemcpyHostToDevice);

    // Allocate device memory for input sizes
    size_t* d_model_param_size;
    cudaMalloc(&d_model_param_size, (num_params) * sizeof(size_t));
    cudaMemcpy(d_model_param_size, model_param_size.data(), num_params * sizeof(size_t), cudaMemcpyHostToDevice);


    size_t shared_mem_size = (num_params) * sizeof(size_t);

    if (is_subblock_schedule) {
        if (threads_per_group == 1) {
            LAUNCH_CACHED_QUANT(num_bits, quant_type, 1, 1, 1);
        } else if (threads_per_group == 2) {
            LAUNCH_CACHED_QUANT(num_bits, quant_type, 1, 1, 2);
        } else if (threads_per_group == 4) {
            LAUNCH_CACHED_QUANT(num_bits, quant_type, 1, 1, 4);
        } else if (threads_per_group == 8) {
            LAUNCH_CACHED_QUANT(num_bits, quant_type, 1, 1, 8);
        } else if (threads_per_group == 16) {
            LAUNCH_CACHED_QUANT(num_bits, quant_type, 1, 1, 16);
        }
    } else if (external_unroll == 1) {
        LAUNCH_CACHED_QUANT(num_bits, quant_type, 1, internal_unroll, max_threads);
    } else if (external_unroll == 2) {
        LAUNCH_CACHED_QUANT(num_bits, quant_type, 2, internal_unroll, max_threads);
    } else if (external_unroll == 3) {
        LAUNCH_CACHED_QUANT(num_bits, quant_type, 3, internal_unroll, max_threads);
    } else if (external_unroll == 4) {
        LAUNCH_CACHED_QUANT(num_bits, quant_type, 4, internal_unroll, max_threads);
    }
}
#endif

std::vector<at::Tensor> sub_quantize_cuda(
    at::Tensor& input_vals,
    std::vector<at::Tensor> param_list,
    size_t dp_param_offset,
    int groups,
    int numBits,
    quantize::Type quantType) {



    auto dtype = at::kFloat;
    auto params_options = at::TensorOptions()
                              .dtype(dtype)
                              .layout(at::kStrided)
                              .device(at::kCUDA)
                              .requires_grad(false);
    const int param_elems = (quantize::requires_offset(quantType)) ? 2 : 1;
    auto params = torch::empty({groups, param_elems}, params_options);

    auto output_options = at::TensorOptions()
                              .dtype(at::kChar)
                              .layout(at::kStrided)
                              .device(at::kCUDA)
                              .requires_grad(false);

    auto output_sizes = input_vals.sizes().vec();
    output_sizes[output_sizes.size() - 1] /= numBits == 8 ? 1 : 2;
    auto output = torch::empty(output_sizes, output_options);

    const int elems_per_group = at::numel(input_vals) / groups;

    if (input_vals.scalar_type() == at::ScalarType::Half) {
        // launch_quant((int8_t*)output.data_ptr(),
        //             (float*)params.data_ptr(),
        //             (__half*)input_vals.data_ptr(),
        //             groups,
        //             elems_per_group,
        //             numBits,
        //             quantType,
        //             at::cuda::getCurrentCUDAStream());
        // return {output, params};
        throw std::runtime_error("Unsupported input tensor data type.");
    } else if (input_vals.scalar_type() == at::ScalarType::Float) {
        // launch_quant((int8_t*)output.data_ptr(),
        //             (float*)params.data_ptr(),
        //             (float*)input_vals.data_ptr(),
        //             groups,
        //             elems_per_group,
        //             numBits,
        //             quantType,
        //             at::cuda::getCurrentCUDAStream());
        // return {output, params};
        throw std::runtime_error("Unsupported input tensor data type.");
    } else if (input_vals.scalar_type() == at::ScalarType::BFloat16) {
        launch_sub_quant(
            (int8_t*)output.data_ptr(),
            (float*)params.data_ptr(),
            (__nv_bfloat16*)input_vals.data_ptr(),
            at::numel(input_vals),
            param_list,
            dp_param_offset,
            groups,
            elems_per_group,
            numBits,
            quantType,
            at::cuda::getCurrentCUDAStream());
        return {output, params};
    } else {
        throw std::runtime_error("Unsupported input tensor data type.");
    }
}