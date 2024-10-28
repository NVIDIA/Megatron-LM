// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

// This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”). 
// All Bytedance's Modifications are Copyright 2024 Bytedance Ltd. and/or its affiliates. 

#include <cstdio>
#include "dequantization_utils.h"
#include "ds_kernel_utils.h"
#include "memory_access_utils.h"
#include "quantization_utils.h"
#include "reduction_utils.h"
#include "fast_hadamard_transform.h"
#include "fast_hadamard_transform_common.h"
using rop = reduce::ROpType;

/*
TODO(cmikeh2): Add implementation that better handles larger nodes. It would like make sense
to leverage some parallel reductions here to improve performance.
*/

template <int numBits, int numTensors, int totalChunks, quantize::Type quantType>
__global__ void __launch_bounds__(1024) dequant_reduce_ht(float* reduced_data,
                                                       const int8_t* input_data,
                                                       const float* input_scales,
                                                       int elems_per_out_group,
                                                       int64_t elems_per_in_tensor,
                                                       int groups_per_in_tensor,
                                                       int elems_per_in_group,
                                                       int num_tensors)
{
    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    // NOTE(cmikeh2): This probably could be hardcoded to a larger number,
    // but that means even stronger restrictions on the number of elements per group
    // A performance analysis here might be beneficial
    constexpr int mem_granularity = (numBits == 8) ? 4 : 2; // mem_granularity = 16 / sizeof(float) / (8 / numBits)
    constexpr int elems_per_load = mem_granularity / sizeof(int8_t);  // div by 1
    constexpr int storage_values = 16 / sizeof(float); //for each thread, each chunk operate on 16bytes, 4 float32 elements. 
                                                        //If you want to change 16, you should also change totalChunks

    const int64_t block_offset = tb.group_index().x * elems_per_in_group;
    const int elem_offset = tb.thread_index().x * elems_per_load;
    const int64_t base_offset = block_offset + elem_offset;
    const int stride = tb.group_dim().x * elems_per_load;

    float local_buffer[totalChunks * storage_values];

    quantize::GroupStats<quantType, float> stats;

#pragma unroll
    for (int i = 0; i < totalChunks; i++) {
        float* iteration_buffer = local_buffer + i * storage_values;

#pragma unroll
        for (int j = 0; j < storage_values; j++) {
            iteration_buffer[j] = reduce::init<rop::Add, float>();
        }

        const int64_t iter_offset = i * stride + base_offset;
        const int iter_scale_idx = iter_offset / elems_per_in_group;
        bool do_loads = i * stride + elem_offset < elems_per_in_group;

        if (numTensors > 0) {
#pragma unroll
            for (int j = 0; j < numTensors; j++) {
                if (do_loads) {
                    int8_t load_buffer[elems_per_load];

                    mem_access::load_global<mem_granularity>(
                        load_buffer, input_data + j * elems_per_in_tensor + iter_offset);

                    quantize::Params<quantType, numBits> params(
                        input_scales + j * groups_per_in_tensor, iter_scale_idx);

                    float dequant_buffer[storage_values];
                    dequantize::chunk<numBits, quantType>(dequant_buffer, load_buffer, params);

#pragma unroll
                    for (int k = 0; k < storage_values; k++) {
                        iteration_buffer[k] =
                            reduce::element<rop::Add>(iteration_buffer[k], dequant_buffer[k]);
                    }
                }
            }
        } else {
#pragma unroll 4
            for (int j = 0; j < num_tensors; j++) {
                if (do_loads) {
                    int8_t load_buffer[elems_per_load];

                    mem_access::load_global<mem_granularity>(
                        load_buffer, input_data + j * elems_per_in_tensor + iter_offset);

                    quantize::Params<quantType, numBits> params(
                        input_scales + j * groups_per_in_tensor, iter_scale_idx);

                    float dequant_buffer[storage_values];
                    dequantize::chunk<numBits, quantType>(dequant_buffer, load_buffer, params);

#pragma unroll
                    for (int k = 0; k < storage_values; k++) {
                        iteration_buffer[k] =
                            reduce::element<rop::Add>(iteration_buffer[k], dequant_buffer[k]);
                    }
                }
            }
        }

    }

    // start fixed 32*32 Hadamard Transform, accroding https://github.com/Dao-AILab/fast-hadamard-transform/blob/master/csrc/fast_hadamard_transform_cuda.cu
    constexpr int kNChunks = 1;
    constexpr int kNElts = quantize::f_per_load;
    constexpr int kLogNElts = cilog2(kNElts);
    constexpr int kNWarps = 32 / kNElts;
    constexpr int kLogWarpSize = cilog2(kNWarps);
#pragma unroll
    for (int i = 0; i < totalChunks; i++) {
        float* iteration_buffer = local_buffer + i * storage_values;
        if (i * stride + elem_offset < elems_per_out_group) {
            hadamard_mult_thread_quant<kLogNElts, kNChunks>(iteration_buffer);
            hadamard_mult_warp_quant<kLogWarpSize, 0, kNChunks, kNElts>(iteration_buffer);
        }
    }

#pragma unroll
    for (int i = 0; i < totalChunks; i++) {
        const int64_t iter_offset = (i * stride + base_offset) * (8 / numBits);
        
        float* data = local_buffer + i * storage_values;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            data[j] *= 0.03125; // hadamard transform back scale when order = 32
        }
        if (i * stride + elem_offset < elems_per_in_group) {
            mem_access::store_global<16>(reduced_data + iter_offset, local_buffer + i * storage_values); //for each thread, each chunk operate on 16bytes, 4 float32 elements. 
                                                                                                        //If you want to change 16, you should also change totalChunks
        }
    }
}

template <int Power>
int32_t pow2_round(int32_t raw_value)
{
    return (((raw_value - 1) >> Power) + 1) << Power;
}

#define LAUNCH_DEQUANT_REDUCE(num_chunks)                      \
    dequant_reduce_ht<numBits, numTensors, num_chunks, quantType> \
        <<<grid, block, 0, stream>>>(reduced_data,             \
                                     input_data,               \
                                     input_scales,             \
                                     elems_per_out_group,      \
                                     elems_per_in_tensor,      \
                                     groups_per_in_tensor,     \
                                     elems_per_in_group,       \
                                     num_tensors);

template <int numBits, int numTensors, quantize::Type quantType>
void launch_dequant_reduce_impl_ht(float* reduced_data,
                                const int8_t* input_data,
                                const float* input_scales,
                                int out_groups,
                                int elems_per_out_group,
                                int64_t elems_per_in_tensor,
                                int groups_per_in_tensor,
                                int elems_per_in_group,
                                int num_tensors,
                                cudaStream_t stream)
{
    // This is a coincidence. This is derived by 8 halves per 16 bytes with 2-way packing for int4
    // Note, this should be changed for float
    constexpr int elems_per_thread = numBits / 2; // elems_per_thread = 16 / sizeof(float) / (8 / numBits)
    const int one_step_threads =
        next_pow2((elems_per_in_group + elems_per_thread - 1) / (elems_per_thread));
    // TODO(cmikeh2): Tune this
    const int threads = (one_step_threads < 1024) ? one_step_threads : 1024;

    dim3 block(threads);
    dim3 grid(out_groups);

    const int elems_per_step = threads * elems_per_thread;
    const int unroll_raw = (elems_per_in_group + elems_per_step - 1) / elems_per_step;

    const int unroll = (unroll_raw >= 4) ? pow2_round<1>(unroll_raw) : unroll_raw;

    if (unroll == 1) {
        // 0-4096 elems
        LAUNCH_DEQUANT_REDUCE(1);
    } else if (unroll == 2) {
        // 4097-8192 etc...
        LAUNCH_DEQUANT_REDUCE(2);
    } else if (unroll == 3) {
        LAUNCH_DEQUANT_REDUCE(3);
    } else if (unroll == 4) {
        LAUNCH_DEQUANT_REDUCE(4);
    } else if (unroll == 6) {
        LAUNCH_DEQUANT_REDUCE(6);
    } else if (unroll == 8) {
        LAUNCH_DEQUANT_REDUCE(8);
    } else if (unroll == 10) {
        LAUNCH_DEQUANT_REDUCE(10);
    } else if (unroll == 12) {
        // 48k limit
        LAUNCH_DEQUANT_REDUCE(12);
    } else {
        assert(false);
    }
}

#define LAUNCH_DEQUANT_REDUCE_IMPL(NUM_BITS, NUM_GPUS, QUANT_TYPE)                   \
    launch_dequant_reduce_impl_ht<NUM_BITS, NUM_GPUS, QUANT_TYPE>(reduced_data,         \
                                                               input_data,           \
                                                               input_scales,         \
                                                               out_groups,           \
                                                               elems_per_out_group,  \
                                                               elems_per_in_tensor,  \
                                                               groups_per_in_tensor, \
                                                               elems_per_in_group,   \
                                                               num_gpus,             \
                                                               stream);

void launch_dequant_reduce_ht(float* reduced_data,
                           const int8_t* input_data,
                           const float* input_scales,
                           int num_gpus,
                           int num_bits,
                           quantize::Type quant_type,
                           int out_groups,
                           int elems_per_out_group,
                           int64_t elems_per_in_tensor,
                           int groups_per_in_tensor,
                           int elems_per_in_group,
                           cudaStream_t stream)
{
    if (quant_type == quantize::Type::Symmetric) {
        if (num_bits == 4) { 
            if (num_gpus == 1) {
                LAUNCH_DEQUANT_REDUCE_IMPL(4, 1, quantize::Type::Symmetric);
            } else if (num_gpus == 2) {
                LAUNCH_DEQUANT_REDUCE_IMPL(4, 2, quantize::Type::Symmetric);
            } else if (num_gpus == 4) {
                LAUNCH_DEQUANT_REDUCE_IMPL(4, 4, quantize::Type::Symmetric);
            } else if (num_gpus == 8) {
                LAUNCH_DEQUANT_REDUCE_IMPL(4, 8, quantize::Type::Symmetric);
            } else if (num_gpus == 16) {
                LAUNCH_DEQUANT_REDUCE_IMPL(4, 16, quantize::Type::Symmetric);
            } else {
                LAUNCH_DEQUANT_REDUCE_IMPL(4, -1, quantize::Type::Symmetric);
            }
        } else if (num_bits == 8) {
            if (num_gpus == 1) {
                LAUNCH_DEQUANT_REDUCE_IMPL(8, 1, quantize::Type::Symmetric);
            } else if (num_gpus == 2) {
                LAUNCH_DEQUANT_REDUCE_IMPL(8, 2, quantize::Type::Symmetric);
            } else if (num_gpus == 4) {
                LAUNCH_DEQUANT_REDUCE_IMPL(8, 4, quantize::Type::Symmetric);
            } else if (num_gpus == 8) {
                LAUNCH_DEQUANT_REDUCE_IMPL(8, 8, quantize::Type::Symmetric);
            } else if (num_gpus == 16) {
                LAUNCH_DEQUANT_REDUCE_IMPL(8, 16, quantize::Type::Symmetric);
            } else {
                LAUNCH_DEQUANT_REDUCE_IMPL(8, -1, quantize::Type::Symmetric);
            }
        }
    } else if (quant_type == quantize::Type::Asymmetric) {
        if (num_bits == 4) {
            if (num_gpus == 8) {
                LAUNCH_DEQUANT_REDUCE_IMPL(4, 8, quantize::Type::Asymmetric);
            } else if (num_gpus == 16) {
                LAUNCH_DEQUANT_REDUCE_IMPL(4, 16, quantize::Type::Asymmetric);
            } else {
                LAUNCH_DEQUANT_REDUCE_IMPL(4, -1, quantize::Type::Asymmetric);
            }
        } else if (num_bits == 8) {
            if (num_gpus == 8) {
                LAUNCH_DEQUANT_REDUCE_IMPL(8, 8, quantize::Type::Asymmetric);
            } else if (num_gpus == 16) {
                LAUNCH_DEQUANT_REDUCE_IMPL(8, 16, quantize::Type::Asymmetric);
            } else {
                LAUNCH_DEQUANT_REDUCE_IMPL(8, -1, quantize::Type::Asymmetric);
            }
        }
    }
}
