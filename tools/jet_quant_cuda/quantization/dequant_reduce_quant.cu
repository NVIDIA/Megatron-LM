#include <cstdio>
#include "dequantization_utils.h"
#include "ds_kernel_utils.h"
#include "memory_access_utils.h"
#include "quantization_utils.h"
#include "reduction_utils.h"

using rop = reduce::ROpType;

template <int inNumBits, int outNumBits, int numTensors, int totalChunks, quantize::Type quantType>
__global__ void __launch_bounds__(1024) dequant_reduce_quant(int8_t* reduced_data,
                                                       float* reduced_scales,
                                                       const int8_t* input_data,
                                                       const float* input_scales,
                                                       int elems_per_out_group,
                                                       int elems_per_in_tensor,
                                                       int groups_per_in_tensor,
                                                       int elems_per_in_group,
                                                       int num_tensors)
{
    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    // NOTE(cmikeh2): This probably could be hardcoded to a larger number,
    // but that means even stronger restrictions on the number of elements per group
    // A performance analysis here might be beneficial
    constexpr int mem_granularity = (inNumBits == 8) ? 4 : 2; // mem_granularity = 16 / sizeof(float) / (8 / numBits)
    constexpr int elems_per_load = mem_granularity / sizeof(int8_t);  // div by 1
    constexpr int storage_values = 16 / sizeof(float);

    const int block_offset = tb.group_index().x * elems_per_in_group;
    const int elem_offset = tb.thread_index().x * elems_per_load;
    const int base_offset = block_offset + elem_offset;
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

        const int iter_offset = i * stride + base_offset;
        const int iter_scale_idx = iter_offset / elems_per_in_group;
        bool do_loads = i * stride + elem_offset < elems_per_in_group;

        if (numTensors > 0) {
#pragma unroll
            for (int j = 0; j < numTensors; j++) {
                if (do_loads) {
                    int8_t load_buffer[elems_per_load];

                    mem_access::load_global<mem_granularity>(
                        load_buffer, input_data + j * elems_per_in_tensor + iter_offset);

                    quantize::Params<quantType, inNumBits> params(
                        input_scales + j * groups_per_in_tensor, iter_scale_idx);

                    float dequant_buffer[storage_values];
                    dequantize::chunk<inNumBits, quantType>(dequant_buffer, load_buffer, params);

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

                    quantize::Params<quantType, inNumBits> params(
                        input_scales + j * groups_per_in_tensor, iter_scale_idx);

                    float dequant_buffer[storage_values];
                    dequantize::chunk<inNumBits, quantType>(dequant_buffer, load_buffer, params);

#pragma unroll
                    for (int k = 0; k < storage_values; k++) {
                        iteration_buffer[k] =
                            reduce::element<rop::Add>(iteration_buffer[k], dequant_buffer[k]);
                    }
                }
            }
        }

#pragma unroll
        for (int j = 0; j < storage_values; j++) { stats.update(iteration_buffer[j]); }
    }

    auto params = stats.template get_params<outNumBits, 1024>(tb, warp);

    if (tb.thread_index().x == 0) { params.store(reduced_scales, tb.group_index().x); }

    constexpr int out_mem_granularity = (outNumBits == 8) ? 4 : 2;
    constexpr int elems_per_out = out_mem_granularity / sizeof(int8_t);  // div by 1
    const int block_offset_out = tb.group_index().x * elems_per_out_group;
    const int elem_offset_out = tb.thread_index().x * elems_per_out;
    const int base_offset_out = block_offset_out + elem_offset_out;
    const int stride_out = tb.group_dim().x * elems_per_out;

#pragma unroll
    for (int i = 0; i < totalChunks; i++) {
        const int iter_offset = i * stride_out + base_offset_out;
        if (i * stride_out + elem_offset_out < elems_per_out_group) {
            int8_t local_output[elems_per_out];
            quantize::_chunk<outNumBits, quantType>(
                local_output, local_buffer + i * storage_values, params);
            mem_access::store_global<out_mem_granularity>(reduced_data + iter_offset, local_output);
        }
    }
}

template <int Power>
int32_t pow2_round(int32_t raw_value)
{
    return (((raw_value - 1) >> Power) + 1) << Power;
}

#define LAUNCH_DEQUANT_REDUCE_QUANT(num_chunks)                      \
    dequant_reduce_quant<inNumBits, outNumBits, numTensors, num_chunks, quantType> \
        <<<grid, block, 0, stream>>>(reduced_data,             \
                                     reduced_scales,           \
                                     input_data,               \
                                     input_scales,             \
                                     elems_per_out_group,      \
                                     elems_per_in_tensor,      \
                                     groups_per_in_tensor,     \
                                     elems_per_in_group,       \
                                     num_tensors);

template <int inNumBits, int outNumBits, int numTensors, quantize::Type quantType>
void launch_dequant_reduce_quant_impl(int8_t* reduced_data,
                                float* reduced_scales,
                                const int8_t* input_data,
                                const float* input_scales,
                                int out_groups,
                                int elems_per_out_group,
                                int elems_per_in_tensor,
                                int groups_per_in_tensor,
                                int elems_per_in_group,
                                int num_tensors,
                                cudaStream_t stream)
{
    // This is a coincidence. This is derived by 8 halves per 16 bytes with 2-way packing for int4
    constexpr int elems_per_thread = inNumBits / 2; // elems_per_thread = 16 / sizeof(float) / (8 / numBits)
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
        LAUNCH_DEQUANT_REDUCE_QUANT(1);
    } else if (unroll == 2) {
        // 4097-8192 etc...
        LAUNCH_DEQUANT_REDUCE_QUANT(2);
    } else if (unroll == 3) {
        LAUNCH_DEQUANT_REDUCE_QUANT(3);
    } else if (unroll == 4) {
        LAUNCH_DEQUANT_REDUCE_QUANT(4);
    } else if (unroll == 6) {
        LAUNCH_DEQUANT_REDUCE_QUANT(6);
    } else if (unroll == 8) {
        LAUNCH_DEQUANT_REDUCE_QUANT(8);
    } else if (unroll == 10) {
        LAUNCH_DEQUANT_REDUCE_QUANT(10);
    } else if (unroll == 12) {
        // 48k limit
        LAUNCH_DEQUANT_REDUCE_QUANT(12);
    } else {
        assert(false);
    }
}

#define LAUNCH_DEQUANT_REDUCE_QUANT_IMPL(IN_NUM_BITS, OUT_NUM_BITS, NUM_GPUS, QUANT_TYPE)                   \
    launch_dequant_reduce_quant_impl<IN_NUM_BITS, OUT_NUM_BITS, NUM_GPUS, QUANT_TYPE>(reduced_data,         \
                                                               reduced_scales,       \
                                                               input_data,           \
                                                               input_scales,         \
                                                               out_groups,           \
                                                               elems_per_out_group,  \
                                                               elems_per_in_tensor,  \
                                                               groups_per_in_tensor, \
                                                               elems_per_in_group,   \
                                                               num_gpus,             \
                                                               stream);

void launch_dequant_reduce_quant(int8_t* reduced_data,
                           float* reduced_scales,
                           const int8_t* input_data,
                           const float* input_scales,
                           int num_gpus,
                           int in_num_bits,
                           int out_num_bits,
                           quantize::Type quant_type,
                           int out_groups,
                           int elems_per_out_group,
                           int elems_per_in_tensor,
                           int groups_per_in_tensor,
                           int elems_per_in_group,
                           cudaStream_t stream)
{
    if (quant_type == quantize::Type::Symmetric) {
        if (in_num_bits == 4) {
            if (out_num_bits == 8){
                if (num_gpus == 1) {
                    LAUNCH_DEQUANT_REDUCE_QUANT_IMPL(4, 8, 1, quantize::Type::Symmetric);
                } else if (num_gpus == 2) {
                    LAUNCH_DEQUANT_REDUCE_QUANT_IMPL(4, 8, 2, quantize::Type::Symmetric);
                } else if (num_gpus == 4) {
                    LAUNCH_DEQUANT_REDUCE_QUANT_IMPL(4, 8, 4, quantize::Type::Symmetric);
                } else if (num_gpus == 8) {
                    LAUNCH_DEQUANT_REDUCE_QUANT_IMPL(4, 8, 8, quantize::Type::Symmetric);
                } else if (num_gpus == 16) {
                    LAUNCH_DEQUANT_REDUCE_QUANT_IMPL(4, 8, 16, quantize::Type::Symmetric);
                } else {
                    LAUNCH_DEQUANT_REDUCE_QUANT_IMPL(4, 8, -1, quantize::Type::Symmetric);
                }
            } else if (out_num_bits == 4) {
                if (num_gpus == 1) {
                    LAUNCH_DEQUANT_REDUCE_QUANT_IMPL(4, 4, 1, quantize::Type::Symmetric);
                } else if (num_gpus == 2) {
                    LAUNCH_DEQUANT_REDUCE_QUANT_IMPL(4, 4, 2, quantize::Type::Symmetric);
                } else if (num_gpus == 4) {
                    LAUNCH_DEQUANT_REDUCE_QUANT_IMPL(4, 4, 4, quantize::Type::Symmetric);
                } else if (num_gpus == 8) {
                    LAUNCH_DEQUANT_REDUCE_QUANT_IMPL(4, 4, 8, quantize::Type::Symmetric);
                } else if (num_gpus == 16) {
                    LAUNCH_DEQUANT_REDUCE_QUANT_IMPL(4, 4, 16, quantize::Type::Symmetric);
                } else {
                    LAUNCH_DEQUANT_REDUCE_QUANT_IMPL(4, 4, -1, quantize::Type::Symmetric);
                }
            }
        } else if (in_num_bits == 8) {
            if (out_num_bits == 8){
                if (num_gpus == 1) {
                    LAUNCH_DEQUANT_REDUCE_QUANT_IMPL(8, 8, 1, quantize::Type::Symmetric);
                } else if (num_gpus == 2) {
                    LAUNCH_DEQUANT_REDUCE_QUANT_IMPL(8, 8, 2, quantize::Type::Symmetric);
                } else if (num_gpus == 4) {
                    LAUNCH_DEQUANT_REDUCE_QUANT_IMPL(8, 8, 4, quantize::Type::Symmetric);
                } else if (num_gpus == 8) {
                    LAUNCH_DEQUANT_REDUCE_QUANT_IMPL(8, 8, 8, quantize::Type::Symmetric);
                } else if (num_gpus == 16) {
                    LAUNCH_DEQUANT_REDUCE_QUANT_IMPL(8, 8, 16, quantize::Type::Symmetric);
                } else {
                    LAUNCH_DEQUANT_REDUCE_QUANT_IMPL(8, 8, -1, quantize::Type::Symmetric);
                }
            } else if (out_num_bits == 4) {
                if (num_gpus == 1) {
                    LAUNCH_DEQUANT_REDUCE_QUANT_IMPL(8, 4, 1, quantize::Type::Symmetric);
                } else if (num_gpus == 2) {
                    LAUNCH_DEQUANT_REDUCE_QUANT_IMPL(8, 4, 2, quantize::Type::Symmetric);
                } else if (num_gpus == 4) {
                    LAUNCH_DEQUANT_REDUCE_QUANT_IMPL(8, 4, 4, quantize::Type::Symmetric);
                } else if (num_gpus == 8) {
                    LAUNCH_DEQUANT_REDUCE_QUANT_IMPL(8, 4, 8, quantize::Type::Symmetric);
                } else if (num_gpus == 16) {
                    LAUNCH_DEQUANT_REDUCE_QUANT_IMPL(8, 4, 16, quantize::Type::Symmetric);
                } else {
                    LAUNCH_DEQUANT_REDUCE_QUANT_IMPL(8, 4, -1, quantize::Type::Symmetric);
                }
            }


        }
    } else if (quant_type == quantize::Type::Asymmetric) {
        assert(false);
    }
}