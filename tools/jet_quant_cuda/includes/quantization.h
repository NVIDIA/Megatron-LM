// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

// This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”). 
// All Bytedance's Modifications are Copyright 2024 Bytedance Ltd. and/or its affiliates. 

#pragma once

#include <cuda_fp16.h>
#include "ds_kernel_utils.h"
#include <vector>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

namespace quantize {

enum class Type { Symmetric, Asymmetric };

struct PackedInt4 {
    int8_t high : 4;
    int8_t low : 4;
};

DS_HD_INLINE bool requires_offset(Type qType) { return qType == Type::Asymmetric; }

}  // namespace quantize

template <typename T>
void launch_dequantize_kernel(T* dequant_data,
                              const int8_t* q_data,
                              const float* q_params,
                              quantize::Type q_type,
                              int num_bits,
                              int elems_per_group,
                              int64_t total_elems,
                              cudaStream_t stream);

template <typename T>
at::Tensor fused_dequantize_add_cuda(
    at::Tensor& quantized_data,
    at::Tensor& params,
    at::Tensor& output_buffer,
    std::vector<at::Tensor> param_list,
    size_t dp_param_offset,
    int groups,
    int num_bits,
    quantize::Type quant_type);

std::vector<at::Tensor> sub_quantize_cuda(
    at::Tensor& input_vals,
    std::vector<at::Tensor> params_list,
    size_t dp_param_offset,
    int groups,
    int numBits,
    quantize::Type quantType);

void launch_quant(int8_t* output_data,
                  float* params,
                  const __half* input_data,
                  const int groups,
                  const int elems_per_group,
                  const int num_bits,
                  const quantize::Type quant_type,
                  cudaStream_t stream);

void launch_quant(int8_t* output_data,
                  float* params,
                  const float* input_data,
                  const int groups,
                  const int elems_per_group,
                  const int num_bits,
                  const quantize::Type quant_type,
                  cudaStream_t stream);

void launch_quant(int8_t* output_data,
                  float* params,
                  const __nv_bfloat16* input_data,
                  const int groups,
                  const int elems_per_group,
                  const int num_bits,
                  const quantize::Type quant_type,
                  cudaStream_t stream);

void launch_quant_ht(int8_t* output_data,
                  float* params,
                  const __half* input_data,
                  const int groups,
                  const int elems_per_group,
                  const int num_bits,
                  const quantize::Type quant_type,
                  cudaStream_t stream);

void launch_quant_ht(int8_t* output_data,
                  float* params,
                  const float* input_data,
                  const int groups,
                  const int elems_per_group,
                  const int num_bits,
                  const quantize::Type quant_type,
                  cudaStream_t stream);

void launch_quant_ht(int8_t* output_data,
                  float* params,
                  const __nv_bfloat16* input_data,
                  const int groups,
                  const int elems_per_group,
                  const int num_bits,
                  const quantize::Type quant_type,
                  cudaStream_t stream);

template <typename T>
void launch_dequantize_kernel(T* dequant_data,
                              const int8_t* q_data,
                              const float* q_params,
                              quantize::Type q_type,
                              int num_bits,
                              int elems_per_group,
                              int total_elems,
                              cudaStream_t stream);

void launch_swizzled_quant(int8_t* q_data,
                           float* q_scales,
                           const __half* input_data,
                           int num_bits,
                           quantize::Type q_type,
                           int groups,
                           int elems_per_group,
                           int pipelining,
                           int nodes,
                           int devices_per_node,
                           cudaStream_t stream);

void launch_swizzled_quant(int8_t* q_data,
                           float* q_scales,
                           const float* input_data,
                           int num_bits,
                           quantize::Type q_type,
                           int groups,
                           int elems_per_group,
                           int pipelining,
                           int nodes,
                           int devices_per_node,
                           cudaStream_t stream);

void launch_swizzled_quant_ht(int8_t* q_data,
                           float* q_scales,
                           const __half* input_data,
                           int num_bits,
                           quantize::Type q_type,
                           int groups,
                           int elems_per_group,
                           int pipelining,
                           int nodes,
                           int devices_per_node,
                           cudaStream_t stream);

void launch_swizzled_quant_ht(int8_t* q_data,
                           float* q_scales,
                           const float* input_data,
                           int num_bits,
                           quantize::Type q_type,
                           int groups,
                           int elems_per_group,
                           int pipelining,
                           int nodes,
                           int devices_per_node,
                           cudaStream_t stream);

void launch_dequant_reduce(float* reduced_data,
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
                           cudaStream_t stream);

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
                           int64_t elems_per_in_tensor,
                           int groups_per_in_tensor,
                           int elems_per_in_group,
                           cudaStream_t stream);

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
                            cudaStream_t stream);

template <typename T>
void launch_fake_quantize_kernel(T* vals,
                                 int total_count,
                                 int group_num,
                                 int num_bits,
                                 cudaStream_t stream);
template <typename T>
void launch_sr_fake_quantize_kernel(T* vals,
                                    int total_count,
                                    int group_num,
                                    int num_bits,
                                    cudaStream_t stream);
template <typename T>
void launch_fake_quantize_kernel_asym(T* vals,
                                      int total_count,
                                      int group_num,
                                      int num_bits,
                                      cudaStream_t stream);
template <typename T>
void launch_sr_fake_quantize_kernel_asym(T* vals,
                                         int total_count,
                                         int group_num,
                                         int num_bits,
                                         cudaStream_t stream);

void launch_dequantize_int4_to_half_experimental(uint8_t* data_in,
                                                 half* data_out,
                                                 half* scale_buffer,
                                                 half* min_val_buffer,
                                                 int num_group,
                                                 int group_size,
                                                 cudaStream_t stream);

void launch_dequantize_int8_to_half_experimental(uint8_t* data_in,
                                                 half* data_out,
                                                 half* scale_buffer,
                                                 half* min_val_buffer,
                                                 int num_group,
                                                 int group_size,
                                                 cudaStream_t stream);
