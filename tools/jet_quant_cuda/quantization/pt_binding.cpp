// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cassert>
#include <vector>
#include "quantization.h"
#include "hadamard_binding.h" 

template <typename T>
at::Tensor dequantize(at::Tensor& quantized_data,
                      at::Tensor& params,
                      at::Tensor& output_buffer,
                      int groups,
                      int num_bits,
                      quantize::Type quant_type)
{
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

    launch_dequantize_kernel((T*)output_buffer.data_ptr(),
                             (const int8_t*)quantized_data.data_ptr(),
                             (const float*)params.data_ptr(),
                             quant_type,
                             num_bits,
                             elems_per_group,
                             total_elems,
                             at::cuda::getCurrentCUDAStream());

    return output_buffer;
}

at::Tensor dequantize_reduce(at::Tensor& input_vals,
                            at::Tensor& input_scales,
                            at::Tensor& output_buffer,
                            int num_groups,
                            int num_bits,
                            quantize::Type quant_type,
                            int chunks)
{

    const int granularity = 8 / num_bits;
    std::vector<long int> original_sz(input_vals.sizes().begin(), input_vals.sizes().end());
    const long int expected_buffer_elements = (original_sz.back() / chunks) * granularity * (input_vals.numel() / original_sz.back());

    TORCH_CHECK(output_buffer.dtype() == at::kFloat, "Output buffer must be of float type.");
    TORCH_CHECK(output_buffer.device().is_cuda(), "Output buffer must be on CUDA device.");
    TORCH_CHECK(output_buffer.is_contiguous(), "Output buffer must be contiguous.");
    TORCH_CHECK(output_buffer.numel() == expected_buffer_elements, "Output buffer does not have the correct number of elements.");

    const int64_t elems_per_in_tensor = at::numel(input_vals) / chunks;
    const int out_groups = num_groups / chunks;
    const int elems_per_in_group = elems_per_in_tensor / (num_groups / chunks); // number of bytes per in group
    const int elems_per_out_group = elems_per_in_group * (8 / num_bits) * sizeof(float); // number of bytes per out group

    launch_dequant_reduce((float*)output_buffer.data_ptr(),
                          (const int8_t*)input_vals.data_ptr(),
                          (const float*)input_scales.data_ptr(),
                          chunks,
                          num_bits,
                          quant_type,
                          out_groups,
                          elems_per_out_group,
                          elems_per_in_tensor,
                          num_groups / chunks,
                          elems_per_in_group,
                          at::cuda::getCurrentCUDAStream());
    return output_buffer;
}

at::Tensor dequantize_reduce_ht(at::Tensor& input_vals,
                            at::Tensor& input_scales,
                            at::Tensor& output_buffer,
                            int num_groups,
                            int num_bits,
                            quantize::Type quant_type,
                            int chunks)
{

    const int granularity = 8 / num_bits;
    std::vector<long int> original_sz(input_vals.sizes().begin(), input_vals.sizes().end());
    const long int expected_buffer_elements = (original_sz.back() / chunks) * granularity * (input_vals.numel() / original_sz.back());

    TORCH_CHECK(output_buffer.dtype() == at::kFloat, "Output buffer must be of float type.");
    TORCH_CHECK(output_buffer.device().is_cuda(), "Output buffer must be on CUDA device.");
    TORCH_CHECK(output_buffer.is_contiguous(), "Output buffer must be contiguous.");
    TORCH_CHECK(output_buffer.numel() == expected_buffer_elements, "Output buffer does not have the correct number of elements.");

    const int64_t elems_per_in_tensor = at::numel(input_vals) / chunks;
    const int out_groups = num_groups / chunks;
    const int elems_per_in_group = elems_per_in_tensor / (num_groups / chunks); // number of bytes per in group
    const int elems_per_out_group = elems_per_in_group * (8 / num_bits) * sizeof(float); // number of bytes per out group

    launch_dequant_reduce_ht((float*)output_buffer.data_ptr(),
                          (const int8_t*)input_vals.data_ptr(),
                          (const float*)input_scales.data_ptr(),
                          chunks,
                          num_bits,
                          quant_type,
                          out_groups,
                          elems_per_out_group,
                          elems_per_in_tensor,
                          num_groups / chunks,
                          elems_per_in_group,
                          at::cuda::getCurrentCUDAStream());
    return output_buffer;
}

std::vector<at::Tensor> ds_swizzle_quant(at::Tensor& input_vals,
                                         int groups,
                                         int num_bits,
                                         quantize::Type quant_type,
                                         int pipeline_size,
                                         int nodes,
                                         int devices_per_node)
{
    auto scales_options = at::TensorOptions()
                              .dtype(at::kFloat)
                              .layout(at::kStrided)
                              .device(at::kCUDA)
                              .requires_grad(false);
    const int scales_elems = (quantize::requires_offset(quant_type)) ? 2 : 1;
    auto scales = torch::empty({groups, scales_elems}, scales_options);

    auto output_options = at::TensorOptions()
                              .dtype(at::kChar)
                              .layout(at::kStrided)
                              .device(at::kCUDA)
                              .requires_grad(false);

    const int quantization_scalar = 8 / num_bits;
    const int64_t compressed_vals = at::numel(input_vals) / quantization_scalar;

    auto output = torch::empty({compressed_vals}, output_options);
    const int elems_per_group = at::numel(input_vals) / groups;
    
    // Check data type and launch appropriate kernel
    if (input_vals.scalar_type() == at::ScalarType::Half) {
        launch_swizzled_quant((int8_t*)output.data_ptr(),
                            (float*)scales.data_ptr(),
                            (__half*)input_vals.data_ptr(),
                            num_bits,
                            quant_type,
                            groups,
                            elems_per_group,
                            pipeline_size,
                            nodes,
                            devices_per_node,
                            at::cuda::getCurrentCUDAStream());
    } else if (input_vals.scalar_type() == at::ScalarType::Float) {
        launch_swizzled_quant((int8_t*)output.data_ptr(),
                            (float*)scales.data_ptr(),
                            (float*)input_vals.data_ptr(),
                            num_bits,
                            quant_type,
                            groups,
                            elems_per_group,
                            pipeline_size,
                            nodes,
                            devices_per_node,
                            at::cuda::getCurrentCUDAStream());
    }


    return {output, scales};
}

std::vector<at::Tensor> ds_swizzle_quant_ht(at::Tensor& input_vals,
                                         int groups,
                                         int num_bits,
                                         quantize::Type quant_type,
                                         int pipeline_size,
                                         int nodes,
                                         int devices_per_node)
{
    auto scales_options = at::TensorOptions()
                              .dtype(at::kFloat)
                              .layout(at::kStrided)
                              .device(at::kCUDA)
                              .requires_grad(false);
    const int scales_elems = (quantize::requires_offset(quant_type)) ? 2 : 1;
    auto scales = torch::empty({groups, scales_elems}, scales_options);

    auto output_options = at::TensorOptions()
                              .dtype(at::kChar)
                              .layout(at::kStrided)
                              .device(at::kCUDA)
                              .requires_grad(false);

    const int quantization_scalar = 8 / num_bits;
    const int64_t compressed_vals = at::numel(input_vals) / quantization_scalar;

    auto output = torch::empty({compressed_vals}, output_options);
    const int elems_per_group = at::numel(input_vals) / groups;
    
    // Check data type and launch appropriate kernel
    if (input_vals.scalar_type() == at::ScalarType::Half) {
        launch_swizzled_quant_ht((int8_t*)output.data_ptr(),
                            (float*)scales.data_ptr(),
                            (__half*)input_vals.data_ptr(),
                            num_bits,
                            quant_type,
                            groups,
                            elems_per_group,
                            pipeline_size,
                            nodes,
                            devices_per_node,
                            at::cuda::getCurrentCUDAStream());
    } else if (input_vals.scalar_type() == at::ScalarType::Float) {
        launch_swizzled_quant_ht((int8_t*)output.data_ptr(),
                            (float*)scales.data_ptr(),
                            (float*)input_vals.data_ptr(),
                            num_bits,
                            quant_type,
                            groups,
                            elems_per_group,
                            pipeline_size,
                            nodes,
                            devices_per_node,
                            at::cuda::getCurrentCUDAStream());
    }


    return {output, scales};
}

std::vector<at::Tensor> stochastic_quantize(at::Tensor& input_vals,
                                            int groups,
                                            int numBits,
                                            quantize::Type quantType)
{
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
        launch_quant((int8_t*)output.data_ptr(),
                    (float*)params.data_ptr(),
                    (__half*)input_vals.data_ptr(),
                    groups,
                    elems_per_group,
                    numBits,
                    quantType,
                    at::cuda::getCurrentCUDAStream());
        return {output, params};
    } else if (input_vals.scalar_type() == at::ScalarType::Float) {
        launch_quant((int8_t*)output.data_ptr(),
                    (float*)params.data_ptr(),
                    (float*)input_vals.data_ptr(),
                    groups,
                    elems_per_group,
                    numBits,
                    quantType,
                    at::cuda::getCurrentCUDAStream());
        return {output, params};
    } else if (input_vals.scalar_type() == at::ScalarType::BFloat16) {
        launch_quant((int8_t*)output.data_ptr(),
                    (float*)params.data_ptr(),
                    (__nv_bfloat16*)input_vals.data_ptr(),
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

std::vector<at::Tensor> stochastic_quantize_ht(at::Tensor& input_vals,
                                            int groups,
                                            int numBits,
                                            quantize::Type quantType)
{
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
        launch_quant_ht((int8_t*)output.data_ptr(),
                    (float*)params.data_ptr(),
                    (__half*)input_vals.data_ptr(),
                    groups,
                    elems_per_group,
                    numBits,
                    quantType,
                    at::cuda::getCurrentCUDAStream());
        return {output, params};
    } else if (input_vals.scalar_type() == at::ScalarType::Float) {
        launch_quant_ht((int8_t*)output.data_ptr(),
                    (float*)params.data_ptr(),
                    (float*)input_vals.data_ptr(),
                    groups,
                    elems_per_group,
                    numBits,
                    quantType,
                    at::cuda::getCurrentCUDAStream());
        return {output, params};
    } else if (input_vals.scalar_type() == at::ScalarType::BFloat16) {
        launch_quant_ht((int8_t*)output.data_ptr(),
                    (float*)params.data_ptr(),
                    (__nv_bfloat16*)input_vals.data_ptr(),
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

std::vector<at::Tensor> dequantize_reduce_quant(at::Tensor& input_vals,
                                            at::Tensor& input_scales,
                                            int in_groups,
                                            int in_num_bits,
                                            int out_num_bits,
                                            quantize::Type quant_type,
                                            int chunks)
{
    auto scales_options = at::TensorOptions()
                              .dtype(at::kFloat)
                              .layout(at::kStrided)
                              .device(at::kCUDA)
                              .requires_grad(false);
    const int scales_elems = (quantize::requires_offset(quant_type)) ? 2 : 1;
    const int out_groups = in_groups / chunks;
    auto scales = torch::empty({out_groups, scales_elems}, scales_options);

    auto output_options = at::TensorOptions()
                              .dtype(at::kChar)
                              .layout(at::kStrided)
                              .device(at::kCUDA)
                              .requires_grad(false);

    std::vector<long int> sz(input_vals.sizes().begin(), input_vals.sizes().end());
    sz[sz.size() - 1] = sz.back() / chunks ;
    sz[sz.size() - 1] = sz.back() * out_num_bits / in_num_bits ;
    const int64_t elems_per_in_tensor = at::numel(input_vals) / chunks;
    auto output = torch::empty(sz, output_options);

    const int elems_per_in_group = elems_per_in_tensor / (in_groups / chunks); // number of bytes per in group
    // const int elems_per_out_group = elems_per_in_tensor / out_groups; 
    const int elems_per_out_group = elems_per_in_group * out_num_bits / in_num_bits ; // number of bytes of per out group

    launch_dequant_reduce_quant((int8_t*)output.data_ptr(),
                                (float*)scales.data_ptr(),
                                (const int8_t*)input_vals.data_ptr(),
                                (const float*)input_scales.data_ptr(),
                                chunks,
                                in_num_bits,
                                out_num_bits,
                                quant_type,
                                out_groups,
                                elems_per_out_group,
                                elems_per_in_tensor,
                                in_groups / chunks,
                                elems_per_in_group,
                                at::cuda::getCurrentCUDAStream());
    return {output, scales};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    pybind11::enum_<quantize::Type>(m, "QuantizationType")
        .value("Symmetric", quantize::Type::Symmetric)
        .value("Asymmetric", quantize::Type::Asymmetric)
        .export_values();
    m.def("swizzle_quant", &ds_swizzle_quant);
    m.def("swizzle_quant_ht32", &ds_swizzle_quant_ht, "swizzle quantization with Hadamard Transfomr 32*32");
    m.def("dequantize_reduce_quant", &dequantize_reduce_quant);
    m.def("dequantize_reduce", &dequantize_reduce);
    m.def("dequantize_reduce_ht32", &dequantize_reduce_ht);
    m.def("stochastic_quantize", &stochastic_quantize);
    m.def("stochastic_quantize_ht32", &stochastic_quantize_ht);
    m.def("dequantize_half", &dequantize<__half>);
    m.def("dequantize_fp32", &dequantize<float>);
    m.def("dequantize_bf16", &dequantize<__nv_bfloat16>);

    // extern void init_hadamard(py::module_ &m);
    init_hadamard(m);

    // m.def("dequantize_int4_to_half_experimental",
    //       &dequantize_int4_to_half_experimental,
    //       "Dequantize int4 to half (experimental)");
    // m.def("dequantize_int8_to_half_experimental",
    //       &dequantize_int8_to_half_experimental,
    //       "Dequantize int8 to half (experimental)");
}
